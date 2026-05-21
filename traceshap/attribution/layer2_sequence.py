from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from traceshap.models.step import CanonicalStep
from traceshap.models.enums import StepType
from traceshap.models.trajectory import Trajectory
from traceshap.attribution.base import LayerResult


class InterventionType(Enum):
    PREFIX_REMOVAL = "prefix_removal"
    CONTIGUOUS_REMOVAL = "contiguous_removal"
    RETRY_COLLAPSE = "retry_collapse"


@dataclass
class Intervention:
    intervention_type: InterventionType
    target_step_id: str
    removed_step_ids: list[str]
    resulting_sequence: list[str]


def _step_label(step: CanonicalStep) -> str:
    return step.tool_name or step.step_type.value


def generate_legal_interventions(
    steps: list[CanonicalStep],
    target: CanonicalStep,
) -> list[Intervention]:
    interventions: list[Intervention] = []
    idx = next((i for i, s in enumerate(steps) if s.step_id == target.step_id), None)
    if idx is None:
        return interventions

    labels = [_step_label(s) for s in steps]

    if target.attempt_index > 0:
        retry_ids = [s.step_id for s in steps
                     if s.tool_name == target.tool_name
                     and s.input_hash == target.input_hash
                     and s.attempt_index > 0]
        if retry_ids:
            remaining = [l for i, l in enumerate(labels)
                         if steps[i].step_id not in retry_ids]
            interventions.append(Intervention(
                intervention_type=InterventionType.RETRY_COLLAPSE,
                target_step_id=target.step_id,
                removed_step_ids=retry_ids,
                resulting_sequence=remaining,
            ))

    removed = [steps[idx].step_id]
    remaining = [l for i, l in enumerate(labels) if i != idx]
    interventions.append(Intervention(
        intervention_type=InterventionType.CONTIGUOUS_REMOVAL,
        target_step_id=target.step_id,
        removed_step_ids=removed,
        resulting_sequence=remaining,
    ))

    if idx == len(steps) - 1 and len(steps) > 1:
        remaining = labels[:idx]
        interventions.append(Intervention(
            intervention_type=InterventionType.PREFIX_REMOVAL,
            target_step_id=target.step_id,
            removed_step_ids=[target.step_id],
            resulting_sequence=remaining,
        ))

    return interventions


class TransitionModel:
    def __init__(self):
        self._transition_counts: Counter = Counter()
        self._state_counts: Counter = Counter()
        self._outcome_by_final: dict[str, list[float]] = {}
        self._global_success_rate: float = 0.5

    def fit(self, trajectories: list[Trajectory]) -> None:
        successes = 0
        total = 0
        for traj in trajectories:
            if traj.outcome is None or traj.outcome.quality_score is None:
                continue
            total += 1
            if traj.outcome.success:
                successes += 1

            labels = [_step_label(s) for s in traj.steps]
            for i in range(len(labels) - 1):
                self._transition_counts[(labels[i], labels[i + 1])] += 1
                self._state_counts[labels[i]] += 1

            if labels:
                final = labels[-1]
                self._outcome_by_final.setdefault(final, []).append(
                    traj.outcome.quality_score
                )

        self._global_success_rate = successes / total if total > 0 else 0.5

    def predict(self, sequence: list[str]) -> float:
        if not sequence:
            return self._global_success_rate

        log_prob = 0.0
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            count = self._transition_counts[pair]
            state_count = self._state_counts[sequence[i]]
            if state_count > 0:
                log_prob += math.log((count + 1) / (state_count + len(self._state_counts) + 1))

        final = sequence[-1]
        outcomes = self._outcome_by_final.get(final, [])
        if outcomes:
            base = sum(outcomes) / len(outcomes)
        else:
            base = self._global_success_rate

        transition_factor = math.exp(log_prob) if log_prob != 0 else 1.0
        return max(0.0, min(1.0, base * min(transition_factor, 2.0)))


class Layer2Sequence:
    def __init__(self, num_samples: int = 200):
        self._num_samples = num_samples
        self._model = TransitionModel()

    @property
    def layer_id(self) -> int:
        return 2

    def fit(self, trajectories: list[Trajectory]) -> None:
        self._model.fit(trajectories)

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        results: list[LayerResult] = []

        if trajectory.outcome is None or trajectory.outcome.quality_score is None:
            for step in trajectory.steps:
                results.append(LayerResult(
                    layer=2, step_id=step.step_id,
                    quality_delta=0.0, cost_delta=step.cost or 0.0,
                    latency_delta=step.duration_ms, risk_delta=0.0,
                    confidence_lower=0.0, confidence_upper=0.0,
                    evidence="no outcome available",
                ))
            return results

        full_labels = [_step_label(s) for s in trajectory.steps]
        factual_score = self._model.predict(full_labels)

        for step in trajectory.steps:
            interventions = generate_legal_interventions(trajectory.steps, step)
            if not interventions:
                results.append(LayerResult(
                    layer=2, step_id=step.step_id,
                    quality_delta=0.0, cost_delta=step.cost or 0.0,
                    latency_delta=step.duration_ms, risk_delta=0.0,
                    confidence_lower=0.0, confidence_upper=0.0,
                    evidence="no legal interventions",
                ))
                continue

            deltas: list[float] = []
            for intervention in interventions:
                counterfactual_score = self._model.predict(intervention.resulting_sequence)
                delta = counterfactual_score - factual_score
                deltas.append(delta)

            mean_delta = sum(deltas) / len(deltas)
            if len(deltas) > 1:
                variance = sum((d - mean_delta) ** 2 for d in deltas) / (len(deltas) - 1)
                se = math.sqrt(variance / len(deltas))
            else:
                se = abs(mean_delta) * 0.3

            results.append(LayerResult(
                layer=2,
                step_id=step.step_id,
                quality_delta=mean_delta,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=mean_delta - 1.96 * se,
                confidence_upper=mean_delta + 1.96 * se,
                evidence=f"sequence estimator: {len(interventions)} interventions, mean_delta={mean_delta:.4f}",
            ))

        return results
