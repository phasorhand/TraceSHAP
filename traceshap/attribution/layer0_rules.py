from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from traceshap.models.step import CanonicalStep
from traceshap.models.trajectory import Trajectory
from traceshap.attribution.base import LayerResult


@dataclass
class RuleVerdict:
    step_id: str
    rule_name: str
    severity: float
    recommendation: str


class RepetitionRule:
    def __init__(self, threshold: int = 3, similarity: float = 0.9):
        self.threshold = threshold
        self.similarity = similarity

    def evaluate(self, steps: list[CanonicalStep]) -> list[RuleVerdict]:
        verdicts: list[RuleVerdict] = []
        groups: dict[str, list[CanonicalStep]] = {}
        for step in steps:
            key = f"{step.tool_name or step.step_type.value}:{step.input_hash}"
            groups.setdefault(key, []).append(step)

        for key, group in groups.items():
            if len(group) >= self.threshold:
                for step in group[self.threshold - 1:]:
                    verdicts.append(RuleVerdict(
                        step_id=step.step_id,
                        rule_name="repetition",
                        severity=min(1.0, len(group) / (self.threshold * 2)),
                        recommendation=f"Excessive repetition ({len(group)} attempts)",
                    ))
        return verdicts


class NoOpRule:
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold

    def evaluate(self, steps: list[CanonicalStep]) -> list[RuleVerdict]:
        verdicts: list[RuleVerdict] = []
        for step in steps:
            if step.input_hash == step.output_hash:
                verdicts.append(RuleVerdict(
                    step_id=step.step_id,
                    rule_name="no_op",
                    severity=0.8,
                    recommendation=f"No-op detected: input equals output",
                ))
        return verdicts


class LoopDetectionRule:
    def __init__(self, max_cycle: int = 2):
        self.max_cycle = max_cycle

    def evaluate(self, steps: list[CanonicalStep]) -> list[RuleVerdict]:
        verdicts: list[RuleVerdict] = []
        names = [s.tool_name or s.step_type.value for s in steps]

        for cycle_len in range(1, len(names) // 2 + 1):
            for start in range(len(names) - cycle_len * (self.max_cycle + 1) + 1):
                pattern = names[start:start + cycle_len]
                repeat_count = 0
                pos = start
                while pos + cycle_len <= len(names):
                    if names[pos:pos + cycle_len] == pattern:
                        repeat_count += 1
                        pos += cycle_len
                    else:
                        break
                if repeat_count > self.max_cycle:
                    for i in range(start + cycle_len * self.max_cycle, start + cycle_len * repeat_count):
                        if i < len(steps):
                            verdicts.append(RuleVerdict(
                                step_id=steps[i].step_id,
                                rule_name="loop",
                                severity=min(1.0, repeat_count / (self.max_cycle * 2)),
                                recommendation=f"Loop detected: pattern repeated {repeat_count} times (max {self.max_cycle})",
                            ))
                    break
            else:
                continue
            break

        return verdicts


class Layer0Rules:
    def __init__(self, rules: list | None = None):
        self._rules = rules or [
            RepetitionRule(threshold=3),
            NoOpRule(similarity_threshold=0.95),
            LoopDetectionRule(max_cycle=2),
        ]

    @property
    def layer_id(self) -> int:
        return 0

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        all_verdicts: dict[str, list[RuleVerdict]] = {}
        for rule in self._rules:
            for verdict in rule.evaluate(trajectory.steps):
                all_verdicts.setdefault(verdict.step_id, []).append(verdict)

        results: list[LayerResult] = []
        for step in trajectory.steps:
            verdicts = all_verdicts.get(step.step_id, [])
            max_severity = max((v.severity for v in verdicts), default=0.0)
            evidence_parts = [f"{v.rule_name}: {v.recommendation}" for v in verdicts]
            evidence = "; ".join(evidence_parts) if evidence_parts else "no rule violation"

            results.append(LayerResult(
                layer=0,
                step_id=step.step_id,
                quality_delta=-max_severity * 0.1 if max_severity > 0 else 0.0,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=-max_severity * 0.15 if max_severity > 0 else 0.0,
                confidence_upper=-max_severity * 0.05 if max_severity > 0 else 0.0,
                evidence=evidence,
            ))

        return results
