from __future__ import annotations

import math
from dataclasses import dataclass

from traceshap.models.step import CanonicalStep
from traceshap.models.trajectory import Trajectory
from traceshap.attribution.base import LayerResult


@dataclass
class CohortStats:
    step_key: str
    present_success: int
    present_total: int
    absent_success: int
    absent_total: int
    smoothing: float = 1.0

    @property
    def lift(self) -> float:
        p_present = (self.present_success + self.smoothing) / (self.present_total + 2 * self.smoothing)
        p_absent = (self.absent_success + self.smoothing) / (self.absent_total + 2 * self.smoothing)
        if p_absent == 0:
            return float("inf")
        return p_present / p_absent

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        z = 1.96 if confidence == 0.95 else 2.576
        log_lift = math.log(self.lift) if self.lift > 0 and self.lift != float("inf") else 0.0

        p1 = (self.present_success + self.smoothing) / (self.present_total + 2 * self.smoothing)
        p2 = (self.absent_success + self.smoothing) / (self.absent_total + 2 * self.smoothing)
        n1 = self.present_total + 2 * self.smoothing
        n2 = self.absent_total + 2 * self.smoothing

        se = math.sqrt((1 - p1) / (p1 * n1) + (1 - p2) / (p2 * n2)) if p1 > 0 and p2 > 0 else 1.0

        return (math.exp(log_lift - z * se), math.exp(log_lift + z * se))


def _step_key(step: CanonicalStep) -> str:
    return step.tool_name or step.step_type.value


class Layer1Lift:
    def __init__(
        self,
        min_support: int = 50,
        smoothing: float = 1.0,
        confidence_level: float = 0.95,
    ):
        self._min_support = min_support
        self._smoothing = smoothing
        self._confidence_level = confidence_level
        self._cohort_stats: dict[str, CohortStats] = {}

    @property
    def layer_id(self) -> int:
        return 1

    def fit(self, trajectories: list[Trajectory]) -> None:
        step_presence: dict[str, list[bool]] = {}
        outcomes: list[bool] = []

        for traj in trajectories:
            if traj.outcome is None or traj.outcome.success is None:
                continue
            outcomes.append(traj.outcome.success)
            present_keys = {_step_key(s) for s in traj.steps}
            all_keys = step_presence.keys() | present_keys
            for key in all_keys:
                step_presence.setdefault(key, []).append(key in present_keys)

        for idx in range(len(outcomes)):
            for key in step_presence:
                if len(step_presence[key]) <= idx:
                    step_presence[key].append(False)

        self._cohort_stats = {}
        for key, presences in step_presence.items():
            present_success = sum(1 for i, p in enumerate(presences) if p and i < len(outcomes) and outcomes[i])
            present_total = sum(1 for p in presences if p)
            absent_success = sum(1 for i, p in enumerate(presences) if not p and i < len(outcomes) and outcomes[i])
            absent_total = sum(1 for p in presences if not p)

            if present_total + absent_total >= self._min_support:
                self._cohort_stats[key] = CohortStats(
                    step_key=key,
                    present_success=present_success,
                    present_total=present_total,
                    absent_success=absent_success,
                    absent_total=absent_total,
                    smoothing=self._smoothing,
                )

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        results: list[LayerResult] = []

        for step in trajectory.steps:
            key = _step_key(step)
            stats = self._cohort_stats.get(key)

            if stats is None:
                results.append(LayerResult(
                    layer=1,
                    step_id=step.step_id,
                    quality_delta=0.0,
                    cost_delta=step.cost or 0.0,
                    latency_delta=step.duration_ms,
                    risk_delta=0.0,
                    confidence_lower=0.0,
                    confidence_upper=0.0,
                    evidence=f"insufficient support for '{key}'",
                ))
                continue

            lift = stats.lift
            ci = stats.confidence_interval(self._confidence_level)
            quality_delta = (lift - 1.0) * 0.5

            results.append(LayerResult(
                layer=1,
                step_id=step.step_id,
                quality_delta=quality_delta,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=(ci[0] - 1.0) * 0.5,
                confidence_upper=(ci[1] - 1.0) * 0.5,
                evidence=f"lift={lift:.3f} (CI: [{ci[0]:.3f}, {ci[1]:.3f}]), n_present={stats.present_total}, n_absent={stats.absent_total}",
            ))

        return results
