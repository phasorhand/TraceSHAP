from __future__ import annotations

from traceshap.models.trajectory import Trajectory
from traceshap.models.outcome import StepAttribution, ConfidenceInterval
from traceshap.models.enums import Verdict
from traceshap.attribution.base import AttributionLayer, LayerResult, merge_layer_results


class AttributionEngine:
    def __init__(self, layers: list[AttributionLayer]):
        self._layers = sorted(layers, key=lambda l: l.layer_id)

    async def analyze(self, trajectory: Trajectory) -> list[StepAttribution]:
        step_results: dict[str, list[LayerResult]] = {}

        for layer in self._layers:
            results = await layer.analyze(trajectory)
            for result in results:
                step_results.setdefault(result.step_id, []).append(result)

        attributions: list[StepAttribution] = []
        for step in trajectory.steps:
            results = step_results.get(step.step_id, [])
            if not results:
                attributions.append(StepAttribution(
                    step_id=step.step_id,
                    step_name=step.tool_name or step.step_type.value,
                    node_id=step.node_id,
                    quality_delta=None,
                    cost_delta=step.cost,
                    latency_delta=step.duration_ms,
                    risk_delta=None,
                    layer_scores={},
                    confidence=None,
                    verdict=Verdict.INSUFFICIENT_EVIDENCE,
                    causal_hypothesis=None,
                    evidence=[],
                    calibration=None,
                ))
                continue

            merged = merge_layer_results(
                step_id=step.step_id,
                step_name=step.tool_name or step.step_type.value,
                node_id=step.node_id,
                results=results,
            )

            merged.verdict = self._classify_verdict(merged, trajectory)
            attributions.append(merged)

        return attributions

    @staticmethod
    def _classify_verdict(attr: StepAttribution, trajectory: Trajectory) -> Verdict:
        if trajectory.outcome is None:
            return Verdict.INSUFFICIENT_EVIDENCE
        if attr.confidence is None or attr.quality_delta is None:
            return Verdict.INSUFFICIENT_EVIDENCE
        return Verdict.REVIEW
