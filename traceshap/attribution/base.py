from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from traceshap.models.outcome import ConfidenceInterval, StepAttribution, CalibrationMetrics
from traceshap.models.enums import Verdict
from traceshap.models.trajectory import Trajectory


@dataclass
class LayerResult:
    layer: int
    step_id: str
    quality_delta: float
    cost_delta: float
    latency_delta: float
    risk_delta: float
    confidence_lower: float
    confidence_upper: float
    evidence: str


class AttributionLayer(Protocol):
    @property
    def layer_id(self) -> int: ...

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]: ...


def merge_layer_results(
    step_id: str,
    step_name: str,
    node_id: str | None,
    results: list[LayerResult],
) -> StepAttribution:
    if not results:
        raise ValueError("Cannot merge empty results")

    highest = max(results, key=lambda r: r.layer)
    layer_scores = {r.layer: r.quality_delta for r in results}

    return StepAttribution(
        step_id=step_id,
        step_name=step_name,
        node_id=node_id,
        quality_delta=highest.quality_delta,
        cost_delta=highest.cost_delta,
        latency_delta=highest.latency_delta,
        risk_delta=highest.risk_delta,
        layer_scores=layer_scores,
        confidence=ConfidenceInterval(
            lower=highest.confidence_lower,
            point=highest.quality_delta,
            upper=highest.confidence_upper,
        ),
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        causal_hypothesis=None,
        evidence=[r.evidence for r in results],
        calibration=None,
    )
