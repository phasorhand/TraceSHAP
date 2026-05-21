from dataclasses import dataclass

from traceshap.models.enums import Verdict


@dataclass
class Outcome:
    success: bool | None
    quality_score: float | None
    token_cost: int
    latency_ms: int
    custom_metrics: dict
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    score_confidence: float | None = None
    label_delay_ms: int | None = None


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    point: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass
class CalibrationMetrics:
    auc: float | None = None
    rmse: float | None = None
    coverage: float | None = None
    ood_score: float | None = None


@dataclass
class StepAttribution:
    step_id: str
    step_name: str
    node_id: str | None
    quality_delta: float | None
    cost_delta: float | None
    latency_delta: float | None
    risk_delta: float | None
    layer_scores: dict[int, float]
    confidence: ConfidenceInterval | None
    verdict: Verdict
    causal_hypothesis: object | None
    evidence: list[str]
    calibration: CalibrationMetrics | None
