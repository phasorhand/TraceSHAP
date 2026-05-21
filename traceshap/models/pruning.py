from dataclasses import dataclass
from datetime import datetime

from traceshap.models.enums import DecisionStatus, RiskLevel, ReplayCapability
from traceshap.models.outcome import StepAttribution


@dataclass(frozen=True)
class Savings:
    token_reduction: int
    cost_reduction: float
    latency_reduction_ms: int
    quality_impact_range: tuple[float, float]


@dataclass
class ValidationPlan:
    replay_required: bool
    replay_mode: ReplayCapability
    min_replay_count: int
    ab_test_recommended: bool
    human_review_required: bool


@dataclass
class PruneCandidate:
    target_type: str
    target_id: str
    evidence: list[StepAttribution]
    estimated_savings: Savings
    required_validation: ValidationPlan
    decision_status: DecisionStatus


@dataclass
class PruningReport:
    trace_id: str | None
    timestamp: datetime
    total_steps: int
    candidates: list[PruneCandidate]
    estimated_savings: Savings
    risk_assessment: RiskLevel
