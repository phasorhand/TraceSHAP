from datetime import datetime, timezone

from traceshap.models.trajectory import Trajectory
from traceshap.models.outcome import StepAttribution
from traceshap.models.enums import DecisionStatus, RiskLevel, ReplayCapability, Verdict
from traceshap.models.pruning import Savings, ValidationPlan, PruneCandidate, PruningReport
from traceshap.config import PruningConfig
from traceshap.pruning.safety import classify_step, is_first_or_last


class PruningAdvisor:
    def __init__(self, config: PruningConfig):
        self._config = config

    def analyze(
        self,
        trajectory: Trajectory,
        attributions: list[StepAttribution],
    ) -> PruningReport:
        attr_map = {a.step_id: a for a in attributions}
        candidates: list[PruneCandidate] = []

        for step in trajectory.steps:
            attr = attr_map.get(step.step_id)
            if attr is None:
                continue

            first_last = is_first_or_last(step.step_id, trajectory.steps)
            verdict = classify_step(attr, step, self._config, first_last)

            if verdict == Verdict.PRUNE_CANDIDATE:
                candidate = PruneCandidate(
                    target_type="step",
                    target_id=step.tool_name or step.step_type.value,
                    evidence=[attr],
                    estimated_savings=Savings(
                        token_reduction=step.tokens.total_tokens if step.tokens else 0,
                        cost_reduction=step.cost or 0.0,
                        latency_reduction_ms=step.duration_ms,
                        quality_impact_range=(
                            attr.confidence.lower if attr.confidence else 0.0,
                            attr.confidence.upper if attr.confidence else 0.0,
                        ),
                    ),
                    required_validation=self._build_validation_plan(step, attr),
                    decision_status=DecisionStatus.CANDIDATE,
                )
                candidates.append(candidate)

        total_savings = Savings(
            token_reduction=sum(c.estimated_savings.token_reduction for c in candidates),
            cost_reduction=sum(c.estimated_savings.cost_reduction for c in candidates),
            latency_reduction_ms=sum(c.estimated_savings.latency_reduction_ms for c in candidates),
            quality_impact_range=self._aggregate_quality_range(candidates),
        )

        return PruningReport(
            trace_id=trajectory.trace_id,
            timestamp=datetime.now(timezone.utc),
            total_steps=len(trajectory.steps),
            candidates=candidates,
            estimated_savings=total_savings,
            risk_assessment=self._assess_risk(candidates, trajectory),
        )

    @staticmethod
    def _build_validation_plan(step, attr: StepAttribution) -> ValidationPlan:
        replay_mode = ReplayCapability.RECORDED_IO_REPLAY
        if step.side_effect_class.is_safe_for_auto_replay():
            replay_mode = ReplayCapability.RECORDED_IO_REPLAY
        else:
            replay_mode = ReplayCapability.DRY_RUN_MOCKED

        ci_width = attr.confidence.width if attr.confidence else 1.0
        min_replays = max(5, int(20 * ci_width))

        return ValidationPlan(
            replay_required=True,
            replay_mode=replay_mode,
            min_replay_count=min_replays,
            ab_test_recommended=ci_width > 0.1,
            human_review_required=not step.side_effect_class.is_safe_for_auto_replay(),
        )

    @staticmethod
    def _aggregate_quality_range(candidates: list[PruneCandidate]) -> tuple[float, float]:
        if not candidates:
            return (0.0, 0.0)
        lower = sum(c.estimated_savings.quality_impact_range[0] for c in candidates)
        upper = sum(c.estimated_savings.quality_impact_range[1] for c in candidates)
        return (lower, upper)

    @staticmethod
    def _assess_risk(candidates: list[PruneCandidate], trajectory: Trajectory) -> RiskLevel:
        if not candidates:
            return RiskLevel.LOW
        ratio = len(candidates) / len(trajectory.steps) if trajectory.steps else 0
        if ratio > 0.3:
            return RiskLevel.HIGH
        if ratio > 0.1:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
