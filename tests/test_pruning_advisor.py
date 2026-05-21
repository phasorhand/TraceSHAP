import pytest
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome, Verdict,
    DecisionStatus, RiskLevel,
)
from traceshap.models.outcome import StepAttribution, ConfidenceInterval
from traceshap.models.pruning import Savings, PruneCandidate, PruningReport
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          cost: float = 0.001) -> CanonicalStep:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=["s1"], node_id=None,
        tool_name=name, step_type=step_type, attempt_index=0,
        loop_iteration=None, input_hash="a", output_hash="b",
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=cost,
        start_time=now, end_time=later,
    )


def _attr(step_id: str, quality_delta: float, ci_lower: float, ci_upper: float,
          cost_delta: float = 0.001) -> StepAttribution:
    return StepAttribution(
        step_id=step_id, step_name="test", node_id=None,
        quality_delta=quality_delta, cost_delta=cost_delta,
        latency_delta=1000, risk_delta=0.0,
        layer_scores={0: quality_delta},
        confidence=ConfidenceInterval(lower=ci_lower, point=quality_delta, upper=ci_upper),
        verdict=Verdict.REVIEW,
        causal_hypothesis=None, evidence=["test evidence"], calibration=None,
    )


def _trajectory(steps: list[CanonicalStep]) -> Trajectory:
    return Trajectory(
        trace_id="t1", spans=[], steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=True, quality_score=0.9, token_cost=100,
                        latency_ms=5000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
    )


class TestPruningAdvisor:
    def test_generates_report(self):
        steps = [
            _step("s1", "plan", StepType.DECISION),
            _step("s2", "search", cost=0.005),
            _step("s3", "summarize", StepType.DECISION),
        ]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
            _attr("s2", quality_delta=-0.01, ci_lower=-0.03, ci_upper=0.01,
                  cost_delta=0.005),
            _attr("s3", quality_delta=-0.15, ci_lower=-0.20, ci_upper=-0.10),
        ]
        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10,
                               protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert isinstance(report, PruningReport)
        assert report.total_steps == 3

    def test_prune_candidate_found(self):
        steps = [
            _step("s1", "plan", StepType.DECISION),
            _step("s2", "useless_tool", cost=0.005),
            _step("s3", "summarize", StepType.DECISION),
        ]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
            _attr("s2", quality_delta=-0.01, ci_lower=-0.02, ci_upper=0.01,
                  cost_delta=0.005),
            _attr("s3", quality_delta=-0.15, ci_lower=-0.20, ci_upper=-0.10),
        ]
        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10,
                               protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert len(report.candidates) >= 1
        candidate = report.candidates[0]
        assert candidate.decision_status == DecisionStatus.CANDIDATE
        assert candidate.estimated_savings.cost_reduction > 0

    def test_no_candidates_when_all_important(self):
        steps = [
            _step("s1", "plan", StepType.DECISION),
            _step("s2", "critical_tool", cost=0.005),
            _step("s3", "summarize", StepType.DECISION),
        ]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.3, ci_lower=-0.35, ci_upper=-0.25),
            _attr("s2", quality_delta=-0.25, ci_lower=-0.30, ci_upper=-0.20),
            _attr("s3", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
        ]
        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10,
                               protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert len(report.candidates) == 0

    def test_validation_plan_generated(self):
        steps = [
            _step("s1", "plan", StepType.DECISION),
            _step("s2", "useless_tool", cost=0.005),
            _step("s3", "summarize", StepType.DECISION),
        ]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
            _attr("s2", quality_delta=0.0, ci_lower=-0.01, ci_upper=0.01,
                  cost_delta=0.005),
            _attr("s3", quality_delta=-0.15, ci_lower=-0.20, ci_upper=-0.10),
        ]
        config = PruningConfig(prune_epsilon=0.05, protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert len(report.candidates) >= 1
        vplan = report.candidates[0].required_validation
        assert vplan.replay_required is True
        assert vplan.min_replay_count > 0

    def test_risk_assessment(self):
        steps = [_step("s1", "plan", StepType.DECISION), _step("s2", "tool")]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
            _attr("s2", quality_delta=0.0, ci_lower=-0.01, ci_upper=0.01,
                  cost_delta=0.005),
        ]
        config = PruningConfig(protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert report.risk_assessment in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)
