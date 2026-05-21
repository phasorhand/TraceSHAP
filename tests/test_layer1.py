import pytest
import math
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.attribution.layer1_lift import Layer1Lift, CohortStats


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          offset_sec: int = 0) -> CanonicalStep:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=[f"s-{step_id}"],
        node_id=None, tool_name=name, step_type=step_type,
        attempt_index=0, loop_iteration=None,
        input_hash="a", output_hash="b",
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=0.001,
        start_time=start, end_time=end,
    )


def _trajectory(trace_id: str, steps: list[CanonicalStep],
                success: bool = True, quality: float = 0.9) -> Trajectory:
    return Trajectory(
        trace_id=trace_id, spans=[], steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=success, quality_score=quality,
                        token_cost=100, latency_ms=5000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test",
                                agent_version="v1", task_type="qa"),
    )


class TestCohortStats:
    def test_lift_positive(self):
        stats = CohortStats(
            step_key="search_web",
            present_success=80, present_total=100,
            absent_success=50, absent_total=100,
            smoothing=0.0,
        )
        assert stats.lift > 1.0

    def test_lift_negative(self):
        stats = CohortStats(
            step_key="bad_tool",
            present_success=20, present_total=100,
            absent_success=60, absent_total=100,
            smoothing=0.0,
        )
        assert stats.lift < 1.0

    def test_lift_with_smoothing(self):
        stats = CohortStats(
            step_key="tool",
            present_success=5, present_total=5,
            absent_success=0, absent_total=5,
            smoothing=1.0,
        )
        assert stats.lift > 1.0
        assert stats.lift < float("inf")

    def test_confidence_interval(self):
        stats = CohortStats(
            step_key="tool",
            present_success=80, present_total=100,
            absent_success=50, absent_total=100,
            smoothing=0.0,
        )
        ci = stats.confidence_interval(confidence=0.95)
        assert ci[0] < stats.lift
        assert ci[1] > stats.lift


class TestLayer1Lift:
    async def test_analyze_with_cohort(self):
        trajs = []
        for i in range(60):
            has_search = i < 40
            success = (i < 32) if has_search else (i < 50)
            steps = [_step(f"s{i}-1", "plan", StepType.DECISION)]
            if has_search:
                steps.append(_step(f"s{i}-2", "search_web"))
            trajs.append(_trajectory(f"t{i}", steps, success=success,
                                     quality=0.9 if success else 0.3))

        layer = Layer1Lift(min_support=10, smoothing=1.0, confidence_level=0.95)
        layer.fit(trajs)
        results = await layer.analyze(trajs[0])
        assert len(results) > 0
        assert all(r.layer == 1 for r in results)

    async def test_insufficient_support(self):
        trajs = [_trajectory(f"t{i}", [_step(f"s{i}", "rare_tool")]) for i in range(3)]
        layer = Layer1Lift(min_support=50)
        layer.fit(trajs)
        results = await layer.analyze(trajs[0])
        assert all(r.quality_delta == 0.0 for r in results)

    async def test_no_outcome_skipped(self):
        traj = Trajectory(
            trace_id="t-no-outcome", spans=[],
            steps=[_step("s1", "tool")],
            span_tree=SpanNode(span_id="root"),
            outcome=None,
            metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
        )
        layer = Layer1Lift(min_support=5)
        layer.fit([traj])
        results = await layer.analyze(traj)
        assert all(r.quality_delta == 0.0 for r in results)
