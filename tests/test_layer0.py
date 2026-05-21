import pytest
from datetime import datetime, timezone

from traceshap.models import (
    TraceSHAPSpan, SpanKind, TokenUsage, CanonicalStep, StepType,
    SideEffect, SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.attribution.layer0_rules import (
    Layer0Rules, RepetitionRule, LoopDetectionRule, NoOpRule, RuleVerdict,
)


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          input_hash: str = "a", output_hash: str = "b",
          attempt_index: int = 0, cost: float = 0.001,
          offset_sec: int = 0) -> CanonicalStep:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=[f"s-{step_id}"],
        node_id=None, tool_name=name, step_type=step_type,
        attempt_index=attempt_index, loop_iteration=None,
        input_hash=input_hash, output_hash=output_hash,
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=cost,
        start_time=start, end_time=end,
    )


def _trajectory(steps: list[CanonicalStep]) -> Trajectory:
    return Trajectory(
        trace_id="t1", spans=[], steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=True, quality_score=0.9, token_cost=100,
                        latency_ms=5000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
    )


class TestRepetitionRule:
    async def test_detects_retries(self):
        steps = [
            _step("s1", "search", input_hash="x", attempt_index=0, offset_sec=0),
            _step("s2", "search", input_hash="x", attempt_index=1, offset_sec=1),
            _step("s3", "search", input_hash="x", attempt_index=2, offset_sec=2),
        ]
        rule = RepetitionRule(threshold=2)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) >= 1

    async def test_no_flag_below_threshold(self):
        steps = [
            _step("s1", "search", input_hash="x", attempt_index=0),
        ]
        rule = RepetitionRule(threshold=2)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) == 0


class TestNoOpRule:
    async def test_detects_noop(self):
        steps = [
            _step("s1", "transform", input_hash="abc", output_hash="abc"),
        ]
        rule = NoOpRule(similarity_threshold=1.0)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) == 1
        assert "no-op" in flagged[0].recommendation.lower()

    async def test_no_flag_different_output(self):
        steps = [
            _step("s1", "transform", input_hash="abc", output_hash="def"),
        ]
        rule = NoOpRule(similarity_threshold=1.0)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) == 0


class TestLoopDetectionRule:
    async def test_detects_loop(self):
        steps = [
            _step("s1", "A", offset_sec=0),
            _step("s2", "B", offset_sec=1),
            _step("s3", "A", offset_sec=2),
            _step("s4", "B", offset_sec=3),
            _step("s5", "A", offset_sec=4),
            _step("s6", "B", offset_sec=5),
        ]
        rule = LoopDetectionRule(max_cycle=2)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) >= 1


class TestLayer0Rules:
    async def test_analyze_returns_layer_results(self):
        steps = [
            _step("s1", "search", input_hash="x", attempt_index=0, offset_sec=0),
            _step("s2", "search", input_hash="x", attempt_index=1, offset_sec=1),
            _step("s3", "search", input_hash="x", attempt_index=2, offset_sec=2),
        ]
        traj = _trajectory(steps)
        layer = Layer0Rules()
        results = await layer.analyze(traj)
        assert len(results) == 3
        assert all(r.layer == 0 for r in results)

    async def test_clean_trajectory_no_flags(self):
        steps = [
            _step("s1", "plan", step_type=StepType.DECISION, offset_sec=0),
            _step("s2", "search", offset_sec=1),
            _step("s3", "summarize", step_type=StepType.DECISION, offset_sec=2),
        ]
        traj = _trajectory(steps)
        layer = Layer0Rules()
        results = await layer.analyze(traj)
        assert all(r.quality_delta == 0.0 for r in results)
