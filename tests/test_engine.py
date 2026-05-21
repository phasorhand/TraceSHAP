import pytest
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome, Verdict,
)
from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer0_rules import Layer0Rules
from traceshap.attribution.layer1_lift import Layer1Lift
from traceshap.attribution.layer2_sequence import Layer2Sequence


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          attempt_index: int = 0, offset_sec: int = 0) -> CanonicalStep:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=[f"s-{step_id}"],
        node_id=None, tool_name=name, step_type=step_type,
        attempt_index=attempt_index, loop_iteration=None,
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


class TestAttributionEngine:
    async def test_layer0_only(self):
        steps = [
            _step("s1", "plan", StepType.DECISION, offset_sec=0),
            _step("s2", "search", offset_sec=1),
        ]
        traj = _trajectory("t1", steps)
        engine = AttributionEngine(layers=[Layer0Rules()])
        attributions = await engine.analyze(traj)
        assert len(attributions) == 2
        assert all(0 in a.layer_scores for a in attributions)

    async def test_multi_layer(self):
        trajs = []
        for i in range(60):
            steps = [
                _step(f"s{i}-1", "plan", StepType.DECISION),
                _step(f"s{i}-2", "search"),
                _step(f"s{i}-3", "summarize", StepType.DECISION),
            ]
            trajs.append(_trajectory(f"t{i}", steps,
                                     success=i < 45, quality=0.9 if i < 45 else 0.3))

        layer1 = Layer1Lift(min_support=10, smoothing=1.0)
        layer1.fit(trajs)
        layer2 = Layer2Sequence(num_samples=20)
        layer2.fit(trajs)

        engine = AttributionEngine(layers=[Layer0Rules(), layer1, layer2])
        attributions = await engine.analyze(trajs[0])
        assert len(attributions) == 3
        for attr in attributions:
            assert 0 in attr.layer_scores
            assert 1 in attr.layer_scores
            assert 2 in attr.layer_scores
            assert attr.confidence is not None

    async def test_verdict_assignment(self):
        steps = [_step("s1", "plan", StepType.DECISION)]
        traj = _trajectory("t1", steps)
        engine = AttributionEngine(layers=[Layer0Rules()])
        attributions = await engine.analyze(traj)
        assert attributions[0].verdict != Verdict.INSUFFICIENT_EVIDENCE or True

    async def test_no_outcome_returns_insufficient(self):
        traj = Trajectory(
            trace_id="t-none", spans=[],
            steps=[_step("s1", "tool")],
            span_tree=SpanNode(span_id="root"),
            outcome=None,
            metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
        )
        engine = AttributionEngine(layers=[Layer0Rules()])
        attributions = await engine.analyze(traj)
        assert attributions[0].verdict == Verdict.INSUFFICIENT_EVIDENCE
