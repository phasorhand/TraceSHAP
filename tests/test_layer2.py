import pytest
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.attribution.layer2_sequence import (
    Layer2Sequence, TransitionModel, generate_legal_interventions,
    InterventionType,
)


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          input_hash: str = "a", output_hash: str = "b",
          attempt_index: int = 0, offset_sec: int = 0) -> CanonicalStep:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=[f"s-{step_id}"],
        node_id=None, tool_name=name, step_type=step_type,
        attempt_index=attempt_index, loop_iteration=None,
        input_hash=input_hash, output_hash=output_hash,
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


class TestGenerateLegalInterventions:
    def test_retry_collapse(self):
        steps = [
            _step("s1", "search", attempt_index=0, offset_sec=0),
            _step("s2", "search", attempt_index=1, offset_sec=1),
            _step("s3", "search", attempt_index=2, offset_sec=2),
        ]
        interventions = generate_legal_interventions(steps, steps[2])
        types = [i.intervention_type for i in interventions]
        assert InterventionType.RETRY_COLLAPSE in types

    def test_contiguous_removal(self):
        steps = [
            _step("s1", "plan", StepType.DECISION, offset_sec=0),
            _step("s2", "search", offset_sec=1),
            _step("s3", "search", offset_sec=2),
            _step("s4", "summarize", StepType.DECISION, offset_sec=3),
        ]
        interventions = generate_legal_interventions(steps, steps[1])
        types = [i.intervention_type for i in interventions]
        assert InterventionType.CONTIGUOUS_REMOVAL in types

    def test_prefix_removal_for_last_step(self):
        steps = [
            _step("s1", "plan", StepType.DECISION, offset_sec=0),
            _step("s2", "search", offset_sec=1),
            _step("s3", "summarize", StepType.DECISION, offset_sec=2),
        ]
        interventions = generate_legal_interventions(steps, steps[2])
        types = [i.intervention_type for i in interventions]
        assert InterventionType.PREFIX_REMOVAL in types


class TestTransitionModel:
    def test_fit_and_predict(self):
        trajs = []
        for i in range(50):
            steps = [
                _step(f"s{i}-1", "plan", StepType.DECISION),
                _step(f"s{i}-2", "search"),
                _step(f"s{i}-3", "summarize", StepType.DECISION),
            ]
            trajs.append(_trajectory(f"t{i}", steps,
                                     success=i < 40, quality=0.9 if i < 40 else 0.3))

        model = TransitionModel()
        model.fit(trajs)

        full_seq = ["decision", "action", "decision"]
        pred_full = model.predict(full_seq)
        assert 0.0 <= pred_full <= 1.0

        short_seq = ["decision", "decision"]
        pred_short = model.predict(short_seq)
        assert 0.0 <= pred_short <= 1.0


class TestLayer2Sequence:
    async def test_analyze_returns_results(self):
        trajs = []
        for i in range(50):
            steps = [
                _step(f"s{i}-1", "plan", StepType.DECISION),
                _step(f"s{i}-2", "search"),
                _step(f"s{i}-3", "summarize", StepType.DECISION),
            ]
            trajs.append(_trajectory(f"t{i}", steps,
                                     success=i < 40, quality=0.9 if i < 40 else 0.3))

        layer = Layer2Sequence(num_samples=20)
        layer.fit(trajs)
        results = await layer.analyze(trajs[0])
        assert len(results) == 3
        assert all(r.layer == 2 for r in results)

    async def test_no_outcome_returns_zero(self):
        traj = Trajectory(
            trace_id="t-none", spans=[],
            steps=[_step("s1", "tool")],
            span_tree=SpanNode(span_id="root"),
            outcome=None,
            metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
        )
        layer = Layer2Sequence()
        layer.fit([traj])
        results = await layer.analyze(traj)
        assert all(r.quality_delta == 0.0 for r in results)
