"""Tests for Layer3Replay attribution layer (Task 5)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from traceshap.attribution.replay.capsule import RecordedIO, ReplayCapsule
from traceshap.attribution.replay.engine import ReplayEngine
from traceshap.attribution.layer3_replay import Layer3Replay
from traceshap.models import (
    CanonicalStep,
    StepType,
    SideEffect,
    TokenUsage,
    SpanNode,
    TrajectoryMeta,
    Trajectory,
    Outcome,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _step(step_id: str, offset_sec: int = 0) -> CanonicalStep:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id,
        raw_span_ids=[f"span-{step_id}"],
        node_id=None,
        tool_name=f"tool_{step_id}",
        step_type=StepType.ACTION,
        attempt_index=0,
        loop_iteration=None,
        input_hash=f"in-{step_id}",
        output_hash=f"out-{step_id}",
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.9,
        tokens=TokenUsage(10, 20, 30),
        cost=0.001,
        start_time=start,
        end_time=end,
    )


def _recorded_io(step_id: str) -> RecordedIO:
    return RecordedIO(
        step_id=step_id,
        tool_name=f"tool_{step_id}",
        input_hash=f"in-{step_id}",
        input_data={"q": step_id},
        output_data={"result": step_id},
        side_effect_class="network_io",
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )


def _make_trajectory(trace_id: str, step_ids: list[str]) -> Trajectory:
    steps = [_step(sid, i) for i, sid in enumerate(step_ids)]
    return Trajectory(
        trace_id=trace_id,
        spans=[],
        steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(
            success=True,
            quality_score=0.9,
            token_cost=100,
            latency_ms=3000,
            custom_metrics={},
        ),
        metadata=TrajectoryMeta(
            framework="langgraph",
            agent_name="test-agent",
            agent_version="v1",
            task_type="qa",
        ),
    )


def _make_capsule(trace_id: str, step_ids: list[str]) -> ReplayCapsule:
    return ReplayCapsule(
        capsule_id="cap-001",
        trace_id=trace_id,
        created_at=datetime(2026, 1, 1, 12, 30, 0),
        model_id="gpt-4o",
        model_config={"temperature": 0.0},
        recorded_ios=[_recorded_io(sid) for sid in step_ids],
    )


TRACE_ID = "trace-layer3"
STEP_IDS = ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# Test: layer_id
# ---------------------------------------------------------------------------

class TestLayer3LayerId:
    """test_layer3_layer_id — verify layer_id == 3."""

    def test_layer3_layer_id(self):
        engine = ReplayEngine()
        layer = Layer3Replay(engine)
        assert layer.layer_id == 3


# ---------------------------------------------------------------------------
# Test: analyze with registered capsule
# ---------------------------------------------------------------------------

class TestLayer3AnalyzeWithCapsule:
    """test_layer3_analyze_with_capsule — registered capsule returns len(steps) results all with layer=3."""

    @pytest.mark.asyncio
    async def test_layer3_analyze_with_capsule(self):
        engine = ReplayEngine()
        capsule = _make_capsule(TRACE_ID, STEP_IDS)
        engine.register_capsule(capsule)

        trajectory = _make_trajectory(TRACE_ID, STEP_IDS)
        layer = Layer3Replay(engine)
        results = await layer.analyze(trajectory)

        assert len(results) == len(STEP_IDS)
        assert all(r.layer == 3 for r in results)

    @pytest.mark.asyncio
    async def test_layer3_step_ids_match(self):
        engine = ReplayEngine()
        capsule = _make_capsule(TRACE_ID, STEP_IDS)
        engine.register_capsule(capsule)

        trajectory = _make_trajectory(TRACE_ID, STEP_IDS)
        layer = Layer3Replay(engine)
        results = await layer.analyze(trajectory)

        result_step_ids = {r.step_id for r in results}
        assert result_step_ids == set(STEP_IDS)


# ---------------------------------------------------------------------------
# Test: analyze without capsule (fallback)
# ---------------------------------------------------------------------------

class TestLayer3AnalyzeWithoutCapsule:
    """test_layer3_analyze_without_capsule — without capsule, evidence='no replay capsule available'."""

    @pytest.mark.asyncio
    async def test_layer3_analyze_without_capsule(self):
        engine = ReplayEngine()  # no capsule registered

        trajectory = _make_trajectory(TRACE_ID, STEP_IDS)
        layer = Layer3Replay(engine)
        results = await layer.analyze(trajectory)

        assert len(results) == len(STEP_IDS)
        assert all(r.layer == 3 for r in results)
        assert all(r.evidence == "no replay capsule available" for r in results)

    @pytest.mark.asyncio
    async def test_layer3_fallback_zero_deltas(self):
        engine = ReplayEngine()  # no capsule registered

        trajectory = _make_trajectory(TRACE_ID, STEP_IDS)
        layer = Layer3Replay(engine)
        results = await layer.analyze(trajectory)

        assert all(r.quality_delta == 0.0 for r in results)


# ---------------------------------------------------------------------------
# Test: SHAP values sum approximately to full_value - empty_value
# ---------------------------------------------------------------------------

class TestLayer3ResultsSumApproximately:
    """test_layer3_results_sum_approximately — SHAP values sum ≈ full_value - empty_value."""

    @pytest.mark.asyncio
    async def test_layer3_results_sum_approximately(self):
        engine = ReplayEngine()
        capsule = _make_capsule(TRACE_ID, STEP_IDS)
        engine.register_capsule(capsule)

        trajectory = _make_trajectory(TRACE_ID, STEP_IDS)
        layer = Layer3Replay(engine)
        results = await layer.analyze(trajectory)

        shap_sum = sum(r.quality_delta for r in results)

        # Compute full_value (no ablation) and empty_value (all ablated)
        full_outcome = await engine.replay_without(capsule, [])
        empty_outcome = await engine.replay_without(capsule, STEP_IDS)
        expected_gain = full_outcome.quality_score - empty_outcome.quality_score

        assert shap_sum == pytest.approx(expected_gain, abs=1e-6)
