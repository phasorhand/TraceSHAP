"""Integration tests for Layer3Replay + Layer4Causal with AttributionEngine (Task 8)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from traceshap.attribution.causal.graph_builder import TrajectoryGraphBuilder
from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer3_replay import Layer3Replay
from traceshap.attribution.layer4_causal import Layer4Causal
from traceshap.attribution.replay.capsule import ReplayCapsule, RecordedIO
from traceshap.attribution.replay.engine import ReplayEngine
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


TRACE_ID = "trace-layer34-integration"
STEP_IDS = ["s1", "s2", "s3", "s4"]


# ---------------------------------------------------------------------------
# Test: engine with layer3 and layer4
# ---------------------------------------------------------------------------

class TestEngineWithLayer3AndLayer4:
    """test_engine_with_layer3_and_layer4 — AttributionEngine with both layers,
    verify layer_scores has both 3 and 4."""

    @pytest.mark.asyncio
    async def test_engine_with_layer3_and_layer4(self):
        # Set up Layer 3
        replay_engine = ReplayEngine()
        capsule = _make_capsule(TRACE_ID, STEP_IDS)
        replay_engine.register_capsule(capsule)
        layer3 = Layer3Replay(replay_engine)

        # Set up Layer 4
        graph_builder = TrajectoryGraphBuilder()
        layer4 = Layer4Causal(graph_builder)

        engine = AttributionEngine(layers=[layer3, layer4])
        trajectory = _make_trajectory(TRACE_ID, STEP_IDS)

        attributions = await engine.analyze(trajectory)

        assert len(attributions) == len(STEP_IDS)
        for attr in attributions:
            assert 3 in attr.layer_scores, f"Layer 3 missing from step {attr.step_id}"
            assert 4 in attr.layer_scores, f"Layer 4 missing from step {attr.step_id}"


# ---------------------------------------------------------------------------
# Test: engine layer4 only
# ---------------------------------------------------------------------------

class TestEngineLayer4Only:
    """test_engine_layer4_only — Only Layer 4, verify layer_scores has 4 but not 3."""

    @pytest.mark.asyncio
    async def test_engine_layer4_only(self):
        graph_builder = TrajectoryGraphBuilder()
        layer4 = Layer4Causal(graph_builder)

        engine = AttributionEngine(layers=[layer4])
        trajectory = _make_trajectory(TRACE_ID, STEP_IDS)

        attributions = await engine.analyze(trajectory)

        assert len(attributions) == len(STEP_IDS)
        for attr in attributions:
            assert 4 in attr.layer_scores, f"Layer 4 missing from step {attr.step_id}"
            assert 3 not in attr.layer_scores, f"Layer 3 should not be present for step {attr.step_id}"


# ---------------------------------------------------------------------------
# Test: layer4 hypotheses API
# ---------------------------------------------------------------------------

class TestLayer4HypothesesApi:
    """test_layer4_hypotheses_api — Call build_hypotheses, verify all are "associational"."""

    def test_layer4_hypotheses_api(self):
        graph_builder = TrajectoryGraphBuilder()
        layer4 = Layer4Causal(graph_builder)
        trajectory = _make_trajectory(TRACE_ID, STEP_IDS)

        hypotheses = layer4.build_hypotheses(trajectory)

        assert len(hypotheses) == len(STEP_IDS)
        assert all(h.hypothesis_type == "associational" for h in hypotheses)
