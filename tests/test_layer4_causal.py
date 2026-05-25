"""Tests for Layer4Causal attribution layer (Task 8)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from traceshap.attribution.causal.graph_builder import TrajectoryGraphBuilder
from traceshap.attribution.layer4_causal import Layer4Causal
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


TRACE_ID = "trace-layer4"
STEP_IDS_3 = ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# Test: layer_id
# ---------------------------------------------------------------------------

class TestLayer4LayerId:
    """test_layer4_layer_id — verify layer_id == 4."""

    def test_layer4_layer_id(self):
        builder = TrajectoryGraphBuilder()
        layer = Layer4Causal(builder)
        assert layer.layer_id == 4


# ---------------------------------------------------------------------------
# Test: analyze returns results per step
# ---------------------------------------------------------------------------

class TestLayer4AnalyzeReturnsResultsPerStep:
    """test_layer4_analyze_returns_results_per_step — 3 steps → 3 results, all layer=4."""

    @pytest.mark.asyncio
    async def test_layer4_analyze_returns_results_per_step(self):
        builder = TrajectoryGraphBuilder()
        layer = Layer4Causal(builder)
        trajectory = _make_trajectory(TRACE_ID, STEP_IDS_3)

        results = await layer.analyze(trajectory)

        assert len(results) == 3
        assert all(r.layer == 4 for r in results)

    @pytest.mark.asyncio
    async def test_layer4_step_ids_match(self):
        builder = TrajectoryGraphBuilder()
        layer = Layer4Causal(builder)
        trajectory = _make_trajectory(TRACE_ID, STEP_IDS_3)

        results = await layer.analyze(trajectory)

        result_step_ids = {r.step_id for r in results}
        assert result_step_ids == set(STEP_IDS_3)


# ---------------------------------------------------------------------------
# Test: hypotheses are associational
# ---------------------------------------------------------------------------

class TestLayer4HypothesesAreAssociational:
    """test_layer4_hypotheses_are_associational — all results have "associational" in evidence."""

    @pytest.mark.asyncio
    async def test_layer4_hypotheses_are_associational(self):
        builder = TrajectoryGraphBuilder()
        layer = Layer4Causal(builder)
        trajectory = _make_trajectory(TRACE_ID, STEP_IDS_3)

        results = await layer.analyze(trajectory)

        assert all("associational" in r.evidence for r in results)


# ---------------------------------------------------------------------------
# Test: empty trajectory
# ---------------------------------------------------------------------------

class TestLayer4EmptyTrajectory:
    """test_layer4_empty_trajectory — empty steps → empty results."""

    @pytest.mark.asyncio
    async def test_layer4_empty_trajectory(self):
        builder = TrajectoryGraphBuilder()
        layer = Layer4Causal(builder)
        trajectory = _make_trajectory(TRACE_ID, [])  # no steps

        results = await layer.analyze(trajectory)

        assert results == []
