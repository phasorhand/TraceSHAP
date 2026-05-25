from __future__ import annotations

from datetime import datetime

import pytest

from traceshap.attribution.causal.graph_builder import TrajectoryGraphBuilder
from traceshap.attribution.causal.models import EdgeType
from traceshap.models.enums import SideEffect, SpanKind, StepType
from traceshap.models.span import TraceSHAPSpan, TokenUsage
from traceshap.models.step import CanonicalStep
from traceshap.models.trajectory import SpanNode, Trajectory, TrajectoryMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, 12, 0, 0)
_LATER = datetime(2024, 1, 1, 12, 0, 1)


def _make_span(span_id: str, input_text: str = "", output_text: str = "") -> TraceSHAPSpan:
    return TraceSHAPSpan(
        trace_id="trace-test",
        span_id=span_id,
        parent_span_id=None,
        span_kind=SpanKind.LLM,
        name=span_id,
        input={"text": input_text},
        output={"text": output_text},
        start_time=_NOW,
        end_time=_LATER,
        tokens=None,
        cost=None,
        metadata={},
        raw_attributes={},
        semconv_version="1.0",
    )


def _make_step(
    step_id: str,
    span_ids: list[str],
    input_hash: str = "",
    output_hash: str = "",
) -> CanonicalStep:
    return CanonicalStep(
        step_id=step_id,
        raw_span_ids=span_ids,
        node_id=None,
        tool_name=None,
        step_type=StepType.ACTION,
        attempt_index=0,
        loop_iteration=None,
        input_hash=input_hash,
        output_hash=output_hash,
        side_effect_class=SideEffect.PURE,
        framework_mapping_confidence=1.0,
        tokens=None,
        cost=None,
        start_time=_NOW,
        end_time=_LATER,
    )


def _make_trajectory(steps: list[CanonicalStep], spans: list[TraceSHAPSpan]) -> Trajectory:
    return Trajectory(
        trace_id="trace-test",
        spans=spans,
        steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=None,
        metadata=TrajectoryMeta(
            framework="test",
            agent_name="test-agent",
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adjacent_steps_get_temporal_edge():
    """Two sequential steps with no hash match should yield a TEMPORAL edge."""
    span_a = _make_span("span-a", output_text="short")
    span_b = _make_span("span-b", input_text="something else")
    step_a = _make_step("s1", ["span-a"], input_hash="h1", output_hash="h_out_a")
    step_b = _make_step("s2", ["span-b"], input_hash="h_in_b", output_hash="h2")

    traj = _make_trajectory([step_a, step_b], [span_a, span_b])
    edges = TrajectoryGraphBuilder().build(traj)

    temporal_edges = [e for e in edges if e.edge_type == EdgeType.TEMPORAL]
    assert len(temporal_edges) >= 1
    assert any(e.source_step_id == "s1" and e.target_step_id == "s2" for e in temporal_edges)


def test_data_dependency_detected():
    """When step_a.output_hash == step_b.input_hash a DATA_DEPENDENCY edge is created."""
    shared_hash = "shared-hash-abc"

    span_a = _make_span("span-a")
    span_b = _make_span("span-b")
    step_a = _make_step("s1", ["span-a"], input_hash="h1", output_hash=shared_hash)
    step_b = _make_step("s2", ["span-b"], input_hash=shared_hash, output_hash="h2")

    traj = _make_trajectory([step_a, step_b], [span_a, span_b])
    edges = TrajectoryGraphBuilder().build(traj)

    data_edges = [e for e in edges if e.edge_type == EdgeType.DATA_DEPENDENCY]
    assert len(data_edges) >= 1
    assert any(e.source_step_id == "s1" and e.target_step_id == "s2" for e in data_edges)


def test_non_adjacent_no_temporal():
    """Steps s1 and s3 (non-adjacent) should NOT have a TEMPORAL edge between them."""
    span_a = _make_span("span-a", output_text="short")
    span_b = _make_span("span-b", input_text="other", output_text="short too")
    span_c = _make_span("span-c", input_text="yet another")
    step_a = _make_step("s1", ["span-a"], input_hash="h1", output_hash="ho1")
    step_b = _make_step("s2", ["span-b"], input_hash="hi2", output_hash="ho2")
    step_c = _make_step("s3", ["span-c"], input_hash="hi3", output_hash="ho3")

    traj = _make_trajectory([step_a, step_b, step_c], [span_a, span_b, span_c])
    edges = TrajectoryGraphBuilder().build(traj)

    # s1 -> s3 must NOT be TEMPORAL
    s1_s3_temporal = [
        e for e in edges
        if e.source_step_id == "s1" and e.target_step_id == "s3" and e.edge_type == EdgeType.TEMPORAL
    ]
    assert s1_s3_temporal == []


def test_empty_trajectory():
    """An empty steps list must return an empty edge list."""
    traj = _make_trajectory([], [])
    edges = TrajectoryGraphBuilder().build(traj)
    assert edges == []


def test_data_dependency_by_content():
    """When span_a's output text (len > 20) appears in span_b's input text, DATA_DEPENDENCY is detected."""
    long_output = "This is a long output text that exceeds twenty characters"
    span_a = _make_span("span-a", output_text=long_output)
    span_b = _make_span("span-b", input_text=f"Prefix: {long_output} :suffix")

    # Use different hashes so hash-based detection does NOT fire
    step_a = _make_step("s1", ["span-a"], input_hash="h1", output_hash="out_hash_a")
    step_b = _make_step("s2", ["span-b"], input_hash="in_hash_b", output_hash="h2")

    traj = _make_trajectory([step_a, step_b], [span_a, span_b])
    edges = TrajectoryGraphBuilder().build(traj)

    data_edges = [e for e in edges if e.edge_type == EdgeType.DATA_DEPENDENCY]
    assert len(data_edges) >= 1
    assert any(e.source_step_id == "s1" and e.target_step_id == "s2" for e in data_edges)
