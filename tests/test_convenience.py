import pytest
from datetime import datetime, timezone

from traceshap.models import (
    TraceSHAPSpan, SpanKind, TokenUsage, SpanNode,
    TrajectoryMeta, Trajectory, Outcome, Verdict,
)
from traceshap.convenience import quick_analyze, spans_to_trajectory


class TestSpansToTrajectory:
    def test_converts_spans(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spans = [
            TraceSHAPSpan(
                trace_id="t1", span_id="s1", parent_span_id=None,
                span_kind=SpanKind.AGENT, name="agent",
                input={}, output={"result": "ok"},
                start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
                tokens=None, cost=None, metadata={},
                raw_attributes={}, semconv_version="test",
            ),
            TraceSHAPSpan(
                trace_id="t1", span_id="s2", parent_span_id="s1",
                span_kind=SpanKind.LLM, name="gpt-4o",
                input={"prompt": "hi"}, output={"text": "hello"},
                start_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                tokens=TokenUsage(10, 20, 30), cost=0.001,
                metadata={}, raw_attributes={}, semconv_version="test",
            ),
        ]
        traj = spans_to_trajectory(spans, trace_id="t1")
        assert traj.trace_id == "t1"
        assert len(traj.spans) == 2
        assert len(traj.steps) == 2
        assert traj.span_tree is not None

    def test_with_outcome(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spans = [
            TraceSHAPSpan(
                trace_id="t1", span_id="s1", parent_span_id=None,
                span_kind=SpanKind.TOOL, name="search",
                input={"q": "test"}, output={"r": "found"},
                start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                tokens=None, cost=None, metadata={},
                raw_attributes={}, semconv_version="test",
            ),
        ]
        outcome = Outcome(success=True, quality_score=0.9, token_cost=30,
                          latency_ms=1000, custom_metrics={})
        traj = spans_to_trajectory(spans, trace_id="t1", outcome=outcome)
        assert traj.outcome is not None
        assert traj.outcome.quality_score == 0.9


class TestQuickAnalyze:
    async def test_analyze_spans(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spans = [
            TraceSHAPSpan(
                trace_id="t1", span_id="s1", parent_span_id=None,
                span_kind=SpanKind.AGENT, name="agent",
                input={}, output={},
                start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
                tokens=None, cost=None, metadata={},
                raw_attributes={}, semconv_version="test",
            ),
            TraceSHAPSpan(
                trace_id="t1", span_id="s2", parent_span_id="s1",
                span_kind=SpanKind.LLM, name="gpt-4o",
                input={"prompt": "hi"}, output={"text": "hello"},
                start_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                tokens=TokenUsage(10, 20, 30), cost=0.001,
                metadata={}, raw_attributes={}, semconv_version="test",
            ),
        ]
        result = await quick_analyze(spans, trace_id="t1", layers=[0])
        assert "attributions" in result
        assert len(result["attributions"]) == 2
        assert "trajectory" in result

    async def test_analyze_with_pruning(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spans = [
            TraceSHAPSpan(
                trace_id="t1", span_id="s1", parent_span_id=None,
                span_kind=SpanKind.AGENT, name="agent",
                input={}, output={},
                start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
                tokens=None, cost=None, metadata={},
                raw_attributes={}, semconv_version="test",
            ),
        ]
        outcome = Outcome(success=True, quality_score=0.9, token_cost=30,
                          latency_ms=5000, custom_metrics={})
        result = await quick_analyze(spans, trace_id="t1", layers=[0],
                                     outcome=outcome, include_pruning=True)
        assert "pruning_report" in result
        assert result["pruning_report"].total_steps == 1
