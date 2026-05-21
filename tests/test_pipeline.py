import pytest
from datetime import datetime, timezone

from traceshap.models import (
    TraceSHAPSpan, SpanKind, SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.ingestion.sources.base import SpanSource
from traceshap.pipeline import TraceSHAPPipeline
from traceshap.storage.sqlite import SQLiteBackend
from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer0_rules import Layer0Rules
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig


class FakeSource(SpanSource):
    def __init__(self, span_batches: list[list[TraceSHAPSpan]]):
        self._batches = list(span_batches)

    async def connect(self) -> None:
        pass

    async def poll(self) -> list[TraceSHAPSpan]:
        if self._batches:
            return self._batches.pop(0)
        return []

    async def close(self) -> None:
        pass


def _make_spans(trace_id: str) -> list[TraceSHAPSpan]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        TraceSHAPSpan(
            trace_id=trace_id, span_id=f"{trace_id}-root", parent_span_id=None,
            span_kind=SpanKind.AGENT, name="agent", input={}, output={"result": "ok"},
            start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
            tokens=None, cost=None, metadata={}, raw_attributes={}, semconv_version="test",
        ),
        TraceSHAPSpan(
            trace_id=trace_id, span_id=f"{trace_id}-llm", parent_span_id=f"{trace_id}-root",
            span_kind=SpanKind.LLM, name="gpt-4o", input={"prompt": "hi"}, output={"text": "hello"},
            start_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            tokens=None, cost=None, metadata={}, raw_attributes={}, semconv_version="test",
        ),
        TraceSHAPSpan(
            trace_id=trace_id, span_id=f"{trace_id}-tool", parent_span_id=f"{trace_id}-root",
            span_kind=SpanKind.TOOL, name="search", input={"q": "test"}, output={"r": "found"},
            start_time=datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc),
            tokens=None, cost=None, metadata={}, raw_attributes={}, semconv_version="test",
        ),
    ]


class TestPipeline:
    async def test_ingest_single_trace(self, tmp_path):
        spans = _make_spans("t1")
        source = FakeSource([spans])
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.initialize()

        pipeline = TraceSHAPPipeline(source=source, storage=backend)
        processed = await pipeline.ingest_once()
        assert processed == 1

        trajectory = await backend.get_trajectory("t1")
        assert trajectory is not None
        assert len(trajectory.spans) == 3
        assert len(trajectory.steps) == 3

        await backend.close()

    async def test_ingest_multiple_traces(self, tmp_path):
        source = FakeSource([_make_spans("t1"), _make_spans("t2")])
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.initialize()

        pipeline = TraceSHAPPipeline(source=source, storage=backend)
        count1 = await pipeline.ingest_once()
        count2 = await pipeline.ingest_once()
        assert count1 + count2 == 2

        t1 = await backend.get_trajectory("t1")
        t2 = await backend.get_trajectory("t2")
        assert t1 is not None
        assert t2 is not None

        await backend.close()

    async def test_ingest_empty_returns_zero(self, tmp_path):
        source = FakeSource([])
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.initialize()

        pipeline = TraceSHAPPipeline(source=source, storage=backend)
        count = await pipeline.ingest_once()
        assert count == 0

        await backend.close()


class TestPipelineWithAttribution:
    async def test_full_pipeline_analyze(self, tmp_path):
        spans = _make_spans("t1")
        source = FakeSource([spans])
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.initialize()

        pipeline = TraceSHAPPipeline(source=source, storage=backend)
        await pipeline.ingest_once()

        trajectory = await backend.get_trajectory("t1")
        assert trajectory is not None

        engine = AttributionEngine(layers=[Layer0Rules()])
        attributions = await engine.analyze(trajectory)
        assert len(attributions) == 3

        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(trajectory, attributions)
        assert report.total_steps == 3
        assert report.risk_assessment is not None

        await backend.close()
