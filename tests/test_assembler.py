import pytest
from datetime import datetime, timezone

from traceshap.models import TraceSHAPSpan, SpanKind, TokenUsage
from traceshap.ingestion.sources.base import SpanSource


class MockSource(SpanSource):
    def __init__(self, spans: list[TraceSHAPSpan]):
        self._spans = spans
        self._index = 0

    async def connect(self) -> None:
        pass

    async def poll(self) -> list[TraceSHAPSpan]:
        if self._index < len(self._spans):
            batch = [self._spans[self._index]]
            self._index += 1
            return batch
        return []

    async def close(self) -> None:
        pass


def make_span(trace_id: str, span_id: str, parent: str | None = None,
              kind: SpanKind = SpanKind.LLM, name: str = "test",
              offset_sec: int = 0, duration_sec: int = 1) -> TraceSHAPSpan:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + duration_sec, tzinfo=timezone.utc)
    return TraceSHAPSpan(
        trace_id=trace_id, span_id=span_id, parent_span_id=parent,
        span_kind=kind, name=name, input={}, output={},
        start_time=start, end_time=end, tokens=None, cost=None,
        metadata={}, raw_attributes={}, semconv_version="test",
    )


class TestSpanSource:
    async def test_mock_source_polls(self):
        spans = [make_span("t1", "s1"), make_span("t1", "s2")]
        source = MockSource(spans)
        await source.connect()
        batch1 = await source.poll()
        assert len(batch1) == 1
        batch2 = await source.poll()
        assert len(batch2) == 1
        batch3 = await source.poll()
        assert len(batch3) == 0
        await source.close()
