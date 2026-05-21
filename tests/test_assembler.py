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


from traceshap.ingestion.assembler import SpanBuffer, TreeAssembler
from traceshap.models import SpanNode


class TestSpanBuffer:
    def test_add_and_get_by_trace(self):
        buf = SpanBuffer()
        s1 = make_span("t1", "s1")
        s2 = make_span("t1", "s2", parent="s1")
        s3 = make_span("t2", "s3")
        buf.add(s1)
        buf.add(s2)
        buf.add(s3)
        assert len(buf.get_spans("t1")) == 2
        assert len(buf.get_spans("t2")) == 1
        assert len(buf.get_spans("t999")) == 0

    def test_flush_trace(self):
        buf = SpanBuffer()
        buf.add(make_span("t1", "s1"))
        buf.add(make_span("t1", "s2"))
        flushed = buf.flush("t1")
        assert len(flushed) == 2
        assert len(buf.get_spans("t1")) == 0

    def test_pending_trace_ids(self):
        buf = SpanBuffer()
        buf.add(make_span("t1", "s1"))
        buf.add(make_span("t2", "s2"))
        assert buf.pending_trace_ids() == {"t1", "t2"}


class TestTreeAssembler:
    def test_build_simple_tree(self):
        spans = [
            make_span("t1", "root", parent=None),
            make_span("t1", "child1", parent="root"),
            make_span("t1", "child2", parent="root"),
            make_span("t1", "grandchild", parent="child1"),
        ]
        tree = TreeAssembler.build(spans)
        assert tree.span_id == "root"
        assert len(tree.children) == 2
        child1 = next(c for c in tree.children if c.span_id == "child1")
        assert len(child1.children) == 1
        assert child1.children[0].span_id == "grandchild"

    def test_build_single_span(self):
        spans = [make_span("t1", "only")]
        tree = TreeAssembler.build(spans)
        assert tree.span_id == "only"
        assert tree.children == []

    def test_build_with_missing_parent(self):
        spans = [
            make_span("t1", "s1", parent="missing"),
            make_span("t1", "s2", parent="s1"),
        ]
        tree = TreeAssembler.build(spans)
        assert tree.span_id == "s1"
        assert len(tree.children) == 1
