from traceshap.models.span import TraceSHAPSpan
from traceshap.models.trajectory import SpanNode


class SpanBuffer:
    def __init__(self):
        self._traces: dict[str, list[TraceSHAPSpan]] = {}

    def add(self, span: TraceSHAPSpan) -> None:
        self._traces.setdefault(span.trace_id, []).append(span)

    def get_spans(self, trace_id: str) -> list[TraceSHAPSpan]:
        return list(self._traces.get(trace_id, []))

    def flush(self, trace_id: str) -> list[TraceSHAPSpan]:
        return self._traces.pop(trace_id, [])

    def pending_trace_ids(self) -> set[str]:
        return set(self._traces.keys())


class TreeAssembler:
    @staticmethod
    def build(spans: list[TraceSHAPSpan]) -> SpanNode:
        nodes: dict[str, SpanNode] = {}
        for span in spans:
            nodes[span.span_id] = SpanNode(span_id=span.span_id)

        root: SpanNode | None = None
        for span in spans:
            node = nodes[span.span_id]
            if span.parent_span_id and span.parent_span_id in nodes:
                nodes[span.parent_span_id].children.append(node)
            else:
                if root is None:
                    root = node

        return root or SpanNode(span_id="empty")
