from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field

from traceshap.models.span import TraceSHAPSpan, TokenUsage
from traceshap.models.enums import SpanKind


@dataclass
class _PendingSpan:
    run_id: str
    span_kind: SpanKind
    name: str
    inputs: dict
    metadata: dict
    parent_run_id: str | None
    start_time: datetime


class LangGraphSpanCollector:
    def __init__(self, trace_id: str):
        self._trace_id = trace_id
        self._pending: dict[str, _PendingSpan] = {}
        self._completed: list[TraceSHAPSpan] = []

    def on_llm_start(
        self, run_id: str, name: str, inputs: dict,
        metadata: dict | None = None, parent_run_id: str | None = None,
    ) -> None:
        self._pending[run_id] = _PendingSpan(
            run_id=run_id, span_kind=SpanKind.LLM, name=name,
            inputs=inputs, metadata=metadata or {},
            parent_run_id=parent_run_id,
            start_time=datetime.now(timezone.utc),
        )

    def on_llm_end(
        self, run_id: str, outputs: dict,
        token_usage: dict | None = None,
    ) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return

        tokens = None
        if token_usage:
            tokens = TokenUsage(
                input_tokens=token_usage.get("input", 0) or 0,
                output_tokens=token_usage.get("output", 0) or 0,
                total_tokens=token_usage.get("total", 0) or 0,
            )

        self._completed.append(TraceSHAPSpan(
            trace_id=self._trace_id,
            span_id=pending.run_id,
            parent_span_id=pending.parent_run_id,
            span_kind=pending.span_kind,
            name=pending.name,
            input=pending.inputs,
            output=outputs,
            start_time=pending.start_time,
            end_time=datetime.now(timezone.utc),
            tokens=tokens,
            cost=None,
            metadata=pending.metadata,
            raw_attributes={"framework": "langgraph"},
            semconv_version="langgraph-native",
        ))

    def on_tool_start(
        self, run_id: str, name: str, inputs: dict,
        metadata: dict | None = None, parent_run_id: str | None = None,
    ) -> None:
        self._pending[run_id] = _PendingSpan(
            run_id=run_id, span_kind=SpanKind.TOOL, name=name,
            inputs=inputs, metadata=metadata or {},
            parent_run_id=parent_run_id,
            start_time=datetime.now(timezone.utc),
        )

    def on_tool_end(self, run_id: str, outputs: dict) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return

        self._completed.append(TraceSHAPSpan(
            trace_id=self._trace_id,
            span_id=pending.run_id,
            parent_span_id=pending.parent_run_id,
            span_kind=pending.span_kind,
            name=pending.name,
            input=pending.inputs,
            output=outputs,
            start_time=pending.start_time,
            end_time=datetime.now(timezone.utc),
            tokens=None,
            cost=None,
            metadata=pending.metadata,
            raw_attributes={"framework": "langgraph"},
            semconv_version="langgraph-native",
        ))

    def on_chain_start(
        self, run_id: str, name: str, inputs: dict,
        metadata: dict | None = None, parent_run_id: str | None = None,
    ) -> None:
        self._pending[run_id] = _PendingSpan(
            run_id=run_id, span_kind=SpanKind.AGENT, name=name,
            inputs=inputs, metadata=metadata or {},
            parent_run_id=parent_run_id,
            start_time=datetime.now(timezone.utc),
        )

    def on_chain_end(self, run_id: str, outputs: dict) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return

        self._completed.append(TraceSHAPSpan(
            trace_id=self._trace_id,
            span_id=pending.run_id,
            parent_span_id=pending.parent_run_id,
            span_kind=pending.span_kind,
            name=pending.name,
            input=pending.inputs,
            output=outputs,
            start_time=pending.start_time,
            end_time=datetime.now(timezone.utc),
            tokens=None,
            cost=None,
            metadata=pending.metadata,
            raw_attributes={"framework": "langgraph"},
            semconv_version="langgraph-native",
        ))

    def get_spans(self) -> list[TraceSHAPSpan]:
        return list(self._completed)

    def clear(self) -> None:
        self._pending.clear()
        self._completed.clear()
