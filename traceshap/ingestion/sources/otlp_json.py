from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from traceshap.models.span import TraceSHAPSpan, TokenUsage
from traceshap.models.enums import SpanKind
from traceshap.ingestion.sources.base import SpanSource

OTEL_KIND_MAP = {
    0: SpanKind.CUSTOM,
    1: SpanKind.AGENT,
    2: SpanKind.AGENT,
    3: SpanKind.LLM,
    4: SpanKind.LLM,
    5: SpanKind.CUSTOM,
}

TOOL_ATTRIBUTE_KEYS = frozenset({"tool.name", "traceloop.entity.name"})


def _get_attr(attributes: list[dict], key: str) -> str | int | None:
    for attr in attributes:
        if attr.get("key") == key:
            val = attr.get("value", {})
            if "stringValue" in val:
                return val["stringValue"]
            if "intValue" in val:
                return int(val["intValue"])
            if "doubleValue" in val:
                return float(val["doubleValue"])
    return None


def _nano_to_datetime(nano_str: str) -> datetime:
    nanos = int(nano_str)
    seconds = nanos / 1_000_000_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _attrs_to_dict(attributes: list[dict]) -> dict:
    result = {}
    for attr in attributes:
        key = attr.get("key", "")
        val = attr.get("value", {})
        if "stringValue" in val:
            result[key] = val["stringValue"]
        elif "intValue" in val:
            result[key] = int(val["intValue"])
        elif "doubleValue" in val:
            result[key] = float(val["doubleValue"])
        elif "boolValue" in val:
            result[key] = val["boolValue"]
    return result


def _infer_span_kind(otel_kind: int, attributes: list[dict]) -> SpanKind:
    for key in TOOL_ATTRIBUTE_KEYS:
        if _get_attr(attributes, key) is not None:
            return SpanKind.TOOL

    if _get_attr(attributes, "gen_ai.system") is not None:
        return SpanKind.LLM

    return OTEL_KIND_MAP.get(otel_kind, SpanKind.CUSTOM)


def _extract_tokens(attributes: list[dict]) -> TokenUsage | None:
    prompt = _get_attr(attributes, "llm.token_count.prompt")
    completion = _get_attr(attributes, "llm.token_count.completion")
    gen_input = _get_attr(attributes, "gen_ai.usage.input_tokens")
    gen_output = _get_attr(attributes, "gen_ai.usage.output_tokens")

    input_tokens = int(prompt or gen_input or 0)
    output_tokens = int(completion or gen_output or 0)

    if input_tokens == 0 and output_tokens == 0:
        return None

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


class OTLPJsonSource(SpanSource):
    def __init__(self, path: str, source_hint: str = "otlp"):
        self._path = Path(path)
        self._source_hint = source_hint
        self._polled = False

    async def connect(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"OTLP source path not found: {self._path}")

    async def poll(self) -> list[TraceSHAPSpan]:
        if self._polled:
            return []
        self._polled = True

        files: list[Path] = []
        if self._path.is_file():
            files = [self._path]
        elif self._path.is_dir():
            files = sorted(self._path.glob("*.json"))

        all_spans: list[TraceSHAPSpan] = []
        for file_path in files:
            with open(file_path) as f:
                data = json.load(f)
            spans = self._parse_otlp(data)
            all_spans.extend(spans)

        return all_spans

    async def close(self) -> None:
        pass

    def _parse_otlp(self, data: dict) -> list[TraceSHAPSpan]:
        spans: list[TraceSHAPSpan] = []

        for resource_span in data.get("resourceSpans", []):
            for scope_span in resource_span.get("scopeSpans", []):
                for raw_span in scope_span.get("spans", []):
                    span = self._convert_span(raw_span)
                    if span:
                        spans.append(span)

        return spans

    def _convert_span(self, raw: dict) -> TraceSHAPSpan | None:
        attributes = raw.get("attributes", [])
        otel_kind = raw.get("kind", 0)
        span_kind = _infer_span_kind(otel_kind, attributes)

        parent_id = raw.get("parentSpanId", "")
        if not parent_id:
            parent_id = None

        name = raw.get("name", "unknown")
        tool_name = _get_attr(attributes, "tool.name")
        if tool_name and span_kind == SpanKind.TOOL:
            name = tool_name

        return TraceSHAPSpan(
            trace_id=raw.get("traceId", "unknown"),
            span_id=raw.get("spanId", "unknown"),
            parent_span_id=parent_id,
            span_kind=span_kind,
            name=name,
            input={},
            output={},
            start_time=_nano_to_datetime(raw.get("startTimeUnixNano", "0")),
            end_time=_nano_to_datetime(raw.get("endTimeUnixNano", "0")),
            tokens=_extract_tokens(attributes),
            cost=None,
            metadata={},
            raw_attributes=_attrs_to_dict(attributes),
            semconv_version=f"otlp-{self._source_hint}",
        )
