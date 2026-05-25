"""Shared OTLP parsing utilities used by all OTLP-based span sources."""
from __future__ import annotations

from datetime import datetime, timezone

from traceshap.models.span import TraceSHAPSpan, TokenUsage
from traceshap.models.enums import SpanKind

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OTEL_KIND_MAP: dict[int, SpanKind] = {
    0: SpanKind.CUSTOM,
    1: SpanKind.AGENT,
    2: SpanKind.AGENT,
    3: SpanKind.LLM,
    4: SpanKind.LLM,
    5: SpanKind.CUSTOM,
}

TOOL_ATTRIBUTE_KEYS: frozenset[str] = frozenset({"tool.name", "traceloop.entity.name"})

# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------


def get_attr(attributes: list[dict], key: str) -> str | int | float | None:
    """Return the typed value of the first attribute matching *key*, or None."""
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


def nano_to_datetime(nano_str: str) -> datetime:
    """Convert a nanosecond Unix timestamp string to a UTC-aware datetime."""
    nanos = int(nano_str)
    seconds = nanos / 1_000_000_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def attrs_to_dict(attributes: list[dict]) -> dict:
    """Convert an OTel attribute list to a plain Python dict."""
    result: dict = {}
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


def infer_span_kind(otel_kind: int, attributes: list[dict]) -> SpanKind:
    """Infer a TraceSHAP SpanKind from the OTel numeric kind and span attributes."""
    for key in TOOL_ATTRIBUTE_KEYS:
        if get_attr(attributes, key) is not None:
            return SpanKind.TOOL

    if get_attr(attributes, "gen_ai.system") is not None:
        return SpanKind.LLM

    return OTEL_KIND_MAP.get(otel_kind, SpanKind.CUSTOM)


def extract_tokens(attributes: list[dict]) -> TokenUsage | None:
    """Extract token usage from OTel attributes, returning None if absent."""
    prompt = get_attr(attributes, "llm.token_count.prompt")
    completion = get_attr(attributes, "llm.token_count.completion")
    gen_input = get_attr(attributes, "gen_ai.usage.input_tokens")
    gen_output = get_attr(attributes, "gen_ai.usage.output_tokens")

    input_tokens = int(prompt or gen_input or 0)
    output_tokens = int(completion or gen_output or 0)

    if input_tokens == 0 and output_tokens == 0:
        return None

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


# ---------------------------------------------------------------------------
# Span conversion
# ---------------------------------------------------------------------------


def convert_otlp_span(raw: dict, source_hint: str = "otlp") -> TraceSHAPSpan | None:
    """Convert a single raw OTel span dict into a TraceSHAPSpan."""
    attributes = raw.get("attributes", [])
    otel_kind = raw.get("kind", 0)
    span_kind = infer_span_kind(otel_kind, attributes)

    parent_id = raw.get("parentSpanId", "")
    if not parent_id:
        parent_id = None

    name = raw.get("name", "unknown")
    tool_name = get_attr(attributes, "tool.name")
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
        start_time=nano_to_datetime(raw.get("startTimeUnixNano", "0")),
        end_time=nano_to_datetime(raw.get("endTimeUnixNano", "0")),
        tokens=extract_tokens(attributes),
        cost=None,
        metadata={},
        raw_attributes=attrs_to_dict(attributes),
        semconv_version=f"otlp-{source_hint}",
    )


def parse_otlp_resource_spans(
    data: dict, source_hint: str = "otlp"
) -> list[TraceSHAPSpan]:
    """Parse all spans from an OTLP JSON payload (resourceSpans envelope)."""
    spans: list[TraceSHAPSpan] = []

    for resource_span in data.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            for raw_span in scope_span.get("spans", []):
                span = convert_otlp_span(raw_span, source_hint)
                if span:
                    spans.append(span)

    return spans
