from __future__ import annotations

import pytest
from datetime import datetime, timezone

from traceshap.ingestion.sources.otlp_common import (
    get_attr,
    nano_to_datetime,
    attrs_to_dict,
    infer_span_kind,
    extract_tokens,
    convert_otlp_span,
    parse_otlp_resource_spans,
    OTEL_KIND_MAP,
    TOOL_ATTRIBUTE_KEYS,
)
from traceshap.models.enums import SpanKind
from traceshap.models.span import TraceSHAPSpan


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _attr(key: str, **value_kwargs) -> dict:
    """Build a single OTel attribute dict."""
    return {"key": key, "value": value_kwargs}


# ---------------------------------------------------------------------------
# get_attr
# ---------------------------------------------------------------------------

class TestGetAttr:
    def test_get_attr_string(self):
        attrs = [_attr("service.name", stringValue="my-service")]
        assert get_attr(attrs, "service.name") == "my-service"

    def test_get_attr_int(self):
        attrs = [_attr("token.count", intValue="42")]
        result = get_attr(attrs, "token.count")
        assert result == 42
        assert isinstance(result, int)

    def test_get_attr_double(self):
        attrs = [_attr("latency", doubleValue=1.5)]
        result = get_attr(attrs, "latency")
        assert result == pytest.approx(1.5)
        assert isinstance(result, float)

    def test_get_attr_missing_key(self):
        attrs = [_attr("other.key", stringValue="x")]
        assert get_attr(attrs, "missing.key") is None

    def test_get_attr_empty_list(self):
        assert get_attr([], "any.key") is None


# ---------------------------------------------------------------------------
# nano_to_datetime
# ---------------------------------------------------------------------------

class TestNanoToDatetime:
    def test_nano_to_datetime(self):
        # 1_000_000_000 ns == 1970-01-01 00:00:01 UTC
        result = nano_to_datetime("1000000000")
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc
        assert result.year == 1970
        assert result.second == 1

    def test_nano_to_datetime_timezone(self):
        result = nano_to_datetime("1735689600000000000")
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# attrs_to_dict
# ---------------------------------------------------------------------------

class TestAttrsToDict:
    def test_attrs_to_dict(self):
        attrs = [
            _attr("str.key", stringValue="hello"),
            _attr("int.key", intValue="7"),
            _attr("float.key", doubleValue=3.14),
            _attr("bool.key", boolValue=True),
        ]
        result = attrs_to_dict(attrs)
        assert result == {
            "str.key": "hello",
            "int.key": 7,
            "float.key": pytest.approx(3.14),
            "bool.key": True,
        }

    def test_attrs_to_dict_empty(self):
        assert attrs_to_dict([]) == {}


# ---------------------------------------------------------------------------
# infer_span_kind
# ---------------------------------------------------------------------------

class TestInferSpanKind:
    def test_infer_span_kind_tool(self):
        attrs = [_attr("tool.name", stringValue="search_web")]
        assert infer_span_kind(0, attrs) == SpanKind.TOOL

    def test_infer_span_kind_llm(self):
        attrs = [_attr("gen_ai.system", stringValue="openai")]
        assert infer_span_kind(0, attrs) == SpanKind.LLM

    def test_infer_span_kind_default(self):
        # otel kind=1 maps to AGENT in OTEL_KIND_MAP
        attrs = []
        assert infer_span_kind(1, attrs) == SpanKind.AGENT

    def test_tool_takes_priority_over_llm(self):
        attrs = [
            _attr("tool.name", stringValue="my_tool"),
            _attr("gen_ai.system", stringValue="openai"),
        ]
        assert infer_span_kind(3, attrs) == SpanKind.TOOL

    def test_infer_span_kind_unknown_otel_kind(self):
        attrs = []
        assert infer_span_kind(99, attrs) == SpanKind.CUSTOM


# ---------------------------------------------------------------------------
# extract_tokens
# ---------------------------------------------------------------------------

class TestExtractTokens:
    def test_extract_tokens_gen_ai(self):
        attrs = [
            _attr("gen_ai.usage.input_tokens", intValue="200"),
            _attr("gen_ai.usage.output_tokens", intValue="80"),
        ]
        result = extract_tokens(attrs)
        assert result is not None
        assert result.input_tokens == 200
        assert result.output_tokens == 80
        assert result.total_tokens == 280

    def test_extract_tokens_none(self):
        attrs = [_attr("some.other.attr", stringValue="value")]
        assert extract_tokens(attrs) is None

    def test_extract_tokens_llm_prefix(self):
        attrs = [
            _attr("llm.token_count.prompt", intValue="100"),
            _attr("llm.token_count.completion", intValue="50"),
        ]
        result = extract_tokens(attrs)
        assert result is not None
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150


# ---------------------------------------------------------------------------
# convert_otlp_span
# ---------------------------------------------------------------------------

class TestConvertOtlpSpan:
    def _raw_span(self, **overrides) -> dict:
        base = {
            "traceId": "trace1",
            "spanId": "span1",
            "parentSpanId": "",
            "name": "my_span",
            "kind": 0,
            "startTimeUnixNano": "1000000000",
            "endTimeUnixNano": "2000000000",
            "attributes": [],
        }
        base.update(overrides)
        return base

    def test_convert_returns_traceshap_span(self):
        raw = self._raw_span()
        result = convert_otlp_span(raw)
        assert isinstance(result, TraceSHAPSpan)
        assert result.trace_id == "trace1"
        assert result.span_id == "span1"
        assert result.parent_span_id is None  # empty string → None

    def test_convert_source_hint_in_semconv(self):
        raw = self._raw_span()
        result = convert_otlp_span(raw, source_hint="myformat")
        assert result.semconv_version == "otlp-myformat"

    def test_convert_tool_name_overrides_name(self):
        raw = self._raw_span(
            name="generic_name",
            attributes=[_attr("tool.name", stringValue="real_tool")],
        )
        result = convert_otlp_span(raw)
        assert result.name == "real_tool"
        assert result.span_kind == SpanKind.TOOL

    def test_convert_parent_id_preserved(self):
        raw = self._raw_span(parentSpanId="parent99")
        result = convert_otlp_span(raw)
        assert result.parent_span_id == "parent99"


# ---------------------------------------------------------------------------
# parse_otlp_resource_spans
# ---------------------------------------------------------------------------

class TestParseOtlpResourceSpans:
    def _sample_data(self) -> dict:
        return {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "t1",
                                    "spanId": "s1",
                                    "parentSpanId": "",
                                    "name": "agent_run",
                                    "kind": 1,
                                    "startTimeUnixNano": "1000000000",
                                    "endTimeUnixNano": "2000000000",
                                    "attributes": [],
                                },
                                {
                                    "traceId": "t1",
                                    "spanId": "s2",
                                    "parentSpanId": "s1",
                                    "name": "llm_call",
                                    "kind": 3,
                                    "startTimeUnixNano": "1100000000",
                                    "endTimeUnixNano": "1900000000",
                                    "attributes": [
                                        _attr("gen_ai.system", stringValue="openai"),
                                    ],
                                },
                            ]
                        }
                    ]
                }
            ]
        }

    def test_parse_returns_all_spans(self):
        data = self._sample_data()
        spans = parse_otlp_resource_spans(data)
        assert len(spans) == 2
        assert all(isinstance(s, TraceSHAPSpan) for s in spans)

    def test_parse_empty_data(self):
        assert parse_otlp_resource_spans({}) == []

    def test_parse_source_hint_propagated(self):
        data = self._sample_data()
        spans = parse_otlp_resource_spans(data, source_hint="custom")
        assert all("otlp-custom" in s.semconv_version for s in spans)
