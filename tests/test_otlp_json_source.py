import pytest
import json
from datetime import datetime, timezone
from pathlib import Path

from traceshap.models import TraceSHAPSpan, SpanKind
from traceshap.ingestion.sources.otlp_json import OTLPJsonSource


def _sample_otlp_data():
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "my-agent"}}]},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "abc123",
                                "spanId": "span1",
                                "parentSpanId": "",
                                "name": "agent_run",
                                "kind": 1,
                                "startTimeUnixNano": "1735689600000000000",
                                "endTimeUnixNano": "1735689605000000000",
                                "attributes": [
                                    {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
                                    {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4o"}},
                                ],
                            },
                            {
                                "traceId": "abc123",
                                "spanId": "span2",
                                "parentSpanId": "span1",
                                "name": "llm_call",
                                "kind": 3,
                                "startTimeUnixNano": "1735689601000000000",
                                "endTimeUnixNano": "1735689602000000000",
                                "attributes": [
                                    {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
                                    {"key": "llm.token_count.prompt", "value": {"intValue": "100"}},
                                    {"key": "llm.token_count.completion", "value": {"intValue": "50"}},
                                ],
                            },
                            {
                                "traceId": "abc123",
                                "spanId": "span3",
                                "parentSpanId": "span1",
                                "name": "search_web",
                                "kind": 3,
                                "startTimeUnixNano": "1735689603000000000",
                                "endTimeUnixNano": "1735689604000000000",
                                "attributes": [
                                    {"key": "tool.name", "value": {"stringValue": "search_web"}},
                                ],
                            },
                        ]
                    }
                ]
            }
        ]
    }


class TestOTLPJsonSource:
    async def test_load_from_file(self, tmp_path):
        data = _sample_otlp_data()
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(data))

        source = OTLPJsonSource(str(path))
        await source.connect()
        spans = await source.poll()
        assert len(spans) == 3
        assert all(isinstance(s, TraceSHAPSpan) for s in spans)
        assert all(s.trace_id == "abc123" for s in spans)

    async def test_span_kinds_mapped(self, tmp_path):
        data = _sample_otlp_data()
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(data))

        source = OTLPJsonSource(str(path))
        await source.connect()
        spans = await source.poll()
        span_map = {s.span_id: s for s in spans}

        assert span_map["span3"].name == "search_web"

    async def test_parent_ids_preserved(self, tmp_path):
        data = _sample_otlp_data()
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(data))

        source = OTLPJsonSource(str(path))
        await source.connect()
        spans = await source.poll()
        span_map = {s.span_id: s for s in spans}

        assert span_map["span1"].parent_span_id is None
        assert span_map["span2"].parent_span_id == "span1"
        assert span_map["span3"].parent_span_id == "span1"

    async def test_token_extraction(self, tmp_path):
        data = _sample_otlp_data()
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(data))

        source = OTLPJsonSource(str(path))
        await source.connect()
        spans = await source.poll()
        span_map = {s.span_id: s for s in spans}

        llm_span = span_map["span2"]
        assert llm_span.tokens is not None
        assert llm_span.tokens.input_tokens == 100
        assert llm_span.tokens.output_tokens == 50

    async def test_directory_of_files(self, tmp_path):
        data = _sample_otlp_data()
        (tmp_path / "trace1.json").write_text(json.dumps(data))

        data2 = _sample_otlp_data()
        data2["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"] = "def456"
        data2["resourceSpans"][0]["scopeSpans"][0]["spans"][1]["traceId"] = "def456"
        data2["resourceSpans"][0]["scopeSpans"][0]["spans"][2]["traceId"] = "def456"
        (tmp_path / "trace2.json").write_text(json.dumps(data2))

        source = OTLPJsonSource(str(tmp_path))
        await source.connect()
        spans = await source.poll()
        trace_ids = {s.trace_id for s in spans}
        assert "abc123" in trace_ids
        assert "def456" in trace_ids

    async def test_poll_returns_empty_after_first(self, tmp_path):
        data = _sample_otlp_data()
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(data))

        source = OTLPJsonSource(str(path))
        await source.connect()
        spans1 = await source.poll()
        assert len(spans1) > 0
        spans2 = await source.poll()
        assert len(spans2) == 0
