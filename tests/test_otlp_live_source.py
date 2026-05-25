import pytest

from traceshap.ingestion.sources.otlp_live import OTLPLiveSource
from traceshap.models.span import TraceSHAPSpan

# ---------------------------------------------------------------------------
# Sample payload
# ---------------------------------------------------------------------------

SAMPLE_OTLP_PAYLOAD = {
    "resourceSpans": [{
        "scopeSpans": [{
            "spans": [
                {
                    "traceId": "abc123",
                    "spanId": "span-1",
                    "name": "test_span",
                    "kind": 3,
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000001000000000",
                    "attributes": [
                        {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
                    ],
                },
                {
                    "traceId": "abc123",
                    "spanId": "span-2",
                    "parentSpanId": "span-1",
                    "name": "web_search",
                    "kind": 0,
                    "startTimeUnixNano": "1700000001000000000",
                    "endTimeUnixNano": "1700000002000000000",
                    "attributes": [
                        {"key": "tool.name", "value": {"stringValue": "web_search"}},
                    ],
                },
            ]
        }]
    }]
}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOTLPLiveSource:
    def test_otlp_live_source_creation(self):
        """Create with custom host/port and verify attributes."""
        source = OTLPLiveSource(host="127.0.0.1", port=9999, source_hint="myapp")
        assert source.host == "127.0.0.1"
        assert source.port == 9999
        assert source.source_hint == "myapp"

    def test_ingest_spans_to_buffer(self):
        """Call ingest() with OTLP payload, verify buffer has correct count."""
        source = OTLPLiveSource()
        source.ingest(SAMPLE_OTLP_PAYLOAD)
        with source._lock:
            assert len(source._buffer) == 2

    def test_poll_returns_and_clears_buffer(self):
        """Ingest then poll returns spans and clears buffer; second poll returns empty."""
        source = OTLPLiveSource()
        source.ingest(SAMPLE_OTLP_PAYLOAD)

        spans = source.poll()
        assert len(spans) == 2
        assert all(isinstance(s, TraceSHAPSpan) for s in spans)

        # Second poll must be empty (buffer cleared)
        second = source.poll()
        assert second == []

    def test_poll_empty_when_no_data(self):
        """Poll without ingest returns empty list."""
        source = OTLPLiveSource()
        assert source.poll() == []

    def test_ingest_with_source_hint(self):
        """Ingest with source_hint='openclaw', verify semconv_version."""
        source = OTLPLiveSource(source_hint="openclaw")
        source.ingest(SAMPLE_OTLP_PAYLOAD)
        spans = source.poll()
        assert len(spans) == 2
        for span in spans:
            assert span.semconv_version == "otlp-openclaw"

    def test_max_buffer_size(self):
        """Set max_buffer_size=1, ingest 2 spans, only 1 kept."""
        source = OTLPLiveSource(max_buffer_size=1)
        source.ingest(SAMPLE_OTLP_PAYLOAD)  # payload has 2 spans
        with source._lock:
            assert len(source._buffer) == 1
