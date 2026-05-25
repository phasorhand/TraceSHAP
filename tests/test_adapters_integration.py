"""Integration tests verifying the new span sources work correctly together
and with existing shared utilities.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from traceshap.ingestion.sources.otlp_json import OTLPJsonSource
from traceshap.ingestion.sources.otlp_live import OTLPLiveSource
from traceshap.ingestion.sources.hermes_jsonl import HermesJsonlSource
from traceshap.ingestion.sources.otlp_common import parse_otlp_resource_spans
from traceshap.ingestion.sources.base import SpanSource
from traceshap.models.enums import SpanKind

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

OTLP_DATA = {
    "resourceSpans": [{
        "scopeSpans": [{
            "spans": [
                {
                    "traceId": "trace-abc",
                    "spanId": "span-1",
                    "name": "chat gpt-4o",
                    "kind": 3,
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000001000000000",
                    "attributes": [
                        {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
                        {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "100"}},
                        {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "50"}},
                    ],
                },
                {
                    "traceId": "trace-abc",
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

HERMES_ENTRY = {
    "conversations": [
        {"from": "system", "value": "You are helpful."},
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "Hi there!"},
    ],
    "timestamp": "2026-05-22T10:00:00Z",
    "model": "hermes-3",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_otlp_json(data: dict) -> str:
    """Write OTLP dict to a temporary JSON file; return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
    except Exception:
        os.unlink(path)
        raise
    return path


def _write_hermes_jsonl(entries: list[dict]) -> str:
    """Write Hermes entries to a temporary JSONL file; return the path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        os.unlink(path)
        raise
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOTLPJsonAndCommonProduceSameSpans:
    """OTLPJsonSource.poll() and parse_otlp_resource_spans() must agree."""

    async def test_otlp_json_and_common_produce_same_spans(self):
        path = _write_otlp_json(OTLP_DATA)
        try:
            # --- parse via OTLPJsonSource ---
            source = OTLPJsonSource(path)
            await source.connect()
            source_spans = await source.poll()
            await source.close()

            # --- parse directly via shared utility ---
            common_spans = parse_otlp_resource_spans(OTLP_DATA)

            # Both paths must produce the same number of spans.
            assert len(source_spans) == len(common_spans), (
                f"OTLPJsonSource returned {len(source_spans)} span(s) but "
                f"parse_otlp_resource_spans returned {len(common_spans)}"
            )

            # Build comparable (trace_id, span_id, span_kind) tuples.
            source_keys = {
                (s.trace_id, s.span_id, s.span_kind) for s in source_spans
            }
            common_keys = {
                (s.trace_id, s.span_id, s.span_kind) for s in common_spans
            }

            assert source_keys == common_keys, (
                f"Span key mismatch:\n  source={source_keys}\n  common={common_keys}"
            )
        finally:
            os.unlink(path)


class TestOTLPLiveSourceIngestMatchesJsonSource:
    """OTLPLiveSource.ingest() + poll() must return identical spans to OTLPJsonSource."""

    async def test_otlp_live_source_ingest_matches_json_source(self):
        path = _write_otlp_json(OTLP_DATA)
        try:
            # --- JSON file source ---
            json_source = OTLPJsonSource(path)
            await json_source.connect()
            json_spans = await json_source.poll()
            await json_source.close()

            # --- Live source (no HTTP server, just direct ingest) ---
            live_source = OTLPLiveSource()
            live_source.ingest(OTLP_DATA)
            live_spans = live_source.poll()

            # Span counts must match.
            assert len(live_spans) == len(json_spans), (
                f"Live source produced {len(live_spans)} span(s); "
                f"JSON source produced {len(json_spans)}"
            )

            # span_kind and name must be identical across both result sets
            # (order may vary, so compare as sorted tuples of string values).
            json_keys = sorted((s.span_kind.value, s.name) for s in json_spans)
            live_keys = sorted((s.span_kind.value, s.name) for s in live_spans)

            assert json_keys == live_keys, (
                f"Span (kind, name) mismatch:\n  json={json_keys}\n  live={live_keys}"
            )
        finally:
            os.unlink(path)


class TestHermesSourceSpanStructure:
    """HermesJsonlSource must produce correctly typed spans from a JSONL file."""

    async def test_hermes_source_span_structure(self):
        path = _write_hermes_jsonl([HERMES_ENTRY])
        try:
            source = HermesJsonlSource(path)
            await source.connect()
            spans = await source.poll()
            await source.close()

            # One entry with 3 conversations → 3 spans.
            assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"

            # All spans in the same entry share one deterministic trace_id.
            trace_ids = {s.trace_id for s in spans}
            assert len(trace_ids) == 1, (
                f"Expected 1 unique trace_id, found {len(trace_ids)}: {trace_ids}"
            )

            # Map role → span_kind for assertion.
            role_to_kind = {s.metadata["role"]: s.span_kind for s in spans}

            assert role_to_kind["system"] == SpanKind.CUSTOM, (
                f"'system' role should map to CUSTOM, got {role_to_kind['system']}"
            )
            assert role_to_kind["gpt"] == SpanKind.LLM, (
                f"'gpt' role should map to LLM, got {role_to_kind['gpt']}"
            )
        finally:
            os.unlink(path)


class TestAllSourcesImplementInterface:
    """Every new source must satisfy the SpanSource ABC contract."""

    def test_all_sources_implement_interface(self):
        otlp_json_source = OTLPJsonSource("/tmp/dummy.json")
        otlp_live_source = OTLPLiveSource()

        for source in (otlp_json_source, otlp_live_source):
            assert isinstance(source, SpanSource), (
                f"{type(source).__name__} is not a SpanSource subclass"
            )
            assert hasattr(source, "connect"), (
                f"{type(source).__name__} is missing 'connect'"
            )
            assert hasattr(source, "poll"), (
                f"{type(source).__name__} is missing 'poll'"
            )
            assert hasattr(source, "close"), (
                f"{type(source).__name__} is missing 'close'"
            )
