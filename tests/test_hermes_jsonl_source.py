import json
import os
import tempfile

import pytest

from traceshap.models import TraceSHAPSpan, SpanKind
from traceshap.ingestion.sources.hermes_jsonl import HermesJsonlSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_CONVERSATIONS = [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "Search for cats"},
    {"from": "gpt", "value": "<think>I should search</think>\nSearching..."},
    {"from": "tool", "value": {"results": ["cat1", "cat2"]}},
    {"from": "gpt", "value": "Found 2 results about cats."},
]

_SAMPLE_ENTRY = {
    "conversations": _SAMPLE_CONVERSATIONS,
    "timestamp": "2026-05-22T10:30:00Z",
    "model": "hermes-3-llama-3.1-70b",
    "completed": True,
}


def _write_jsonl(entries: list[dict]) -> str:
    """Write entries to a temporary JSONL file and return the path."""
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


class TestHermesJsonlSource:
    async def test_hermes_source_single_entry(self):
        """1 entry with 5 messages should produce 5 spans."""
        path = _write_jsonl([_SAMPLE_ENTRY])
        try:
            source = HermesJsonlSource(path)
            await source.connect()
            spans = await source.poll()
            assert len(spans) == 5
            assert all(isinstance(s, TraceSHAPSpan) for s in spans)
        finally:
            os.unlink(path)

    async def test_hermes_span_kinds(self):
        """system→CUSTOM, human→CUSTOM, gpt→LLM, tool→TOOL."""
        path = _write_jsonl([_SAMPLE_ENTRY])
        try:
            source = HermesJsonlSource(path)
            await source.connect()
            spans = await source.poll()

            role_to_kind = {s.metadata["role"]: s.span_kind for s in spans}

            assert role_to_kind["system"] == SpanKind.CUSTOM
            assert role_to_kind["human"] == SpanKind.CUSTOM
            assert role_to_kind["gpt"] == SpanKind.LLM
            assert role_to_kind["tool"] == SpanKind.TOOL
        finally:
            os.unlink(path)

    async def test_hermes_poll_once(self):
        """Second poll must return an empty list."""
        path = _write_jsonl([_SAMPLE_ENTRY])
        try:
            source = HermesJsonlSource(path)
            await source.connect()
            spans1 = await source.poll()
            assert len(spans1) > 0
            spans2 = await source.poll()
            assert spans2 == []
        finally:
            os.unlink(path)

    async def test_hermes_multiple_entries(self):
        """2 entries → 10 spans, 2 distinct trace_ids."""
        entry2 = dict(_SAMPLE_ENTRY, timestamp="2026-05-22T11:00:00Z")
        path = _write_jsonl([_SAMPLE_ENTRY, entry2])
        try:
            source = HermesJsonlSource(path)
            await source.connect()
            spans = await source.poll()
            assert len(spans) == 10
            trace_ids = {s.trace_id for s in spans}
            assert len(trace_ids) == 2
        finally:
            os.unlink(path)

    async def test_hermes_trace_id_deterministic(self, tmp_path):
        """Same file content must produce the same trace_ids on repeated reads."""
        path = tmp_path / "sample.jsonl"
        path.write_text(json.dumps(_SAMPLE_ENTRY) + "\n")

        source1 = HermesJsonlSource(str(path))
        await source1.connect()
        spans1 = await source1.poll()

        source2 = HermesJsonlSource(str(path))
        await source2.connect()
        spans2 = await source2.poll()

        assert {s.trace_id for s in spans1} == {s.trace_id for s in spans2}

    async def test_hermes_connect_missing_file(self):
        """connect() must raise FileNotFoundError for a non-existent path."""
        source = HermesJsonlSource("/tmp/does_not_exist_traceshap_hermes.jsonl")
        with pytest.raises(FileNotFoundError):
            await source.connect()

    async def test_hermes_value_wrapping(self):
        """String values are wrapped in {text: …}; dict values passed as-is."""
        path = _write_jsonl([_SAMPLE_ENTRY])
        try:
            source = HermesJsonlSource(path)
            await source.connect()
            spans = await source.poll()

            # Index 3 is the "tool" message with a dict value
            tool_span = spans[3]
            assert tool_span.output == {"results": ["cat1", "cat2"]}

            # Index 0 is the "system" message with a string value
            system_span = spans[0]
            assert system_span.output == {"text": "You are a helpful assistant."}
        finally:
            os.unlink(path)

    async def test_hermes_metadata_fields(self):
        """Each span metadata must contain 'model' and 'role'."""
        path = _write_jsonl([_SAMPLE_ENTRY])
        try:
            source = HermesJsonlSource(path)
            await source.connect()
            spans = await source.poll()
            for span in spans:
                assert "model" in span.metadata
                assert "role" in span.metadata
                assert span.metadata["model"] == "hermes-3-llama-3.1-70b"
        finally:
            os.unlink(path)

    async def test_hermes_semconv_version(self):
        """All spans must carry semconv_version='hermes-sharegpt'."""
        path = _write_jsonl([_SAMPLE_ENTRY])
        try:
            source = HermesJsonlSource(path)
            await source.connect()
            spans = await source.poll()
            assert all(s.semconv_version == "hermes-sharegpt" for s in spans)
        finally:
            os.unlink(path)

    async def test_hermes_span_ids_unique(self):
        """All span_ids within a single entry must be unique."""
        path = _write_jsonl([_SAMPLE_ENTRY])
        try:
            source = HermesJsonlSource(path)
            await source.connect()
            spans = await source.poll()
            span_ids = [s.span_id for s in spans]
            assert len(span_ids) == len(set(span_ids))
        finally:
            os.unlink(path)
