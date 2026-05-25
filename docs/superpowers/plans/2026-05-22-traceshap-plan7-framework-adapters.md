# Framework Adapters Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OTLPLiveSource (real-time OTLP HTTP endpoint), HermesJsonlSource (ShareGPT JSONL import), and extract shared OTLP parsing logic into a common module.

**Architecture:** Extract shared OTLP parsing functions from OTLPJsonSource into otlp_common.py. OTLPLiveSource embeds a lightweight HTTP endpoint receiving POST /v1/traces, buffering spans for poll(). HermesJsonlSource converts ShareGPT JSONL conversation format into TraceSHAPSpan objects. Both implement the existing SpanSource ABC.

**Tech Stack:** Python aiohttp (for HTTP server), existing dataclasses, pytest, pytest-asyncio

---

### Task 1: Extract OTLP Common Parsing Module

**Files:**
- Create: `traceshap/ingestion/sources/otlp_common.py`
- Modify: `traceshap/ingestion/sources/otlp_json.py`
- Test: `tests/test_otlp_common.py`

- [ ] **Step 1: Write tests for shared parsing functions**

```python
# tests/test_otlp_common.py
from datetime import datetime, timezone

from traceshap.ingestion.sources.otlp_common import (
    get_attr,
    nano_to_datetime,
    attrs_to_dict,
    infer_span_kind,
    extract_tokens,
)
from traceshap.models.enums import SpanKind


def test_get_attr_string():
    attrs = [{"key": "foo", "value": {"stringValue": "bar"}}]
    assert get_attr(attrs, "foo") == "bar"
    assert get_attr(attrs, "missing") is None


def test_get_attr_int():
    attrs = [{"key": "count", "value": {"intValue": "42"}}]
    assert get_attr(attrs, "count") == 42


def test_get_attr_double():
    attrs = [{"key": "score", "value": {"doubleValue": 3.14}}]
    assert get_attr(attrs, "score") == 3.14


def test_nano_to_datetime():
    nanos = str(1_700_000_000_000_000_000)
    dt = nano_to_datetime(nanos)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc


def test_attrs_to_dict():
    attrs = [
        {"key": "a", "value": {"stringValue": "hello"}},
        {"key": "b", "value": {"intValue": "5"}},
        {"key": "c", "value": {"boolValue": True}},
    ]
    result = attrs_to_dict(attrs)
    assert result == {"a": "hello", "b": 5, "c": True}


def test_infer_span_kind_tool():
    attrs = [{"key": "tool.name", "value": {"stringValue": "web_search"}}]
    assert infer_span_kind(0, attrs) == SpanKind.TOOL


def test_infer_span_kind_llm():
    attrs = [{"key": "gen_ai.system", "value": {"stringValue": "openai"}}]
    assert infer_span_kind(0, attrs) == SpanKind.LLM


def test_infer_span_kind_default():
    assert infer_span_kind(1, []) == SpanKind.AGENT
    assert infer_span_kind(0, []) == SpanKind.CUSTOM


def test_extract_tokens_gen_ai():
    attrs = [
        {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "100"}},
        {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "50"}},
    ]
    tokens = extract_tokens(attrs)
    assert tokens is not None
    assert tokens.input_tokens == 100
    assert tokens.output_tokens == 50
    assert tokens.total_tokens == 150


def test_extract_tokens_none():
    assert extract_tokens([]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_otlp_common.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Create otlp_common.py by extracting from otlp_json.py**

```python
# traceshap/ingestion/sources/otlp_common.py
from __future__ import annotations

from datetime import datetime, timezone

from traceshap.models.span import TraceSHAPSpan, TokenUsage
from traceshap.models.enums import SpanKind

OTEL_KIND_MAP = {
    0: SpanKind.CUSTOM,
    1: SpanKind.AGENT,
    2: SpanKind.AGENT,
    3: SpanKind.LLM,
    4: SpanKind.LLM,
    5: SpanKind.CUSTOM,
}

TOOL_ATTRIBUTE_KEYS = frozenset({"tool.name", "traceloop.entity.name"})


def get_attr(attributes: list[dict], key: str) -> str | int | float | None:
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
    nanos = int(nano_str)
    seconds = nanos / 1_000_000_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def attrs_to_dict(attributes: list[dict]) -> dict:
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


def infer_span_kind(otel_kind: int, attributes: list[dict]) -> SpanKind:
    for key in TOOL_ATTRIBUTE_KEYS:
        if get_attr(attributes, key) is not None:
            return SpanKind.TOOL

    if get_attr(attributes, "gen_ai.system") is not None:
        return SpanKind.LLM

    return OTEL_KIND_MAP.get(otel_kind, SpanKind.CUSTOM)


def extract_tokens(attributes: list[dict]) -> TokenUsage | None:
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


def convert_otlp_span(
    raw: dict,
    source_hint: str = "otlp",
) -> TraceSHAPSpan | None:
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


def parse_otlp_resource_spans(data: dict, source_hint: str = "otlp") -> list[TraceSHAPSpan]:
    spans: list[TraceSHAPSpan] = []
    for resource_span in data.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            for raw_span in scope_span.get("spans", []):
                span = convert_otlp_span(raw_span, source_hint)
                if span:
                    spans.append(span)
    return spans
```

- [ ] **Step 4: Refactor otlp_json.py to use otlp_common**

```python
# traceshap/ingestion/sources/otlp_json.py
from __future__ import annotations

import json
from pathlib import Path

from traceshap.models.span import TraceSHAPSpan
from traceshap.ingestion.sources.base import SpanSource
from traceshap.ingestion.sources.otlp_common import parse_otlp_resource_spans


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
            spans = parse_otlp_resource_spans(data, self._source_hint)
            all_spans.extend(spans)

        return all_spans

    async def close(self) -> None:
        pass
```

- [ ] **Step 5: Run all OTLP tests to verify refactor**

Run: `python -m pytest tests/test_otlp_common.py tests/test_otlp_json_source.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add traceshap/ingestion/sources/otlp_common.py traceshap/ingestion/sources/otlp_json.py tests/test_otlp_common.py
git commit -m "refactor: extract OTLP parsing into otlp_common module"
```

---

### Task 2: OTLPLiveSource (Real-time OTLP HTTP Endpoint)

**Files:**
- Create: `traceshap/ingestion/sources/otlp_live.py`
- Test: `tests/test_otlp_live_source.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_otlp_live_source.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock

from traceshap.ingestion.sources.otlp_live import OTLPLiveSource


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


def test_otlp_live_source_creation():
    source = OTLPLiveSource(host="127.0.0.1", port=0)
    assert source._host == "127.0.0.1"


def test_ingest_spans_to_buffer():
    source = OTLPLiveSource(host="127.0.0.1", port=0)
    source.ingest(SAMPLE_OTLP_PAYLOAD)
    assert len(source._buffer) == 2


@pytest.mark.asyncio
async def test_poll_returns_and_clears_buffer():
    source = OTLPLiveSource(host="127.0.0.1", port=0)
    source.ingest(SAMPLE_OTLP_PAYLOAD)
    spans = await source.poll()
    assert len(spans) == 2
    assert spans[0].trace_id == "abc123"
    # Buffer should be cleared
    spans2 = await source.poll()
    assert len(spans2) == 0


@pytest.mark.asyncio
async def test_poll_empty_when_no_data():
    source = OTLPLiveSource(host="127.0.0.1", port=0)
    spans = await source.poll()
    assert spans == []


def test_ingest_with_source_hint():
    source = OTLPLiveSource(host="127.0.0.1", port=0, source_hint="openclaw")
    source.ingest(SAMPLE_OTLP_PAYLOAD)
    assert len(source._buffer) == 2
    assert source._buffer[0].semconv_version == "otlp-openclaw"


def test_max_buffer_size():
    source = OTLPLiveSource(host="127.0.0.1", port=0, max_buffer_size=1)
    source.ingest(SAMPLE_OTLP_PAYLOAD)
    assert len(source._buffer) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_otlp_live_source.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement OTLPLiveSource**

```python
# traceshap/ingestion/sources/otlp_live.py
from __future__ import annotations

import threading

from traceshap.models.span import TraceSHAPSpan
from traceshap.ingestion.sources.base import SpanSource
from traceshap.ingestion.sources.otlp_common import parse_otlp_resource_spans


class OTLPLiveSource(SpanSource):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4318,
        source_hint: str = "otlp",
        auth_token: str | None = None,
        max_buffer_size: int = 10000,
    ):
        self._host = host
        self._port = port
        self._source_hint = source_hint
        self._auth_token = auth_token
        self._max_buffer_size = max_buffer_size
        self._buffer: list[TraceSHAPSpan] = []
        self._lock = threading.Lock()
        self._app = None
        self._runner = None

    async def connect(self) -> None:
        try:
            from aiohttp import web
        except ImportError:
            return

        self._app = web.Application()
        self._app.router.add_post("/v1/traces", self._handle_traces)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    async def _handle_traces(self, request) -> "web.Response":
        from aiohttp import web

        if self._auth_token:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {self._auth_token}":
                return web.Response(status=401, text="Unauthorized")

        data = await request.json()
        self.ingest(data)
        return web.Response(status=200, text="OK")

    def ingest(self, data: dict) -> None:
        spans = parse_otlp_resource_spans(data, self._source_hint)
        with self._lock:
            remaining = self._max_buffer_size - len(self._buffer)
            self._buffer.extend(spans[:max(0, remaining)])

    async def poll(self) -> list[TraceSHAPSpan]:
        with self._lock:
            spans = list(self._buffer)
            self._buffer.clear()
        return spans

    async def close(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_otlp_live_source.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add traceshap/ingestion/sources/otlp_live.py tests/test_otlp_live_source.py
git commit -m "feat: add OTLPLiveSource for real-time OTLP HTTP span ingestion"
```

---

### Task 3: HermesJsonlSource

**Files:**
- Create: `traceshap/ingestion/sources/hermes_jsonl.py`
- Test: `tests/test_hermes_jsonl_source.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hermes_jsonl_source.py
import json
import os
import tempfile
import pytest

from traceshap.ingestion.sources.hermes_jsonl import HermesJsonlSource
from traceshap.models.enums import SpanKind


SAMPLE_ENTRY = {
    "conversations": [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": "Search for cats"},
        {"from": "gpt", "value": "<think>I should search</think>\nSearching..."},
        {"from": "tool", "value": {"results": ["cat1", "cat2"]}},
        {"from": "gpt", "value": "Found 2 results about cats."},
    ],
    "timestamp": "2026-05-22T10:30:00Z",
    "model": "hermes-3-llama-3.1-70b",
    "completed": True,
}


def _write_jsonl(entries: list[dict]) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


@pytest.mark.asyncio
async def test_hermes_source_single_entry():
    path = _write_jsonl([SAMPLE_ENTRY])
    try:
        source = HermesJsonlSource(path)
        await source.connect()
        spans = await source.poll()
        assert len(spans) == 5
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_hermes_span_kinds():
    path = _write_jsonl([SAMPLE_ENTRY])
    try:
        source = HermesJsonlSource(path)
        await source.connect()
        spans = await source.poll()
        kinds = [s.span_kind for s in spans]
        assert kinds[0] == SpanKind.CUSTOM      # system
        assert kinds[1] == SpanKind.CUSTOM      # human
        assert kinds[2] == SpanKind.LLM         # gpt
        assert kinds[3] == SpanKind.TOOL        # tool
        assert kinds[4] == SpanKind.LLM         # gpt
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_hermes_poll_once():
    path = _write_jsonl([SAMPLE_ENTRY])
    try:
        source = HermesJsonlSource(path)
        await source.connect()
        spans1 = await source.poll()
        spans2 = await source.poll()
        assert len(spans1) == 5
        assert len(spans2) == 0
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_hermes_multiple_entries():
    path = _write_jsonl([SAMPLE_ENTRY, SAMPLE_ENTRY])
    try:
        source = HermesJsonlSource(path)
        await source.connect()
        spans = await source.poll()
        assert len(spans) == 10
        trace_ids = {s.trace_id for s in spans}
        assert len(trace_ids) == 2
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_hermes_trace_id_deterministic():
    path = _write_jsonl([SAMPLE_ENTRY])
    try:
        source1 = HermesJsonlSource(path)
        await source1.connect()
        spans1 = await source1.poll()

        source2 = HermesJsonlSource(path)
        await source2.connect()
        spans2 = await source2.poll()

        assert spans1[0].trace_id == spans2[0].trace_id
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_hermes_connect_missing_file():
    source = HermesJsonlSource("/nonexistent/path.jsonl")
    with pytest.raises(FileNotFoundError):
        await source.connect()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hermes_jsonl_source.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement HermesJsonlSource**

```python
# traceshap/ingestion/sources/hermes_jsonl.py
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from traceshap.models.span import TraceSHAPSpan
from traceshap.models.enums import SpanKind
from traceshap.ingestion.sources.base import SpanSource

ROLE_TO_KIND = {
    "system": SpanKind.CUSTOM,
    "human": SpanKind.CUSTOM,
    "gpt": SpanKind.LLM,
    "tool": SpanKind.TOOL,
}


class HermesJsonlSource(SpanSource):
    def __init__(self, path: str):
        self._path = Path(path)
        self._polled = False

    async def connect(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Hermes trajectory not found: {self._path}")

    async def poll(self) -> list[TraceSHAPSpan]:
        if self._polled:
            return []
        self._polled = True

        spans: list[TraceSHAPSpan] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                spans.extend(self._convert_entry(entry))
        return spans

    async def close(self) -> None:
        pass

    def _convert_entry(self, entry: dict) -> list[TraceSHAPSpan]:
        trace_id = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()[:32]

        ts_str = entry.get("timestamp")
        base_time = (
            datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts_str
            else datetime.now(timezone.utc)
        )
        model = entry.get("model", "unknown")

        conversations = entry.get("conversations", [])
        spans: list[TraceSHAPSpan] = []

        for i, msg in enumerate(conversations):
            role = msg.get("from", "unknown")
            value = msg.get("value", "")

            span_kind = ROLE_TO_KIND.get(role, SpanKind.CUSTOM)

            if isinstance(value, dict):
                input_data = value
                output_data = value
                name = f"{role}_structured"
            else:
                input_data = {"text": str(value)}
                output_data = {"text": str(value)}
                name = f"{role}_{i}"

            span_id = hashlib.sha256(
                f"{trace_id}_{i}".encode()
            ).hexdigest()[:16]

            spans.append(TraceSHAPSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=None,
                span_kind=span_kind,
                name=name,
                input=input_data,
                output=output_data,
                start_time=base_time,
                end_time=base_time,
                tokens=None,
                cost=None,
                metadata={"model": model, "role": role},
                raw_attributes={},
                semconv_version="hermes-sharegpt",
            ))

        return spans
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hermes_jsonl_source.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add traceshap/ingestion/sources/hermes_jsonl.py tests/test_hermes_jsonl_source.py
git commit -m "feat: add HermesJsonlSource for ShareGPT JSONL trajectory import"
```

---

### Task 4: Update Factory, Config, and Source Exports

**Files:**
- Modify: `traceshap/ingestion/sources/factory.py`
- Modify: `traceshap/config.py`
- Test: `tests/test_source_factory.py` (update existing)

- [ ] **Step 1: Update factory to support new source types**

```python
# traceshap/ingestion/sources/factory.py
from traceshap.config import SourceConfig
from traceshap.ingestion.sources.base import SpanSource
from traceshap.ingestion.sources.langfuse import LangfuseSource
from traceshap.ingestion.sources.otlp_json import OTLPJsonSource


def create_source(config: SourceConfig) -> SpanSource:
    if config.type == "langfuse":
        return LangfuseSource(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
            poll_batch_size=config.poll_interval_seconds,
        )

    if config.type == "otlp_json":
        return OTLPJsonSource(
            path=config.otlp_endpoint,
            source_hint=config.source_hint or "otlp",
        )

    if config.type == "otlp_live":
        from traceshap.ingestion.sources.otlp_live import OTLPLiveSource
        return OTLPLiveSource(
            host=config.otlp_live_host or "0.0.0.0",
            port=config.otlp_live_port or 4318,
            source_hint=config.source_hint or "otlp",
            auth_token=config.otlp_live_auth_token or None,
            max_buffer_size=config.otlp_live_max_buffer or 10000,
        )

    if config.type == "hermes_jsonl":
        from traceshap.ingestion.sources.hermes_jsonl import HermesJsonlSource
        return HermesJsonlSource(path=config.otlp_endpoint)

    raise ValueError(
        f"Unknown source type: '{config.type}'. "
        f"Supported: langfuse, otlp_json, otlp_live, hermes_jsonl"
    )
```

- [ ] **Step 2: Add new config fields to SourceConfig**

Add the following fields to `SourceConfig` in `traceshap/config.py`:

```python
@dataclass
class SourceConfig:
    type: str = "langfuse"
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    poll_interval_seconds: int = 10
    otlp_endpoint: str = ""
    source_hint: str = ""
    otlp_live_host: str = ""
    otlp_live_port: int = 4318
    otlp_live_auth_token: str = ""
    otlp_live_max_buffer: int = 10000
```

- [ ] **Step 3: Write tests for new factory cases**

Add to existing `tests/test_source_factory.py`:

```python
def test_create_otlp_live_source():
    config = SourceConfig(
        type="otlp_live",
        source_hint="openclaw",
        otlp_live_host="127.0.0.1",
        otlp_live_port=4320,
    )
    source = create_source(config)
    from traceshap.ingestion.sources.otlp_live import OTLPLiveSource
    assert isinstance(source, OTLPLiveSource)


def test_create_hermes_jsonl_source(tmp_path):
    jsonl_file = tmp_path / "trajectories.jsonl"
    jsonl_file.write_text("")
    config = SourceConfig(
        type="hermes_jsonl",
        otlp_endpoint=str(jsonl_file),
    )
    source = create_source(config)
    from traceshap.ingestion.sources.hermes_jsonl import HermesJsonlSource
    assert isinstance(source, HermesJsonlSource)


def test_create_unknown_source_error_message():
    config = SourceConfig(type="magic")
    try:
        create_source(config)
        assert False, "Should raise"
    except ValueError as e:
        assert "hermes_jsonl" in str(e)
        assert "otlp_live" in str(e)
```

- [ ] **Step 4: Run all factory tests**

Run: `python -m pytest tests/test_source_factory.py -v`
Expected: All pass (old + 3 new)

- [ ] **Step 5: Commit**

```bash
git add traceshap/ingestion/sources/factory.py traceshap/config.py tests/test_source_factory.py
git commit -m "feat: add otlp_live and hermes_jsonl to source factory and config"
```

---

### Task 5: Integration Tests

**Files:**
- Create: `tests/test_adapters_integration.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/test_adapters_integration.py
import json
import os
import tempfile
import pytest

from traceshap.ingestion.sources.otlp_json import OTLPJsonSource
from traceshap.ingestion.sources.otlp_live import OTLPLiveSource
from traceshap.ingestion.sources.hermes_jsonl import HermesJsonlSource
from traceshap.ingestion.sources.otlp_common import parse_otlp_resource_spans
from traceshap.models.enums import SpanKind


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


@pytest.mark.asyncio
async def test_otlp_json_and_common_produce_same_spans():
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(OTLP_DATA, f)

        source = OTLPJsonSource(path=path, source_hint="test")
        await source.connect()
        json_spans = await source.poll()

        common_spans = parse_otlp_resource_spans(OTLP_DATA, "test")

        assert len(json_spans) == len(common_spans)
        for js, cs in zip(json_spans, common_spans):
            assert js.trace_id == cs.trace_id
            assert js.span_id == cs.span_id
            assert js.span_kind == cs.span_kind
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_otlp_live_source_ingest_matches_json_source():
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(OTLP_DATA, f)

        json_source = OTLPJsonSource(path=path, source_hint="test")
        await json_source.connect()
        json_spans = await json_source.poll()

        live_source = OTLPLiveSource(host="127.0.0.1", port=0, source_hint="test")
        live_source.ingest(OTLP_DATA)
        live_spans = await live_source.poll()

        assert len(json_spans) == len(live_spans)
        for js, ls in zip(json_spans, live_spans):
            assert js.span_kind == ls.span_kind
            assert js.name == ls.name
    finally:
        os.unlink(path)


HERMES_ENTRY = {
    "conversations": [
        {"from": "system", "value": "You are helpful."},
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "Hi there!"},
    ],
    "timestamp": "2026-05-22T10:00:00Z",
    "model": "hermes-3",
}


@pytest.mark.asyncio
async def test_hermes_source_span_structure():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(HERMES_ENTRY) + "\n")

        source = HermesJsonlSource(path)
        await source.connect()
        spans = await source.poll()

        assert len(spans) == 3
        assert all(s.trace_id == spans[0].trace_id for s in spans)
        assert spans[0].span_kind == SpanKind.CUSTOM
        assert spans[2].span_kind == SpanKind.LLM
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_all_sources_implement_interface():
    from traceshap.ingestion.sources.base import SpanSource

    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(OTLP_DATA, f)

        sources = [
            OTLPJsonSource(path=path),
            OTLPLiveSource(host="127.0.0.1", port=0),
        ]

        for source in sources:
            assert isinstance(source, SpanSource)
            assert hasattr(source, "connect")
            assert hasattr(source, "poll")
            assert hasattr(source, "close")
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_adapters_integration.py -v`
Expected: 4 passed

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_adapters_integration.py
git commit -m "test: add integration tests for OTLP common, live source, and Hermes adapter"
```
