# TraceSHAP Plan 4: Framework Adapters + Source Factory

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LangGraph native adapter (`instrument()`), OTLP JSON file source (for OpenClaw/generic OTel import), source factory, and top-level convenience API so users can integrate TraceSHAP with real frameworks.

**Architecture:** LangGraph adapter wraps a compiled graph's callbacks to capture spans in real-time. OTLP JSON source reads exported traces from files (common export path for OpenClaw/Hermes). Source factory instantiates the right source from config. Top-level `traceshap.instrument()` and `traceshap.analyze()` provide the simplest possible API.

**Tech Stack:** langgraph (optional dependency), existing TraceSHAP core

---

### Task 1: LangGraph Callback Handler

**Files:**
- Create: `traceshap/adapters/__init__.py`
- Create: `traceshap/adapters/langgraph.py`
- Create: `tests/test_langgraph_adapter.py`

- [ ] **Step 1: Write tests**

`tests/test_langgraph_adapter.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.models import TraceSHAPSpan, SpanKind, TokenUsage
from traceshap.adapters.langgraph import LangGraphSpanCollector


class TestLangGraphSpanCollector:
    def test_on_llm_start_end(self):
        collector = LangGraphSpanCollector(trace_id="t1")

        collector.on_llm_start(
            run_id="run1",
            name="gpt-4o",
            inputs={"messages": [{"role": "user", "content": "hi"}]},
            metadata={"node_id": "agent"},
            parent_run_id=None,
        )
        collector.on_llm_end(
            run_id="run1",
            outputs={"text": "hello"},
            token_usage={"input": 10, "output": 20, "total": 30},
        )

        spans = collector.get_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.trace_id == "t1"
        assert span.span_kind == SpanKind.LLM
        assert span.name == "gpt-4o"
        assert span.tokens is not None
        assert span.tokens.total_tokens == 30
        assert span.metadata.get("node_id") == "agent"

    def test_on_tool_start_end(self):
        collector = LangGraphSpanCollector(trace_id="t1")

        collector.on_tool_start(
            run_id="run2",
            name="search_web",
            inputs={"query": "test"},
            metadata={},
            parent_run_id="run1",
        )
        collector.on_tool_end(
            run_id="run2",
            outputs={"results": ["found"]},
        )

        spans = collector.get_spans()
        assert len(spans) == 1
        assert spans[0].span_kind == SpanKind.TOOL
        assert spans[0].name == "search_web"
        assert spans[0].parent_span_id == "run1"

    def test_on_chain_start_end(self):
        collector = LangGraphSpanCollector(trace_id="t1")

        collector.on_chain_start(
            run_id="run3",
            name="agent_node",
            inputs={"state": {}},
            metadata={"node_id": "planner"},
            parent_run_id=None,
        )
        collector.on_chain_end(
            run_id="run3",
            outputs={"next": "tool_call"},
        )

        spans = collector.get_spans()
        assert len(spans) == 1
        assert spans[0].span_kind == SpanKind.AGENT
        assert spans[0].metadata.get("node_id") == "planner"

    def test_multiple_events(self):
        collector = LangGraphSpanCollector(trace_id="t1")

        collector.on_chain_start(run_id="r1", name="agent", inputs={}, metadata={"node_id": "a"}, parent_run_id=None)
        collector.on_llm_start(run_id="r2", name="gpt-4o", inputs={"prompt": "hi"}, metadata={}, parent_run_id="r1")
        collector.on_llm_end(run_id="r2", outputs={"text": "hello"}, token_usage={"input": 5, "output": 10, "total": 15})
        collector.on_tool_start(run_id="r3", name="search", inputs={"q": "x"}, metadata={}, parent_run_id="r1")
        collector.on_tool_end(run_id="r3", outputs={"r": "y"})
        collector.on_chain_end(run_id="r1", outputs={"done": True})

        spans = collector.get_spans()
        assert len(spans) == 3
        kinds = {s.span_kind for s in spans}
        assert SpanKind.AGENT in kinds
        assert SpanKind.LLM in kinds
        assert SpanKind.TOOL in kinds

    def test_clear_resets(self):
        collector = LangGraphSpanCollector(trace_id="t1")
        collector.on_chain_start(run_id="r1", name="a", inputs={}, metadata={}, parent_run_id=None)
        collector.on_chain_end(run_id="r1", outputs={})
        assert len(collector.get_spans()) == 1
        collector.clear()
        assert len(collector.get_spans()) == 0

    def test_high_mapping_confidence(self):
        collector = LangGraphSpanCollector(trace_id="t1")
        collector.on_chain_start(run_id="r1", name="node", inputs={}, metadata={"node_id": "x"}, parent_run_id=None)
        collector.on_chain_end(run_id="r1", outputs={})
        span = collector.get_spans()[0]
        assert span.metadata.get("node_id") == "x"
        assert span.semconv_version == "langgraph-native"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_langgraph_adapter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement LangGraphSpanCollector**

`traceshap/adapters/__init__.py`:
```python
```

`traceshap/adapters/langgraph.py`:
```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_langgraph_adapter.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/adapters/ tests/test_langgraph_adapter.py
git commit -m "feat: LangGraph span collector with LLM/tool/chain event capture"
```

---

### Task 2: OTLP JSON File Source

**Files:**
- Create: `traceshap/ingestion/sources/otlp_json.py`
- Create: `tests/test_otlp_json_source.py`

- [ ] **Step 1: Write tests**

`tests/test_otlp_json_source.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_otlp_json_source.py -v`
Expected: FAIL

- [ ] **Step 3: Implement OTLP JSON source**

`traceshap/ingestion/sources/otlp_json.py`:
```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_otlp_json_source.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/ingestion/sources/otlp_json.py tests/test_otlp_json_source.py
git commit -m "feat: OTLP JSON file source for importing OTel trace exports"
```

---

### Task 3: Source Factory

**Files:**
- Create: `traceshap/ingestion/sources/factory.py`
- Create: `tests/test_source_factory.py`

- [ ] **Step 1: Write tests**

`tests/test_source_factory.py`:
```python
import pytest
import json

from traceshap.config import SourceConfig
from traceshap.ingestion.sources.factory import create_source
from traceshap.ingestion.sources.langfuse import LangfuseSource
from traceshap.ingestion.sources.otlp_json import OTLPJsonSource


class TestSourceFactory:
    def test_create_langfuse_source(self):
        config = SourceConfig(
            type="langfuse",
            langfuse_host="https://cloud.langfuse.com",
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        )
        source = create_source(config)
        assert isinstance(source, LangfuseSource)

    def test_create_otlp_json_source(self, tmp_path):
        data = {"resourceSpans": []}
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(data))

        config = SourceConfig(type="otlp_json", otlp_endpoint=str(path))
        source = create_source(config)
        assert isinstance(source, OTLPJsonSource)

    def test_unknown_type_raises(self):
        config = SourceConfig(type="unknown_source")
        with pytest.raises(ValueError, match="Unknown source type"):
            create_source(config)

    def test_otlp_json_with_source_hint(self, tmp_path):
        data = {"resourceSpans": []}
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(data))

        config = SourceConfig(type="otlp_json", otlp_endpoint=str(path), source_hint="openclaw")
        source = create_source(config)
        assert isinstance(source, OTLPJsonSource)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_source_factory.py -v`
Expected: FAIL

- [ ] **Step 3: Implement source factory**

`traceshap/ingestion/sources/factory.py`:
```python
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

    raise ValueError(f"Unknown source type: '{config.type}'. Supported: langfuse, otlp_json")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_source_factory.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/ingestion/sources/factory.py tests/test_source_factory.py
git commit -m "feat: source factory for creating SpanSource from config"
```

---

### Task 4: Top-Level Convenience API

**Files:**
- Create: `traceshap/convenience.py`
- Modify: `traceshap/__init__.py`
- Create: `tests/test_convenience.py`

- [ ] **Step 1: Write tests**

`tests/test_convenience.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.models import (
    TraceSHAPSpan, SpanKind, TokenUsage, SpanNode,
    TrajectoryMeta, Trajectory, Outcome, Verdict,
)
from traceshap.convenience import quick_analyze, spans_to_trajectory


class TestSpansToTrajectory:
    def test_converts_spans(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spans = [
            TraceSHAPSpan(
                trace_id="t1", span_id="s1", parent_span_id=None,
                span_kind=SpanKind.AGENT, name="agent",
                input={}, output={"result": "ok"},
                start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
                tokens=None, cost=None, metadata={},
                raw_attributes={}, semconv_version="test",
            ),
            TraceSHAPSpan(
                trace_id="t1", span_id="s2", parent_span_id="s1",
                span_kind=SpanKind.LLM, name="gpt-4o",
                input={"prompt": "hi"}, output={"text": "hello"},
                start_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                tokens=TokenUsage(10, 20, 30), cost=0.001,
                metadata={}, raw_attributes={}, semconv_version="test",
            ),
        ]
        traj = spans_to_trajectory(spans, trace_id="t1")
        assert traj.trace_id == "t1"
        assert len(traj.spans) == 2
        assert len(traj.steps) == 2
        assert traj.span_tree is not None

    def test_with_outcome(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spans = [
            TraceSHAPSpan(
                trace_id="t1", span_id="s1", parent_span_id=None,
                span_kind=SpanKind.TOOL, name="search",
                input={"q": "test"}, output={"r": "found"},
                start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                tokens=None, cost=None, metadata={},
                raw_attributes={}, semconv_version="test",
            ),
        ]
        outcome = Outcome(success=True, quality_score=0.9, token_cost=30,
                          latency_ms=1000, custom_metrics={})
        traj = spans_to_trajectory(spans, trace_id="t1", outcome=outcome)
        assert traj.outcome is not None
        assert traj.outcome.quality_score == 0.9


class TestQuickAnalyze:
    async def test_analyze_spans(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spans = [
            TraceSHAPSpan(
                trace_id="t1", span_id="s1", parent_span_id=None,
                span_kind=SpanKind.AGENT, name="agent",
                input={}, output={},
                start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
                tokens=None, cost=None, metadata={},
                raw_attributes={}, semconv_version="test",
            ),
            TraceSHAPSpan(
                trace_id="t1", span_id="s2", parent_span_id="s1",
                span_kind=SpanKind.LLM, name="gpt-4o",
                input={"prompt": "hi"}, output={"text": "hello"},
                start_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                tokens=TokenUsage(10, 20, 30), cost=0.001,
                metadata={}, raw_attributes={}, semconv_version="test",
            ),
        ]
        result = await quick_analyze(spans, trace_id="t1", layers=[0])
        assert "attributions" in result
        assert len(result["attributions"]) == 2
        assert "trajectory" in result

    async def test_analyze_with_pruning(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spans = [
            TraceSHAPSpan(
                trace_id="t1", span_id="s1", parent_span_id=None,
                span_kind=SpanKind.AGENT, name="agent",
                input={}, output={},
                start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
                tokens=None, cost=None, metadata={},
                raw_attributes={}, semconv_version="test",
            ),
        ]
        outcome = Outcome(success=True, quality_score=0.9, token_cost=30,
                          latency_ms=5000, custom_metrics={})
        result = await quick_analyze(spans, trace_id="t1", layers=[0],
                                     outcome=outcome, include_pruning=True)
        assert "pruning_report" in result
        assert result["pruning_report"].total_steps == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_convenience.py -v`
Expected: FAIL

- [ ] **Step 3: Implement convenience module**

`traceshap/convenience.py`:
```python
from __future__ import annotations

from traceshap.models.span import TraceSHAPSpan
from traceshap.models.trajectory import Trajectory, SpanNode, TrajectoryMeta
from traceshap.models.outcome import Outcome, StepAttribution
from traceshap.models.pruning import PruningReport
from traceshap.ingestion.normalizer import StepNormalizer
from traceshap.ingestion.assembler import TreeAssembler
from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer0_rules import Layer0Rules
from traceshap.attribution.layer1_lift import Layer1Lift
from traceshap.attribution.layer2_sequence import Layer2Sequence
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig


def spans_to_trajectory(
    spans: list[TraceSHAPSpan],
    trace_id: str,
    outcome: Outcome | None = None,
    framework: str = "unknown",
    agent_name: str = "default",
) -> Trajectory:
    sorted_spans = sorted(spans, key=lambda s: s.start_time)
    normalizer = StepNormalizer()
    steps = normalizer.normalize(sorted_spans)
    span_tree = TreeAssembler.build(sorted_spans)

    return Trajectory(
        trace_id=trace_id,
        spans=sorted_spans,
        steps=steps,
        span_tree=span_tree,
        outcome=outcome,
        metadata=TrajectoryMeta(
            framework=framework,
            agent_name=agent_name,
        ),
    )


async def quick_analyze(
    spans: list[TraceSHAPSpan],
    trace_id: str,
    layers: list[int] | None = None,
    outcome: Outcome | None = None,
    framework: str = "unknown",
    agent_name: str = "default",
    include_pruning: bool = False,
    training_trajectories: list[Trajectory] | None = None,
) -> dict:
    if layers is None:
        layers = [0]

    trajectory = spans_to_trajectory(
        spans, trace_id=trace_id, outcome=outcome,
        framework=framework, agent_name=agent_name,
    )

    layer_objs = []
    for layer_id in layers:
        if layer_id == 0:
            layer_objs.append(Layer0Rules())
        elif layer_id == 1:
            l1 = Layer1Lift()
            if training_trajectories:
                l1.fit(training_trajectories)
            layer_objs.append(l1)
        elif layer_id == 2:
            l2 = Layer2Sequence()
            if training_trajectories:
                l2.fit(training_trajectories)
            layer_objs.append(l2)

    engine = AttributionEngine(layers=layer_objs)
    attributions = await engine.analyze(trajectory)

    result: dict = {
        "trajectory": trajectory,
        "attributions": attributions,
    }

    if include_pruning:
        config = PruningConfig()
        advisor = PruningAdvisor(config)
        report = advisor.analyze(trajectory, attributions)
        result["pruning_report"] = report

    return result
```

- [ ] **Step 4: Update `traceshap/__init__.py` with convenience re-exports**

Read the current `traceshap/__init__.py` first. Then add these imports at the bottom (before `__all__`):
```python
from traceshap.convenience import spans_to_trajectory, quick_analyze
```

And add `"spans_to_trajectory"` and `"quick_analyze"` to the `__all__` list.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_convenience.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add traceshap/convenience.py traceshap/__init__.py tests/test_convenience.py
git commit -m "feat: convenience API (spans_to_trajectory, quick_analyze) with top-level exports"
```

---

### Task 5: LangGraph instrument() Wrapper

**Files:**
- Create: `traceshap/adapters/instrument.py`
- Modify: `traceshap/__init__.py`
- Create: `tests/test_instrument.py`

- [ ] **Step 1: Write tests**

`tests/test_instrument.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.adapters.instrument import InstrumentedApp
from traceshap.models import TraceSHAPSpan, SpanKind


class FakeGraph:
    """Simulates a LangGraph compiled graph for testing."""
    def __init__(self):
        self.callbacks = []

    async def ainvoke(self, inputs: dict, config: dict | None = None):
        for cb in self.callbacks:
            cb.on_chain_start(run_id="r1", name="agent", inputs=inputs,
                              metadata={"node_id": "agent"}, parent_run_id=None)
            cb.on_llm_start(run_id="r2", name="gpt-4o",
                            inputs={"prompt": "test"}, metadata={},
                            parent_run_id="r1")
            cb.on_llm_end(run_id="r2", outputs={"text": "response"},
                          token_usage={"input": 10, "output": 20, "total": 30})
            cb.on_tool_start(run_id="r3", name="search", inputs={"q": "test"},
                             metadata={}, parent_run_id="r1")
            cb.on_tool_end(run_id="r3", outputs={"result": "found"})
            cb.on_chain_end(run_id="r1", outputs={"result": "done"})
        return {"result": "done"}


class TestInstrumentedApp:
    async def test_captures_spans(self):
        graph = FakeGraph()
        app = InstrumentedApp(graph, framework="langgraph")

        result = await app.ainvoke({"input": "test"})
        assert result == {"result": "done"}

        spans = app.get_spans()
        assert len(spans) == 3
        kinds = {s.span_kind for s in spans}
        assert SpanKind.AGENT in kinds
        assert SpanKind.LLM in kinds
        assert SpanKind.TOOL in kinds

    async def test_trace_id_generated(self):
        graph = FakeGraph()
        app = InstrumentedApp(graph, framework="langgraph")
        await app.ainvoke({"input": "test"})

        spans = app.get_spans()
        trace_ids = {s.trace_id for s in spans}
        assert len(trace_ids) == 1
        assert list(trace_ids)[0] != ""

    async def test_multiple_invocations(self):
        graph = FakeGraph()
        app = InstrumentedApp(graph, framework="langgraph")

        await app.ainvoke({"input": "test1"})
        await app.ainvoke({"input": "test2"})

        traces = app.get_traces()
        assert len(traces) == 2
        trace_ids = {t_id for t_id in traces}
        assert len(trace_ids) == 2

    async def test_analyze_last(self):
        graph = FakeGraph()
        app = InstrumentedApp(graph, framework="langgraph")
        await app.ainvoke({"input": "test"})

        result = await app.analyze_last(layers=[0])
        assert "attributions" in result
        assert "trajectory" in result
        assert len(result["attributions"]) == 3

    async def test_framework_metadata(self):
        graph = FakeGraph()
        app = InstrumentedApp(graph, framework="langgraph")
        await app.ainvoke({"input": "test"})

        spans = app.get_spans()
        assert all(s.semconv_version == "langgraph-native" for s in spans)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_instrument.py -v`
Expected: FAIL

- [ ] **Step 3: Implement InstrumentedApp**

`traceshap/adapters/instrument.py`:
```python
from __future__ import annotations

import uuid

from traceshap.models.span import TraceSHAPSpan
from traceshap.models.outcome import Outcome
from traceshap.adapters.langgraph import LangGraphSpanCollector
from traceshap.convenience import quick_analyze


class InstrumentedApp:
    def __init__(self, graph, framework: str = "langgraph", agent_name: str = "default"):
        self._graph = graph
        self._framework = framework
        self._agent_name = agent_name
        self._traces: dict[str, list[TraceSHAPSpan]] = {}
        self._last_trace_id: str | None = None

    async def ainvoke(self, inputs: dict, config: dict | None = None, **kwargs):
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        collector = LangGraphSpanCollector(trace_id=trace_id)

        self._graph.callbacks = [collector]
        result = await self._graph.ainvoke(inputs, config=config, **kwargs)

        spans = collector.get_spans()
        self._traces[trace_id] = spans
        self._last_trace_id = trace_id

        return result

    def get_spans(self, trace_id: str | None = None) -> list[TraceSHAPSpan]:
        tid = trace_id or self._last_trace_id
        if tid is None:
            return []
        return self._traces.get(tid, [])

    def get_traces(self) -> dict[str, list[TraceSHAPSpan]]:
        return dict(self._traces)

    async def analyze_last(
        self,
        layers: list[int] | None = None,
        outcome: Outcome | None = None,
        include_pruning: bool = False,
    ) -> dict:
        if self._last_trace_id is None:
            raise RuntimeError("No traces captured yet. Call ainvoke() first.")

        spans = self._traces[self._last_trace_id]
        return await quick_analyze(
            spans,
            trace_id=self._last_trace_id,
            layers=layers,
            outcome=outcome,
            framework=self._framework,
            agent_name=self._agent_name,
            include_pruning=include_pruning,
        )
```

- [ ] **Step 4: Add `instrument()` function to `traceshap/__init__.py`**

Add this import and function:
```python
from traceshap.adapters.instrument import InstrumentedApp

def instrument(graph, framework: str = "langgraph", agent_name: str = "default") -> InstrumentedApp:
    return InstrumentedApp(graph, framework=framework, agent_name=agent_name)
```

And add `"instrument"`, `"InstrumentedApp"` to `__all__`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_instrument.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add traceshap/adapters/instrument.py traceshap/__init__.py tests/test_instrument.py
git commit -m "feat: instrument() wrapper for LangGraph with automatic span capture and analysis"
```

---

### Task 6: Full Integration Test

**Files:**
- Create: `tests/test_adapter_integration.py`

- [ ] **Step 1: Write integration test**

`tests/test_adapter_integration.py`:
```python
import pytest
import json
from datetime import datetime, timezone

from traceshap.adapters.langgraph import LangGraphSpanCollector
from traceshap.adapters.instrument import InstrumentedApp
from traceshap.ingestion.sources.otlp_json import OTLPJsonSource
from traceshap.ingestion.sources.factory import create_source
from traceshap.ingestion.normalizer import StepNormalizer
from traceshap.ingestion.assembler import TreeAssembler
from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer0_rules import Layer0Rules
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.convenience import spans_to_trajectory, quick_analyze
from traceshap.config import SourceConfig, PruningConfig
from traceshap.models import Outcome, Verdict
from traceshap.storage.sqlite import SQLiteBackend
from traceshap.pipeline import TraceSHAPPipeline
from traceshap.ingestion.sources.base import SpanSource
from traceshap.models import TraceSHAPSpan


class FakeGraph:
    def __init__(self):
        self.callbacks = []

    async def ainvoke(self, inputs, config=None, **kwargs):
        for cb in self.callbacks:
            cb.on_chain_start(run_id="r1", name="planner", inputs=inputs,
                              metadata={"node_id": "planner"}, parent_run_id=None)
            cb.on_llm_start(run_id="r2", name="gpt-4o",
                            inputs={"messages": [{"role": "user", "content": "analyze"}]},
                            metadata={}, parent_run_id="r1")
            cb.on_llm_end(run_id="r2", outputs={"content": "I'll search for info"},
                          token_usage={"input": 50, "output": 100, "total": 150})
            cb.on_tool_start(run_id="r3", name="search_web",
                             inputs={"query": "traceshap"}, metadata={},
                             parent_run_id="r1")
            cb.on_tool_end(run_id="r3", outputs={"results": ["TraceSHAP is great"]})
            cb.on_llm_start(run_id="r4", name="gpt-4o",
                            inputs={"messages": [{"role": "user", "content": "summarize"}]},
                            metadata={}, parent_run_id="r1")
            cb.on_llm_end(run_id="r4", outputs={"content": "Summary: TraceSHAP works"},
                          token_usage={"input": 100, "output": 50, "total": 150})
            cb.on_chain_end(run_id="r1", outputs={"result": "Summary: TraceSHAP works"})
        return {"result": "Summary: TraceSHAP works"}


class TestLangGraphE2E:
    async def test_instrument_analyze_prune(self):
        graph = FakeGraph()
        app = InstrumentedApp(graph, framework="langgraph", agent_name="test-agent")

        result = await app.ainvoke({"query": "What is TraceSHAP?"})
        assert result["result"] == "Summary: TraceSHAP works"

        analysis = await app.analyze_last(
            layers=[0],
            outcome=Outcome(success=True, quality_score=0.85,
                            token_cost=300, latency_ms=5000, custom_metrics={}),
            include_pruning=True,
        )

        assert len(analysis["attributions"]) == 4
        assert analysis["pruning_report"].total_steps == 4
        assert analysis["pruning_report"].risk_assessment is not None


class TestOTLPJsonE2E:
    async def test_otlp_import_then_analyze(self, tmp_path):
        otlp_data = {
            "resourceSpans": [{
                "resource": {"attributes": []},
                "scopeSpans": [{
                    "spans": [
                        {
                            "traceId": "otlp-trace-1",
                            "spanId": "s1",
                            "parentSpanId": "",
                            "name": "agent_run",
                            "kind": 1,
                            "startTimeUnixNano": "1735689600000000000",
                            "endTimeUnixNano": "1735689605000000000",
                            "attributes": [],
                        },
                        {
                            "traceId": "otlp-trace-1",
                            "spanId": "s2",
                            "parentSpanId": "s1",
                            "name": "gpt-4o",
                            "kind": 3,
                            "startTimeUnixNano": "1735689601000000000",
                            "endTimeUnixNano": "1735689602000000000",
                            "attributes": [
                                {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
                                {"key": "llm.token_count.prompt", "value": {"intValue": "50"}},
                                {"key": "llm.token_count.completion", "value": {"intValue": "100"}},
                            ],
                        },
                        {
                            "traceId": "otlp-trace-1",
                            "spanId": "s3",
                            "parentSpanId": "s1",
                            "name": "search",
                            "kind": 3,
                            "startTimeUnixNano": "1735689603000000000",
                            "endTimeUnixNano": "1735689604000000000",
                            "attributes": [
                                {"key": "tool.name", "value": {"stringValue": "search"}},
                            ],
                        },
                    ]
                }]
            }]
        }

        path = tmp_path / "traces.json"
        path.write_text(json.dumps(otlp_data))

        source = OTLPJsonSource(str(path))
        await source.connect()
        spans = await source.poll()
        assert len(spans) == 3

        traj = spans_to_trajectory(spans, trace_id="otlp-trace-1")
        assert len(traj.steps) == 3

        engine = AttributionEngine(layers=[Layer0Rules()])
        attributions = await engine.analyze(traj)
        assert len(attributions) == 3

        await source.close()


class TestSourceFactoryE2E:
    async def test_factory_to_pipeline(self, tmp_path):
        otlp_data = {
            "resourceSpans": [{
                "resource": {"attributes": []},
                "scopeSpans": [{
                    "spans": [
                        {
                            "traceId": "factory-t1",
                            "spanId": "fs1",
                            "parentSpanId": "",
                            "name": "agent",
                            "kind": 1,
                            "startTimeUnixNano": "1735689600000000000",
                            "endTimeUnixNano": "1735689605000000000",
                            "attributes": [],
                        },
                    ]
                }]
            }]
        }

        path = tmp_path / "traces.json"
        path.write_text(json.dumps(otlp_data))

        config = SourceConfig(type="otlp_json", otlp_endpoint=str(path))
        source = create_source(config)
        assert source is not None

        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.initialize()

        pipeline = TraceSHAPPipeline(source=source, storage=backend)
        processed = await pipeline.ingest_once()
        assert processed == 1

        traj = await backend.get_trajectory("factory-t1")
        assert traj is not None
        assert len(traj.steps) == 1

        await backend.close()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_adapter_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_adapter_integration.py
git commit -m "feat: end-to-end integration tests for LangGraph adapter, OTLP source, and source factory"
```

---

## Summary

After completing all 6 tasks, you will have:

1. **LangGraph Span Collector** — captures LLM/tool/chain events as TraceSHAPSpan objects
2. **OTLP JSON Source** — imports OTel traces from JSON files (common export format for OpenClaw/Hermes/generic OTel)
3. **Source Factory** — creates SpanSource from config (langfuse, otlp_json)
4. **Convenience API** — `spans_to_trajectory()` and `quick_analyze()` for one-line analysis
5. **`instrument()` Wrapper** — wraps LangGraph graph with automatic span capture and `analyze_last()`
6. **Integration Tests** — E2E flows for LangGraph, OTLP import, and factory→pipeline

**Next plans:**
- **Plan 5**: Web Dashboard (React + Vite frontend)
