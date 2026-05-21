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
