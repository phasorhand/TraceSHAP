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
