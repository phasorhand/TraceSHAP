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
