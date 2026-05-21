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
