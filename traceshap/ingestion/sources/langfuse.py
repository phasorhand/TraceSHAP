from datetime import datetime, timezone

from langfuse import Langfuse

from traceshap.models.span import TraceSHAPSpan, TokenUsage
from traceshap.models.enums import SpanKind
from traceshap.ingestion.sources.base import SpanSource

LANGFUSE_TO_SPAN_KIND = {
    "GENERATION": SpanKind.LLM,
    "SPAN": SpanKind.AGENT,
    "EVENT": SpanKind.CUSTOM,
}


class LangfuseSource(SpanSource):
    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
        poll_batch_size: int = 50,
    ):
        self._public_key = public_key
        self._secret_key = secret_key
        self._host = host
        self._poll_batch_size = poll_batch_size
        self._client: Langfuse | None = None
        self._last_poll_time: datetime | None = None

    async def connect(self) -> None:
        self._client = Langfuse(
            public_key=self._public_key,
            secret_key=self._secret_key,
            host=self._host,
        )
        self._last_poll_time = datetime.now(timezone.utc)

    async def poll(self) -> list[TraceSHAPSpan]:
        if not self._client:
            raise RuntimeError("Source not connected. Call connect() first.")

        traces = self._client.fetch_traces(limit=self._poll_batch_size)
        spans: list[TraceSHAPSpan] = []

        for trace in traces.data:
            trace_detail = self._client.fetch_trace(trace.id)
            for obs in trace_detail.data.observations or []:
                span = self._observation_to_span(trace.id, obs)
                if span:
                    spans.append(span)

        self._last_poll_time = datetime.now(timezone.utc)
        return spans

    async def close(self) -> None:
        if self._client:
            self._client.flush()
            self._client = None

    @staticmethod
    def _observation_to_span(trace_id: str, obs) -> TraceSHAPSpan | None:
        span_kind = LANGFUSE_TO_SPAN_KIND.get(obs.type, SpanKind.CUSTOM)

        tokens = None
        if obs.usage:
            tokens = TokenUsage(
                input_tokens=getattr(obs.usage, "input", 0) or 0,
                output_tokens=getattr(obs.usage, "output", 0) or 0,
                total_tokens=getattr(obs.usage, "total", 0) or 0,
            )

        start_time = obs.start_time or datetime.now(timezone.utc)
        end_time = obs.end_time or obs.start_time or datetime.now(timezone.utc)

        return TraceSHAPSpan(
            trace_id=trace_id,
            span_id=obs.id,
            parent_span_id=obs.parent_observation_id,
            span_kind=span_kind,
            name=obs.name or "unknown",
            input=obs.input or {},
            output=obs.output or {},
            start_time=start_time,
            end_time=end_time,
            tokens=tokens,
            cost=getattr(obs, "calculated_total_cost", None),
            metadata=obs.metadata or {},
            raw_attributes={
                "langfuse_type": obs.type,
                "model": getattr(obs, "model", None),
                "level": getattr(obs, "level", None),
            },
            semconv_version="langfuse-v2",
        )
