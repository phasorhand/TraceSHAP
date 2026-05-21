from traceshap.models.trajectory import Trajectory, TrajectoryMeta, SpanNode
from traceshap.ingestion.sources.base import SpanSource
from traceshap.ingestion.assembler import SpanBuffer, TreeAssembler
from traceshap.ingestion.normalizer import StepNormalizer
from traceshap.storage.backend import StorageBackend


class TraceSHAPPipeline:
    def __init__(
        self,
        source: SpanSource,
        storage: StorageBackend,
        normalizer: StepNormalizer | None = None,
        framework: str = "unknown",
        agent_name: str = "default",
    ):
        self._source = source
        self._storage = storage
        self._normalizer = normalizer or StepNormalizer()
        self._buffer = SpanBuffer()
        self._framework = framework
        self._agent_name = agent_name

    async def ingest_once(self) -> int:
        spans = await self._source.poll()
        if not spans:
            return 0

        for span in spans:
            self._buffer.add(span)

        processed = 0
        for trace_id in list(self._buffer.pending_trace_ids()):
            trace_spans = self._buffer.flush(trace_id)
            if not trace_spans:
                continue

            sorted_spans = sorted(trace_spans, key=lambda s: s.start_time)
            span_tree = TreeAssembler.build(sorted_spans)
            steps = self._normalizer.normalize(sorted_spans)

            trajectory = Trajectory(
                trace_id=trace_id,
                spans=sorted_spans,
                steps=steps,
                span_tree=span_tree,
                outcome=None,
                metadata=TrajectoryMeta(
                    framework=self._framework,
                    agent_name=self._agent_name,
                ),
            )

            await self._storage.save_trajectory(trajectory)
            processed += 1

        return processed
