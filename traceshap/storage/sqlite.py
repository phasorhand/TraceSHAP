import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from traceshap.models import (
    TraceSHAPSpan, SpanKind, TokenUsage, CanonicalStep, StepType,
    SideEffect, SpanNode, TrajectoryMeta, Trajectory, Outcome,
    StepAttribution, DecisionStatus,
)
from traceshap.storage.backend import StorageBackend, QueryFilter, CohortFilter
from traceshap.storage.orm import (
    Base, TrajectoryRow, SpanRow, CanonicalStepRow,
    AttributionRunRow, StepAttributionRow, PruneCandidateRow,
)


class SQLiteBackend(StorageBackend):
    def __init__(self, db_path: str):
        url = f"sqlite+aiosqlite:///{db_path}" if db_path != ":memory:" else "sqlite+aiosqlite:///:memory:"
        self._engine = create_async_engine(url)
        self._session_factory = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_trajectory(self, trajectory: Trajectory) -> None:
        async with self._session_factory() as session:
            traj_row = TrajectoryRow(
                trace_id=trajectory.trace_id,
                framework=trajectory.metadata.framework,
                agent_name=trajectory.metadata.agent_name,
                agent_version=trajectory.metadata.agent_version,
                task_type=trajectory.metadata.task_type,
                outcome_success=trajectory.outcome.success if trajectory.outcome else None,
                outcome_quality=trajectory.outcome.quality_score if trajectory.outcome else None,
                outcome_token_cost=trajectory.outcome.token_cost if trajectory.outcome else None,
                outcome_latency_ms=trajectory.outcome.latency_ms if trajectory.outcome else None,
                metadata_json=json.dumps({}),
            )
            session.add(traj_row)
            await session.flush()

            for span in trajectory.spans:
                span_row = SpanRow(
                    span_id=span.span_id,
                    trace_id=span.trace_id,
                    parent_span_id=span.parent_span_id,
                    span_kind=span.span_kind.value,
                    name=span.name,
                    input_data=json.dumps(span.input),
                    output_data=json.dumps(span.output),
                    start_time=span.start_time.isoformat(),
                    end_time=span.end_time.isoformat(),
                    tokens_input=span.tokens.input_tokens if span.tokens else None,
                    tokens_output=span.tokens.output_tokens if span.tokens else None,
                    tokens_total=span.tokens.total_tokens if span.tokens else None,
                    cost=span.cost,
                    raw_attributes=json.dumps(span.raw_attributes),
                    semconv_version=span.semconv_version,
                )
                session.add(span_row)

            for step in trajectory.steps:
                step_row = CanonicalStepRow(
                    step_id=step.step_id,
                    trace_id=trajectory.trace_id,
                    raw_span_ids_json=json.dumps(step.raw_span_ids),
                    node_id=step.node_id,
                    tool_name=step.tool_name,
                    step_type=step.step_type.value,
                    side_effect=step.side_effect_class.value,
                    attempt_index=step.attempt_index,
                    loop_iteration=step.loop_iteration,
                    input_hash=step.input_hash,
                    output_hash=step.output_hash,
                    mapping_confidence=step.framework_mapping_confidence,
                    tokens_total=step.tokens.total_tokens if step.tokens else None,
                    cost=step.cost,
                    start_time=step.start_time.isoformat(),
                    end_time=step.end_time.isoformat(),
                )
                session.add(step_row)

            await session.commit()

    async def get_trajectory(self, trace_id: str) -> Trajectory | None:
        async with self._session_factory() as session:
            traj_row = await session.get(TrajectoryRow, trace_id)
            if traj_row is None:
                return None

            span_result = await session.execute(
                select(SpanRow).where(SpanRow.trace_id == trace_id)
            )
            span_rows = span_result.scalars().all()

            step_result = await session.execute(
                select(CanonicalStepRow).where(CanonicalStepRow.trace_id == trace_id)
            )
            step_rows = step_result.scalars().all()

        spans = [self._row_to_span(r) for r in span_rows]
        steps = [self._row_to_step(r) for r in step_rows]
        span_tree = self._build_span_tree(spans)
        outcome = self._row_to_outcome(traj_row)
        metadata = TrajectoryMeta(
            framework=traj_row.framework,
            agent_name=traj_row.agent_name,
            agent_version=traj_row.agent_version,
            task_type=traj_row.task_type,
        )

        return Trajectory(
            trace_id=trace_id,
            spans=sorted(spans, key=lambda s: s.start_time),
            steps=sorted(steps, key=lambda s: s.start_time),
            span_tree=span_tree,
            outcome=outcome,
            metadata=metadata,
        )

    async def query_trajectories(self, filters: QueryFilter) -> list[Trajectory]:
        async with self._session_factory() as session:
            query = select(TrajectoryRow)
            if filters.agent_name:
                query = query.where(TrajectoryRow.agent_name == filters.agent_name)
            if filters.framework:
                query = query.where(TrajectoryRow.framework == filters.framework)
            if filters.agent_version:
                query = query.where(TrajectoryRow.agent_version == filters.agent_version)
            if filters.task_type:
                query = query.where(TrajectoryRow.task_type == filters.task_type)
            query = query.limit(filters.limit).offset(filters.offset)
            result = await session.execute(query)
            rows = result.scalars().all()

        trajectories = []
        for row in rows:
            t = await self.get_trajectory(row.trace_id)
            if t:
                trajectories.append(t)
        return trajectories

    async def save_attribution_run(
        self, run_id: str, trace_id: str, config_hash: str,
        code_version: str, layers: list[int],
        attributions: list[StepAttribution],
    ) -> None:
        async with self._session_factory() as session:
            run_row = AttributionRunRow(
                run_id=run_id,
                trace_id=trace_id,
                config_hash=config_hash,
                code_version=code_version,
                layers_used=json.dumps(layers),
            )
            session.add(run_row)
            await session.flush()

            for attr in attributions:
                attr_row = StepAttributionRow(
                    run_id=run_id,
                    step_id=attr.step_id,
                    layer=max(attr.layer_scores.keys()) if attr.layer_scores else 0,
                    quality_delta=attr.quality_delta,
                    cost_delta=attr.cost_delta,
                    latency_delta=attr.latency_delta,
                    confidence_lo=attr.confidence.lower if attr.confidence else None,
                    confidence_hi=attr.confidence.upper if attr.confidence else None,
                    verdict=attr.verdict.value,
                    evidence_json=json.dumps(attr.evidence),
                )
                session.add(attr_row)

            await session.commit()

    async def update_candidate_status(
        self, candidate_id: str, status: DecisionStatus,
    ) -> None:
        async with self._session_factory() as session:
            row = await session.get(PruneCandidateRow, candidate_id)
            if row:
                row.decision_status = status.value
                row.updated_at = datetime.now(timezone.utc).isoformat()
                await session.commit()

    async def close(self) -> None:
        await self._engine.dispose()

    @staticmethod
    def _row_to_span(row: SpanRow) -> TraceSHAPSpan:
        tokens = None
        if row.tokens_total is not None:
            tokens = TokenUsage(
                input_tokens=row.tokens_input or 0,
                output_tokens=row.tokens_output or 0,
                total_tokens=row.tokens_total,
            )
        return TraceSHAPSpan(
            trace_id=row.trace_id,
            span_id=row.span_id,
            parent_span_id=row.parent_span_id,
            span_kind=SpanKind(row.span_kind),
            name=row.name,
            input=json.loads(row.input_data),
            output=json.loads(row.output_data),
            start_time=datetime.fromisoformat(row.start_time),
            end_time=datetime.fromisoformat(row.end_time),
            tokens=tokens,
            cost=row.cost,
            metadata={},
            raw_attributes=json.loads(row.raw_attributes),
            semconv_version=row.semconv_version,
        )

    @staticmethod
    def _row_to_step(row: CanonicalStepRow) -> CanonicalStep:
        tokens_total = row.tokens_total
        tokens = TokenUsage(0, 0, tokens_total) if tokens_total is not None else None
        return CanonicalStep(
            step_id=row.step_id,
            raw_span_ids=json.loads(row.raw_span_ids_json),
            node_id=row.node_id,
            tool_name=row.tool_name,
            step_type=StepType(row.step_type),
            attempt_index=row.attempt_index,
            loop_iteration=row.loop_iteration,
            input_hash=row.input_hash,
            output_hash=row.output_hash,
            side_effect_class=SideEffect(row.side_effect),
            framework_mapping_confidence=row.mapping_confidence,
            tokens=tokens,
            cost=row.cost,
            start_time=datetime.fromisoformat(row.start_time),
            end_time=datetime.fromisoformat(row.end_time),
        )

    @staticmethod
    def _row_to_outcome(row: TrajectoryRow) -> Outcome | None:
        if row.outcome_success is None and row.outcome_quality is None:
            return None
        return Outcome(
            success=row.outcome_success,
            quality_score=row.outcome_quality,
            token_cost=row.outcome_token_cost or 0,
            latency_ms=row.outcome_latency_ms or 0,
            custom_metrics={},
        )

    @staticmethod
    def _build_span_tree(spans: list[TraceSHAPSpan]) -> SpanNode:
        nodes: dict[str, SpanNode] = {}
        for span in spans:
            nodes[span.span_id] = SpanNode(span_id=span.span_id)
        root = None
        for span in spans:
            node = nodes[span.span_id]
            if span.parent_span_id and span.parent_span_id in nodes:
                nodes[span.parent_span_id].children.append(node)
            else:
                root = node
        return root or SpanNode(span_id="empty")
