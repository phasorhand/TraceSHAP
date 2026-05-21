import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from traceshap.storage.orm import Base


@pytest.fixture
async def async_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(async_engine):
    factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


class TestORMTables:
    async def test_all_tables_created(self, async_engine):
        async with async_engine.connect() as conn:
            table_names = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
        expected = {
            "trajectories", "spans", "canonical_steps",
            "attribution_runs", "step_attributions",
            "prune_candidates", "cohort_stats", "markov_models",
        }
        assert expected.issubset(set(table_names))

    async def test_insert_trajectory(self, session):
        from traceshap.storage.orm import TrajectoryRow
        row = TrajectoryRow(
            trace_id="t1",
            framework="langgraph",
            agent_name="my-agent",
            agent_version="v1",
            task_type="qa",
            outcome_success=True,
            outcome_quality=0.85,
            outcome_token_cost=1500,
            outcome_latency_ms=3000,
        )
        session.add(row)
        await session.commit()
        result = await session.get(TrajectoryRow, "t1")
        assert result is not None
        assert result.agent_name == "my-agent"

    async def test_insert_span(self, session):
        from traceshap.storage.orm import TrajectoryRow, SpanRow
        session.add(TrajectoryRow(
            trace_id="t2", framework="langgraph", agent_name="a",
        ))
        await session.flush()
        span = SpanRow(
            span_id="s1",
            trace_id="t2",
            parent_span_id=None,
            span_kind="llm",
            name="gpt-4o",
            input_data="{}",
            output_data="{}",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-01T00:00:01Z",
            raw_attributes="{}",
            semconv_version="otel-genai-v0.1",
        )
        session.add(span)
        await session.commit()
        result = await session.get(SpanRow, "s1")
        assert result.trace_id == "t2"


import json
from datetime import datetime, timezone
from traceshap.storage.sqlite import SQLiteBackend
from traceshap.models import (
    TraceSHAPSpan, SpanKind, TokenUsage, CanonicalStep, StepType,
    SideEffect, SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.storage.backend import QueryFilter


@pytest.fixture
async def backend(tmp_path):
    db_path = str(tmp_path / "test.db")
    backend = SQLiteBackend(db_path)
    await backend.initialize()
    yield backend
    await backend.close()


def _make_trajectory(trace_id: str = "t1") -> Trajectory:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
    span = TraceSHAPSpan(
        trace_id=trace_id, span_id=f"{trace_id}-s1", parent_span_id=None,
        span_kind=SpanKind.LLM, name="gpt-4o", input={"prompt": "hi"},
        output={"text": "hello"}, start_time=now, end_time=later,
        tokens=TokenUsage(10, 20, 30), cost=0.001,
        metadata={}, raw_attributes={}, semconv_version="otel-genai-v0.1",
    )
    step = CanonicalStep(
        step_id=f"{trace_id}-step1", raw_span_ids=[f"{trace_id}-s1"],
        node_id="llm_node", tool_name=None, step_type=StepType.DECISION,
        attempt_index=0, loop_iteration=None, input_hash="abc", output_hash="def",
        side_effect_class=SideEffect.PURE, framework_mapping_confidence=0.95,
        tokens=TokenUsage(10, 20, 30), cost=0.001, start_time=now, end_time=later,
    )
    return Trajectory(
        trace_id=trace_id,
        spans=[span],
        steps=[step],
        span_tree=SpanNode(span_id=f"{trace_id}-s1"),
        outcome=Outcome(
            success=True, quality_score=0.9, token_cost=30, latency_ms=2000,
            custom_metrics={}, evaluator_id="judge-v1",
        ),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test-agent", agent_version="v1"),
    )


class TestSQLiteBackend:
    async def test_save_and_get_trajectory(self, backend):
        t = _make_trajectory("t1")
        await backend.save_trajectory(t)
        result = await backend.get_trajectory("t1")
        assert result is not None
        assert result.trace_id == "t1"
        assert len(result.spans) == 1
        assert len(result.steps) == 1
        assert result.outcome.success is True
        assert result.metadata.agent_name == "test-agent"

    async def test_get_nonexistent_returns_none(self, backend):
        result = await backend.get_trajectory("nonexistent")
        assert result is None

    async def test_query_by_agent_name(self, backend):
        await backend.save_trajectory(_make_trajectory("t1"))
        await backend.save_trajectory(_make_trajectory("t2"))
        results = await backend.query_trajectories(
            QueryFilter(agent_name="test-agent")
        )
        assert len(results) == 2

    async def test_query_with_limit(self, backend):
        await backend.save_trajectory(_make_trajectory("t1"))
        await backend.save_trajectory(_make_trajectory("t2"))
        results = await backend.query_trajectories(
            QueryFilter(agent_name="test-agent", limit=1)
        )
        assert len(results) == 1
