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
