import pytest
import asyncio
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport

from traceshap.api.app import create_app
from traceshap.storage.sqlite import SQLiteBackend
from traceshap.models import (
    TraceSHAPSpan, SpanKind, SpanNode, TrajectoryMeta, Trajectory, Outcome, TokenUsage,
)
from traceshap.ingestion.normalizer import StepNormalizer
from traceshap.ingestion.assembler import TreeAssembler


async def _seed_backend(backend: SQLiteBackend):
    await backend.initialize()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    spans = [
        TraceSHAPSpan(
            trace_id="t1", span_id="t1-root", parent_span_id=None,
            span_kind=SpanKind.AGENT, name="agent", input={}, output={"result": "ok"},
            start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
            tokens=None, cost=None, metadata={}, raw_attributes={}, semconv_version="test",
        ),
        TraceSHAPSpan(
            trace_id="t1", span_id="t1-llm", parent_span_id="t1-root",
            span_kind=SpanKind.LLM, name="gpt-4o", input={"prompt": "hi"},
            output={"text": "hello"},
            start_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            tokens=TokenUsage(10, 20, 30), cost=0.001, metadata={},
            raw_attributes={}, semconv_version="test",
        ),
    ]
    normalizer = StepNormalizer()
    sorted_spans = sorted(spans, key=lambda s: s.start_time)
    steps = normalizer.normalize(sorted_spans)
    span_tree = TreeAssembler.build(sorted_spans)
    trajectory = Trajectory(
        trace_id="t1", spans=sorted_spans, steps=steps,
        span_tree=span_tree,
        outcome=Outcome(success=True, quality_score=0.9, token_cost=30,
                        latency_ms=5000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test-agent"),
    )
    await backend.save_trajectory(trajectory)


@pytest.fixture
async def seeded_app(tmp_path):
    db_path = str(tmp_path / "test.db")
    backend = SQLiteBackend(db_path)
    await _seed_backend(backend)
    app = create_app(backend=backend)
    yield app
    await backend.close()


class TestTraceEndpoints:
    async def test_list_traces(self, seeded_app):
        transport = ASGITransport(app=seeded_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/traces")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) >= 1
            assert data[0]["trace_id"] == "t1"

    async def test_get_trace(self, seeded_app):
        transport = ASGITransport(app=seeded_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/traces/t1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["trace_id"] == "t1"
            assert "steps" in data
            assert "spans" in data

    async def test_get_trace_not_found(self, seeded_app):
        transport = ASGITransport(app=seeded_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/traces/nonexistent")
            assert resp.status_code == 404
