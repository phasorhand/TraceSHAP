"""Tests for ReplayCache and ReplayEngine (Task 3 — Recorded-IO Mode)."""
from __future__ import annotations

from datetime import datetime

import pytest
import pytest_asyncio

from traceshap.attribution.replay.capsule import RecordedIO, ReplayCapsule
from traceshap.attribution.replay.cache import ReplayCache
from traceshap.attribution.replay.engine import ReplayEngine
from traceshap.models.outcome import Outcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recorded_io(step_id: str, tool_name: str = "search_web") -> RecordedIO:
    return RecordedIO(
        step_id=step_id,
        tool_name=tool_name,
        input_hash=f"hash-{step_id}",
        input_data={"q": step_id},
        output_data={"result": step_id},
        side_effect_class="network_io",
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )


def _make_capsule(step_ids: list[str]) -> ReplayCapsule:
    ios = [_make_recorded_io(sid) for sid in step_ids]
    return ReplayCapsule(
        capsule_id="cap-001",
        trace_id="trace-xyz",
        created_at=datetime(2026, 1, 1, 12, 30, 0),
        model_id="gpt-4o",
        model_config={"temperature": 0.0},
        recorded_ios=ios,
    )


def _make_outcome(score: float) -> Outcome:
    return Outcome(
        success=True,
        quality_score=score,
        token_cost=0,
        latency_ms=0,
        custom_metrics={},
    )


# ---------------------------------------------------------------------------
# ReplayCache tests
# ---------------------------------------------------------------------------

class TestCacheMiss:
    """test_cache_miss — get on empty cache returns None."""

    def test_cache_miss(self):
        cache = ReplayCache()
        result = cache.get({"step-1", "step-2"})
        assert result is None


class TestCachePutAndGet:
    """test_cache_put_and_get — put then get returns the same object."""

    def test_cache_put_and_get(self):
        cache = ReplayCache()
        outcome = _make_outcome(0.9)
        cache.put({"step-1"}, outcome)
        retrieved = cache.get({"step-1"})
        assert retrieved is outcome


class TestCacheDifferentSets:
    """test_cache_different_sets — different keys return different values."""

    def test_cache_different_sets(self):
        cache = ReplayCache()
        outcome_a = _make_outcome(0.5)
        outcome_b = _make_outcome(0.8)

        cache.put({"step-1"}, outcome_a)
        cache.put({"step-2"}, outcome_b)

        assert cache.get({"step-1"}) is outcome_a
        assert cache.get({"step-2"}) is outcome_b
        assert cache.get({"step-1", "step-2"}) is None


class TestCacheLen:
    """__len__ reflects the number of distinct entries."""

    def test_cache_len(self):
        cache = ReplayCache()
        assert len(cache) == 0
        cache.put({"a"}, _make_outcome(1.0))
        assert len(cache) == 1
        cache.put({"b"}, _make_outcome(0.5))
        assert len(cache) == 2
        # Overwriting same key must not grow the count.
        cache.put({"a"}, _make_outcome(0.0))
        assert len(cache) == 2


# ---------------------------------------------------------------------------
# ReplayEngine tests
# ---------------------------------------------------------------------------

class TestReplayWithoutNoAblation:
    """test_replay_without_no_ablation — empty ablation list → score 1.0."""

    @pytest.mark.asyncio
    async def test_replay_without_no_ablation(self):
        engine = ReplayEngine()
        capsule = _make_capsule(["s1", "s2", "s3"])
        outcome = await engine.replay_without(capsule, [])
        assert outcome.quality_score == pytest.approx(1.0)


class TestReplayWithoutAblation:
    """test_replay_without_ablation — removing 1 of 3 steps → score ~0.67."""

    @pytest.mark.asyncio
    async def test_replay_without_ablation(self):
        engine = ReplayEngine()
        capsule = _make_capsule(["s1", "s2", "s3"])
        outcome = await engine.replay_without(capsule, ["s1"])
        assert outcome.quality_score == pytest.approx(2 / 3)


class TestReplayCachesResults:
    """test_replay_caches_results — second call returns same object (identity)."""

    @pytest.mark.asyncio
    async def test_replay_caches_results(self):
        engine = ReplayEngine()
        capsule = _make_capsule(["s1", "s2", "s3"])

        first = await engine.replay_without(capsule, ["s2"])
        second = await engine.replay_without(capsule, ["s2"])

        assert first is second


class TestGetCapsule:
    """test_get_capsule — register then get returns capsule; unknown id returns None."""

    def test_get_capsule_registered(self):
        engine = ReplayEngine()
        capsule = _make_capsule(["s1"])
        engine.register_capsule(capsule)
        assert engine.get_capsule("trace-xyz") is capsule

    def test_get_capsule_not_registered(self):
        engine = ReplayEngine()
        assert engine.get_capsule("does-not-exist") is None
