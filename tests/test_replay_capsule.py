"""Tests for ReplayCapsule and RecordedIO data models (Layer 3 Replay SHAP)."""
from __future__ import annotations

from datetime import datetime

import pytest

from traceshap.attribution.replay.capsule import (
    EnvironmentSnapshot,
    RecordedIO,
    ReplayCapsule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_recorded_io(
    step_id: str = "step-1",
    tool_name: str = "search_web",
    input_hash: str = "abc123",
) -> RecordedIO:
    return RecordedIO(
        step_id=step_id,
        tool_name=tool_name,
        input_hash=input_hash,
        input_data={"query": "latest news"},
        output_data={"results": ["item1", "item2"]},
        side_effect_class="network_io",
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )


def _make_env_snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.11.9",
        package_versions={"traceshap": "0.1.0", "numpy": "1.26.4"},
        env_vars_hash="deadbeef",
        framework_version="1.0.0",
        timestamp=datetime(2026, 1, 1, 11, 0, 0),
    )


def _make_capsule(
    recorded_ios: list[RecordedIO] | None = None,
    with_snapshot: bool = True,
) -> ReplayCapsule:
    if recorded_ios is None:
        recorded_ios = [_make_recorded_io()]
    return ReplayCapsule(
        capsule_id="cap-001",
        trace_id="trace-xyz",
        created_at=datetime(2026, 1, 1, 12, 30, 0),
        model_id="gpt-4o",
        model_config={"temperature": 0.0, "max_tokens": 1024},
        recorded_ios=recorded_ios,
        environment_snapshot=_make_env_snapshot() if with_snapshot else None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecordedIOCreation:
    """test_recorded_io_creation — create a RecordedIO, verify fields."""

    def test_recorded_io_creation(self):
        rio = _make_recorded_io()

        assert rio.step_id == "step-1"
        assert rio.tool_name == "search_web"
        assert rio.input_hash == "abc123"
        assert rio.input_data == {"query": "latest news"}
        assert rio.output_data == {"results": ["item1", "item2"]}
        assert rio.side_effect_class == "network_io"
        assert rio.timestamp == datetime(2026, 1, 1, 12, 0, 0)


class TestReplayCapsuleCreation:
    """test_replay_capsule_creation — create with EnvironmentSnapshot, verify fields."""

    def test_replay_capsule_creation(self):
        capsule = _make_capsule(with_snapshot=True)

        assert capsule.capsule_id == "cap-001"
        assert capsule.trace_id == "trace-xyz"
        assert capsule.created_at == datetime(2026, 1, 1, 12, 30, 0)
        assert capsule.model_id == "gpt-4o"
        assert capsule.model_config == {"temperature": 0.0, "max_tokens": 1024}
        assert len(capsule.recorded_ios) == 1
        assert isinstance(capsule.environment_snapshot, EnvironmentSnapshot)

        snap = capsule.environment_snapshot
        assert snap.python_version == "3.11.9"
        assert snap.package_versions == {"traceshap": "0.1.0", "numpy": "1.26.4"}
        assert snap.env_vars_hash == "deadbeef"
        assert snap.framework_version == "1.0.0"
        assert snap.timestamp == datetime(2026, 1, 1, 11, 0, 0)

    def test_replay_capsule_no_snapshot_defaults_to_none(self):
        capsule = _make_capsule(with_snapshot=False)
        assert capsule.environment_snapshot is None


class TestCapsuleLookupExact:
    """test_capsule_lookup_recorded_io_exact — lookup matching tool+hash returns correct RecordedIO."""

    def test_capsule_lookup_recorded_io_exact(self):
        rio = _make_recorded_io(tool_name="search_web", input_hash="abc123")
        capsule = _make_capsule(recorded_ios=[rio])

        result = capsule.lookup_io("search_web", "abc123")

        assert result is rio


class TestCapsuleLookupWrongTool:
    """test_capsule_lookup_returns_none_for_wrong_tool — wrong tool name returns None."""

    def test_capsule_lookup_returns_none_for_wrong_tool(self):
        rio = _make_recorded_io(tool_name="search_web", input_hash="abc123")
        capsule = _make_capsule(recorded_ios=[rio])

        result = capsule.lookup_io("calculator", "abc123")

        assert result is None


class TestCapsuleLookupNoMatch:
    """test_capsule_lookup_returns_none_for_no_match — no matching hash returns None."""

    def test_capsule_lookup_returns_none_for_no_match(self):
        rio = _make_recorded_io(tool_name="search_web", input_hash="abc123")
        capsule = _make_capsule(recorded_ios=[rio])

        result = capsule.lookup_io("search_web", "deadbeef")

        assert result is None

    def test_capsule_lookup_empty_ios_returns_none(self):
        capsule = _make_capsule(recorded_ios=[])

        result = capsule.lookup_io("search_web", "abc123")

        assert result is None
