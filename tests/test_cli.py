import asyncio
import json
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner

from traceshap.cli.main import cli
from traceshap.models import (
    TraceSHAPSpan, SpanKind, SpanNode, TrajectoryMeta, Trajectory, Outcome, TokenUsage,
)
from traceshap.storage.sqlite import SQLiteBackend
from traceshap.ingestion.normalizer import StepNormalizer
from traceshap.ingestion.assembler import TreeAssembler


class TestCLIEntryPoint:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "TraceSHAP" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_init_creates_config(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        config_path = tmp_path / "traceshap.yaml"
        assert config_path.exists()
        content = config_path.read_text()
        assert "source:" in content
        assert "attribution:" in content


async def _seed_db(db_path: str):
    backend = SQLiteBackend(db_path)
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
    await backend.close()


class TestCLIAnalyze:
    def test_analyze_trace(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        asyncio.run(_seed_db(db_path))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze", "t1",
            "--db", db_path,
            "--layers", "0",
        ])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert isinstance(output, list)
        assert len(output) >= 1
        assert "step_id" in output[0]
        assert "verdict" in output[0]

    def test_analyze_nonexistent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        asyncio.run(_seed_db(db_path))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze", "nonexistent",
            "--db", db_path,
        ])
        assert result.exit_code != 0 or "not found" in result.output.lower()


class TestCLIReport:
    def test_report_trace(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        asyncio.run(_seed_db(db_path))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "report", "t1",
            "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "t1" in result.output
        assert "step" in result.output.lower() or "attribution" in result.output.lower()


class TestCLIPruneReport:
    def test_prune_report(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        asyncio.run(_seed_db(db_path))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "prune-report",
            "--db", db_path,
            "--agent", "test-agent",
        ])
        assert result.exit_code == 0
        assert "prune" in result.output.lower() or "candidate" in result.output.lower() or "report" in result.output.lower()


class TestCLIExport:
    def test_export_json(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        asyncio.run(_seed_db(db_path))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "export", "t1",
            "--db", db_path,
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["trace_id"] == "t1"
        assert "steps" in data
        assert "spans" in data

    def test_export_csv(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        asyncio.run(_seed_db(db_path))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "export", "t1",
            "--db", db_path,
            "--format", "csv",
        ])
        assert result.exit_code == 0
        assert "step_id" in result.output
