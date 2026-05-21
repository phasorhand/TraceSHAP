# TraceSHAP Plan 3: CLI + REST API + Visualization Library

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a click-based CLI, FastAPI REST API, and SHAP-style visualization library (plotly + matplotlib) so users can interact with TraceSHAP from the terminal, HTTP clients, and Python code.

**Architecture:** CLI commands (click) drive analysis/reporting by instantiating the same core classes (AttributionEngine, PruningAdvisor, SQLiteBackend). The REST API (FastAPI) exposes the same functionality over HTTP. The plot library produces interactive HTML (plotly) and static PNG (matplotlib) for 5 SHAP-style visualizations.

**Tech Stack:** click, FastAPI, uvicorn, plotly, matplotlib, numpy

---

### Task 1: Add Dependencies + CLI Entry Point

**Files:**
- Modify: `pyproject.toml`
- Create: `traceshap/cli/__init__.py`
- Create: `traceshap/cli/main.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write tests for CLI entry point**

`tests/test_cli.py`:
```python
import pytest
from click.testing import CliRunner

from traceshap.cli.main import cli


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Update pyproject.toml with new dependencies**

Add to `pyproject.toml`:
```toml
[project]
# ... existing fields ...
dependencies = [
    "sqlalchemy[asyncio]>=2.0",
    "aiosqlite>=0.19",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "langfuse>=2.0",
    "click>=8.1",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "plotly>=5.18",
    "matplotlib>=3.8",
    "numpy>=1.26",
]

[project.scripts]
traceshap = "traceshap.cli.main:cli"
```

- [ ] **Step 4: Implement CLI entry point**

`traceshap/cli/__init__.py`:
```python
```

`traceshap/cli/main.py`:
```python
import asyncio
from pathlib import Path

import click
import yaml

import traceshap


@click.group()
@click.version_option(version=traceshap.__version__, prog_name="TraceSHAP")
def cli():
    """TraceSHAP — Attribution and ablation analysis for LLM agent trajectories."""
    pass


@cli.command()
@click.option("--dir", "target_dir", default=".", help="Directory to create config in")
def init(target_dir: str):
    """Generate a traceshap.yaml config template."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    config_path = target / "traceshap.yaml"

    template = {
        "source": {
            "type": "langfuse",
            "langfuse_host": "https://cloud.langfuse.com",
            "langfuse_public_key": "",
            "langfuse_secret_key": "",
            "poll_interval_seconds": 10,
        },
        "attribution": {
            "layers": [0, 1, 2],
        },
        "pruning": {
            "prune_epsilon": 0.05,
            "keep_threshold": 0.10,
            "min_trajectories": 10,
            "protect_first_last": True,
            "validation_gate": True,
        },
        "storage": {
            "backend": "sqlite",
            "sqlite_path": "./traceshap.db",
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Config template written to {config_path}")
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml traceshap/cli/ tests/test_cli.py
git commit -m "feat: CLI entry point with click, init command, and new dependencies"
```

---

### Task 2: CLI Analyze + Report Commands

**Files:**
- Modify: `traceshap/cli/main.py`
- Create: `traceshap/cli/helpers.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_cli.py`:
```python
import asyncio
import json
from traceshap.models import (
    TraceSHAPSpan, SpanKind, SpanNode, TrajectoryMeta, Trajectory, Outcome,
    CanonicalStep, StepType, SideEffect, TokenUsage,
)
from traceshap.storage.sqlite import SQLiteBackend
from datetime import datetime, timezone


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

    from traceshap.ingestion.normalizer import StepNormalizer
    from traceshap.ingestion.assembler import TreeAssembler
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestCLIAnalyze -v`
Expected: FAIL

- [ ] **Step 3: Implement helpers**

`traceshap/cli/helpers.py`:
```python
from __future__ import annotations

import asyncio
from pathlib import Path

from traceshap.config import TraceSHAPConfig, load_config
from traceshap.storage.sqlite import SQLiteBackend
from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer0_rules import Layer0Rules
from traceshap.attribution.layer1_lift import Layer1Lift
from traceshap.attribution.layer2_sequence import Layer2Sequence
from traceshap.models.trajectory import Trajectory
from traceshap.models.outcome import StepAttribution


def run_async(coro):
    return asyncio.run(coro)


async def open_backend(db_path: str) -> SQLiteBackend:
    backend = SQLiteBackend(db_path)
    await backend.initialize()
    return backend


def build_engine(layers: list[int], trajectories: list[Trajectory] | None = None) -> AttributionEngine:
    layer_objs = []
    for layer_id in layers:
        if layer_id == 0:
            layer_objs.append(Layer0Rules())
        elif layer_id == 1:
            l1 = Layer1Lift()
            if trajectories:
                l1.fit(trajectories)
            layer_objs.append(l1)
        elif layer_id == 2:
            l2 = Layer2Sequence()
            if trajectories:
                l2.fit(trajectories)
            layer_objs.append(l2)
    return AttributionEngine(layers=layer_objs)


def attribution_to_dict(attr: StepAttribution) -> dict:
    return {
        "step_id": attr.step_id,
        "step_name": attr.step_name,
        "node_id": attr.node_id,
        "quality_delta": attr.quality_delta,
        "cost_delta": attr.cost_delta,
        "latency_delta": attr.latency_delta,
        "risk_delta": attr.risk_delta,
        "layer_scores": {str(k): v for k, v in attr.layer_scores.items()},
        "confidence": {
            "lower": attr.confidence.lower,
            "point": attr.confidence.point,
            "upper": attr.confidence.upper,
        } if attr.confidence else None,
        "verdict": attr.verdict.value,
        "evidence": attr.evidence,
    }
```

- [ ] **Step 4: Implement analyze and report commands**

Add to `traceshap/cli/main.py` (append after `init` command):
```python
import json
from traceshap.cli.helpers import run_async, open_backend, build_engine, attribution_to_dict
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig
from traceshap.storage.backend import QueryFilter


@cli.command()
@click.argument("trace_id")
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--layers", default="0,1,2", help="Comma-separated layer IDs")
def analyze(trace_id: str, db: str, layers: str):
    """Run attribution analysis on a trajectory."""
    layer_ids = [int(x.strip()) for x in layers.split(",")]

    async def _run():
        backend = await open_backend(db)
        try:
            trajectory = await backend.get_trajectory(trace_id)
            if trajectory is None:
                click.echo(f"Error: Trajectory '{trace_id}' not found.", err=True)
                raise SystemExit(1)

            training_trajs = None
            if any(lid in (1, 2) for lid in layer_ids):
                training_trajs = await backend.query_trajectories(QueryFilter(limit=200))

            engine = build_engine(layer_ids, training_trajs)
            attributions = await engine.analyze(trajectory)
            click.echo(json.dumps([attribution_to_dict(a) for a in attributions], indent=2))
        finally:
            await backend.close()

    run_async(_run())


@cli.command()
@click.argument("trace_id")
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--layers", default="0", help="Comma-separated layer IDs")
def report(trace_id: str, db: str, layers: str):
    """Print a human-readable attribution report for a trajectory."""
    layer_ids = [int(x.strip()) for x in layers.split(",")]

    async def _run():
        backend = await open_backend(db)
        try:
            trajectory = await backend.get_trajectory(trace_id)
            if trajectory is None:
                click.echo(f"Error: Trajectory '{trace_id}' not found.", err=True)
                raise SystemExit(1)

            training_trajs = None
            if any(lid in (1, 2) for lid in layer_ids):
                training_trajs = await backend.query_trajectories(QueryFilter(limit=200))

            engine = build_engine(layer_ids, training_trajs)
            attributions = await engine.analyze(trajectory)

            click.echo(f"\n{'='*60}")
            click.echo(f"  TraceSHAP Attribution Report: {trace_id}")
            click.echo(f"{'='*60}")
            click.echo(f"  Steps: {len(trajectory.steps)}  |  Layers: {layer_ids}")
            if trajectory.outcome:
                click.echo(f"  Outcome: success={trajectory.outcome.success}, "
                           f"quality={trajectory.outcome.quality_score}")
            click.echo(f"{'─'*60}")

            for attr in attributions:
                verdict_icon = {"keep": "✓", "review": "?", "prune_candidate": "✗",
                                "insufficient_evidence": "—"}.get(attr.verdict.value, " ")
                click.echo(f"\n  [{verdict_icon}] {attr.step_name} ({attr.step_id})")
                click.echo(f"      quality_delta: {attr.quality_delta}")
                click.echo(f"      cost_delta:    {attr.cost_delta}")
                click.echo(f"      latency_delta: {attr.latency_delta}")
                if attr.confidence:
                    click.echo(f"      confidence:    [{attr.confidence.lower:.4f}, "
                               f"{attr.confidence.upper:.4f}]")
                click.echo(f"      verdict:       {attr.verdict.value}")
                if attr.evidence:
                    click.echo(f"      evidence:      {'; '.join(attr.evidence[:3])}")

            click.echo(f"\n{'='*60}\n")
        finally:
            await backend.close()

    run_async(_run())
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add traceshap/cli/ tests/test_cli.py
git commit -m "feat: CLI analyze and report commands with helpers"
```

---

### Task 3: CLI Prune-Report + Export Commands

**Files:**
- Modify: `traceshap/cli/main.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_cli.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestCLIPruneReport -v`
Expected: FAIL

- [ ] **Step 3: Implement prune-report and export commands**

Add to `traceshap/cli/main.py`:
```python
import csv
import io


@cli.command("prune-report")
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--agent", required=True, help="Agent name to analyze")
@click.option("--min-trajectories", default=10, type=int, help="Minimum trajectories needed")
def prune_report(db: str, agent: str, min_trajectories: int):
    """Generate a pruning report across trajectories for an agent."""

    async def _run():
        backend = await open_backend(db)
        try:
            trajectories = await backend.query_trajectories(
                QueryFilter(agent_name=agent, limit=200)
            )
            if not trajectories:
                click.echo(f"No trajectories found for agent '{agent}'.")
                return

            engine = build_engine([0, 1, 2], trajectories)
            config = PruningConfig(min_trajectories=min_trajectories)
            advisor = PruningAdvisor(config)

            all_candidates = []
            for traj in trajectories:
                attributions = await engine.analyze(traj)
                report = advisor.analyze(traj, attributions)
                all_candidates.extend(report.candidates)

            click.echo(f"\n{'='*60}")
            click.echo(f"  TraceSHAP Pruning Report: {agent}")
            click.echo(f"{'='*60}")
            click.echo(f"  Trajectories analyzed: {len(trajectories)}")
            click.echo(f"  Total candidates: {len(all_candidates)}")
            click.echo(f"{'─'*60}")

            if all_candidates:
                for c in all_candidates:
                    click.echo(f"\n  ✗ {c.target_id} ({c.target_type})")
                    click.echo(f"      cost_reduction:    ${c.estimated_savings.cost_reduction:.4f}")
                    click.echo(f"      token_reduction:   {c.estimated_savings.token_reduction}")
                    click.echo(f"      latency_reduction: {c.estimated_savings.latency_reduction_ms}ms")
                    click.echo(f"      quality_impact:    [{c.estimated_savings.quality_impact_range[0]:.4f}, "
                               f"{c.estimated_savings.quality_impact_range[1]:.4f}]")
                    click.echo(f"      status:            {c.decision_status.value}")
            else:
                click.echo("\n  No pruning candidates found.")

            click.echo(f"\n{'='*60}\n")
        finally:
            await backend.close()

    run_async(_run())


@cli.command()
@click.argument("trace_id")
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json",
              help="Output format")
def export(trace_id: str, db: str, fmt: str):
    """Export a trajectory in JSON or CSV format."""

    async def _run():
        backend = await open_backend(db)
        try:
            trajectory = await backend.get_trajectory(trace_id)
            if trajectory is None:
                click.echo(f"Error: Trajectory '{trace_id}' not found.", err=True)
                raise SystemExit(1)

            if fmt == "json":
                data = {
                    "trace_id": trajectory.trace_id,
                    "framework": trajectory.metadata.framework,
                    "agent_name": trajectory.metadata.agent_name,
                    "outcome": {
                        "success": trajectory.outcome.success,
                        "quality_score": trajectory.outcome.quality_score,
                        "token_cost": trajectory.outcome.token_cost,
                        "latency_ms": trajectory.outcome.latency_ms,
                    } if trajectory.outcome else None,
                    "steps": [
                        {
                            "step_id": s.step_id,
                            "tool_name": s.tool_name,
                            "step_type": s.step_type.value,
                            "side_effect": s.side_effect_class.value,
                            "attempt_index": s.attempt_index,
                            "cost": s.cost,
                            "duration_ms": s.duration_ms,
                            "start_time": s.start_time.isoformat(),
                            "end_time": s.end_time.isoformat(),
                        }
                        for s in trajectory.steps
                    ],
                    "spans": [
                        {
                            "span_id": sp.span_id,
                            "parent_span_id": sp.parent_span_id,
                            "span_kind": sp.span_kind.value,
                            "name": sp.name,
                            "start_time": sp.start_time.isoformat(),
                            "end_time": sp.end_time.isoformat(),
                        }
                        for sp in trajectory.spans
                    ],
                }
                click.echo(json.dumps(data, indent=2))
            elif fmt == "csv":
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["step_id", "tool_name", "step_type", "side_effect",
                                 "attempt_index", "cost", "duration_ms",
                                 "start_time", "end_time"])
                for s in trajectory.steps:
                    writer.writerow([
                        s.step_id, s.tool_name, s.step_type.value,
                        s.side_effect_class.value, s.attempt_index,
                        s.cost, s.duration_ms,
                        s.start_time.isoformat(), s.end_time.isoformat(),
                    ])
                click.echo(output.getvalue().strip())
        finally:
            await backend.close()

    run_async(_run())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/cli/main.py tests/test_cli.py
git commit -m "feat: CLI prune-report and export commands (JSON + CSV)"
```

---

### Task 4: FastAPI App + Trajectory Endpoints

**Files:**
- Create: `traceshap/api/__init__.py`
- Create: `traceshap/api/app.py`
- Create: `traceshap/api/deps.py`
- Create: `traceshap/api/routes_traces.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write tests**

`tests/test_api.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FastAPI app and trace endpoints**

`traceshap/api/__init__.py`:
```python
```

`traceshap/api/deps.py`:
```python
from traceshap.storage.sqlite import SQLiteBackend

_backend: SQLiteBackend | None = None


def set_backend(backend: SQLiteBackend) -> None:
    global _backend
    _backend = backend


def get_backend() -> SQLiteBackend:
    if _backend is None:
        raise RuntimeError("Backend not initialized")
    return _backend
```

`traceshap/api/routes_traces.py`:
```python
from fastapi import APIRouter, HTTPException, Query

from traceshap.api.deps import get_backend
from traceshap.storage.backend import QueryFilter

router = APIRouter(prefix="/api/traces", tags=["traces"])


@router.get("")
async def list_traces(
    agent_name: str | None = None,
    framework: str | None = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
):
    backend = get_backend()
    filters = QueryFilter(
        agent_name=agent_name,
        framework=framework,
        limit=limit,
        offset=offset,
    )
    trajectories = await backend.query_trajectories(filters)
    return [_trajectory_summary(t) for t in trajectories]


@router.get("/{trace_id}")
async def get_trace(trace_id: str):
    backend = get_backend()
    trajectory = await backend.get_trajectory(trace_id)
    if trajectory is None:
        raise HTTPException(status_code=404, detail=f"Trajectory '{trace_id}' not found")
    return _trajectory_detail(trajectory)


def _trajectory_summary(t) -> dict:
    return {
        "trace_id": t.trace_id,
        "framework": t.metadata.framework,
        "agent_name": t.metadata.agent_name,
        "agent_version": t.metadata.agent_version,
        "task_type": t.metadata.task_type,
        "step_count": len(t.steps),
        "outcome_success": t.outcome.success if t.outcome else None,
        "outcome_quality": t.outcome.quality_score if t.outcome else None,
    }


def _trajectory_detail(t) -> dict:
    return {
        "trace_id": t.trace_id,
        "framework": t.metadata.framework,
        "agent_name": t.metadata.agent_name,
        "agent_version": t.metadata.agent_version,
        "task_type": t.metadata.task_type,
        "outcome": {
            "success": t.outcome.success,
            "quality_score": t.outcome.quality_score,
            "token_cost": t.outcome.token_cost,
            "latency_ms": t.outcome.latency_ms,
        } if t.outcome else None,
        "steps": [
            {
                "step_id": s.step_id,
                "tool_name": s.tool_name,
                "step_type": s.step_type.value,
                "side_effect": s.side_effect_class.value,
                "attempt_index": s.attempt_index,
                "cost": s.cost,
                "duration_ms": s.duration_ms,
                "start_time": s.start_time.isoformat(),
                "end_time": s.end_time.isoformat(),
            }
            for s in t.steps
        ],
        "spans": [
            {
                "span_id": sp.span_id,
                "parent_span_id": sp.parent_span_id,
                "span_kind": sp.span_kind.value,
                "name": sp.name,
                "start_time": sp.start_time.isoformat(),
                "end_time": sp.end_time.isoformat(),
            }
            for sp in t.spans
        ],
    }
```

`traceshap/api/app.py`:
```python
from fastapi import FastAPI

from traceshap.api.deps import set_backend
from traceshap.api.routes_traces import router as traces_router
from traceshap.storage.sqlite import SQLiteBackend


def create_app(backend: SQLiteBackend | None = None) -> FastAPI:
    app = FastAPI(title="TraceSHAP", version="0.1.0")

    if backend is not None:
        set_backend(backend)

    app.include_router(traces_router)

    return app
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_api.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/api/ tests/test_api.py
git commit -m "feat: FastAPI app with trajectory list and detail endpoints"
```

---

### Task 5: Attribution + Pruning API Endpoints

**Files:**
- Create: `traceshap/api/routes_attribution.py`
- Create: `traceshap/api/routes_pruning.py`
- Modify: `traceshap/api/app.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_api.py`:
```python
class TestAttributionEndpoints:
    async def test_get_attribution(self, seeded_app):
        transport = ASGITransport(app=seeded_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/traces/t1/attribution?layers=0")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) >= 1
            assert "step_id" in data[0]
            assert "verdict" in data[0]

    async def test_attribution_not_found(self, seeded_app):
        transport = ASGITransport(app=seeded_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/traces/nonexistent/attribution")
            assert resp.status_code == 404


class TestPruningEndpoints:
    async def test_agent_prune_candidates(self, seeded_app):
        transport = ASGITransport(app=seeded_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/agents/test-agent/prune-candidates")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, dict)
            assert "candidates" in data
            assert "trajectory_count" in data

    async def test_agent_stats(self, seeded_app):
        transport = ASGITransport(app=seeded_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/agents/test-agent/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert "agent_name" in data
            assert "trajectory_count" in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py::TestAttributionEndpoints -v`
Expected: FAIL

- [ ] **Step 3: Implement attribution routes**

`traceshap/api/routes_attribution.py`:
```python
from fastapi import APIRouter, HTTPException, Query

from traceshap.api.deps import get_backend
from traceshap.cli.helpers import build_engine, attribution_to_dict
from traceshap.storage.backend import QueryFilter

router = APIRouter(prefix="/api/traces", tags=["attribution"])


@router.get("/{trace_id}/attribution")
async def get_attribution(
    trace_id: str,
    layers: str = Query(default="0", description="Comma-separated layer IDs"),
):
    backend = get_backend()
    trajectory = await backend.get_trajectory(trace_id)
    if trajectory is None:
        raise HTTPException(status_code=404, detail=f"Trajectory '{trace_id}' not found")

    layer_ids = [int(x.strip()) for x in layers.split(",")]

    training_trajs = None
    if any(lid in (1, 2) for lid in layer_ids):
        training_trajs = await backend.query_trajectories(QueryFilter(limit=200))

    engine = build_engine(layer_ids, training_trajs)
    attributions = await engine.analyze(trajectory)

    return [attribution_to_dict(a) for a in attributions]
```

- [ ] **Step 4: Implement pruning routes**

`traceshap/api/routes_pruning.py`:
```python
from fastapi import APIRouter, HTTPException, Query

from traceshap.api.deps import get_backend
from traceshap.cli.helpers import build_engine, attribution_to_dict
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig
from traceshap.storage.backend import QueryFilter

router = APIRouter(prefix="/api/agents", tags=["pruning"])


@router.get("/{agent_name}/stats")
async def agent_stats(agent_name: str):
    backend = get_backend()
    trajectories = await backend.query_trajectories(
        QueryFilter(agent_name=agent_name, limit=200)
    )
    if not trajectories:
        return {
            "agent_name": agent_name,
            "trajectory_count": 0,
            "avg_quality": None,
            "avg_cost": None,
            "avg_latency_ms": None,
        }

    qualities = [t.outcome.quality_score for t in trajectories
                 if t.outcome and t.outcome.quality_score is not None]
    costs = [t.outcome.token_cost for t in trajectories if t.outcome]
    latencies = [t.outcome.latency_ms for t in trajectories if t.outcome]

    return {
        "agent_name": agent_name,
        "trajectory_count": len(trajectories),
        "avg_quality": sum(qualities) / len(qualities) if qualities else None,
        "avg_cost": sum(costs) / len(costs) if costs else None,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }


@router.get("/{agent_name}/prune-candidates")
async def agent_prune_candidates(
    agent_name: str,
    layers: str = Query(default="0", description="Comma-separated layer IDs"),
):
    backend = get_backend()
    trajectories = await backend.query_trajectories(
        QueryFilter(agent_name=agent_name, limit=200)
    )

    layer_ids = [int(x.strip()) for x in layers.split(",")]
    engine = build_engine(layer_ids, trajectories if any(lid in (1, 2) for lid in layer_ids) else None)
    config = PruningConfig()
    advisor = PruningAdvisor(config)

    all_candidates = []
    for traj in trajectories:
        attributions = await engine.analyze(traj)
        report = advisor.analyze(traj, attributions)
        for c in report.candidates:
            all_candidates.append({
                "target_type": c.target_type,
                "target_id": c.target_id,
                "decision_status": c.decision_status.value,
                "estimated_savings": {
                    "token_reduction": c.estimated_savings.token_reduction,
                    "cost_reduction": c.estimated_savings.cost_reduction,
                    "latency_reduction_ms": c.estimated_savings.latency_reduction_ms,
                    "quality_impact_range": list(c.estimated_savings.quality_impact_range),
                },
                "validation": {
                    "replay_required": c.required_validation.replay_required,
                    "replay_mode": c.required_validation.replay_mode.value,
                    "min_replay_count": c.required_validation.min_replay_count,
                },
            })

    return {
        "agent_name": agent_name,
        "trajectory_count": len(trajectories),
        "candidates": all_candidates,
    }
```

- [ ] **Step 5: Register new routers in app.py**

Update `traceshap/api/app.py`:
```python
from fastapi import FastAPI

from traceshap.api.deps import set_backend
from traceshap.api.routes_traces import router as traces_router
from traceshap.api.routes_attribution import router as attribution_router
from traceshap.api.routes_pruning import router as pruning_router
from traceshap.storage.sqlite import SQLiteBackend


def create_app(backend: SQLiteBackend | None = None) -> FastAPI:
    app = FastAPI(title="TraceSHAP", version="0.1.0")

    if backend is not None:
        set_backend(backend)

    app.include_router(traces_router)
    app.include_router(attribution_router)
    app.include_router(pruning_router)

    return app
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_api.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add traceshap/api/ tests/test_api.py
git commit -m "feat: attribution and pruning API endpoints"
```

---

### Task 6: SHAP Plots — Force + Waterfall (Plotly)

**Files:**
- Create: `traceshap/plots/__init__.py`
- Create: `traceshap/plots/force.py`
- Create: `traceshap/plots/waterfall.py`
- Create: `tests/test_plots.py`

- [ ] **Step 1: Write tests**

`tests/test_plots.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome, Verdict,
)
from traceshap.models.outcome import StepAttribution, ConfidenceInterval
from traceshap.plots.force import force_plot
from traceshap.plots.waterfall import waterfall_plot


def _step(step_id: str, name: str) -> CanonicalStep:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=["s1"], node_id=None,
        tool_name=name, step_type=StepType.ACTION, attempt_index=0,
        loop_iteration=None, input_hash="a", output_hash="b",
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=0.001,
        start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )


def _attr(step_id: str, name: str, quality_delta: float,
          ci_lower: float, ci_upper: float) -> StepAttribution:
    return StepAttribution(
        step_id=step_id, step_name=name, node_id=None,
        quality_delta=quality_delta, cost_delta=0.001,
        latency_delta=1000, risk_delta=0.0,
        layer_scores={0: quality_delta},
        confidence=ConfidenceInterval(lower=ci_lower, point=quality_delta, upper=ci_upper),
        verdict=Verdict.REVIEW,
        causal_hypothesis=None, evidence=["test"], calibration=None,
    )


@pytest.fixture
def sample_attributions():
    return [
        _attr("s1", "plan", 0.15, 0.10, 0.20),
        _attr("s2", "search_web", 0.05, 0.01, 0.09),
        _attr("s3", "summarize", -0.02, -0.05, 0.01),
        _attr("s4", "validate", 0.10, 0.07, 0.13),
    ]


class TestForcePlot:
    def test_returns_plotly_figure(self, sample_attributions):
        fig = force_plot(sample_attributions, base_value=0.5)
        assert fig is not None
        assert hasattr(fig, "to_html")

    def test_save_html(self, sample_attributions, tmp_path):
        fig = force_plot(sample_attributions, base_value=0.5)
        path = tmp_path / "force.html"
        fig.write_html(str(path))
        assert path.exists()
        content = path.read_text()
        assert "plotly" in content.lower() or "div" in content.lower()


class TestWaterfallPlot:
    def test_returns_plotly_figure(self, sample_attributions):
        fig = waterfall_plot(sample_attributions, base_value=0.5)
        assert fig is not None
        assert hasattr(fig, "to_html")

    def test_save_html(self, sample_attributions, tmp_path):
        fig = waterfall_plot(sample_attributions, base_value=0.5)
        path = tmp_path / "waterfall.html"
        fig.write_html(str(path))
        assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_plots.py -v`
Expected: FAIL

- [ ] **Step 3: Implement force plot**

`traceshap/plots/__init__.py`:
```python
from traceshap.plots.force import force_plot
from traceshap.plots.waterfall import waterfall_plot

__all__ = ["force_plot", "waterfall_plot"]
```

`traceshap/plots/force.py`:
```python
from __future__ import annotations

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def force_plot(
    attributions: list[StepAttribution],
    base_value: float = 0.5,
    title: str = "TraceSHAP Force Plot",
) -> go.Figure:
    sorted_attrs = sorted(attributions, key=lambda a: abs(a.quality_delta or 0), reverse=True)

    names = []
    values = []
    colors = []
    for attr in sorted_attrs:
        delta = attr.quality_delta or 0.0
        names.append(attr.step_name)
        values.append(delta)
        colors.append("rgba(255, 77, 77, 0.8)" if delta >= 0 else "rgba(77, 148, 255, 0.8)")

    cumulative = base_value
    x_starts = []
    x_ends = []
    for v in values:
        x_starts.append(cumulative)
        cumulative += v
        x_ends.append(cumulative)

    fig = go.Figure()

    for i in range(len(names)):
        fig.add_trace(go.Bar(
            y=[names[i]],
            x=[values[i]],
            base=[x_starts[i]],
            orientation="h",
            marker_color=colors[i],
            name=names[i],
            hovertemplate=(
                f"<b>{names[i]}</b><br>"
                f"quality_delta: {values[i]:.4f}<br>"
                f"<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_vline(x=base_value, line_dash="dash", line_color="gray",
                  annotation_text=f"base: {base_value:.3f}")
    fig.add_vline(x=cumulative, line_dash="solid", line_color="black",
                  annotation_text=f"final: {cumulative:.3f}")

    fig.update_layout(
        title=title,
        xaxis_title="Quality Score",
        yaxis_title="Step",
        barmode="overlay",
        height=max(300, len(names) * 50),
        template="plotly_white",
    )

    return fig
```

- [ ] **Step 4: Implement waterfall plot**

`traceshap/plots/waterfall.py`:
```python
from __future__ import annotations

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def waterfall_plot(
    attributions: list[StepAttribution],
    base_value: float = 0.5,
    title: str = "TraceSHAP Waterfall Plot",
) -> go.Figure:
    names = ["Base Value"]
    measures = ["absolute"]
    values = [base_value]
    text = [f"{base_value:.4f}"]

    for attr in attributions:
        delta = attr.quality_delta or 0.0
        names.append(attr.step_name)
        measures.append("relative")
        values.append(delta)
        text.append(f"{delta:+.4f}")

    final_value = base_value + sum(a.quality_delta or 0.0 for a in attributions)
    names.append("Final Value")
    measures.append("total")
    values.append(final_value)
    text.append(f"{final_value:.4f}")

    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=measures,
        x=names,
        y=values,
        text=text,
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)", "width": 1}},
        increasing={"marker": {"color": "rgba(255, 77, 77, 0.8)"}},
        decreasing={"marker": {"color": "rgba(77, 148, 255, 0.8)"}},
        totals={"marker": {"color": "rgba(100, 100, 100, 0.6)"}},
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Quality Score",
        xaxis_title="Step",
        template="plotly_white",
        height=500,
    )

    return fig
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_plots.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add traceshap/plots/ tests/test_plots.py
git commit -m "feat: SHAP force plot and waterfall plot (plotly)"
```

---

### Task 7: SHAP Plots — Beeswarm + Bar + Dependency

**Files:**
- Create: `traceshap/plots/beeswarm.py`
- Create: `traceshap/plots/bar.py`
- Create: `traceshap/plots/dependency.py`
- Modify: `traceshap/plots/__init__.py`
- Modify: `tests/test_plots.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_plots.py`:
```python
from traceshap.plots.beeswarm import beeswarm_plot
from traceshap.plots.bar import bar_plot
from traceshap.plots.dependency import dependency_plot


def _multi_trajectory_attrs():
    """Simulate attributions from multiple trajectories for cross-trajectory plots."""
    import random
    random.seed(42)
    step_names = ["plan", "search_web", "summarize", "validate", "format"]
    all_attrs = []
    for traj_idx in range(20):
        traj_attrs = []
        for name in step_names:
            delta = random.gauss(0.05 if name in ("plan", "validate") else -0.02, 0.05)
            traj_attrs.append(_attr(
                f"t{traj_idx}-{name}", name, delta,
                ci_lower=delta - 0.03, ci_upper=delta + 0.03,
            ))
        all_attrs.append(traj_attrs)
    return all_attrs


class TestBeeswarmPlot:
    def test_returns_figure(self):
        multi_attrs = _multi_trajectory_attrs()
        fig = beeswarm_plot(multi_attrs)
        assert fig is not None
        assert hasattr(fig, "to_html")

    def test_save_html(self, tmp_path):
        multi_attrs = _multi_trajectory_attrs()
        fig = beeswarm_plot(multi_attrs)
        path = tmp_path / "beeswarm.html"
        fig.write_html(str(path))
        assert path.exists()


class TestBarPlot:
    def test_returns_figure(self):
        multi_attrs = _multi_trajectory_attrs()
        fig = bar_plot(multi_attrs, top_k=5)
        assert fig is not None
        assert hasattr(fig, "to_html")

    def test_save_html(self, tmp_path):
        multi_attrs = _multi_trajectory_attrs()
        fig = bar_plot(multi_attrs, top_k=3)
        path = tmp_path / "bar.html"
        fig.write_html(str(path))
        assert path.exists()


class TestDependencyPlot:
    def test_returns_figure(self):
        multi_attrs = _multi_trajectory_attrs()
        fig = dependency_plot(multi_attrs, step_name="plan", color_by="search_web")
        assert fig is not None
        assert hasattr(fig, "to_html")

    def test_save_html(self, tmp_path):
        multi_attrs = _multi_trajectory_attrs()
        fig = dependency_plot(multi_attrs, step_name="plan", color_by="validate")
        path = tmp_path / "dep.html"
        fig.write_html(str(path))
        assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_plots.py::TestBeeswarmPlot -v`
Expected: FAIL

- [ ] **Step 3: Implement beeswarm plot**

`traceshap/plots/beeswarm.py`:
```python
from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def beeswarm_plot(
    multi_trajectory_attrs: list[list[StepAttribution]],
    color_by: str = "cost_delta",
    title: str = "TraceSHAP Beeswarm Plot",
) -> go.Figure:
    step_values: dict[str, list[float]] = defaultdict(list)
    step_colors: dict[str, list[float]] = defaultdict(list)

    for traj_attrs in multi_trajectory_attrs:
        for attr in traj_attrs:
            step_values[attr.step_name].append(attr.quality_delta or 0.0)
            color_val = getattr(attr, color_by, None) or 0.0
            step_colors[attr.step_name].append(color_val)

    mean_abs = {name: sum(abs(v) for v in vals) / len(vals)
                for name, vals in step_values.items()}
    sorted_names = sorted(mean_abs, key=mean_abs.get, reverse=True)

    fig = go.Figure()

    for i, name in enumerate(sorted_names):
        vals = step_values[name]
        cols = step_colors[name]
        jitter = [i + (j % 5 - 2) * 0.08 for j in range(len(vals))]

        fig.add_trace(go.Scatter(
            x=vals,
            y=jitter,
            mode="markers",
            marker=dict(
                size=6,
                color=cols,
                colorscale="RdBu_r",
                showscale=(i == 0),
                colorbar=dict(title=color_by) if i == 0 else None,
                opacity=0.7,
            ),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"quality_delta: %{{x:.4f}}<br>"
                f"{color_by}: %{{marker.color:.4f}}<br>"
                f"<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value (quality_delta)",
        yaxis=dict(
            tickvals=list(range(len(sorted_names))),
            ticktext=sorted_names,
            title="Step Type",
        ),
        template="plotly_white",
        height=max(400, len(sorted_names) * 60),
    )

    return fig
```

- [ ] **Step 4: Implement bar plot**

`traceshap/plots/bar.py`:
```python
from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def bar_plot(
    multi_trajectory_attrs: list[list[StepAttribution]],
    top_k: int = 10,
    title: str = "TraceSHAP Step Importance",
) -> go.Figure:
    step_values: dict[str, list[float]] = defaultdict(list)

    for traj_attrs in multi_trajectory_attrs:
        for attr in traj_attrs:
            step_values[attr.step_name].append(attr.quality_delta or 0.0)

    mean_abs = {name: sum(abs(v) for v in vals) / len(vals)
                for name, vals in step_values.items()}
    mean_signed = {name: sum(vals) / len(vals) for name, vals in step_values.items()}

    sorted_names = sorted(mean_abs, key=mean_abs.get, reverse=True)[:top_k]
    sorted_names.reverse()

    bar_values = [mean_abs[n] for n in sorted_names]
    bar_colors = [
        "rgba(255, 77, 77, 0.8)" if mean_signed[n] >= 0 else "rgba(77, 148, 255, 0.8)"
        for n in sorted_names
    ]

    fig = go.Figure(go.Bar(
        y=sorted_names,
        x=bar_values,
        orientation="h",
        marker_color=bar_colors,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "mean |SHAP|: %{x:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Step Type",
        template="plotly_white",
        height=max(300, len(sorted_names) * 40),
    )

    return fig
```

- [ ] **Step 5: Implement dependency plot**

`traceshap/plots/dependency.py`:
```python
from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def dependency_plot(
    multi_trajectory_attrs: list[list[StepAttribution]],
    step_name: str,
    color_by: str,
    title: str | None = None,
) -> go.Figure:
    if title is None:
        title = f"TraceSHAP Dependency: {step_name} vs {color_by}"

    x_values = []
    y_values = []
    color_values = []

    for traj_attrs in multi_trajectory_attrs:
        attr_map = {a.step_name: a for a in traj_attrs}
        target = attr_map.get(step_name)
        color_attr = attr_map.get(color_by)

        if target is None:
            continue

        x_val = target.cost_delta or 0.0
        y_val = target.quality_delta or 0.0
        c_val = (color_attr.quality_delta or 0.0) if color_attr else 0.0

        x_values.append(x_val)
        y_values.append(y_val)
        color_values.append(c_val)

    fig = go.Figure(go.Scatter(
        x=x_values,
        y=y_values,
        mode="markers",
        marker=dict(
            size=8,
            color=color_values,
            colorscale="RdBu_r",
            showscale=True,
            colorbar=dict(title=f"{color_by} SHAP"),
            opacity=0.8,
        ),
        hovertemplate=(
            f"<b>{step_name}</b><br>"
            f"cost_delta: %{{x:.4f}}<br>"
            f"quality_delta: %{{y:.4f}}<br>"
            f"{color_by} SHAP: %{{marker.color:.4f}}<br>"
            f"<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f"{step_name} cost_delta",
        yaxis_title=f"{step_name} SHAP Value",
        template="plotly_white",
        height=500,
    )

    return fig
```

- [ ] **Step 6: Update plots __init__.py**

`traceshap/plots/__init__.py`:
```python
from traceshap.plots.force import force_plot
from traceshap.plots.waterfall import waterfall_plot
from traceshap.plots.beeswarm import beeswarm_plot
from traceshap.plots.bar import bar_plot
from traceshap.plots.dependency import dependency_plot

__all__ = ["force_plot", "waterfall_plot", "beeswarm_plot", "bar_plot", "dependency_plot"]
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_plots.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add traceshap/plots/ tests/test_plots.py
git commit -m "feat: beeswarm, bar, and dependency SHAP plots (plotly)"
```

---

### Task 8: CLI Plot Commands

**Files:**
- Modify: `traceshap/cli/main.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_cli.py`:
```python
class TestCLIPlot:
    def test_plot_force(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        asyncio.run(_seed_db(db_path))
        output_path = str(tmp_path / "force.html")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "plot", "force", "t1",
            "--db", db_path,
            "--output", output_path,
        ])
        assert result.exit_code == 0
        assert (tmp_path / "force.html").exists()

    def test_plot_waterfall(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        asyncio.run(_seed_db(db_path))
        output_path = str(tmp_path / "waterfall.html")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "plot", "waterfall", "t1",
            "--db", db_path,
            "--output", output_path,
        ])
        assert result.exit_code == 0
        assert (tmp_path / "waterfall.html").exists()

    def test_plot_bar(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        asyncio.run(_seed_db(db_path))
        output_path = str(tmp_path / "bar.html")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "plot", "bar",
            "--db", db_path,
            "--agent", "test-agent",
            "--output", output_path,
        ])
        assert result.exit_code == 0
        assert (tmp_path / "bar.html").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestCLIPlot -v`
Expected: FAIL

- [ ] **Step 3: Implement plot CLI group**

Add to `traceshap/cli/main.py`:
```python
from traceshap.plots.force import force_plot
from traceshap.plots.waterfall import waterfall_plot
from traceshap.plots.beeswarm import beeswarm_plot
from traceshap.plots.bar import bar_plot
from traceshap.plots.dependency import dependency_plot


@cli.group()
def plot():
    """Generate SHAP-style visualizations."""
    pass


@plot.command("force")
@click.argument("trace_id")
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--output", default="force.html", help="Output file path")
@click.option("--layers", default="0", help="Comma-separated layer IDs")
def plot_force(trace_id: str, db: str, output: str, layers: str):
    """Generate a force plot for a single trajectory."""
    layer_ids = [int(x.strip()) for x in layers.split(",")]

    async def _run():
        backend = await open_backend(db)
        try:
            trajectory = await backend.get_trajectory(trace_id)
            if trajectory is None:
                click.echo(f"Error: Trajectory '{trace_id}' not found.", err=True)
                raise SystemExit(1)

            training_trajs = None
            if any(lid in (1, 2) for lid in layer_ids):
                training_trajs = await backend.query_trajectories(QueryFilter(limit=200))

            engine = build_engine(layer_ids, training_trajs)
            attributions = await engine.analyze(trajectory)

            base = trajectory.outcome.quality_score * 0.5 if trajectory.outcome and trajectory.outcome.quality_score else 0.5
            fig = force_plot(attributions, base_value=base)
            fig.write_html(output)
            click.echo(f"Force plot saved to {output}")
        finally:
            await backend.close()

    run_async(_run())


@plot.command("waterfall")
@click.argument("trace_id")
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--output", default="waterfall.html", help="Output file path")
@click.option("--layers", default="0", help="Comma-separated layer IDs")
def plot_waterfall(trace_id: str, db: str, output: str, layers: str):
    """Generate a waterfall plot for a single trajectory."""
    layer_ids = [int(x.strip()) for x in layers.split(",")]

    async def _run():
        backend = await open_backend(db)
        try:
            trajectory = await backend.get_trajectory(trace_id)
            if trajectory is None:
                click.echo(f"Error: Trajectory '{trace_id}' not found.", err=True)
                raise SystemExit(1)

            training_trajs = None
            if any(lid in (1, 2) for lid in layer_ids):
                training_trajs = await backend.query_trajectories(QueryFilter(limit=200))

            engine = build_engine(layer_ids, training_trajs)
            attributions = await engine.analyze(trajectory)

            base = trajectory.outcome.quality_score * 0.5 if trajectory.outcome and trajectory.outcome.quality_score else 0.5
            fig = waterfall_plot(attributions, base_value=base)
            fig.write_html(output)
            click.echo(f"Waterfall plot saved to {output}")
        finally:
            await backend.close()

    run_async(_run())


@plot.command("bar")
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--agent", required=True, help="Agent name")
@click.option("--top-k", default=10, type=int, help="Number of top steps")
@click.option("--output", default="bar.html", help="Output file path")
@click.option("--layers", default="0", help="Comma-separated layer IDs")
def plot_bar_cmd(db: str, agent: str, top_k: int, output: str, layers: str):
    """Generate a bar plot of step importance across trajectories."""
    layer_ids = [int(x.strip()) for x in layers.split(",")]

    async def _run():
        backend = await open_backend(db)
        try:
            trajectories = await backend.query_trajectories(
                QueryFilter(agent_name=agent, limit=200)
            )
            if not trajectories:
                click.echo(f"No trajectories found for agent '{agent}'.")
                return

            engine = build_engine(layer_ids, trajectories if any(lid in (1, 2) for lid in layer_ids) else None)

            multi_attrs = []
            for traj in trajectories:
                attrs = await engine.analyze(traj)
                multi_attrs.append(attrs)

            fig = bar_plot(multi_attrs, top_k=top_k)
            fig.write_html(output)
            click.echo(f"Bar plot saved to {output}")
        finally:
            await backend.close()

    run_async(_run())


@plot.command("beeswarm")
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--agent", required=True, help="Agent name")
@click.option("--color-by", default="cost_delta", help="Color variable")
@click.option("--output", default="beeswarm.html", help="Output file path")
@click.option("--layers", default="0", help="Comma-separated layer IDs")
def plot_beeswarm_cmd(db: str, agent: str, color_by: str, output: str, layers: str):
    """Generate a beeswarm plot across trajectories."""
    layer_ids = [int(x.strip()) for x in layers.split(",")]

    async def _run():
        backend = await open_backend(db)
        try:
            trajectories = await backend.query_trajectories(
                QueryFilter(agent_name=agent, limit=200)
            )
            if not trajectories:
                click.echo(f"No trajectories found for agent '{agent}'.")
                return

            engine = build_engine(layer_ids, trajectories if any(lid in (1, 2) for lid in layer_ids) else None)

            multi_attrs = []
            for traj in trajectories:
                attrs = await engine.analyze(traj)
                multi_attrs.append(attrs)

            fig = beeswarm_plot(multi_attrs, color_by=color_by)
            fig.write_html(output)
            click.echo(f"Beeswarm plot saved to {output}")
        finally:
            await backend.close()

    run_async(_run())


@plot.command("dependency")
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--agent", required=True, help="Agent name")
@click.option("--step-type", required=True, help="Step to analyze")
@click.option("--color-by", required=True, help="Step for color encoding")
@click.option("--output", default="dependency.html", help="Output file path")
@click.option("--layers", default="0", help="Comma-separated layer IDs")
def plot_dependency_cmd(db: str, agent: str, step_type: str, color_by: str, output: str, layers: str):
    """Generate a dependency plot for step interactions."""
    layer_ids = [int(x.strip()) for x in layers.split(",")]

    async def _run():
        backend = await open_backend(db)
        try:
            trajectories = await backend.query_trajectories(
                QueryFilter(agent_name=agent, limit=200)
            )
            if not trajectories:
                click.echo(f"No trajectories found for agent '{agent}'.")
                return

            engine = build_engine(layer_ids, trajectories if any(lid in (1, 2) for lid in layer_ids) else None)

            multi_attrs = []
            for traj in trajectories:
                attrs = await engine.analyze(traj)
                multi_attrs.append(attrs)

            fig = dependency_plot(multi_attrs, step_name=step_type, color_by=color_by)
            fig.write_html(output)
            click.echo(f"Dependency plot saved to {output}")
        finally:
            await backend.close()

    run_async(_run())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/cli/main.py tests/test_cli.py
git commit -m "feat: CLI plot commands (force, waterfall, bar, beeswarm, dependency)"
```

---

### Task 9: CLI Serve Command + Full Integration Test

**Files:**
- Modify: `traceshap/cli/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_api.py`:
```python
class TestFullIntegration:
    async def test_ingest_then_analyze_then_plot(self, seeded_app, tmp_path):
        transport = ASGITransport(app=seeded_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            traces_resp = await client.get("/api/traces")
            assert traces_resp.status_code == 200
            traces = traces_resp.json()
            assert len(traces) >= 1

            trace_id = traces[0]["trace_id"]

            detail_resp = await client.get(f"/api/traces/{trace_id}")
            assert detail_resp.status_code == 200
            detail = detail_resp.json()
            assert len(detail["steps"]) >= 1

            attr_resp = await client.get(f"/api/traces/{trace_id}/attribution?layers=0")
            assert attr_resp.status_code == 200
            attrs = attr_resp.json()
            assert len(attrs) >= 1
            assert all("verdict" in a for a in attrs)

            stats_resp = await client.get("/api/agents/test-agent/stats")
            assert stats_resp.status_code == 200
            stats = stats_resp.json()
            assert stats["trajectory_count"] >= 1
```

- [ ] **Step 2: Implement serve command**

Add to `traceshap/cli/main.py`:
```python
@cli.command()
@click.option("--db", default="./traceshap.db", help="Database path")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8080, type=int, help="Port to bind to")
def serve(db: str, host: str, port: int):
    """Start the TraceSHAP API server."""
    import uvicorn
    from traceshap.api.app import create_app
    from traceshap.storage.sqlite import SQLiteBackend

    async def _init():
        backend = SQLiteBackend(db)
        await backend.initialize()
        return backend

    backend = run_async(_init())
    app = create_app(backend=backend)
    click.echo(f"TraceSHAP server starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
```

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add traceshap/cli/main.py tests/test_api.py
git commit -m "feat: CLI serve command and full integration test"
```

---

## Summary

After completing all 9 tasks, you will have:

1. **CLI entry point** with `traceshap --version` and `traceshap init`
2. **CLI analyze + report** — JSON and human-readable attribution output
3. **CLI prune-report + export** — cross-trajectory pruning and JSON/CSV export
4. **FastAPI app** — trajectory list and detail endpoints
5. **Attribution + pruning API** — HTTP endpoints for analysis
6. **Force + waterfall plots** — single-trajectory SHAP visualizations (plotly)
7. **Beeswarm + bar + dependency plots** — cross-trajectory visualizations
8. **CLI plot commands** — all 5 plot types accessible from CLI
9. **Serve command** — full API server with integration test

**Next plans:**
- **Plan 4**: Web Dashboard (React + Vite frontend)
- **Plan 5**: LangGraph native adapter + bridge extractors
