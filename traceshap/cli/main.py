import asyncio
import json
from pathlib import Path

import click
import yaml

import traceshap
from traceshap.cli.helpers import run_async, open_backend, build_engine, attribution_to_dict
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig
from traceshap.storage.backend import QueryFilter


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
