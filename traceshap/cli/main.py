import asyncio
import csv
import io
import json
from pathlib import Path

import click
import yaml

import traceshap
from traceshap.cli.helpers import run_async, open_backend, build_engine, attribution_to_dict
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig
from traceshap.storage.backend import QueryFilter
from traceshap.plots.force import force_plot
from traceshap.plots.waterfall import waterfall_plot
from traceshap.plots.beeswarm import beeswarm_plot
from traceshap.plots.bar import bar_plot
from traceshap.plots.dependency import dependency_plot


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
