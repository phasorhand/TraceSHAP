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
