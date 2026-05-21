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
