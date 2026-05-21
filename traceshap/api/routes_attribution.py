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
