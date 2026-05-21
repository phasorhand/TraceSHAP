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
