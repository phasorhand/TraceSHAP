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
