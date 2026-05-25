"""Layer 4 attribution — Graph-based causal / associational hypothesis generation.

Uses a :class:`TrajectoryGraphBuilder` to construct a causal dependency graph
and derives per-step effect scores and :class:`CausalHypothesis` objects from
that graph.
"""
from __future__ import annotations

from collections import deque

from traceshap.attribution.base import LayerResult
from traceshap.attribution.causal.graph_builder import TrajectoryGraphBuilder
from traceshap.attribution.causal.models import CausalEdge, CausalHypothesis, EdgeType
from traceshap.models.trajectory import Trajectory

# Base confidence values for each edge type used in _compute_confidence.
_EDGE_TYPE_BASE_CONFIDENCE: dict[EdgeType, float] = {
    EdgeType.CONTROL_FLOW: 0.7,
    EdgeType.DATA_DEPENDENCY: 0.5,
    EdgeType.TEMPORAL: 0.2,
}


class Layer4Causal:
    """Attribution layer that generates associational hypotheses from graph structure.

    Parameters
    ----------
    graph_builder:
        A :class:`TrajectoryGraphBuilder` used to derive causal edges from the
        trajectory.
    """

    def __init__(self, graph_builder: TrajectoryGraphBuilder) -> None:
        self._graph_builder = graph_builder

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    @property
    def layer_id(self) -> int:
        return 4

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        """Compute Layer 4 attributions for every step in *trajectory*.

        Returns
        -------
        list[LayerResult]
            One :class:`LayerResult` per step (empty list if no steps), with
            ``layer=4`` and ``quality_delta`` set to the fraction of downstream
            steps reachable from that step.
        """
        steps = trajectory.steps
        if not steps:
            return []

        edges = self._graph_builder.build(trajectory)
        n = len(steps)
        denom = max(n - 1, 1)

        results: list[LayerResult] = []
        for step in steps:
            downstream = self._find_downstream(step.step_id, edges)
            effect = len(downstream) / denom

            # Collect outgoing edges for this step to compute confidence.
            outgoing = [e for e in edges if e.source_step_id == step.step_id]
            confidence = self._compute_confidence(outgoing)

            results.append(LayerResult(
                layer=4,
                step_id=step.step_id,
                quality_delta=effect,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=max(0.0, confidence - 0.1),
                confidence_upper=min(1.0, confidence + 0.1),
                evidence="associational hypothesis from graph structure",
            ))

        return results

    def build_hypotheses(self, trajectory: Trajectory) -> list[CausalHypothesis]:
        """Build associational :class:`CausalHypothesis` objects for every step.

        In the MVP all hypotheses are tagged as ``"associational"``; no causal
        inference is performed.

        Returns
        -------
        list[CausalHypothesis]
            One hypothesis per step, ordered to match ``trajectory.steps``.
        """
        steps = trajectory.steps
        if not steps:
            return []

        edges = self._graph_builder.build(trajectory)
        n = len(steps)
        denom = max(n - 1, 1)

        hypotheses: list[CausalHypothesis] = []
        for step in steps:
            downstream = self._find_downstream(step.step_id, edges)
            magnitude = len(downstream) / denom

            outgoing = [e for e in edges if e.source_step_id == step.step_id]
            confidence = self._compute_confidence(outgoing)

            effect_direction = "positive" if magnitude > 0.3 else "neutral"

            hypotheses.append(CausalHypothesis(
                hypothesis_type="associational",
                source_step_id=step.step_id,
                target="outcome",
                effect_direction=effect_direction,
                effect_magnitude=magnitude,
                downstream_effects=list(downstream),
                evidence_sources=["graph_structure"],
                confidence=confidence,
            ))

        return hypotheses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_downstream(self, step_id: str, edges: list[CausalEdge]) -> list[str]:
        """BFS from *step_id* following *edges*, returning all reachable step IDs.

        The source step itself is not included in the result.

        Parameters
        ----------
        step_id:
            The starting node.
        edges:
            All edges in the graph.

        Returns
        -------
        list[str]
            Reachable step IDs (order determined by BFS traversal).
        """
        # Build adjacency list for quick lookup.
        adjacency: dict[str, list[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge.source_step_id, []).append(edge.target_step_id)

        visited: set[str] = set()
        queue: deque[str] = deque([step_id])

        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Exclude the source itself (it may never be added, but guard anyway).
        visited.discard(step_id)
        return list(visited)

    def _compute_confidence(self, edges: list[CausalEdge]) -> float:
        """Average base confidence across *edges* (outgoing from one step).

        Base values per edge type:
        - ``CONTROL_FLOW``   → 0.7
        - ``DATA_DEPENDENCY`` → 0.5
        - ``TEMPORAL``        → 0.2

        Returns 0.2 if *edges* is empty; caps the result at 0.95.

        Parameters
        ----------
        edges:
            Outgoing edges from a single step.

        Returns
        -------
        float
            Confidence in [0, 0.95].
        """
        if not edges:
            return 0.2

        total = sum(_EDGE_TYPE_BASE_CONFIDENCE.get(e.edge_type, 0.2) for e in edges)
        avg = total / len(edges)
        return min(avg, 0.95)
