"""Layer 3 attribution — Replay + Kernel SHAP.

Uses a :class:`ReplayEngine` to evaluate counterfactual coalitions of agent
steps and applies Kernel SHAP weighted least-squares to compute per-step
attribution values.
"""
from __future__ import annotations

from traceshap.attribution.base import LayerResult
from traceshap.attribution.replay.budget import ReplayBudget, sample_coalitions
from traceshap.attribution.replay.engine import ReplayEngine
from traceshap.attribution.replay.kernel_shap import kernel_shap_from_coalitions
from traceshap.models.trajectory import Trajectory


class Layer3Replay:
    """Attribution layer that uses recorded-IO replay and Kernel SHAP.

    Parameters
    ----------
    replay_engine:
        The :class:`ReplayEngine` managing capsule registration and replay.
    budget_multiplier:
        Scale factor for the replay budget relative to trajectory length.
        Defaults to 2.0 (i.e. ``max_replays = int(n_steps * budget_multiplier)``).
    """

    def __init__(self, replay_engine: ReplayEngine, budget_multiplier: float = 2.0) -> None:
        self._engine = replay_engine
        self._budget_multiplier = budget_multiplier

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    @property
    def layer_id(self) -> int:
        return 3

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        """Compute Layer 3 SHAP attributions for every step in *trajectory*.

        If no :class:`ReplayCapsule` is registered for the trajectory's
        ``trace_id``, returns zero-delta results with an explanatory evidence
        string via :meth:`_fallback_no_capsule`.

        Parameters
        ----------
        trajectory:
            The agent trajectory to attribute.

        Returns
        -------
        list[LayerResult]
            One :class:`LayerResult` per step, with ``layer=3`` and
            ``quality_delta`` set to the Kernel SHAP value.
        """
        capsule = self._engine.get_capsule(trajectory.trace_id)
        if capsule is None:
            return self._fallback_no_capsule(trajectory)

        step_ids = [step.step_id for step in trajectory.steps]
        n = len(step_ids)

        budget = ReplayBudget.from_trajectory(n, self._budget_multiplier)
        coalitions = sample_coalitions(step_ids, budget.max_replays)

        all_step_ids_set = set(step_ids)

        # Evaluate each coalition by replaying with the complement ablated.
        coalition_values: dict[frozenset[str], float] = {}
        for coalition in coalitions:
            ablated = list(all_step_ids_set - coalition)
            outcome = await self._engine.replay_without(capsule, ablated)
            coalition_values[frozenset(coalition)] = outcome.quality_score

        shap_values = kernel_shap_from_coalitions(step_ids, coalition_values)

        results: list[LayerResult] = []
        for step in trajectory.steps:
            phi = shap_values.get(step.step_id, 0.0)
            results.append(LayerResult(
                layer=3,
                step_id=step.step_id,
                quality_delta=phi,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=phi,
                confidence_upper=phi,
                evidence="replay SHAP (recorded-IO)",
            ))

        return results

    def _fallback_no_capsule(self, trajectory: Trajectory) -> list[LayerResult]:
        """Return zero-delta results when no capsule is available.

        Parameters
        ----------
        trajectory:
            The trajectory whose steps need placeholder results.

        Returns
        -------
        list[LayerResult]
            One :class:`LayerResult` per step, all with ``quality_delta=0.0``
            and ``evidence="no replay capsule available"``.
        """
        return [
            LayerResult(
                layer=3,
                step_id=step.step_id,
                quality_delta=0.0,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                evidence="no replay capsule available",
            )
            for step in trajectory.steps
        ]
