"""ReplayEngine — recorded-IO mode replay for TraceSHAP Layer 3.

The engine replays a :class:`ReplayCapsule` with a subset of its recorded
steps removed (*ablated*) and returns an :class:`Outcome` whose
``quality_score`` is proportional to the fraction of steps that remain active.

MVP quality formula
-------------------
    quality_score = active_steps / total_steps

where ``active_steps`` is the count of :class:`RecordedIO` entries whose
``step_id`` is NOT in *ablated_step_ids*, and ``total_steps`` is the total
number of recorded IOs in the capsule.

Special case: if the capsule has no recorded IOs the quality score is 1.0.

Results are memoised in a :class:`ReplayCache` keyed on the frozenset of
ablated step IDs, so repeated calls with the same arguments return the
*same* :class:`Outcome` object.
"""
from __future__ import annotations

from traceshap.attribution.replay.cache import ReplayCache
from traceshap.attribution.replay.capsule import ReplayCapsule
from traceshap.models.outcome import Outcome


class ReplayEngine:
    """Manages capsule registration and counterfactual replay."""

    def __init__(self) -> None:
        self._capsules: dict[str, ReplayCapsule] = {}
        self._cache = ReplayCache()

    # ------------------------------------------------------------------
    # Capsule registry
    # ------------------------------------------------------------------

    def register_capsule(self, capsule: ReplayCapsule) -> None:
        """Store *capsule* indexed by its ``trace_id``."""
        self._capsules[capsule.trace_id] = capsule

    def get_capsule(self, trace_id: str) -> ReplayCapsule | None:
        """Return the capsule for *trace_id*, or ``None`` if not registered."""
        return self._capsules.get(trace_id)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    async def replay_without(
        self,
        capsule: ReplayCapsule,
        ablated_step_ids: list[str],
    ) -> Outcome:
        """Simulate a replay with the given steps removed.

        Returns a cached :class:`Outcome` if an identical ablation set was
        already evaluated; otherwise computes and caches a fresh one.

        Parameters
        ----------
        capsule:
            The :class:`ReplayCapsule` whose recorded IOs are used.
        ablated_step_ids:
            Step IDs to exclude from the replay.  Order does not matter.

        Returns
        -------
        Outcome
            ``quality_score`` = active_steps / total_steps (1.0 when total is 0).
        """
        ablated_set: set[str] = set(ablated_step_ids)

        cached = self._cache.get(ablated_set)
        if cached is not None:
            return cached

        total_steps = len(capsule.recorded_ios)
        if total_steps == 0:
            quality_score = 1.0
        else:
            active_steps = sum(
                1 for rio in capsule.recorded_ios
                if rio.step_id not in ablated_set
            )
            quality_score = active_steps / total_steps

        outcome = Outcome(
            success=True,
            quality_score=quality_score,
            token_cost=0,
            latency_ms=0,
            custom_metrics={},
        )

        self._cache.put(ablated_set, outcome)
        return outcome
