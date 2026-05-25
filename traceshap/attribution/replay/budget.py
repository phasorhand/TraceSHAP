"""Budget management and coalition sampling for Layer 3 Replay SHAP."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import chain, combinations


@dataclass
class ReplayBudget:
    """Tracks how many agent replays are permitted for SHAP coalition evaluation.

    Attributes:
        max_replays: Maximum number of replay calls allowed.
        used: Number of replay calls consumed so far.
    """

    max_replays: int
    used: int = 0

    @classmethod
    def from_trajectory(cls, n_steps: int, multiplier: float = 2.0) -> ReplayBudget:
        """Create a budget sized proportionally to the trajectory length.

        Args:
            n_steps: Number of steps in the agent trajectory.
            multiplier: Scale factor applied to *n_steps* (default 2.0).

        Returns:
            A new :class:`ReplayBudget` with ``max_replays = int(n_steps * multiplier)``.
        """
        return cls(max_replays=int(n_steps * multiplier))

    @property
    def remaining(self) -> int:
        """Number of replays still available (never negative)."""
        return max(0, self.max_replays - self.used)

    def consume(self, n: int = 1) -> None:
        """Mark *n* replays as consumed.

        Args:
            n: Number of replays to consume (default 1).

        Raises:
            RuntimeError: If consuming *n* replays would exceed *max_replays*.
        """
        if self.used + n > self.max_replays:
            raise RuntimeError(
                f"Budget exhausted: consuming {n} would bring used to "
                f"{self.used + n}, exceeding max_replays={self.max_replays}."
            )
        self.used += n


def _all_subsets(steps: list[str]):
    """Yield every subset of *steps* as a frozenset, smallest first."""
    return (frozenset(combo) for r in range(len(steps) + 1) for combo in combinations(steps, r))


def sample_coalitions(steps: list[str], budget: int) -> list[set[str]]:
    """Sample coalitions of agent steps for SHAP value estimation.

    The returned list always begins with the empty coalition and the full
    coalition (all steps), then is filled with unique random coalitions up to
    *budget* entries.  When ``budget >= 2**len(steps)``, every possible
    coalition is returned exhaustively (the list may be shorter than *budget*).

    Args:
        steps: Ordered list of step identifiers in the trajectory.
        budget: Desired total number of coalitions (including empty and full).

    Returns:
        A list of unique sets, each being a subset of *steps*.
    """
    n = len(steps)
    total_possible = 2 ** n

    if budget >= total_possible:
        # Return all coalitions exhaustively in a deterministic order.
        return [set(s) for s in _all_subsets(steps)]

    # Always include empty set and full set first.
    empty: frozenset[str] = frozenset()
    full: frozenset[str] = frozenset(steps)

    seen: set[frozenset[str]] = {empty, full}
    result: list[frozenset[str]] = [empty, full]

    # Fill remaining slots with randomly sampled unique coalitions.
    needed = budget - 2
    attempts = 0
    max_attempts = needed * 20 + 1000  # guard against infinite loop

    while len(result) < budget and attempts < max_attempts:
        size = random.randint(0, n)
        coalition = frozenset(random.sample(steps, size))
        if coalition not in seen:
            seen.add(coalition)
            result.append(coalition)
        attempts += 1

    # Fallback: if random sampling couldn't fill the budget (e.g. small n),
    # top-up deterministically from the exhaustive list.
    if len(result) < budget:
        for subset in _all_subsets(steps):
            if subset not in seen:
                seen.add(subset)
                result.append(subset)
            if len(result) == budget:
                break

    return [set(s) for s in result]
