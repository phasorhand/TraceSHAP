"""ReplayCache — memoisation layer for replay_without results.

Keys are *frozensets* of ablated step IDs so order does not matter.
"""
from __future__ import annotations

from traceshap.models.outcome import Outcome


class ReplayCache:
    """Thread-unsafe in-memory cache mapping ablated-step sets → Outcome."""

    def __init__(self) -> None:
        self._store: dict[frozenset[str], Outcome] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, ablated: set[str]) -> Outcome | None:
        """Return the cached Outcome for *ablated*, or ``None`` on a miss."""
        return self._store.get(frozenset(ablated))

    def put(self, ablated: set[str], outcome: Outcome) -> None:
        """Store *outcome* under the key derived from *ablated*."""
        self._store[frozenset(ablated)] = outcome

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)
