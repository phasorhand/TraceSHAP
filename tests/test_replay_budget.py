"""Tests for ReplayBudget and sample_coalitions (Layer 3 Replay SHAP)."""
from __future__ import annotations

import pytest

from traceshap.attribution.replay.budget import ReplayBudget, sample_coalitions


# ---------------------------------------------------------------------------
# ReplayBudget tests
# ---------------------------------------------------------------------------

class TestBudgetFromTrajectory:
    """test_budget_from_trajectory — verify max_replays = n_steps * multiplier."""

    def test_default_multiplier(self):
        budget = ReplayBudget.from_trajectory(n_steps=10)
        assert budget.max_replays == 20  # 10 * 2.0

    def test_custom_multiplier(self):
        budget = ReplayBudget.from_trajectory(n_steps=5, multiplier=3.0)
        assert budget.max_replays == 15  # 5 * 3.0

    def test_fractional_multiplier_truncates(self):
        budget = ReplayBudget.from_trajectory(n_steps=3, multiplier=1.5)
        assert budget.max_replays == 4  # int(3 * 1.5) = int(4.5) = 4

    def test_used_starts_at_zero(self):
        budget = ReplayBudget.from_trajectory(n_steps=10)
        assert budget.used == 0


class TestBudgetConsume:
    """test_budget_consume — consume reduces remaining."""

    def test_remaining_initially_equals_max(self):
        budget = ReplayBudget(max_replays=10)
        assert budget.remaining == 10

    def test_consume_single_reduces_remaining(self):
        budget = ReplayBudget(max_replays=10)
        budget.consume()
        assert budget.remaining == 9
        assert budget.used == 1

    def test_consume_multiple_reduces_correctly(self):
        budget = ReplayBudget(max_replays=10)
        budget.consume(3)
        assert budget.remaining == 7
        assert budget.used == 3

    def test_remaining_never_negative(self):
        budget = ReplayBudget(max_replays=5, used=5)
        assert budget.remaining == 0

    def test_consume_exactly_to_limit(self):
        budget = ReplayBudget(max_replays=5)
        budget.consume(5)
        assert budget.remaining == 0


class TestBudgetConsumeRaisesOnOverbudget:
    """test_budget_consume_raises_on_overbudget — RuntimeError when exhausted."""

    def test_raises_when_already_exhausted(self):
        budget = ReplayBudget(max_replays=5, used=5)
        with pytest.raises(RuntimeError):
            budget.consume()

    def test_raises_when_consume_would_exceed(self):
        budget = ReplayBudget(max_replays=5, used=4)
        with pytest.raises(RuntimeError):
            budget.consume(2)  # would make used=6, exceeding max=5

    def test_no_raise_at_exact_limit(self):
        budget = ReplayBudget(max_replays=5, used=4)
        budget.consume(1)  # exactly hits limit — no error
        assert budget.used == 5


# ---------------------------------------------------------------------------
# sample_coalitions tests
# ---------------------------------------------------------------------------

class TestSampleCoalitionsIncludesEmptyAndFull:
    """test_sample_coalitions_includes_empty_and_full — empty set and full set present."""

    def test_empty_set_present(self):
        steps = ["a", "b", "c"]
        coalitions = sample_coalitions(steps, budget=8)
        assert frozenset() in [frozenset(c) for c in coalitions]

    def test_full_set_present(self):
        steps = ["a", "b", "c"]
        coalitions = sample_coalitions(steps, budget=8)
        full = frozenset(steps)
        assert full in [frozenset(c) for c in coalitions]

    def test_empty_and_full_with_budget_2(self):
        steps = ["x", "y"]
        coalitions = sample_coalitions(steps, budget=2)
        frozen = [frozenset(c) for c in coalitions]
        assert frozenset() in frozen
        assert frozenset(steps) in frozen


class TestSampleCoalitionsRespectsBudget:
    """test_sample_coalitions_respects_budget — len(result) == budget."""

    def test_budget_respected_small(self):
        steps = ["a", "b", "c", "d", "e"]
        budget = 6
        coalitions = sample_coalitions(steps, budget=budget)
        assert len(coalitions) == budget

    def test_budget_respected_larger(self):
        steps = [f"step_{i}" for i in range(8)]
        budget = 20
        coalitions = sample_coalitions(steps, budget=budget)
        assert len(coalitions) == budget

    def test_no_duplicate_coalitions(self):
        steps = ["a", "b", "c", "d"]
        budget = 10
        coalitions = sample_coalitions(steps, budget=budget)
        frozen = [frozenset(c) for c in coalitions]
        assert len(frozen) == len(set(frozen))


class TestSampleCoalitionsSmallNExact:
    """test_sample_coalitions_small_n_exact — when budget >= 2^n, returns at most 2^n."""

    def test_budget_exceeds_all_coalitions(self):
        steps = ["a", "b", "c"]
        # 2^3 = 8 total coalitions
        coalitions = sample_coalitions(steps, budget=100)
        assert len(coalitions) <= 8

    def test_exhaustive_returns_all_coalitions(self):
        steps = ["a", "b"]
        # 2^2 = 4 total coalitions
        coalitions = sample_coalitions(steps, budget=50)
        assert len(coalitions) == 4
        frozen = {frozenset(c) for c in coalitions}
        expected = {frozenset(), frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})}
        assert frozen == expected

    def test_budget_exactly_equals_2_to_n(self):
        steps = ["a", "b", "c"]
        # 2^3 = 8
        coalitions = sample_coalitions(steps, budget=8)
        assert len(coalitions) == 8
