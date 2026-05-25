"""Tests for Kernel SHAP computation via weighted least squares."""

import pytest

from traceshap.attribution.replay.kernel_shap import kernel_shap_from_coalitions


# ---------------------------------------------------------------------------
# test_kernel_shap_two_steps_simple
# ---------------------------------------------------------------------------

def test_kernel_shap_two_steps_simple():
    """2 steps, all 4 coalitions provided; SHAP values should sum to full - empty."""
    step_ids = ["A", "B"]
    coalition_values = {
        frozenset(): 0.0,
        frozenset(["A"]): 0.6,
        frozenset(["B"]): 0.3,
        frozenset(["A", "B"]): 1.0,
    }

    result = kernel_shap_from_coalitions(step_ids, coalition_values)

    assert set(result.keys()) == {"A", "B"}
    total = sum(result.values())
    expected_total = 1.0 - 0.0
    assert abs(total - expected_total) < 1e-6, f"SHAP sum {total} != {expected_total}"
    # A contributes more than B
    assert result["A"] > result["B"]


# ---------------------------------------------------------------------------
# test_kernel_shap_three_steps
# ---------------------------------------------------------------------------

def test_kernel_shap_three_steps():
    """3 steps with all 8 coalitions; step with highest individual lift gets highest SHAP."""
    step_ids = ["A", "B", "C"]
    # C has dominant individual contribution
    coalition_values = {
        frozenset(): 0.0,
        frozenset(["A"]): 0.1,
        frozenset(["B"]): 0.1,
        frozenset(["C"]): 0.7,
        frozenset(["A", "B"]): 0.2,
        frozenset(["A", "C"]): 0.8,
        frozenset(["B", "C"]): 0.8,
        frozenset(["A", "B", "C"]): 1.0,
    }

    result = kernel_shap_from_coalitions(step_ids, coalition_values)

    assert set(result.keys()) == {"A", "B", "C"}
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-6, f"SHAP sum {total} != 1.0"
    # C should dominate
    assert result["C"] > result["A"]
    assert result["C"] > result["B"]


# ---------------------------------------------------------------------------
# test_kernel_shap_single_step
# ---------------------------------------------------------------------------

def test_kernel_shap_single_step():
    """1 step: SHAP value must equal full_val - empty_val."""
    step_ids = ["only"]
    coalition_values = {
        frozenset(): 2.0,
        frozenset(["only"]): 5.0,
    }

    result = kernel_shap_from_coalitions(step_ids, coalition_values)

    assert set(result.keys()) == {"only"}
    assert abs(result["only"] - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# test_kernel_shap_equal_contribution
# ---------------------------------------------------------------------------

def test_kernel_shap_equal_contribution():
    """2 symmetric steps: SHAP values should be approximately equal."""
    step_ids = ["X", "Y"]
    # Perfectly symmetric: marginal of each alone is identical
    coalition_values = {
        frozenset(): 0.0,
        frozenset(["X"]): 0.5,
        frozenset(["Y"]): 0.5,
        frozenset(["X", "Y"]): 1.0,
    }

    result = kernel_shap_from_coalitions(step_ids, coalition_values)

    assert set(result.keys()) == {"X", "Y"}
    assert abs(result["X"] - result["Y"]) < 1e-6
    assert abs(result["X"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# test_kernel_shap_insufficient_coalitions_returns_uniform
# ---------------------------------------------------------------------------

def test_kernel_shap_insufficient_coalitions_returns_uniform():
    """Only empty + full coalitions provided → uniform split of (full - empty) / n."""
    step_ids = ["P", "Q", "R"]
    coalition_values = {
        frozenset(): 1.0,
        frozenset(["P", "Q", "R"]): 4.0,
    }

    result = kernel_shap_from_coalitions(step_ids, coalition_values)

    assert set(result.keys()) == {"P", "Q", "R"}
    expected = (4.0 - 1.0) / 3
    for step in step_ids:
        assert abs(result[step] - expected) < 1e-9, (
            f"step {step}: {result[step]} != {expected}"
        )
