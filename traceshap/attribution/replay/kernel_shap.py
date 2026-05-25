"""Kernel SHAP solver using weighted least squares (numpy).

Given a set of coalition values produced by replaying an LLM agent trajectory
with different subsets of steps masked, this module computes SHAP attribution
values that satisfy the efficiency axiom (SHAP values sum to full_val - empty_val).
"""

from math import comb

import numpy as np


def kernel_shap_from_coalitions(
    step_ids: list[str],
    coalition_values: dict[frozenset[str], float],
) -> dict[str, float]:
    """Compute Kernel SHAP attributions from pre-evaluated coalition values.

    Parameters
    ----------
    step_ids:
        Ordered list of step identifiers that form the grand coalition.
    coalition_values:
        Mapping from coalition (as a frozenset of step ids) to its value.
        Must include at least the empty coalition (frozenset()) and the
        grand coalition (frozenset(step_ids)).

    Returns
    -------
    dict[str, float]
        Mapping from each step_id to its SHAP attribution value.
        The values sum to ``full_val - empty_val``.

    Notes
    -----
    The Kernel SHAP weight for a coalition of size *k* out of *n* steps is:

        w(k) = (n - 1) / (C(n, k) * k * (n - k))   for 0 < k < n

    Interior coalitions (0 < |S| < n) are used to build a weighted least-
    squares system.  If no interior coalitions are present only the boundary
    constraint is available, so the efficiency axiom gives a uniform split.
    """
    n = len(step_ids)
    empty_val = coalition_values.get(frozenset(), 0.0)
    grand_set = frozenset(step_ids)
    full_val = coalition_values.get(grand_set, empty_val)
    gain = full_val - empty_val

    # Single-step shortcut — no interior coalitions possible.
    if n == 1:
        return {step_ids[0]: gain}

    # Index mapping: step_id -> column index.
    step_index = {sid: j for j, sid in enumerate(step_ids)}

    # Collect interior coalitions (0 < size < n).
    interior = [
        (coalition, value)
        for coalition, value in coalition_values.items()
        if 0 < len(coalition) < n
    ]

    # If there are no interior coalitions return a uniform split.
    if not interior:
        uniform = gain / n
        return {sid: uniform for sid in step_ids}

    m = len(interior)

    # Build design matrix X (m × n) and weight vector w (m,).
    X = np.zeros((m, n), dtype=float)
    w = np.zeros(m, dtype=float)
    y = np.zeros(m, dtype=float)

    for i, (coalition, value) in enumerate(interior):
        k = len(coalition)
        for sid in coalition:
            X[i, step_index[sid]] = 1.0
        # Kernel SHAP weight formula.
        w[i] = (n - 1) / (comb(n, k) * k * (n - k))
        y[i] = value - empty_val

    # Weighted least squares: minimise (Xφ - y)^T W (Xφ - y)
    # Normal equations: (X^T W X + λI) φ = X^T W y
    W = np.diag(w)
    XtW = X.T @ W          # (n × m)
    XtWX = XtW @ X         # (n × n)
    XtWy = XtW @ y         # (n,)

    # Tikhonov regularisation for numerical stability.
    reg = 1e-8 * np.eye(n)
    A = XtWX + reg

    phi = np.linalg.solve(A, XtWy)

    # Enforce the efficiency constraint: sum(phi) == gain.
    # Scale phi uniformly so that it sums exactly to gain.
    current_sum = float(phi.sum())
    if abs(current_sum) > 1e-12:
        phi = phi * (gain / current_sum)
    else:
        # Fall back to uniform when the solver produces an all-zero vector.
        phi = np.full(n, gain / n)

    return {sid: float(phi[j]) for j, sid in enumerate(step_ids)}
