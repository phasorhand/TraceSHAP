"""TraceSHAP: Attribution and ablation analysis for LLM agent trajectories."""

__version__ = "0.1.0"

from traceshap.convenience import spans_to_trajectory, quick_analyze

__all__ = [
    "__version__",
    "spans_to_trajectory",
    "quick_analyze",
]
