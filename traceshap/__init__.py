"""TraceSHAP: Attribution and ablation analysis for LLM agent trajectories."""

__version__ = "0.1.0"

from traceshap.convenience import spans_to_trajectory, quick_analyze
from traceshap.adapters.instrument import InstrumentedApp


def instrument(graph, framework: str = "langgraph", agent_name: str = "default") -> InstrumentedApp:
    return InstrumentedApp(graph, framework=framework, agent_name=agent_name)


__all__ = [
    "__version__",
    "spans_to_trajectory",
    "quick_analyze",
    "instrument",
    "InstrumentedApp",
]
