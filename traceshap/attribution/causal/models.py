from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    CONTROL_FLOW = "control_flow"
    DATA_DEPENDENCY = "data_dependency"
    TEMPORAL = "temporal"


@dataclass
class CausalEdge:
    source_step_id: str
    target_step_id: str
    edge_type: EdgeType
    confidence: float
    evidence: str


@dataclass
class CausalHypothesis:
    hypothesis_type: str  # "causal" or "associational"
    source_step_id: str
    target: str  # "outcome" or a step_id
    effect_direction: str  # "positive", "negative", or "neutral"
    effect_magnitude: float
    downstream_effects: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def is_causal(self) -> bool:
        return self.hypothesis_type == "causal"
