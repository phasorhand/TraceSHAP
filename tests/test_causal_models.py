from __future__ import annotations

import pytest

from traceshap.attribution.causal.models import CausalEdge, CausalHypothesis, EdgeType


def test_edge_type_values():
    assert EdgeType.CONTROL_FLOW.value == "control_flow"
    assert EdgeType.DATA_DEPENDENCY.value == "data_dependency"
    assert EdgeType.TEMPORAL.value == "temporal"


def test_causal_edge_creation():
    edge = CausalEdge(
        source_step_id="step_1",
        target_step_id="step_2",
        edge_type=EdgeType.DATA_DEPENDENCY,
        confidence=0.85,
        evidence="Step 1 output is consumed by Step 2",
    )
    assert edge.source_step_id == "step_1"
    assert edge.target_step_id == "step_2"
    assert edge.edge_type == EdgeType.DATA_DEPENDENCY
    assert edge.confidence == 0.85
    assert edge.evidence == "Step 1 output is consumed by Step 2"


def test_causal_hypothesis_associational():
    hyp = CausalHypothesis(
        hypothesis_type="associational",
        source_step_id="step_3",
        target="outcome",
        effect_direction="positive",
        effect_magnitude=0.4,
        downstream_effects=["step_4", "step_5"],
        evidence_sources=["correlation_analysis"],
        confidence=0.6,
    )
    assert hyp.hypothesis_type == "associational"
    assert hyp.is_causal is False


def test_causal_hypothesis_causal():
    hyp = CausalHypothesis(
        hypothesis_type="causal",
        source_step_id="step_1",
        target="step_2",
        effect_direction="negative",
        effect_magnitude=0.75,
        downstream_effects=[],
        evidence_sources=["replay_experiment", "shap_values"],
        confidence=0.9,
    )
    assert hyp.hypothesis_type == "causal"
    assert hyp.is_causal is True
