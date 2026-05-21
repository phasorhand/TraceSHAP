import pytest
from traceshap.attribution.base import LayerResult, merge_layer_results


class TestLayerResult:
    def test_create(self):
        result = LayerResult(
            layer=0,
            step_id="step-001",
            quality_delta=-0.1,
            cost_delta=0.002,
            latency_delta=500,
            risk_delta=0.0,
            confidence_lower=-0.15,
            confidence_upper=-0.05,
            evidence="Layer 0: repetition detected (3 retries)",
        )
        assert result.layer == 0
        assert result.step_id == "step-001"
        assert result.quality_delta == -0.1

    def test_merge_single_layer(self):
        results = [
            LayerResult(layer=0, step_id="s1", quality_delta=-0.1,
                        cost_delta=0.001, latency_delta=100, risk_delta=0.0,
                        confidence_lower=-0.15, confidence_upper=-0.05,
                        evidence="repetition rule"),
        ]
        merged = merge_layer_results("s1", "search_web", "node1", results)
        assert merged.step_id == "s1"
        assert merged.quality_delta == -0.1
        assert merged.layer_scores == {0: -0.1}
        assert merged.confidence is not None
        assert len(merged.evidence) == 1

    def test_merge_multiple_layers(self):
        results = [
            LayerResult(layer=0, step_id="s1", quality_delta=0.0,
                        cost_delta=0.001, latency_delta=100, risk_delta=0.0,
                        confidence_lower=-0.02, confidence_upper=0.02,
                        evidence="no rule violation"),
            LayerResult(layer=1, step_id="s1", quality_delta=-0.05,
                        cost_delta=0.001, latency_delta=100, risk_delta=0.0,
                        confidence_lower=-0.10, confidence_upper=0.0,
                        evidence="lift = 0.95, below baseline"),
            LayerResult(layer=2, step_id="s1", quality_delta=-0.08,
                        cost_delta=0.001, latency_delta=100, risk_delta=0.0,
                        confidence_lower=-0.12, confidence_upper=-0.04,
                        evidence="sequence estimator: removal hurts quality"),
        ]
        merged = merge_layer_results("s1", "search_web", "node1", results)
        assert merged.layer_scores == {0: 0.0, 1: -0.05, 2: -0.08}
        assert merged.quality_delta == -0.08
        assert merged.confidence.lower == -0.12
        assert merged.confidence.upper == -0.04
        assert len(merged.evidence) == 3

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError):
            merge_layer_results("s1", "test", None, [])
