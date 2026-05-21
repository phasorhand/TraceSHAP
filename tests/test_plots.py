import pytest
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome, Verdict,
)
from traceshap.models.outcome import StepAttribution, ConfidenceInterval
from traceshap.plots.force import force_plot
from traceshap.plots.waterfall import waterfall_plot


def _step(step_id: str, name: str) -> CanonicalStep:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=["s1"], node_id=None,
        tool_name=name, step_type=StepType.ACTION, attempt_index=0,
        loop_iteration=None, input_hash="a", output_hash="b",
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=0.001,
        start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )


def _attr(step_id: str, name: str, quality_delta: float,
          ci_lower: float, ci_upper: float) -> StepAttribution:
    return StepAttribution(
        step_id=step_id, step_name=name, node_id=None,
        quality_delta=quality_delta, cost_delta=0.001,
        latency_delta=1000, risk_delta=0.0,
        layer_scores={0: quality_delta},
        confidence=ConfidenceInterval(lower=ci_lower, point=quality_delta, upper=ci_upper),
        verdict=Verdict.REVIEW,
        causal_hypothesis=None, evidence=["test"], calibration=None,
    )


@pytest.fixture
def sample_attributions():
    return [
        _attr("s1", "plan", 0.15, 0.10, 0.20),
        _attr("s2", "search_web", 0.05, 0.01, 0.09),
        _attr("s3", "summarize", -0.02, -0.05, 0.01),
        _attr("s4", "validate", 0.10, 0.07, 0.13),
    ]


class TestForcePlot:
    def test_returns_plotly_figure(self, sample_attributions):
        fig = force_plot(sample_attributions, base_value=0.5)
        assert fig is not None
        assert hasattr(fig, "to_html")

    def test_save_html(self, sample_attributions, tmp_path):
        fig = force_plot(sample_attributions, base_value=0.5)
        path = tmp_path / "force.html"
        fig.write_html(str(path))
        assert path.exists()
        content = path.read_text()
        assert "plotly" in content.lower() or "div" in content.lower()


class TestWaterfallPlot:
    def test_returns_plotly_figure(self, sample_attributions):
        fig = waterfall_plot(sample_attributions, base_value=0.5)
        assert fig is not None
        assert hasattr(fig, "to_html")

    def test_save_html(self, sample_attributions, tmp_path):
        fig = waterfall_plot(sample_attributions, base_value=0.5)
        path = tmp_path / "waterfall.html"
        fig.write_html(str(path))
        assert path.exists()
