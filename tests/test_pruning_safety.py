import pytest
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage, Verdict,
)
from traceshap.models.outcome import StepAttribution, ConfidenceInterval
from traceshap.pruning.safety import (
    is_protected_step, is_first_or_last, classify_step,
    PROTECTED_STEP_TYPES,
)
from traceshap.config import PruningConfig


def _step(step_id: str, step_type: StepType = StepType.ACTION) -> CanonicalStep:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=["s1"], node_id=None,
        tool_name="test", step_type=step_type, attempt_index=0,
        loop_iteration=None, input_hash="a", output_hash="b",
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8, tokens=None, cost=0.001,
        start_time=now, end_time=now,
    )


def _attr(step_id: str, quality_delta: float,
          ci_lower: float, ci_upper: float,
          cost_delta: float = 0.001) -> StepAttribution:
    return StepAttribution(
        step_id=step_id, step_name="test", node_id=None,
        quality_delta=quality_delta, cost_delta=cost_delta,
        latency_delta=100, risk_delta=0.0,
        layer_scores={0: quality_delta},
        confidence=ConfidenceInterval(lower=ci_lower, point=quality_delta, upper=ci_upper),
        verdict=Verdict.REVIEW,
        causal_hypothesis=None, evidence=[], calibration=None,
    )


class TestProtectedStep:
    def test_validation_is_protected(self):
        assert is_protected_step(_step("s1", StepType.VALIDATION))

    def test_action_is_not_protected(self):
        assert not is_protected_step(_step("s1", StepType.ACTION))

    def test_decision_is_not_protected(self):
        assert not is_protected_step(_step("s1", StepType.DECISION))


class TestFirstOrLast:
    def test_first_step(self):
        steps = [_step("s1"), _step("s2"), _step("s3")]
        assert is_first_or_last("s1", steps)

    def test_last_step(self):
        steps = [_step("s1"), _step("s2"), _step("s3")]
        assert is_first_or_last("s3", steps)

    def test_middle_step(self):
        steps = [_step("s1"), _step("s2"), _step("s3")]
        assert not is_first_or_last("s2", steps)


class TestClassifyStep:
    def test_prune_candidate(self):
        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10)
        attr = _attr("s1", quality_delta=-0.01, ci_lower=-0.03, ci_upper=0.01,
                      cost_delta=0.005)
        step = _step("s1")
        verdict = classify_step(attr, step, config, is_first_last=False)
        assert verdict == Verdict.PRUNE_CANDIDATE

    def test_keep_strong_negative(self):
        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10)
        attr = _attr("s1", quality_delta=-0.20, ci_lower=-0.25, ci_upper=-0.15)
        step = _step("s1")
        verdict = classify_step(attr, step, config, is_first_last=False)
        assert verdict == Verdict.KEEP

    def test_review_ambiguous(self):
        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10)
        attr = _attr("s1", quality_delta=-0.07, ci_lower=-0.09, ci_upper=-0.05)
        step = _step("s1")
        verdict = classify_step(attr, step, config, is_first_last=False)
        assert verdict == Verdict.REVIEW

    def test_protected_always_keep(self):
        config = PruningConfig()
        attr = _attr("s1", quality_delta=0.0, ci_lower=-0.01, ci_upper=0.01)
        step = _step("s1", StepType.VALIDATION)
        verdict = classify_step(attr, step, config, is_first_last=False)
        assert verdict == Verdict.KEEP

    def test_first_step_always_keep(self):
        config = PruningConfig()
        attr = _attr("s1", quality_delta=0.0, ci_lower=-0.01, ci_upper=0.01,
                      cost_delta=0.005)
        step = _step("s1")
        verdict = classify_step(attr, step, config, is_first_last=True)
        assert verdict == Verdict.KEEP

    def test_no_confidence_insufficient(self):
        config = PruningConfig()
        attr = StepAttribution(
            step_id="s1", step_name="test", node_id=None,
            quality_delta=None, cost_delta=0.001,
            latency_delta=100, risk_delta=0.0,
            layer_scores={}, confidence=None,
            verdict=Verdict.REVIEW,
            causal_hypothesis=None, evidence=[], calibration=None,
        )
        step = _step("s1")
        verdict = classify_step(attr, step, config, is_first_last=False)
        assert verdict == Verdict.INSUFFICIENT_EVIDENCE
