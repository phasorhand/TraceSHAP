import pytest
from datetime import datetime, timezone

from traceshap.models import (
    TraceSHAPSpan, SpanKind, TokenUsage, StepType, SideEffect,
)
from traceshap.ingestion.normalizer import StepNormalizer


def _span(span_id: str, parent: str | None, kind: SpanKind, name: str,
          input_data: dict | None = None, output_data: dict | None = None,
          offset_sec: int = 0, duration_sec: int = 1) -> TraceSHAPSpan:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + duration_sec, tzinfo=timezone.utc)
    return TraceSHAPSpan(
        trace_id="t1", span_id=span_id, parent_span_id=parent,
        span_kind=kind, name=name, input=input_data or {}, output=output_data or {},
        start_time=start, end_time=end, tokens=None, cost=None,
        metadata={}, raw_attributes={}, semconv_version="test",
    )


class TestStepNormalizer:
    def test_single_span_becomes_one_step(self):
        spans = [_span("s1", None, SpanKind.TOOL, "search_web")]
        normalizer = StepNormalizer()
        steps = normalizer.normalize(spans)
        assert len(steps) == 1
        assert steps[0].raw_span_ids == ["s1"]
        assert steps[0].tool_name == "search_web"

    def test_llm_span_is_decision_type(self):
        spans = [_span("s1", None, SpanKind.LLM, "gpt-4o")]
        normalizer = StepNormalizer()
        steps = normalizer.normalize(spans)
        assert steps[0].step_type == StepType.DECISION

    def test_tool_span_is_action_type(self):
        spans = [_span("s1", None, SpanKind.TOOL, "calculator")]
        normalizer = StepNormalizer()
        steps = normalizer.normalize(spans)
        assert steps[0].step_type == StepType.ACTION

    def test_guardrail_span_is_validation_type(self):
        spans = [_span("s1", None, SpanKind.GUARDRAIL, "safety_check")]
        normalizer = StepNormalizer()
        steps = normalizer.normalize(spans)
        assert steps[0].step_type == StepType.VALIDATION
        assert steps[0].is_protected

    def test_evaluator_span_is_validation_type(self):
        spans = [_span("s1", None, SpanKind.EVALUATOR, "quality_judge")]
        normalizer = StepNormalizer()
        steps = normalizer.normalize(spans)
        assert steps[0].step_type == StepType.VALIDATION

    def test_retry_detection(self):
        spans = [
            _span("s1", None, SpanKind.TOOL, "search_web",
                  input_data={"query": "hello"}, offset_sec=0),
            _span("s2", None, SpanKind.TOOL, "search_web",
                  input_data={"query": "hello"}, offset_sec=2),
            _span("s3", None, SpanKind.TOOL, "search_web",
                  input_data={"query": "hello"}, offset_sec=4),
        ]
        normalizer = StepNormalizer()
        steps = normalizer.normalize(spans)
        assert len(steps) == 3
        assert steps[0].attempt_index == 0
        assert steps[1].attempt_index == 1
        assert steps[2].attempt_index == 2

    def test_default_side_effect_is_irreversible(self):
        spans = [_span("s1", None, SpanKind.TOOL, "unknown_tool")]
        normalizer = StepNormalizer()
        steps = normalizer.normalize(spans)
        assert steps[0].side_effect_class == SideEffect.IRREVERSIBLE_WRITE

    def test_custom_side_effect_mapping(self):
        spans = [_span("s1", None, SpanKind.TOOL, "search_web")]
        normalizer = StepNormalizer(
            side_effect_overrides={"search_web": SideEffect.READ_ONLY}
        )
        steps = normalizer.normalize(spans)
        assert steps[0].side_effect_class == SideEffect.READ_ONLY

    def test_llm_is_pure_side_effect(self):
        spans = [_span("s1", None, SpanKind.LLM, "gpt-4o")]
        normalizer = StepNormalizer()
        steps = normalizer.normalize(spans)
        assert steps[0].side_effect_class == SideEffect.PURE

    def test_multiple_different_spans(self):
        spans = [
            _span("s1", None, SpanKind.LLM, "planner", offset_sec=0),
            _span("s2", None, SpanKind.TOOL, "search_web", offset_sec=2),
            _span("s3", None, SpanKind.LLM, "summarizer", offset_sec=4),
        ]
        normalizer = StepNormalizer()
        steps = normalizer.normalize(spans)
        assert len(steps) == 3
        assert [s.step_type for s in steps] == [StepType.DECISION, StepType.ACTION, StepType.DECISION]
