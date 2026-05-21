from datetime import datetime, timezone

from traceshap.models.enums import (
    SpanKind,
    StepType,
    SideEffect,
    Verdict,
    DecisionStatus,
    RiskLevel,
    ReplayCapability,
)
from traceshap.models.span import TokenUsage, TraceSHAPSpan


class TestSpanKind:
    def test_all_values_exist(self):
        assert SpanKind.LLM.value == "llm"
        assert SpanKind.TOOL.value == "tool"
        assert SpanKind.RETRIEVER.value == "retriever"
        assert SpanKind.AGENT.value == "agent"
        assert SpanKind.RERANKER.value == "reranker"
        assert SpanKind.GUARDRAIL.value == "guardrail"
        assert SpanKind.EVALUATOR.value == "evaluator"
        assert SpanKind.CUSTOM.value == "custom"


class TestSideEffect:
    def test_safety_ordering(self):
        ordered = [SideEffect.PURE, SideEffect.READ_ONLY, SideEffect.IDEMPOTENT_WRITE, SideEffect.IRREVERSIBLE_WRITE]
        assert all(a.value != b.value for a, b in zip(ordered, ordered[1:]))

    def test_is_safe_for_replay(self):
        assert SideEffect.PURE.is_safe_for_auto_replay()
        assert SideEffect.READ_ONLY.is_safe_for_auto_replay()
        assert not SideEffect.IDEMPOTENT_WRITE.is_safe_for_auto_replay()
        assert not SideEffect.IRREVERSIBLE_WRITE.is_safe_for_auto_replay()


class TestVerdict:
    def test_all_values_exist(self):
        assert Verdict.KEEP.value == "keep"
        assert Verdict.REVIEW.value == "review"
        assert Verdict.PRUNE_CANDIDATE.value == "prune_candidate"
        assert Verdict.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"


class TestDecisionStatus:
    def test_lifecycle_values(self):
        assert DecisionStatus.CANDIDATE.value == "candidate"
        assert DecisionStatus.VALIDATED.value == "validated"
        assert DecisionStatus.REJECTED.value == "rejected"
        assert DecisionStatus.APPLIED.value == "applied"
        assert DecisionStatus.ROLLED_BACK.value == "rolled_back"


class TestTokenUsage:
    def test_total(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.total_tokens == 150

    def test_zero(self):
        usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        assert usage.total_tokens == 0


class TestTraceSHAPSpan:
    def test_create_minimal(self):
        span = TraceSHAPSpan(
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            span_kind=SpanKind.LLM,
            name="gpt-4o-generation",
            input={"prompt": "hello"},
            output={"text": "world"},
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            tokens=None,
            cost=None,
            metadata={},
            raw_attributes={},
            semconv_version="otel-genai-v0.1",
        )
        assert span.trace_id == "t1"
        assert span.parent_span_id is None
        assert span.duration_ms == 1000

    def test_duration_ms(self):
        span = TraceSHAPSpan(
            trace_id="t1",
            span_id="s1",
            parent_span_id="s0",
            span_kind=SpanKind.TOOL,
            name="search_web",
            input={},
            output={},
            start_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 2, 500000, tzinfo=timezone.utc),
            tokens=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
            cost=0.001,
            metadata={"framework": "langgraph"},
            raw_attributes={"gen_ai.operation.name": "invoke_agent"},
            semconv_version="otel-genai-v0.1",
        )
        assert span.duration_ms == 2500
        assert span.tokens.total_tokens == 30
