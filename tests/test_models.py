from traceshap.models.enums import (
    SpanKind,
    StepType,
    SideEffect,
    Verdict,
    DecisionStatus,
    RiskLevel,
    ReplayCapability,
)


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
