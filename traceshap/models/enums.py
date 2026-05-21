from enum import Enum


class SpanKind(Enum):
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    AGENT = "agent"
    RERANKER = "reranker"
    GUARDRAIL = "guardrail"
    EVALUATOR = "evaluator"
    CUSTOM = "custom"


class StepType(Enum):
    DECISION = "decision"
    ACTION = "action"
    OBSERVATION = "observation"
    VALIDATION = "validation"
    META = "meta"


class SideEffect(Enum):
    PURE = "pure"
    READ_ONLY = "read_only"
    IDEMPOTENT_WRITE = "idempotent_write"
    IRREVERSIBLE_WRITE = "irreversible_write"

    def is_safe_for_auto_replay(self) -> bool:
        return self in (SideEffect.PURE, SideEffect.READ_ONLY)


class Verdict(Enum):
    KEEP = "keep"
    REVIEW = "review"
    PRUNE_CANDIDATE = "prune_candidate"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DecisionStatus(Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReplayCapability(Enum):
    NONE = "none"
    DRY_RUN_MOCKED = "dry_run_mocked"
    RECORDED_IO_REPLAY = "recorded_io_replay"
    LIVE_SANDBOX_REPLAY = "live_sandbox_replay"
