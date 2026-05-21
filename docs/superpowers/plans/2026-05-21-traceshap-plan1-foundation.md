# TraceSHAP Plan 1: Foundation (Models + Storage + Ingestion + Normalizer)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data pipeline that ingests OTel spans from Langfuse, reconstructs trajectory trees, normalizes spans into CanonicalSteps, and persists everything in SQLite.

**Architecture:** Pipeline-first — a `TraceSHAPPipeline` polls Langfuse for new traces, buffers spans, assembles span trees, normalizes to canonical steps, binds outcomes, and writes to SQLite via async SQLAlchemy. All domain models are plain dataclasses; ORM models are separate.

**Tech Stack:** Python 3.11+, asyncio, SQLAlchemy 2.0 (async), aiosqlite, langfuse SDK, pydantic (config validation), pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-05-21-traceshap-design.md` — Sections 2-5, 8

---

## File Structure

```
traceshap/
├── pyproject.toml
├── traceshap.yaml.example
├── traceshap/
│   ├── __init__.py                    # version, public API re-exports
│   ├── config.py                      # Config loading from traceshap.yaml
│   ├── models/
│   │   ├── __init__.py                # re-exports all models
│   │   ├── enums.py                   # SpanKind, StepType, SideEffect, Verdict, DecisionStatus, RiskLevel
│   │   ├── span.py                    # TokenUsage, TraceSHAPSpan
│   │   ├── step.py                    # CanonicalStep
│   │   ├── trajectory.py             # SpanNode, TrajectoryMeta, Trajectory
│   │   ├── outcome.py                # Outcome, ConfidenceInterval, StepAttribution
│   │   └── pruning.py                # Savings, ValidationPlan, PruneCandidate, PruningReport
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── backend.py                # StorageBackend ABC
│   │   ├── orm.py                    # SQLAlchemy ORM table models
│   │   └── sqlite.py                 # SQLiteBackend implementation
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── sources/
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # SpanSource ABC
│   │   │   └── langfuse.py           # LangfuseSource
│   │   ├── assembler.py              # SpanBuffer, TreeAssembler
│   │   └── normalizer.py             # StepNormalizer
│   └── pipeline.py                   # TraceSHAPPipeline orchestrator
├── tests/
│   ├── conftest.py                   # shared fixtures
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_storage.py
│   ├── test_assembler.py
│   ├── test_normalizer.py
│   └── test_pipeline.py
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `traceshap/__init__.py`
- Create: `traceshap/models/__init__.py`
- Create: `traceshap/storage/__init__.py`
- Create: `traceshap/ingestion/__init__.py`
- Create: `traceshap/ingestion/sources/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "traceshap"
version = "0.1.0"
description = "Attribution and ablation analysis for LLM agent trajectories"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
dependencies = [
    "sqlalchemy[asyncio]>=2.0",
    "aiosqlite>=0.19",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "langfuse>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.4",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100
```

- [ ] **Step 2: Create package __init__ files**

`traceshap/__init__.py`:
```python
"""TraceSHAP: Attribution and ablation analysis for LLM agent trajectories."""

__version__ = "0.1.0"
```

`traceshap/models/__init__.py`:
```python
```

`traceshap/storage/__init__.py`:
```python
```

`traceshap/ingestion/__init__.py`:
```python
```

`traceshap/ingestion/sources/__init__.py`:
```python
```

- [ ] **Step 3: Create test conftest**

`tests/conftest.py`:
```python
import pytest


@pytest.fixture
def sample_trace_id() -> str:
    return "trace-abc-123"


@pytest.fixture
def sample_span_id() -> str:
    return "span-001"
```

- [ ] **Step 4: Install and verify**

Run: `cd /Users/sunxing/Downloads/projects/TraceSHAP && pip install -e ".[dev]"`
Expected: Successful install

Run: `pytest --co -q`
Expected: `no tests ran` (no test files with tests yet)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml traceshap/ tests/
git commit -m "feat: project scaffolding with pyproject.toml and package structure"
```

---

### Task 2: Core Enums

**Files:**
- Create: `traceshap/models/enums.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write tests for enums**

`tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement enums**

`traceshap/models/enums.py`:
```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/models/enums.py tests/test_models.py
git commit -m "feat: core enums (SpanKind, StepType, SideEffect, Verdict, DecisionStatus)"
```

---

### Task 3: Span and Token Models

**Files:**
- Create: `traceshap/models/span.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_models.py`:
```python
from datetime import datetime, timezone
from traceshap.models.span import TokenUsage, TraceSHAPSpan
from traceshap.models.enums import SpanKind


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py::TestTokenUsage -v`
Expected: FAIL

- [ ] **Step 3: Implement span models**

`traceshap/models/span.py`:
```python
from dataclasses import dataclass
from datetime import datetime

from traceshap.models.enums import SpanKind


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class TraceSHAPSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    span_kind: SpanKind
    name: str
    input: dict
    output: dict
    start_time: datetime
    end_time: datetime
    tokens: TokenUsage | None
    cost: float | None
    metadata: dict
    raw_attributes: dict
    semconv_version: str

    @property
    def duration_ms(self) -> int:
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_models.py -v -k "Span or Token"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/models/span.py tests/test_models.py
git commit -m "feat: TraceSHAPSpan and TokenUsage models"
```

---

### Task 4: CanonicalStep Model

**Files:**
- Create: `traceshap/models/step.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_models.py`:
```python
from traceshap.models.step import CanonicalStep
from traceshap.models.enums import StepType, SideEffect


class TestCanonicalStep:
    def test_create(self):
        step = CanonicalStep(
            step_id="step-001",
            raw_span_ids=["s1", "s2"],
            node_id="search_node",
            tool_name="search_web",
            step_type=StepType.ACTION,
            attempt_index=0,
            loop_iteration=None,
            input_hash="abc123",
            output_hash="def456",
            side_effect_class=SideEffect.READ_ONLY,
            framework_mapping_confidence=0.95,
            tokens=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost=0.002,
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
        )
        assert step.step_id == "step-001"
        assert len(step.raw_span_ids) == 2
        assert step.side_effect_class.is_safe_for_auto_replay()
        assert step.duration_ms == 3000

    def test_is_protected_validation(self):
        step = CanonicalStep(
            step_id="step-002",
            raw_span_ids=["s3"],
            node_id=None,
            tool_name=None,
            step_type=StepType.VALIDATION,
            attempt_index=0,
            loop_iteration=None,
            input_hash="x",
            output_hash="y",
            side_effect_class=SideEffect.PURE,
            framework_mapping_confidence=0.5,
            tokens=None,
            cost=None,
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        )
        assert step.is_protected

    def test_is_not_protected_action(self):
        step = CanonicalStep(
            step_id="step-003",
            raw_span_ids=["s4"],
            node_id="tool_node",
            tool_name="calculator",
            step_type=StepType.ACTION,
            attempt_index=0,
            loop_iteration=None,
            input_hash="a",
            output_hash="b",
            side_effect_class=SideEffect.PURE,
            framework_mapping_confidence=0.9,
            tokens=None,
            cost=None,
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        )
        assert not step.is_protected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py::TestCanonicalStep -v`
Expected: FAIL

- [ ] **Step 3: Implement CanonicalStep**

`traceshap/models/step.py`:
```python
from dataclasses import dataclass
from datetime import datetime

from traceshap.models.enums import StepType, SideEffect
from traceshap.models.span import TokenUsage

PROTECTED_STEP_TYPES = frozenset({StepType.VALIDATION})


@dataclass
class CanonicalStep:
    step_id: str
    raw_span_ids: list[str]
    node_id: str | None
    tool_name: str | None
    step_type: StepType
    attempt_index: int
    loop_iteration: int | None
    input_hash: str
    output_hash: str
    side_effect_class: SideEffect
    framework_mapping_confidence: float
    tokens: TokenUsage | None
    cost: float | None
    start_time: datetime
    end_time: datetime

    @property
    def duration_ms(self) -> int:
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)

    @property
    def is_protected(self) -> bool:
        return self.step_type in PROTECTED_STEP_TYPES
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_models.py::TestCanonicalStep -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/models/step.py tests/test_models.py
git commit -m "feat: CanonicalStep model with side-effect classification and protected types"
```

---

### Task 5: Trajectory, Outcome, and Attribution Models

**Files:**
- Create: `traceshap/models/trajectory.py`
- Create: `traceshap/models/outcome.py`
- Create: `traceshap/models/pruning.py`
- Modify: `traceshap/models/__init__.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_models.py`:
```python
from traceshap.models.trajectory import SpanNode, TrajectoryMeta, Trajectory
from traceshap.models.outcome import Outcome, ConfidenceInterval, StepAttribution
from traceshap.models.pruning import Savings, ValidationPlan, PruneCandidate, PruningReport


class TestOutcome:
    def test_create_full(self):
        outcome = Outcome(
            success=True,
            quality_score=0.85,
            token_cost=1500,
            latency_ms=3000,
            custom_metrics={"relevance": 0.9},
            evaluator_id="gpt-4o-judge",
            evaluator_version="v1",
            score_confidence=0.92,
            label_delay_ms=500,
        )
        assert outcome.success is True
        assert outcome.quality_score == 0.85

    def test_create_minimal(self):
        outcome = Outcome(
            success=False,
            quality_score=None,
            token_cost=500,
            latency_ms=1000,
            custom_metrics={},
        )
        assert outcome.evaluator_id is None


class TestConfidenceInterval:
    def test_contains(self):
        ci = ConfidenceInterval(lower=0.1, point=0.3, upper=0.5)
        assert ci.contains(0.3)
        assert ci.contains(0.1)
        assert not ci.contains(0.05)

    def test_width(self):
        ci = ConfidenceInterval(lower=-0.1, point=0.0, upper=0.1)
        assert ci.width == pytest.approx(0.2)


class TestStepAttribution:
    def test_create(self):
        attr = StepAttribution(
            step_id="step-001",
            step_name="search_web",
            node_id="search_node",
            quality_delta=-0.15,
            cost_delta=0.002,
            latency_delta=500,
            risk_delta=0.0,
            layer_scores={0: -0.1, 1: -0.2, 2: -0.15},
            confidence=ConfidenceInterval(lower=-0.2, point=-0.15, upper=-0.1),
            verdict=Verdict.KEEP,
            causal_hypothesis=None,
            evidence=["Layer 0: no rule violation", "Layer 2: removal hurts quality by 0.15"],
            calibration=None,
        )
        assert attr.verdict == Verdict.KEEP


class TestSpanNode:
    def test_tree_structure(self):
        root = SpanNode(span_id="s0", children=[
            SpanNode(span_id="s1", children=[]),
            SpanNode(span_id="s2", children=[
                SpanNode(span_id="s3", children=[]),
            ]),
        ])
        assert len(root.children) == 2
        assert root.children[1].children[0].span_id == "s3"


class TestTrajectory:
    def test_create_without_outcome(self):
        t = Trajectory(
            trace_id="t1",
            spans=[],
            steps=[],
            span_tree=SpanNode(span_id="root", children=[]),
            outcome=None,
            metadata=TrajectoryMeta(framework="langgraph", agent_name="my-agent"),
        )
        assert t.outcome is None
        assert t.metadata.framework == "langgraph"


class TestSavings:
    def test_create(self):
        s = Savings(
            token_reduction=500,
            cost_reduction=0.005,
            latency_reduction_ms=800,
            quality_impact_range=(-0.02, 0.01),
        )
        assert s.token_reduction == 500


class TestPruneCandidate:
    def test_create(self):
        candidate = PruneCandidate(
            target_type="node",
            target_id="retry_search",
            evidence=[],
            estimated_savings=Savings(
                token_reduction=200,
                cost_reduction=0.001,
                latency_reduction_ms=300,
                quality_impact_range=(-0.01, 0.005),
            ),
            required_validation=ValidationPlan(
                replay_required=True,
                replay_mode=ReplayCapability.RECORDED_IO_REPLAY,
                min_replay_count=5,
                ab_test_recommended=False,
                human_review_required=False,
            ),
            decision_status=DecisionStatus.CANDIDATE,
        )
        assert candidate.decision_status == DecisionStatus.CANDIDATE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v -k "Outcome or Confidence or StepAttr or SpanNode or Trajectory or Savings or PruneCandidate"`
Expected: FAIL

- [ ] **Step 3: Implement trajectory models**

`traceshap/models/trajectory.py`:
```python
from dataclasses import dataclass, field

from traceshap.models.span import TraceSHAPSpan
from traceshap.models.step import CanonicalStep
from traceshap.models.outcome import Outcome


@dataclass
class SpanNode:
    span_id: str
    children: list["SpanNode"] = field(default_factory=list)


@dataclass
class TrajectoryMeta:
    framework: str
    agent_name: str
    agent_version: str | None = None
    task_type: str | None = None


@dataclass
class Trajectory:
    trace_id: str
    spans: list[TraceSHAPSpan]
    steps: list[CanonicalStep]
    span_tree: SpanNode
    outcome: Outcome | None
    metadata: TrajectoryMeta
```

- [ ] **Step 4: Implement outcome models**

`traceshap/models/outcome.py`:
```python
from dataclasses import dataclass

from traceshap.models.enums import Verdict


@dataclass
class Outcome:
    success: bool | None
    quality_score: float | None
    token_cost: int
    latency_ms: int
    custom_metrics: dict
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    score_confidence: float | None = None
    label_delay_ms: int | None = None


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    point: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass
class CalibrationMetrics:
    auc: float | None = None
    rmse: float | None = None
    coverage: float | None = None
    ood_score: float | None = None


@dataclass
class StepAttribution:
    step_id: str
    step_name: str
    node_id: str | None
    quality_delta: float | None
    cost_delta: float | None
    latency_delta: float | None
    risk_delta: float | None
    layer_scores: dict[int, float]
    confidence: ConfidenceInterval | None
    verdict: Verdict
    causal_hypothesis: object | None
    evidence: list[str]
    calibration: CalibrationMetrics | None
```

- [ ] **Step 5: Implement pruning models**

`traceshap/models/pruning.py`:
```python
from dataclasses import dataclass
from datetime import datetime

from traceshap.models.enums import DecisionStatus, RiskLevel, ReplayCapability
from traceshap.models.outcome import StepAttribution


@dataclass(frozen=True)
class Savings:
    token_reduction: int
    cost_reduction: float
    latency_reduction_ms: int
    quality_impact_range: tuple[float, float]


@dataclass
class ValidationPlan:
    replay_required: bool
    replay_mode: ReplayCapability
    min_replay_count: int
    ab_test_recommended: bool
    human_review_required: bool


@dataclass
class PruneCandidate:
    target_type: str
    target_id: str
    evidence: list[StepAttribution]
    estimated_savings: Savings
    required_validation: ValidationPlan
    decision_status: DecisionStatus


@dataclass
class PruningReport:
    trace_id: str | None
    timestamp: datetime
    total_steps: int
    candidates: list[PruneCandidate]
    estimated_savings: Savings
    risk_assessment: RiskLevel
```

- [ ] **Step 6: Update models __init__ for re-exports**

`traceshap/models/__init__.py`:
```python
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
from traceshap.models.step import CanonicalStep
from traceshap.models.trajectory import SpanNode, TrajectoryMeta, Trajectory
from traceshap.models.outcome import Outcome, ConfidenceInterval, StepAttribution, CalibrationMetrics
from traceshap.models.pruning import Savings, ValidationPlan, PruneCandidate, PruningReport

__all__ = [
    "SpanKind", "StepType", "SideEffect", "Verdict", "DecisionStatus",
    "RiskLevel", "ReplayCapability",
    "TokenUsage", "TraceSHAPSpan",
    "CanonicalStep",
    "SpanNode", "TrajectoryMeta", "Trajectory",
    "Outcome", "ConfidenceInterval", "StepAttribution", "CalibrationMetrics",
    "Savings", "ValidationPlan", "PruneCandidate", "PruningReport",
]
```

- [ ] **Step 7: Run all model tests**

Run: `pytest tests/test_models.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add traceshap/models/ tests/test_models.py
git commit -m "feat: Trajectory, Outcome, StepAttribution, and Pruning models"
```

---

### Task 6: Configuration Loading

**Files:**
- Create: `traceshap/config.py`
- Create: `traceshap.yaml.example`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write tests**

`tests/test_config.py`:
```python
import tempfile
from pathlib import Path
from traceshap.config import TraceSHAPConfig, load_config


MINIMAL_YAML = """\
source:
  type: langfuse
  langfuse_host: https://cloud.langfuse.com
  langfuse_public_key: pk-test
  langfuse_secret_key: sk-test

storage:
  backend: sqlite
  sqlite_path: ./test.db
"""

FULL_YAML = """\
source:
  type: langfuse
  langfuse_host: https://cloud.langfuse.com
  langfuse_public_key: pk-test
  langfuse_secret_key: sk-test
  poll_interval_seconds: 5

attribution:
  layers: [0, 1, 2]
  layer_1:
    min_support: 30
  layer_2:
    num_samples: 100

pruning:
  prune_epsilon: 0.03
  keep_threshold: 0.15
  min_trajectories: 20

storage:
  backend: sqlite
  sqlite_path: ./test.db
  retention_days: 14

server:
  host: 127.0.0.1
  port: 9090
"""


class TestLoadConfig:
    def test_minimal_config(self, tmp_path: Path):
        cfg_path = tmp_path / "traceshap.yaml"
        cfg_path.write_text(MINIMAL_YAML)
        config = load_config(cfg_path)
        assert config.source.type == "langfuse"
        assert config.source.langfuse_host == "https://cloud.langfuse.com"
        assert config.storage.backend == "sqlite"
        # defaults
        assert config.source.poll_interval_seconds == 10
        assert config.attribution.layers == [0, 1, 2]
        assert config.server.port == 8080

    def test_full_config(self, tmp_path: Path):
        cfg_path = tmp_path / "traceshap.yaml"
        cfg_path.write_text(FULL_YAML)
        config = load_config(cfg_path)
        assert config.source.poll_interval_seconds == 5
        assert config.attribution.layers == [0, 1, 2]
        assert config.pruning.prune_epsilon == 0.03
        assert config.storage.retention_days == 14
        assert config.server.port == 9090

    def test_missing_file_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/traceshap.yaml"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL

- [ ] **Step 3: Implement config**

`traceshap/config.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SourceConfig:
    type: str = "langfuse"
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    poll_interval_seconds: int = 10
    otlp_endpoint: str = ""
    source_hint: str = ""


@dataclass
class Layer1Config:
    stratify_by: list[str] = field(default_factory=lambda: ["agent_version", "task_type", "model_version"])
    min_support: int = 50
    confidence_level: float = 0.95


@dataclass
class Layer2Config:
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache: bool = True
    num_samples: int = 200
    calibration_holdout: float = 0.1


@dataclass
class Layer3Config:
    enabled: bool = False
    replay_mode: str = "recorded_io_replay"
    replay_budget_per_trace: int = 40
    replay_concurrency: int = 4


@dataclass
class Layer4Config:
    enabled: bool = False
    requires_layer_3: bool = True
    claim_type: str = "associational"


@dataclass
class AttributionConfig:
    layers: list[int] = field(default_factory=lambda: [0, 1, 2])
    layer_1: Layer1Config = field(default_factory=Layer1Config)
    layer_2: Layer2Config = field(default_factory=Layer2Config)
    layer_3: Layer3Config = field(default_factory=Layer3Config)
    layer_4: Layer4Config = field(default_factory=Layer4Config)


@dataclass
class PruningConfig:
    prune_epsilon: float = 0.05
    keep_threshold: float = 0.10
    min_trajectories: int = 10
    protect_first_last: bool = True
    validation_gate: bool = True


@dataclass
class OutcomeConfig:
    source: str = "langfuse_score"
    score_name: str = "task_success"
    normalization_baseline: str = "historical_p50"


@dataclass
class StorageConfig:
    backend: str = "sqlite"
    sqlite_path: str = "./traceshap.db"
    retention_days: int = 30


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4


@dataclass
class PerformanceConfig:
    embedding_cache_size: int = 10000
    max_concurrent_analyses: int = 8
    layer_2_timeout_seconds: int = 10


@dataclass
class TraceSHAPConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


def _merge_dataclass(dc_instance, raw_dict: dict):
    if raw_dict is None:
        return dc_instance
    for key, value in raw_dict.items():
        if hasattr(dc_instance, key):
            attr = getattr(dc_instance, key)
            if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                _merge_dataclass(attr, value)
            else:
                setattr(dc_instance, key, value)
    return dc_instance


def load_config(path: Path) -> TraceSHAPConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    config = TraceSHAPConfig()
    for section_name in ("source", "attribution", "pruning", "outcome", "storage", "server", "performance"):
        if section_name in raw:
            section = getattr(config, section_name)
            _merge_dataclass(section, raw[section_name])
    return config
```

- [ ] **Step 4: Create example config**

`traceshap.yaml.example`:
```yaml
# TraceSHAP configuration
# Copy to traceshap.yaml and fill in your values.

source:
  type: langfuse                    # langfuse | otlp | hermes_atropos | direct
  langfuse_host: https://cloud.langfuse.com
  langfuse_public_key: pk-...
  langfuse_secret_key: sk-...
  poll_interval_seconds: 10

attribution:
  layers: [0, 1, 2]                # production default; add 3, 4 for experimental
  layer_1:
    min_support: 50
  layer_2:
    embedding_model: all-MiniLM-L6-v2
    num_samples: 200

pruning:
  prune_epsilon: 0.05
  keep_threshold: 0.10
  min_trajectories: 10

outcome:
  source: langfuse_score
  score_name: task_success
  normalization_baseline: historical_p50

storage:
  backend: sqlite
  sqlite_path: ./traceshap.db
  retention_days: 30

server:
  host: 0.0.0.0
  port: 8080
  workers: 4
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add traceshap/config.py traceshap.yaml.example tests/test_config.py
git commit -m "feat: YAML config loading with defaults for all sections"
```

---

### Task 7: Storage Backend ABC and ORM Models

**Files:**
- Create: `traceshap/storage/backend.py`
- Create: `traceshap/storage/orm.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write tests for ORM table creation**

`tests/test_storage.py`:
```python
import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from traceshap.storage.orm import Base


@pytest.fixture
async def async_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(async_engine):
    factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


class TestORMTables:
    async def test_all_tables_created(self, async_engine):
        async with async_engine.connect() as conn:
            table_names = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
        expected = {
            "trajectories", "spans", "canonical_steps",
            "attribution_runs", "step_attributions",
            "prune_candidates", "cohort_stats", "markov_models",
        }
        assert expected.issubset(set(table_names))

    async def test_insert_trajectory(self, session):
        from traceshap.storage.orm import TrajectoryRow
        row = TrajectoryRow(
            trace_id="t1",
            framework="langgraph",
            agent_name="my-agent",
            agent_version="v1",
            task_type="qa",
            outcome_success=True,
            outcome_quality=0.85,
            outcome_token_cost=1500,
            outcome_latency_ms=3000,
        )
        session.add(row)
        await session.commit()
        result = await session.get(TrajectoryRow, "t1")
        assert result is not None
        assert result.agent_name == "my-agent"

    async def test_insert_span(self, session):
        from traceshap.storage.orm import TrajectoryRow, SpanRow
        session.add(TrajectoryRow(
            trace_id="t2", framework="langgraph", agent_name="a",
        ))
        await session.flush()
        span = SpanRow(
            span_id="s1",
            trace_id="t2",
            parent_span_id=None,
            span_kind="llm",
            name="gpt-4o",
            input_data="{}",
            output_data="{}",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-01T00:00:01Z",
            raw_attributes="{}",
            semconv_version="otel-genai-v0.1",
        )
        session.add(span)
        await session.commit()
        result = await session.get(SpanRow, "s1")
        assert result.trace_id == "t2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ORM models**

`traceshap/storage/orm.py`:
```python
from datetime import datetime, timezone

from sqlalchemy import String, Integer, Float, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TrajectoryRow(Base):
    __tablename__ = "trajectories"

    trace_id: Mapped[str] = mapped_column(String, primary_key=True)
    framework: Mapped[str] = mapped_column(String)
    agent_name: Mapped[str] = mapped_column(String)
    agent_version: Mapped[str | None] = mapped_column(String, nullable=True)
    task_type: Mapped[str | None] = mapped_column(String, nullable=True)
    outcome_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    outcome_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome_token_cost: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outcome_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class SpanRow(Base):
    __tablename__ = "spans"

    span_id: Mapped[str] = mapped_column(String, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String, ForeignKey("trajectories.trace_id"))
    parent_span_id: Mapped[str | None] = mapped_column(String, nullable=True)
    span_kind: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    input_data: Mapped[str] = mapped_column(Text)
    output_data: Mapped[str] = mapped_column(Text)
    start_time: Mapped[str] = mapped_column(String)
    end_time: Mapped[str] = mapped_column(String)
    tokens_input: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_output: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_attributes: Mapped[str] = mapped_column(Text, default="{}")
    semconv_version: Mapped[str] = mapped_column(String, default="unknown")


class CanonicalStepRow(Base):
    __tablename__ = "canonical_steps"

    step_id: Mapped[str] = mapped_column(String, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String, ForeignKey("trajectories.trace_id"))
    raw_span_ids_json: Mapped[str] = mapped_column(Text)
    node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    step_type: Mapped[str] = mapped_column(String)
    side_effect: Mapped[str] = mapped_column(String)
    attempt_index: Mapped[int] = mapped_column(Integer, default=0)
    loop_iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_hash: Mapped[str] = mapped_column(String)
    output_hash: Mapped[str] = mapped_column(String)
    mapping_confidence: Mapped[float] = mapped_column(Float)
    tokens_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_time: Mapped[str] = mapped_column(String)
    end_time: Mapped[str] = mapped_column(String)


class AttributionRunRow(Base):
    __tablename__ = "attribution_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String, ForeignKey("trajectories.trace_id"))
    config_hash: Mapped[str] = mapped_column(String)
    code_version: Mapped[str] = mapped_column(String)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    layers_used: Mapped[str] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )


class StepAttributionRow(Base):
    __tablename__ = "step_attributions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("attribution_runs.run_id"))
    step_id: Mapped[str] = mapped_column(String, ForeignKey("canonical_steps.step_id"))
    layer: Mapped[int] = mapped_column(Integer)
    quality_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_lo: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_hi: Mapped[float | None] = mapped_column(Float, nullable=True)
    verdict: Mapped[str] = mapped_column(String)
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    calibration_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class PruneCandidateRow(Base):
    __tablename__ = "prune_candidates"

    candidate_id: Mapped[str] = mapped_column(String, primary_key=True)
    target_type: Mapped[str] = mapped_column(String)
    target_id: Mapped[str] = mapped_column(String)
    savings_json: Mapped[str] = mapped_column(Text)
    validation_plan_json: Mapped[str] = mapped_column(Text)
    decision_status: Mapped[str] = mapped_column(String, default="candidate")
    validated_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: Mapped[str | None] = mapped_column(String, nullable=True)


class CohortStatRow(Base):
    __tablename__ = "cohort_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    step_type: Mapped[str] = mapped_column(String)
    agent_name: Mapped[str] = mapped_column(String)
    agent_version: Mapped[str | None] = mapped_column(String, nullable=True)
    task_type: Mapped[str | None] = mapped_column(String, nullable=True)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    lift_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    lift_ci_lo: Mapped[float | None] = mapped_column(Float, nullable=True)
    lift_ci_hi: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[str | None] = mapped_column(String, nullable=True)


class MarkovModelRow(Base):
    __tablename__ = "markov_models"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String)
    framework: Mapped[str] = mapped_column(String)
    model_blob: Mapped[bytes] = mapped_column()
    training_count: Mapped[int] = mapped_column(Integer, default=0)
    held_out_metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )
```

- [ ] **Step 4: Implement StorageBackend ABC**

`traceshap/storage/backend.py`:
```python
from abc import ABC, abstractmethod

from traceshap.models.trajectory import Trajectory
from traceshap.models.outcome import StepAttribution
from traceshap.models.enums import DecisionStatus


class QueryFilter:
    def __init__(
        self,
        agent_name: str | None = None,
        agent_version: str | None = None,
        task_type: str | None = None,
        framework: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.task_type = task_type
        self.framework = framework
        self.limit = limit
        self.offset = offset


class CohortFilter:
    def __init__(
        self,
        agent_version: str | None = None,
        task_type: str | None = None,
    ):
        self.agent_version = agent_version
        self.task_type = task_type


class StorageBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        ...

    @abstractmethod
    async def save_trajectory(self, trajectory: Trajectory) -> None:
        ...

    @abstractmethod
    async def get_trajectory(self, trace_id: str) -> Trajectory | None:
        ...

    @abstractmethod
    async def query_trajectories(self, filters: QueryFilter) -> list[Trajectory]:
        ...

    @abstractmethod
    async def save_attribution_run(
        self, run_id: str, trace_id: str, config_hash: str,
        code_version: str, layers: list[int],
        attributions: list[StepAttribution],
    ) -> None:
        ...

    @abstractmethod
    async def update_candidate_status(
        self, candidate_id: str, status: DecisionStatus,
    ) -> None:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_storage.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add traceshap/storage/ tests/test_storage.py
git commit -m "feat: StorageBackend ABC and SQLAlchemy ORM models for all tables"
```

---

### Task 8: SQLite Backend Implementation

**Files:**
- Create: `traceshap/storage/sqlite.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Write tests for SQLiteBackend**

Append to `tests/test_storage.py`:
```python
import json
from datetime import datetime, timezone
from traceshap.storage.sqlite import SQLiteBackend
from traceshap.models import (
    TraceSHAPSpan, SpanKind, TokenUsage, CanonicalStep, StepType,
    SideEffect, SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.storage.backend import QueryFilter


@pytest.fixture
async def backend(tmp_path):
    db_path = str(tmp_path / "test.db")
    backend = SQLiteBackend(db_path)
    await backend.initialize()
    yield backend
    await backend.close()


def _make_trajectory(trace_id: str = "t1") -> Trajectory:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
    span = TraceSHAPSpan(
        trace_id=trace_id, span_id=f"{trace_id}-s1", parent_span_id=None,
        span_kind=SpanKind.LLM, name="gpt-4o", input={"prompt": "hi"},
        output={"text": "hello"}, start_time=now, end_time=later,
        tokens=TokenUsage(10, 20, 30), cost=0.001,
        metadata={}, raw_attributes={}, semconv_version="otel-genai-v0.1",
    )
    step = CanonicalStep(
        step_id=f"{trace_id}-step1", raw_span_ids=[f"{trace_id}-s1"],
        node_id="llm_node", tool_name=None, step_type=StepType.DECISION,
        attempt_index=0, loop_iteration=None, input_hash="abc", output_hash="def",
        side_effect_class=SideEffect.PURE, framework_mapping_confidence=0.95,
        tokens=TokenUsage(10, 20, 30), cost=0.001, start_time=now, end_time=later,
    )
    return Trajectory(
        trace_id=trace_id,
        spans=[span],
        steps=[step],
        span_tree=SpanNode(span_id=f"{trace_id}-s1"),
        outcome=Outcome(
            success=True, quality_score=0.9, token_cost=30, latency_ms=2000,
            custom_metrics={}, evaluator_id="judge-v1",
        ),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test-agent", agent_version="v1"),
    )


class TestSQLiteBackend:
    async def test_save_and_get_trajectory(self, backend):
        t = _make_trajectory("t1")
        await backend.save_trajectory(t)
        result = await backend.get_trajectory("t1")
        assert result is not None
        assert result.trace_id == "t1"
        assert len(result.spans) == 1
        assert len(result.steps) == 1
        assert result.outcome.success is True
        assert result.metadata.agent_name == "test-agent"

    async def test_get_nonexistent_returns_none(self, backend):
        result = await backend.get_trajectory("nonexistent")
        assert result is None

    async def test_query_by_agent_name(self, backend):
        await backend.save_trajectory(_make_trajectory("t1"))
        await backend.save_trajectory(_make_trajectory("t2"))
        results = await backend.query_trajectories(
            QueryFilter(agent_name="test-agent")
        )
        assert len(results) == 2

    async def test_query_with_limit(self, backend):
        await backend.save_trajectory(_make_trajectory("t1"))
        await backend.save_trajectory(_make_trajectory("t2"))
        results = await backend.query_trajectories(
            QueryFilter(agent_name="test-agent", limit=1)
        )
        assert len(results) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage.py::TestSQLiteBackend -v`
Expected: FAIL

- [ ] **Step 3: Implement SQLiteBackend**

`traceshap/storage/sqlite.py`:
```python
import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from traceshap.models import (
    TraceSHAPSpan, SpanKind, TokenUsage, CanonicalStep, StepType,
    SideEffect, SpanNode, TrajectoryMeta, Trajectory, Outcome,
    StepAttribution, DecisionStatus,
)
from traceshap.storage.backend import StorageBackend, QueryFilter, CohortFilter
from traceshap.storage.orm import (
    Base, TrajectoryRow, SpanRow, CanonicalStepRow,
    AttributionRunRow, StepAttributionRow, PruneCandidateRow,
)


class SQLiteBackend(StorageBackend):
    def __init__(self, db_path: str):
        url = f"sqlite+aiosqlite:///{db_path}" if db_path != ":memory:" else "sqlite+aiosqlite:///:memory:"
        self._engine = create_async_engine(url)
        self._session_factory = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_trajectory(self, trajectory: Trajectory) -> None:
        async with self._session_factory() as session:
            traj_row = TrajectoryRow(
                trace_id=trajectory.trace_id,
                framework=trajectory.metadata.framework,
                agent_name=trajectory.metadata.agent_name,
                agent_version=trajectory.metadata.agent_version,
                task_type=trajectory.metadata.task_type,
                outcome_success=trajectory.outcome.success if trajectory.outcome else None,
                outcome_quality=trajectory.outcome.quality_score if trajectory.outcome else None,
                outcome_token_cost=trajectory.outcome.token_cost if trajectory.outcome else None,
                outcome_latency_ms=trajectory.outcome.latency_ms if trajectory.outcome else None,
                metadata_json=json.dumps({}),
            )
            session.add(traj_row)
            await session.flush()

            for span in trajectory.spans:
                span_row = SpanRow(
                    span_id=span.span_id,
                    trace_id=span.trace_id,
                    parent_span_id=span.parent_span_id,
                    span_kind=span.span_kind.value,
                    name=span.name,
                    input_data=json.dumps(span.input),
                    output_data=json.dumps(span.output),
                    start_time=span.start_time.isoformat(),
                    end_time=span.end_time.isoformat(),
                    tokens_input=span.tokens.input_tokens if span.tokens else None,
                    tokens_output=span.tokens.output_tokens if span.tokens else None,
                    tokens_total=span.tokens.total_tokens if span.tokens else None,
                    cost=span.cost,
                    raw_attributes=json.dumps(span.raw_attributes),
                    semconv_version=span.semconv_version,
                )
                session.add(span_row)

            for step in trajectory.steps:
                step_row = CanonicalStepRow(
                    step_id=step.step_id,
                    trace_id=trajectory.trace_id,
                    raw_span_ids_json=json.dumps(step.raw_span_ids),
                    node_id=step.node_id,
                    tool_name=step.tool_name,
                    step_type=step.step_type.value,
                    side_effect=step.side_effect_class.value,
                    attempt_index=step.attempt_index,
                    loop_iteration=step.loop_iteration,
                    input_hash=step.input_hash,
                    output_hash=step.output_hash,
                    mapping_confidence=step.framework_mapping_confidence,
                    tokens_total=step.tokens.total_tokens if step.tokens else None,
                    cost=step.cost,
                    start_time=step.start_time.isoformat(),
                    end_time=step.end_time.isoformat(),
                )
                session.add(step_row)

            await session.commit()

    async def get_trajectory(self, trace_id: str) -> Trajectory | None:
        async with self._session_factory() as session:
            traj_row = await session.get(TrajectoryRow, trace_id)
            if traj_row is None:
                return None

            span_result = await session.execute(
                select(SpanRow).where(SpanRow.trace_id == trace_id)
            )
            span_rows = span_result.scalars().all()

            step_result = await session.execute(
                select(CanonicalStepRow).where(CanonicalStepRow.trace_id == trace_id)
            )
            step_rows = step_result.scalars().all()

        spans = [self._row_to_span(r) for r in span_rows]
        steps = [self._row_to_step(r) for r in step_rows]
        span_tree = self._build_span_tree(spans)
        outcome = self._row_to_outcome(traj_row)
        metadata = TrajectoryMeta(
            framework=traj_row.framework,
            agent_name=traj_row.agent_name,
            agent_version=traj_row.agent_version,
            task_type=traj_row.task_type,
        )

        return Trajectory(
            trace_id=trace_id,
            spans=sorted(spans, key=lambda s: s.start_time),
            steps=sorted(steps, key=lambda s: s.start_time),
            span_tree=span_tree,
            outcome=outcome,
            metadata=metadata,
        )

    async def query_trajectories(self, filters: QueryFilter) -> list[Trajectory]:
        async with self._session_factory() as session:
            query = select(TrajectoryRow)
            if filters.agent_name:
                query = query.where(TrajectoryRow.agent_name == filters.agent_name)
            if filters.framework:
                query = query.where(TrajectoryRow.framework == filters.framework)
            if filters.agent_version:
                query = query.where(TrajectoryRow.agent_version == filters.agent_version)
            if filters.task_type:
                query = query.where(TrajectoryRow.task_type == filters.task_type)
            query = query.limit(filters.limit).offset(filters.offset)
            result = await session.execute(query)
            rows = result.scalars().all()

        trajectories = []
        for row in rows:
            t = await self.get_trajectory(row.trace_id)
            if t:
                trajectories.append(t)
        return trajectories

    async def save_attribution_run(
        self, run_id: str, trace_id: str, config_hash: str,
        code_version: str, layers: list[int],
        attributions: list[StepAttribution],
    ) -> None:
        async with self._session_factory() as session:
            run_row = AttributionRunRow(
                run_id=run_id,
                trace_id=trace_id,
                config_hash=config_hash,
                code_version=code_version,
                layers_used=json.dumps(layers),
            )
            session.add(run_row)
            await session.flush()

            for attr in attributions:
                attr_row = StepAttributionRow(
                    run_id=run_id,
                    step_id=attr.step_id,
                    layer=max(attr.layer_scores.keys()) if attr.layer_scores else 0,
                    quality_delta=attr.quality_delta,
                    cost_delta=attr.cost_delta,
                    latency_delta=attr.latency_delta,
                    confidence_lo=attr.confidence.lower if attr.confidence else None,
                    confidence_hi=attr.confidence.upper if attr.confidence else None,
                    verdict=attr.verdict.value,
                    evidence_json=json.dumps(attr.evidence),
                )
                session.add(attr_row)

            await session.commit()

    async def update_candidate_status(
        self, candidate_id: str, status: DecisionStatus,
    ) -> None:
        async with self._session_factory() as session:
            row = await session.get(PruneCandidateRow, candidate_id)
            if row:
                row.decision_status = status.value
                row.updated_at = datetime.now(timezone.utc).isoformat()
                await session.commit()

    async def close(self) -> None:
        await self._engine.dispose()

    @staticmethod
    def _row_to_span(row: SpanRow) -> TraceSHAPSpan:
        tokens = None
        if row.tokens_total is not None:
            tokens = TokenUsage(
                input_tokens=row.tokens_input or 0,
                output_tokens=row.tokens_output or 0,
                total_tokens=row.tokens_total,
            )
        return TraceSHAPSpan(
            trace_id=row.trace_id,
            span_id=row.span_id,
            parent_span_id=row.parent_span_id,
            span_kind=SpanKind(row.span_kind),
            name=row.name,
            input=json.loads(row.input_data),
            output=json.loads(row.output_data),
            start_time=datetime.fromisoformat(row.start_time),
            end_time=datetime.fromisoformat(row.end_time),
            tokens=tokens,
            cost=row.cost,
            metadata={},
            raw_attributes=json.loads(row.raw_attributes),
            semconv_version=row.semconv_version,
        )

    @staticmethod
    def _row_to_step(row: CanonicalStepRow) -> CanonicalStep:
        tokens_total = row.tokens_total
        tokens = TokenUsage(0, 0, tokens_total) if tokens_total is not None else None
        return CanonicalStep(
            step_id=row.step_id,
            raw_span_ids=json.loads(row.raw_span_ids_json),
            node_id=row.node_id,
            tool_name=row.tool_name,
            step_type=StepType(row.step_type),
            attempt_index=row.attempt_index,
            loop_iteration=row.loop_iteration,
            input_hash=row.input_hash,
            output_hash=row.output_hash,
            side_effect_class=SideEffect(row.side_effect),
            framework_mapping_confidence=row.mapping_confidence,
            tokens=tokens,
            cost=row.cost,
            start_time=datetime.fromisoformat(row.start_time),
            end_time=datetime.fromisoformat(row.end_time),
        )

    @staticmethod
    def _row_to_outcome(row: TrajectoryRow) -> Outcome | None:
        if row.outcome_success is None and row.outcome_quality is None:
            return None
        return Outcome(
            success=row.outcome_success,
            quality_score=row.outcome_quality,
            token_cost=row.outcome_token_cost or 0,
            latency_ms=row.outcome_latency_ms or 0,
            custom_metrics={},
        )

    @staticmethod
    def _build_span_tree(spans: list[TraceSHAPSpan]) -> SpanNode:
        nodes: dict[str, SpanNode] = {}
        for span in spans:
            nodes[span.span_id] = SpanNode(span_id=span.span_id)
        root = None
        for span in spans:
            node = nodes[span.span_id]
            if span.parent_span_id and span.parent_span_id in nodes:
                nodes[span.parent_span_id].children.append(node)
            else:
                root = node
        return root or SpanNode(span_id="empty")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_storage.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/storage/sqlite.py tests/test_storage.py
git commit -m "feat: SQLiteBackend with full trajectory save/get/query"
```

---

### Task 9: Span Source ABC and Langfuse Source

**Files:**
- Create: `traceshap/ingestion/sources/base.py`
- Create: `traceshap/ingestion/sources/langfuse.py`
- Create: `tests/test_assembler.py`

- [ ] **Step 1: Write tests with a mock source**

`tests/test_assembler.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.models import TraceSHAPSpan, SpanKind, TokenUsage
from traceshap.ingestion.sources.base import SpanSource


class MockSource(SpanSource):
    def __init__(self, spans: list[TraceSHAPSpan]):
        self._spans = spans
        self._index = 0

    async def connect(self) -> None:
        pass

    async def poll(self) -> list[TraceSHAPSpan]:
        if self._index < len(self._spans):
            batch = [self._spans[self._index]]
            self._index += 1
            return batch
        return []

    async def close(self) -> None:
        pass


def make_span(trace_id: str, span_id: str, parent: str | None = None,
              kind: SpanKind = SpanKind.LLM, name: str = "test",
              offset_sec: int = 0, duration_sec: int = 1) -> TraceSHAPSpan:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + duration_sec, tzinfo=timezone.utc)
    return TraceSHAPSpan(
        trace_id=trace_id, span_id=span_id, parent_span_id=parent,
        span_kind=kind, name=name, input={}, output={},
        start_time=start, end_time=end, tokens=None, cost=None,
        metadata={}, raw_attributes={}, semconv_version="test",
    )


class TestSpanSource:
    async def test_mock_source_polls(self):
        spans = [make_span("t1", "s1"), make_span("t1", "s2")]
        source = MockSource(spans)
        await source.connect()
        batch1 = await source.poll()
        assert len(batch1) == 1
        batch2 = await source.poll()
        assert len(batch2) == 1
        batch3 = await source.poll()
        assert len(batch3) == 0
        await source.close()
```

- [ ] **Step 2: Implement SpanSource ABC**

`traceshap/ingestion/sources/base.py`:
```python
from abc import ABC, abstractmethod

from traceshap.models.span import TraceSHAPSpan


class SpanSource(ABC):
    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def poll(self) -> list[TraceSHAPSpan]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
```

- [ ] **Step 3: Implement LangfuseSource**

`traceshap/ingestion/sources/langfuse.py`:
```python
from datetime import datetime, timezone

from langfuse import Langfuse

from traceshap.models.span import TraceSHAPSpan, TokenUsage
from traceshap.models.enums import SpanKind
from traceshap.ingestion.sources.base import SpanSource

LANGFUSE_TO_SPAN_KIND = {
    "GENERATION": SpanKind.LLM,
    "SPAN": SpanKind.AGENT,
    "EVENT": SpanKind.CUSTOM,
}


class LangfuseSource(SpanSource):
    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
        poll_batch_size: int = 50,
    ):
        self._public_key = public_key
        self._secret_key = secret_key
        self._host = host
        self._poll_batch_size = poll_batch_size
        self._client: Langfuse | None = None
        self._last_poll_time: datetime | None = None

    async def connect(self) -> None:
        self._client = Langfuse(
            public_key=self._public_key,
            secret_key=self._secret_key,
            host=self._host,
        )
        self._last_poll_time = datetime.now(timezone.utc)

    async def poll(self) -> list[TraceSHAPSpan]:
        if not self._client:
            raise RuntimeError("Source not connected. Call connect() first.")

        traces = self._client.fetch_traces(limit=self._poll_batch_size)
        spans: list[TraceSHAPSpan] = []

        for trace in traces.data:
            trace_detail = self._client.fetch_trace(trace.id)
            for obs in trace_detail.data.observations or []:
                span = self._observation_to_span(trace.id, obs)
                if span:
                    spans.append(span)

        self._last_poll_time = datetime.now(timezone.utc)
        return spans

    async def close(self) -> None:
        if self._client:
            self._client.flush()
            self._client = None

    @staticmethod
    def _observation_to_span(trace_id: str, obs) -> TraceSHAPSpan | None:
        span_kind = LANGFUSE_TO_SPAN_KIND.get(obs.type, SpanKind.CUSTOM)

        tokens = None
        if obs.usage:
            tokens = TokenUsage(
                input_tokens=getattr(obs.usage, "input", 0) or 0,
                output_tokens=getattr(obs.usage, "output", 0) or 0,
                total_tokens=getattr(obs.usage, "total", 0) or 0,
            )

        start_time = obs.start_time or datetime.now(timezone.utc)
        end_time = obs.end_time or obs.start_time or datetime.now(timezone.utc)

        return TraceSHAPSpan(
            trace_id=trace_id,
            span_id=obs.id,
            parent_span_id=obs.parent_observation_id,
            span_kind=span_kind,
            name=obs.name or "unknown",
            input=obs.input or {},
            output=obs.output or {},
            start_time=start_time,
            end_time=end_time,
            tokens=tokens,
            cost=getattr(obs, "calculated_total_cost", None),
            metadata=obs.metadata or {},
            raw_attributes={
                "langfuse_type": obs.type,
                "model": getattr(obs, "model", None),
                "level": getattr(obs, "level", None),
            },
            semconv_version="langfuse-v2",
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_assembler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/ingestion/sources/ tests/test_assembler.py
git commit -m "feat: SpanSource ABC and LangfuseSource implementation"
```

---

### Task 10: Span Buffer and Tree Assembler

**Files:**
- Create: `traceshap/ingestion/assembler.py`
- Modify: `tests/test_assembler.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_assembler.py`:
```python
from traceshap.ingestion.assembler import SpanBuffer, TreeAssembler
from traceshap.models import SpanNode


class TestSpanBuffer:
    def test_add_and_get_by_trace(self):
        buf = SpanBuffer()
        s1 = make_span("t1", "s1")
        s2 = make_span("t1", "s2", parent="s1")
        s3 = make_span("t2", "s3")
        buf.add(s1)
        buf.add(s2)
        buf.add(s3)
        assert len(buf.get_spans("t1")) == 2
        assert len(buf.get_spans("t2")) == 1
        assert len(buf.get_spans("t999")) == 0

    def test_flush_trace(self):
        buf = SpanBuffer()
        buf.add(make_span("t1", "s1"))
        buf.add(make_span("t1", "s2"))
        flushed = buf.flush("t1")
        assert len(flushed) == 2
        assert len(buf.get_spans("t1")) == 0

    def test_pending_trace_ids(self):
        buf = SpanBuffer()
        buf.add(make_span("t1", "s1"))
        buf.add(make_span("t2", "s2"))
        assert buf.pending_trace_ids() == {"t1", "t2"}


class TestTreeAssembler:
    def test_build_simple_tree(self):
        spans = [
            make_span("t1", "root", parent=None),
            make_span("t1", "child1", parent="root"),
            make_span("t1", "child2", parent="root"),
            make_span("t1", "grandchild", parent="child1"),
        ]
        tree = TreeAssembler.build(spans)
        assert tree.span_id == "root"
        assert len(tree.children) == 2
        child1 = next(c for c in tree.children if c.span_id == "child1")
        assert len(child1.children) == 1
        assert child1.children[0].span_id == "grandchild"

    def test_build_single_span(self):
        spans = [make_span("t1", "only")]
        tree = TreeAssembler.build(spans)
        assert tree.span_id == "only"
        assert tree.children == []

    def test_build_with_missing_parent(self):
        spans = [
            make_span("t1", "s1", parent="missing"),
            make_span("t1", "s2", parent="s1"),
        ]
        tree = TreeAssembler.build(spans)
        assert tree.span_id == "s1"
        assert len(tree.children) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assembler.py -v -k "Buffer or TreeAssembler"`
Expected: FAIL

- [ ] **Step 3: Implement SpanBuffer and TreeAssembler**

`traceshap/ingestion/assembler.py`:
```python
from traceshap.models.span import TraceSHAPSpan
from traceshap.models.trajectory import SpanNode


class SpanBuffer:
    def __init__(self):
        self._traces: dict[str, list[TraceSHAPSpan]] = {}

    def add(self, span: TraceSHAPSpan) -> None:
        self._traces.setdefault(span.trace_id, []).append(span)

    def get_spans(self, trace_id: str) -> list[TraceSHAPSpan]:
        return list(self._traces.get(trace_id, []))

    def flush(self, trace_id: str) -> list[TraceSHAPSpan]:
        return self._traces.pop(trace_id, [])

    def pending_trace_ids(self) -> set[str]:
        return set(self._traces.keys())


class TreeAssembler:
    @staticmethod
    def build(spans: list[TraceSHAPSpan]) -> SpanNode:
        nodes: dict[str, SpanNode] = {}
        for span in spans:
            nodes[span.span_id] = SpanNode(span_id=span.span_id)

        root: SpanNode | None = None
        for span in spans:
            node = nodes[span.span_id]
            if span.parent_span_id and span.parent_span_id in nodes:
                nodes[span.parent_span_id].children.append(node)
            else:
                if root is None:
                    root = node

        return root or SpanNode(span_id="empty")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_assembler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/ingestion/assembler.py tests/test_assembler.py
git commit -m "feat: SpanBuffer and TreeAssembler for trajectory reconstruction"
```

---

### Task 11: Step Normalizer

**Files:**
- Create: `traceshap/ingestion/normalizer.py`
- Create: `tests/test_normalizer.py`

- [ ] **Step 1: Write tests**

`tests/test_normalizer.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_normalizer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement StepNormalizer**

`traceshap/ingestion/normalizer.py`:
```python
import hashlib
import json
import uuid

from traceshap.models.span import TraceSHAPSpan
from traceshap.models.step import CanonicalStep
from traceshap.models.enums import SpanKind, StepType, SideEffect

SPAN_KIND_TO_STEP_TYPE: dict[SpanKind, StepType] = {
    SpanKind.LLM: StepType.DECISION,
    SpanKind.TOOL: StepType.ACTION,
    SpanKind.RETRIEVER: StepType.OBSERVATION,
    SpanKind.RERANKER: StepType.OBSERVATION,
    SpanKind.AGENT: StepType.DECISION,
    SpanKind.GUARDRAIL: StepType.VALIDATION,
    SpanKind.EVALUATOR: StepType.VALIDATION,
    SpanKind.CUSTOM: StepType.META,
}

SPAN_KIND_DEFAULT_SIDE_EFFECT: dict[SpanKind, SideEffect] = {
    SpanKind.LLM: SideEffect.PURE,
    SpanKind.RETRIEVER: SideEffect.READ_ONLY,
    SpanKind.RERANKER: SideEffect.PURE,
    SpanKind.GUARDRAIL: SideEffect.PURE,
    SpanKind.EVALUATOR: SideEffect.PURE,
    SpanKind.AGENT: SideEffect.PURE,
    SpanKind.CUSTOM: SideEffect.IRREVERSIBLE_WRITE,
}


def _hash_dict(d: dict) -> str:
    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class StepNormalizer:
    def __init__(
        self,
        side_effect_overrides: dict[str, SideEffect] | None = None,
    ):
        self._side_effect_overrides = side_effect_overrides or {}

    def normalize(self, spans: list[TraceSHAPSpan]) -> list[CanonicalStep]:
        sorted_spans = sorted(spans, key=lambda s: s.start_time)
        steps: list[CanonicalStep] = []
        retry_tracker: dict[str, int] = {}

        for span in sorted_spans:
            retry_key = f"{span.name}:{_hash_dict(span.input)}"
            attempt = retry_tracker.get(retry_key, 0)
            retry_tracker[retry_key] = attempt + 1

            step = CanonicalStep(
                step_id=f"step-{uuid.uuid4().hex[:12]}",
                raw_span_ids=[span.span_id],
                node_id=span.metadata.get("node_id"),
                tool_name=span.name if span.span_kind == SpanKind.TOOL else None,
                step_type=SPAN_KIND_TO_STEP_TYPE.get(span.span_kind, StepType.META),
                attempt_index=attempt,
                loop_iteration=None,
                input_hash=_hash_dict(span.input),
                output_hash=_hash_dict(span.output),
                side_effect_class=self._get_side_effect(span),
                framework_mapping_confidence=self._estimate_mapping_confidence(span),
                tokens=span.tokens,
                cost=span.cost,
                start_time=span.start_time,
                end_time=span.end_time,
            )
            steps.append(step)

        return steps

    def _get_side_effect(self, span: TraceSHAPSpan) -> SideEffect:
        if span.name in self._side_effect_overrides:
            return self._side_effect_overrides[span.name]
        if span.span_kind == SpanKind.TOOL:
            return SideEffect.IRREVERSIBLE_WRITE
        return SPAN_KIND_DEFAULT_SIDE_EFFECT.get(span.span_kind, SideEffect.IRREVERSIBLE_WRITE)

    @staticmethod
    def _estimate_mapping_confidence(span: TraceSHAPSpan) -> float:
        if span.metadata.get("node_id"):
            return 0.95
        if span.span_kind in (SpanKind.LLM, SpanKind.TOOL):
            return 0.7
        return 0.5
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_normalizer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/ingestion/normalizer.py tests/test_normalizer.py
git commit -m "feat: StepNormalizer with retry detection, side-effect classification, and span-to-step mapping"
```

---

### Task 12: Pipeline Orchestrator

**Files:**
- Create: `traceshap/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write tests**

`tests/test_pipeline.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.models import (
    TraceSHAPSpan, SpanKind, SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.ingestion.sources.base import SpanSource
from traceshap.pipeline import TraceSHAPPipeline
from traceshap.storage.sqlite import SQLiteBackend


class FakeSource(SpanSource):
    def __init__(self, span_batches: list[list[TraceSHAPSpan]]):
        self._batches = list(span_batches)

    async def connect(self) -> None:
        pass

    async def poll(self) -> list[TraceSHAPSpan]:
        if self._batches:
            return self._batches.pop(0)
        return []

    async def close(self) -> None:
        pass


def _make_spans(trace_id: str) -> list[TraceSHAPSpan]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        TraceSHAPSpan(
            trace_id=trace_id, span_id=f"{trace_id}-root", parent_span_id=None,
            span_kind=SpanKind.AGENT, name="agent", input={}, output={"result": "ok"},
            start_time=now, end_time=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
            tokens=None, cost=None, metadata={}, raw_attributes={}, semconv_version="test",
        ),
        TraceSHAPSpan(
            trace_id=trace_id, span_id=f"{trace_id}-llm", parent_span_id=f"{trace_id}-root",
            span_kind=SpanKind.LLM, name="gpt-4o", input={"prompt": "hi"}, output={"text": "hello"},
            start_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            tokens=None, cost=None, metadata={}, raw_attributes={}, semconv_version="test",
        ),
        TraceSHAPSpan(
            trace_id=trace_id, span_id=f"{trace_id}-tool", parent_span_id=f"{trace_id}-root",
            span_kind=SpanKind.TOOL, name="search", input={"q": "test"}, output={"r": "found"},
            start_time=datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc),
            tokens=None, cost=None, metadata={}, raw_attributes={}, semconv_version="test",
        ),
    ]


class TestPipeline:
    async def test_ingest_single_trace(self, tmp_path):
        spans = _make_spans("t1")
        source = FakeSource([spans])
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.initialize()

        pipeline = TraceSHAPPipeline(source=source, storage=backend)
        processed = await pipeline.ingest_once()
        assert processed == 1

        trajectory = await backend.get_trajectory("t1")
        assert trajectory is not None
        assert len(trajectory.spans) == 3
        assert len(trajectory.steps) == 3

        await backend.close()

    async def test_ingest_multiple_traces(self, tmp_path):
        source = FakeSource([_make_spans("t1"), _make_spans("t2")])
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.initialize()

        pipeline = TraceSHAPPipeline(source=source, storage=backend)
        count1 = await pipeline.ingest_once()
        count2 = await pipeline.ingest_once()
        assert count1 + count2 == 2

        t1 = await backend.get_trajectory("t1")
        t2 = await backend.get_trajectory("t2")
        assert t1 is not None
        assert t2 is not None

        await backend.close()

    async def test_ingest_empty_returns_zero(self, tmp_path):
        source = FakeSource([])
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.initialize()

        pipeline = TraceSHAPPipeline(source=source, storage=backend)
        count = await pipeline.ingest_once()
        assert count == 0

        await backend.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement pipeline**

`traceshap/pipeline.py`:
```python
from traceshap.models.trajectory import Trajectory, TrajectoryMeta, SpanNode
from traceshap.ingestion.sources.base import SpanSource
from traceshap.ingestion.assembler import SpanBuffer, TreeAssembler
from traceshap.ingestion.normalizer import StepNormalizer
from traceshap.storage.backend import StorageBackend


class TraceSHAPPipeline:
    def __init__(
        self,
        source: SpanSource,
        storage: StorageBackend,
        normalizer: StepNormalizer | None = None,
        framework: str = "unknown",
        agent_name: str = "default",
    ):
        self._source = source
        self._storage = storage
        self._normalizer = normalizer or StepNormalizer()
        self._buffer = SpanBuffer()
        self._framework = framework
        self._agent_name = agent_name

    async def ingest_once(self) -> int:
        spans = await self._source.poll()
        if not spans:
            return 0

        for span in spans:
            self._buffer.add(span)

        processed = 0
        for trace_id in list(self._buffer.pending_trace_ids()):
            trace_spans = self._buffer.flush(trace_id)
            if not trace_spans:
                continue

            sorted_spans = sorted(trace_spans, key=lambda s: s.start_time)
            span_tree = TreeAssembler.build(sorted_spans)
            steps = self._normalizer.normalize(sorted_spans)

            trajectory = Trajectory(
                trace_id=trace_id,
                spans=sorted_spans,
                steps=steps,
                span_tree=span_tree,
                outcome=None,
                metadata=TrajectoryMeta(
                    framework=self._framework,
                    agent_name=self._agent_name,
                ),
            )

            await self._storage.save_trajectory(trajectory)
            processed += 1

        return processed
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add traceshap/pipeline.py tests/test_pipeline.py
git commit -m "feat: TraceSHAPPipeline orchestrator (ingest → buffer → assemble → normalize → store)"
```

---

## Summary

After completing all 12 tasks, you will have:

1. **Project scaffolding** with pyproject.toml, dev dependencies, pytest config
2. **Complete domain models**: SpanKind, StepType, SideEffect, Verdict, DecisionStatus, TraceSHAPSpan, CanonicalStep, Trajectory, Outcome, StepAttribution, PruneCandidate, Savings, ValidationPlan
3. **YAML config system** loading all config sections with sensible defaults
4. **SQLite storage backend** with ORM models for all 8 tables, save/get/query trajectories
5. **Ingestion pipeline**: SpanSource ABC → LangfuseSource, SpanBuffer, TreeAssembler
6. **Step Normalizer**: span→step mapping, retry detection, side-effect classification, protected type tagging
7. **Pipeline orchestrator**: `TraceSHAPPipeline.ingest_once()` doing the full ingest cycle

**Next plans:**
- **Plan 2**: Attribution Engine (Layer 0 rules, Layer 1 statistical lift, Layer 2 sequence estimator) + Pruning Advisor
- **Plan 3**: CLI + REST API + Web Dashboard
- **Plan 4**: LangGraph native adapter + bridge extractors
