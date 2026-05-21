# TraceSHAP Design Spec

> Date: 2026-05-21 (rev2, post-review)
> Status: Draft
> Author: star + Claude

## 1. Vision & Goals

TraceSHAP is a production-grade attribution and ablation analysis tool for LLM agent trajectories. It consumes OpenTelemetry spans (at Langfuse trace granularity), normalizes them into canonical steps, computes layered attribution, and produces validated pruning candidates — enabling agents to self-evolve by shedding low-value nodes (negative entropy maintenance).

### Core Questions TraceSHAP Answers

- Which steps in an agent trajectory contribute most (or least) to the outcome?
- Which nodes/tools/steps can be pruned without degrading quality?
- How much cost/latency/tokens can be saved by pruning?
- What validation is needed before a prune candidate can be safely applied?

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary use case | Production monitoring (with debug & batch as secondary) | Core need is continuous attribution pipeline |
| Architecture | Pipeline-first with instrument() convenience | Matches continuous monitoring; sidecar only sends spans |
| Attribution unit | CanonicalStep (normalized from raw spans) | Spans are observations, not interventable units; steps map to actual graph nodes/tool policies |
| Attribution method | 5-layer (rules → statistical lift → sequence estimator → replay SHAP → causal hypothesis) | Progressive depth: fast for prod, deep for analysis |
| Value function | Multi-objective (quality, cost, latency, risk) with optional composite | Avoids batch leakage from premature aggregation |
| Replay strategy | Safety-classified: simulated (prod) + sandboxed replay (deep analysis) | Not all steps are replayable; side effects must be classified |
| Pruning automation | Validated candidates in v0.1, auto-prune interface reserved with validation gate | Prune only after replay/eval/A-B confirmation |
| Frameworks v0.1 | Langfuse/OTLP ingestion + LangGraph native adapter; OpenClaw/Hermes via bridge extractors | OpenClaw is Node.js, Hermes has its own export API; can't assume Python in-process for all |
| Distribution | Library + CLI + Web Dashboard | Full-stack developer experience |

---

## 2. Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                           TraceSHAP                               │
│                                                                   │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────────┐   │
│  │Instrument│──▶│  Ingestion   │──▶│  Step Normalizer        │   │
│  │  Layer   │   │   Layer      │   │  (Span → CanonicalStep) │   │
│  └──────────┘   └──────────────┘   └───────────┬─────────────┘   │
│                                                 │                 │
│                                     ┌───────────▼─────────────┐   │
│                                     │  Attribution Engine     │   │
│                                     │  (Layer 0-4)            │   │
│                                     └───────────┬─────────────┘   │
│                                                 │                 │
│                                     ┌───────────▼─────────────┐   │
│                                     │  Pruning Advisor        │   │
│                                     │  + Validation Planner   │   │
│                                     └───────────┬─────────────┘   │
│                                                 │                 │
│                    ┌──────────────┬──────────────┼──────────┐     │
│                    ▼              ▼              ▼          │     │
│               ┌────────┐   ┌──────────┐   ┌──────────┐    │     │
│               │  CLI   │   │Web Dash  │   │  API     │    │     │
│               └────────┘   └──────────┘   └──────────┘    │     │
│                                                            │     │
│  ┌─────────────────────────────────────────────────────┐   │     │
│  │              Storage Layer (SQLite/PG)               │   │     │
│  └─────────────────────────────────────────────────────┘   │     │
└───────────────────────────────────────────────────────────────────┘

External:
  Langfuse    ──(OTel/API)──▶ Ingestion Layer
  OTel OTLP   ──(gRPC/HTTP)──▶ Ingestion Layer
  LangGraph   ──(instrument)──▶ Instrument Layer      [native adapter]
  OpenClaw    ──(bridge)──▶ Ingestion Layer            [ClawTrace/API extractor]
  Hermes      ──(bridge)──▶ Ingestion Layer            [Atropos trajectory export]
```

### 7 Core Modules

| Module | Responsibility |
|--------|---------------|
| **Instrument Layer** | Framework adapters; `instrument()` for LangGraph (native); bridge extractors for OpenClaw/Hermes |
| **Ingestion Layer** | Consume OTel spans from Langfuse API, OTLP endpoint, or instrument direct; buffer and assemble span trees |
| **Step Normalizer** | Convert raw spans → CanonicalStep; map steps to interventable graph nodes/tool policies; classify side effects |
| **Attribution Engine** | Core computation; Layer 0-4 layered attribution; output per-step scores with confidence intervals |
| **Pruning Advisor** | Convert attribution scores to validated prune candidates with required validation plan |
| **Storage** | Persist trajectories, steps, attributions, models, stats; versioned attribution runs (default SQLite, production PostgreSQL) |
| **Output Layer** | CLI tools + Web Dashboard + REST API |

---

## 3. Instrument Layer

Only collects spans. No analysis. Analysis happens entirely in the pipeline backend.

### LangGraph (Native Adapter)

```python
from traceshap import instrument
app = instrument(langgraph_app, framework="langgraph")
```

### OpenClaw (Bridge Extractor)

OpenClaw is a Node.js/Gateway runtime. TraceSHAP consumes its traces via ClawTrace API or OTLP export, not in-process Python instrumentation.

```python
pipeline = TraceSHAPPipeline(
    source="otlp",
    otlp_endpoint="http://clawtrace:4318",   # ClawTrace OTLP endpoint
    source_hint="openclaw",                    # helps step normalizer
)
```

### Hermes (Bridge Extractor)

Hermes has its own trajectory export API (Atropos). TraceSHAP imports from exported trajectory files or Atropos API.

```python
pipeline = TraceSHAPPipeline(
    source="hermes_atropos",
    atropos_api="http://hermes:8080/trajectories",
)
```

### Langfuse-Only Mode

```python
pipeline = TraceSHAPPipeline(source="langfuse", langfuse_host="...", langfuse_api_key="...")
pipeline.run()
```

### Adapter Capability Matrix

| Capability | LangGraph | OpenClaw | Hermes | Langfuse/OTLP |
|-----------|-----------|----------|--------|---------------|
| Span capture | native | bridge | bridge | native |
| Step mapping | high confidence | medium | medium | depends on instrumentation |
| Outcome binding | native | via ClawTrace | via Atropos | via Langfuse score |
| Dry replay | yes (graph mutation) | no (v0.1) | partial (tool list) | no |
| Live replay | sandbox only | no (v0.1) | sandbox only | no |
| Prune patch | graph node removal | no (v0.1) | tool policy change | no |

### Standardized Span Model

```python
@dataclass
class TraceSHAPSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    span_kind: SpanKind    # LLM / TOOL / RETRIEVER / AGENT / RERANKER / GUARDRAIL / EVALUATOR / CUSTOM
    name: str
    input: dict
    output: dict
    start_time: datetime
    end_time: datetime
    tokens: TokenUsage | None
    cost: float | None
    metadata: dict
    raw_attributes: dict   # original OTel/OpenInference attributes preserved verbatim
    semconv_version: str   # "otel-genai-v0.x" | "openinference-v1.x" | "custom"
```

SpanKind aligned with OpenInference conventions (AGENT, TOOL, RETRIEVER, RERANKER, GUARDRAIL, EVALUATOR, etc.) since these are the most widely adopted in practice. OTel GenAI agent span conventions are still in Development status — TraceSHAP normalizes both.

---

## 4. Step Normalizer (New)

The key insight from review: **spans are observations, not interventable units**. A single graph node may emit multiple spans (LLM call + tool call + retry). A retry span is not an independent decision. TraceSHAP's attribution and pruning must operate on CanonicalSteps, not raw spans.

### CanonicalStep Model

```python
@dataclass
class CanonicalStep:
    step_id: str                         # TraceSHAP-assigned
    raw_span_ids: list[str]              # source spans that form this step
    node_id: str | None                  # framework graph node (e.g., LangGraph node name)
    tool_name: str | None                # tool invoked, if any
    step_type: StepType                  # DECISION / ACTION / OBSERVATION / VALIDATION / META
    attempt_index: int                   # retry number (0 = first attempt)
    loop_iteration: int | None           # if inside a detected loop
    input_hash: str                      # for dedup and similarity detection
    output_hash: str
    side_effect_class: SideEffect        # PURE / READ_ONLY / IDEMPOTENT_WRITE / IRREVERSIBLE_WRITE
    framework_mapping_confidence: float  # 0-1, how confidently this step maps to a graph node
    tokens: TokenUsage | None
    cost: float | None
    start_time: datetime
    end_time: datetime

class SideEffect(Enum):
    PURE = "pure"                        # no external state change
    READ_ONLY = "read_only"              # reads external state but doesn't modify
    IDEMPOTENT_WRITE = "idempotent_write"  # writes but safe to retry
    IRREVERSIBLE_WRITE = "irreversible_write"  # sends email, places order, etc.
```

### Normalization Rules

1. **Span grouping**: Spans sharing the same `node_id` within a single execution pass are grouped into one step
2. **Retry detection**: Sequential spans with same tool_name and similar input_hash → single step with attempt_index > 0
3. **Loop detection**: Repeated step patterns → steps annotated with loop_iteration
4. **Side effect classification**: Based on tool metadata, framework hints, or user-provided mapping. Defaults to `IRREVERSIBLE_WRITE` if unknown (safe default).
5. **Protected step types**: Steps classified as VALIDATION, GUARDRAIL, or EVALUATOR (from span_kind) are auto-tagged as non-prunable

### Pruning Target Mapping

Pruning never operates on span names. It operates on:
- **node_id**: remove/disable a graph node
- **tool_name**: remove a tool from the agent's tool policy
- **step pattern**: disable a specific step sequence (e.g., "retry after search failure")

---

## 5. Ingestion Layer

Reconstructs structured trajectories from a stream of spans, then normalizes to steps.

### Data Models

```python
@dataclass
class Trajectory:
    trace_id: str
    spans: list[TraceSHAPSpan]           # raw, time-ordered
    steps: list[CanonicalStep]           # normalized, time-ordered
    span_tree: SpanNode                  # tree structure (parent-child)
    outcome: Outcome | None              # may arrive late
    metadata: TrajectoryMeta             # framework, agent_name, agent_version, task_type

@dataclass
class Outcome:
    success: bool | None
    quality_score: float | None          # 0-1
    token_cost: int
    latency_ms: int
    custom_metrics: dict
    evaluator_id: str | None             # who/what produced this score
    evaluator_version: str | None
    score_confidence: float | None       # evaluator's confidence
    label_delay_ms: int | None           # time between trace end and score arrival
```

### Value Function: Multi-Objective, Not Premature Composite

Review correctly identified that `quality - α × normalized_cost` with batch-max normalization causes batch leakage (same trace gets different scores in different batches).

**Fix: attribution operates on per-dimension deltas, not a single composite.**

```python
@dataclass
class StepAttribution:
    # Per-dimension removal deltas (what happens if this step is removed)
    quality_delta: float | None          # score_without - score_with (positive = removal hurts)
    cost_delta: float | None             # cost saved by removal (positive = saves money)
    latency_delta: float | None          # latency saved (positive = faster)
    risk_delta: float | None             # risk change (positive = riskier without this step)

    def composite_score(self, weights: dict | None = None) -> float:
        """Optional composite, only when user explicitly configures weights.
        Normalization uses per-agent historical P50 as baseline, not batch max."""
        ...
```

If users want a single number, they configure weights explicitly and normalization uses the agent's historical P50/P95 as baseline (stable across batches), not the current batch max.

### Processing Stages

| Stage | What | Trigger |
|-------|------|---------|
| **Span Buffering** | Buffer spans by trace_id, handle out-of-order arrival | Each span arrival |
| **Tree Assembly** | Rebuild span tree from parent_span_id, detect missing spans | Root span ends or timeout |
| **Step Normalization** | Spans → CanonicalSteps via Step Normalizer | After tree assembly |
| **Outcome Binding** | Bind task result to trajectory | After step normalization |

### Outcome Sources

```python
# 1. From Langfuse score
pipeline = TraceSHAPPipeline(outcome_source="langfuse_score", score_name="task_success")

# 2. User callback
@pipeline.outcome_handler
def evaluate(trajectory: Trajectory) -> Outcome:
    ...

# 3. Infer from span attributes
pipeline = TraceSHAPPipeline(outcome_source="span_attribute", outcome_path="spans[-1].output.success")
```

---

## 6. Attribution Engine

Layered architecture. Each layer independent, can be enabled/disabled. All layers operate on CanonicalSteps (not raw spans).

```python
pipeline = TraceSHAPPipeline(layers=[0, 1, 2])         # production default
pipeline = TraceSHAPPipeline(layers=[0, 1, 2, 3, 4])   # deep analysis
```

### Layer 0: Expert Rules Engine

```python
rules = [
    RepetitionRule(threshold=3, similarity=0.9),
    LoopDetectionRule(max_cycle=2),
    NoOpRule(similarity_threshold=0.95),
    CustomRule(fn=my_rule_function),
]
```

Output: `RuleVerdict(step_id, rule_name, severity, recommendation)`

### Layer 1: Statistical Lift (Cohort-Stratified)

Cross-trajectory association analysis, **stratified by cohort to control confounding:**

```python
@dataclass
class LiftConfig:
    stratify_by: list[str] = ["agent_version", "task_type", "model_version"]
    min_support: int = 50              # minimum trajectories per cohort
    smoothing: float = 1.0             # Laplace smoothing
    confidence_level: float = 0.95     # for CI computation
    multiple_testing: str = "bonferroni"  # correction method
```

For each cohort: `P(success | step_type=X present) vs P(success | step_type=X absent)`

Output: **lift score** with confidence interval per step type per cohort. Explicitly labeled as "statistical association", not causal attribution.

### Layer 2: Sequence-Aware Counterfactual Estimator

Production workhorse. The original design's `random_subset` approach violates trajectory sequence constraints — arbitrary subsets are not valid execution paths. **Fixed to only sample legal interventions.**

**Legal intervention types:**
1. **Prefix removal**: skip the last N steps, estimate outcome from earlier state
2. **Contiguous segment removal**: remove a consecutive block of steps (e.g., a retry loop)
3. **Same-node retry removal**: collapse retries to first attempt only
4. **Tool alternative**: estimate outcome with a different tool at this step
5. **Retrieval perturbation**: vary retrieval top-k or query

```python
def sequence_shap(trajectory: Trajectory, model: SequenceModel) -> dict[str, StepAttribution]:
    attributions = {}
    for step in trajectory.steps:
        interventions = generate_legal_interventions(trajectory, step)
        deltas = []
        for intervention in interventions:
            counterfactual = model.estimate_outcome(intervention.modified_trajectory)
            factual = model.estimate_outcome(trajectory)
            deltas.append(factual - counterfactual)
        
        attributions[step.step_id] = StepAttribution(
            quality_delta=mean([d.quality for d in deltas]),
            cost_delta=step.cost,  # direct measurement
            latency_delta=step.duration_ms,
            confidence=bootstrap_ci(deltas),
            calibration=model.held_out_metrics(),  # AUC, RMSE, coverage
        )
    return attributions
```

**Model options (configurable):**
- Default: learned transition model from historical trajectories
- Alternative: fitted Q-evaluation, doubly robust off-policy estimator

**Latency target:** < 3 seconds for ~20 steps on CPU (no GPU required). Embedding cache keyed by `text_hash + model_id`.

**Every Layer 2 output includes calibration metrics:** held-out AUC/RMSE, OOD detection score, bootstrap confidence intervals.

### Layer 3: Replay SHAP [experimental]

Real agent replay with ablation. **Only runs in offline eval harness, never in production ingestion path.**

#### Replay Safety Model

```python
class ReplayCapability(Enum):
    NONE = "none"                              # cannot replay
    DRY_RUN_MOCKED = "dry_run_mocked"          # mock all external calls
    RECORDED_IO_REPLAY = "recorded_io_replay"  # replay with recorded I/O
    LIVE_SANDBOX_REPLAY = "live_sandbox_replay" # real execution in sandbox

@dataclass
class ReplayCapsule:
    trajectory: Trajectory
    model_id: str                    # exact model version
    prompt_hashes: dict[str, str]    # step_id → prompt template hash
    temperature: float
    tool_schemas: dict[str, dict]    # tool name → schema snapshot
    environment_snapshot: dict       # relevant env state
    recorded_ios: dict[str, Any]     # step_id → recorded external I/O for mock replay
```

**Replay eligibility per step:**
- `PURE` / `READ_ONLY` side effect → eligible for all replay modes
- `IDEMPOTENT_WRITE` → eligible for `DRY_RUN_MOCKED` and `RECORDED_IO_REPLAY`
- `IRREVERSIBLE_WRITE` → only `DRY_RUN_MOCKED` with explicit user opt-in

```python
class ReplayEngine:
    def replay_without(
        self,
        capsule: ReplayCapsule,
        ablate_steps: list[str],
        mode: ReplayCapability = ReplayCapability.RECORDED_IO_REPLAY,
    ) -> Outcome:
        for step_id in ablate_steps:
            step = capsule.get_step(step_id)
            if step.side_effect_class == SideEffect.IRREVERSIBLE_WRITE:
                if mode != ReplayCapability.DRY_RUN_MOCKED:
                    raise ReplaySafetyError(f"Step {step_id} has irreversible side effects")
        ...
```

Cost control: sampling budget — max K replays per trace (default K=2n, n=step count).

### Layer 4: Causal Hypothesis [experimental]

Built on Layer 3 replay data. **Outputs causal hypotheses, not confirmed causal claims.**

Causal graph edges come from three sources with separate confidence levels:

| Edge Type | Source | Confidence |
|-----------|--------|-----------|
| **Control-flow** | Framework graph definition (LangGraph edges) | High |
| **Data-dependency** | Output of step A used as input to step B (hash matching) | Medium |
| **Temporal** | Step A precedes step B | Low (correlation, not causation) |

**Only outputs causal claims when supported by:**
- Randomized data (policy variation across trajectories)
- Replay intervention data (Layer 3)
- Natural experiment (A/B test data)

Otherwise outputs "associational attribution" with explicit disclaimer.

```python
@dataclass
class CausalAttribution:
    step_id: str
    hypothesis_type: str              # "causal" | "associational"
    downstream_effects: list[CausalEdge]
    evidence_sources: list[str]       # "replay", "policy_variation", "temporal_only"
    confidence_by_edge_type: dict[str, float]
```

### Unified Output

```python
@dataclass
class StepAttribution:
    step_id: str
    step_name: str
    node_id: str | None
    quality_delta: float | None       # per-dimension, not premature composite
    cost_delta: float | None
    latency_delta: float | None
    risk_delta: float | None
    layer_scores: dict[int, float]
    confidence: ConfidenceInterval     # lower_bound, point_estimate, upper_bound
    verdict: Verdict                   # KEEP / REVIEW / PRUNE_CANDIDATE / INSUFFICIENT_EVIDENCE
    causal_hypothesis: CausalAttribution | None  # Layer 4 only
    evidence: list[str]
    calibration: CalibrationMetrics | None  # Layer 2+: AUC, RMSE, coverage
```

Note: `INSUFFICIENT_EVIDENCE` verdict for trajectories that lack outcome, are out-of-distribution, or have too few peers for statistical analysis. Never silently degrade to PRUNE/KEEP.

---

## 7. Pruning Advisor

### Decision Logic

```python
def classify_step(attr: StepAttribution, config: PruningConfig) -> Verdict:
    # Insufficient data → cannot make a call
    if attr.confidence is None or attr.quality_delta is None:
        return Verdict.INSUFFICIENT_EVIDENCE
    
    # Protected step types are never pruned
    if attr.step_type in PROTECTED_TYPES:
        return Verdict.KEEP
    
    # removal_delta semantics: quality_delta = score_without - score_with
    # quality_delta >= -epsilon means "removal doesn't hurt (much)"
    # Must check lower bound of CI, not just point estimate
    if (attr.confidence.lower_bound >= -config.prune_epsilon
            and attr.cost_delta > 0):
        return Verdict.PRUNE_CANDIDATE
    
    if attr.confidence.lower_bound < -config.keep_threshold:
        return Verdict.KEEP
    
    return Verdict.REVIEW

PROTECTED_TYPES = {
    StepType.VALIDATION,     # evaluator/validator steps
    StepType.GUARDRAIL,      # safety guardrails
    StepType.AUTH,           # permission/security checks
    StepType.INITIALIZATION, # state setup
    StepType.FINALIZATION,   # state cleanup
}
```

### Output

```python
@dataclass
class PruneCandidate:
    target_type: str                 # "node" | "tool_policy" | "step_pattern"
    target_id: str                   # node_id, tool_name, or pattern descriptor
    evidence: list[StepAttribution]  # supporting attributions across trajectories
    estimated_savings: Savings
    required_validation: ValidationPlan
    decision_status: DecisionStatus  # CANDIDATE | VALIDATED | REJECTED | APPLIED | ROLLED_BACK

@dataclass
class ValidationPlan:
    replay_required: bool            # does this need replay confirmation?
    replay_mode: ReplayCapability    # what level of replay is needed?
    min_replay_count: int            # how many replays to confirm?
    ab_test_recommended: bool        # should this be A/B tested?
    human_review_required: bool      # does a human need to approve?

@dataclass
class PruningReport:
    trace_id: str | None             # None for cross-trajectory reports
    timestamp: datetime
    total_steps: int
    candidates: list[PruneCandidate]
    estimated_savings: Savings
    risk_assessment: RiskLevel

@dataclass
class Savings:
    token_reduction: int
    cost_reduction: float
    latency_reduction_ms: int
    quality_impact_range: tuple[float, float]  # (lower_bound, upper_bound) of quality change
```

### Auto-Pruner Interface (v0.1: interface only, execution requires validation gate)

```python
class AutoPruner(ABC):
    @abstractmethod
    def generate_patch(self, candidate: PruneCandidate, graph_definition: Any) -> PrunePatch:
        """Generate a patch proposal, NOT directly mutate the graph."""
        ...

    @abstractmethod
    def apply_patch(self, patch: PrunePatch, graph_definition: Any) -> Any:
        """Apply only after validation gate passes."""
        ...

@dataclass
class PrunePatch:
    candidate: PruneCandidate
    patch_type: str                  # "remove_node" | "disable_tool" | "modify_policy"
    patch_payload: dict              # framework-specific mutation descriptor
    validation_status: DecisionStatus
    validated_by: str | None         # "replay", "ab_test", "human", None
```

### Safety Constraints

- **Never prune first/last steps** — agent entry and final output always KEEP
- **Protected step types** — VALIDATION, GUARDRAIL, AUTH, INITIALIZATION, FINALIZATION never auto-pruned
- **Causal chain protection** — if step A is PRUNE_CANDIDATE but is causal upstream of a KEEP step, downgrade to REVIEW
- **Cross-trajectory consistency** — single-trajectory verdict is always REVIEW; PRUNE_CANDIDATE requires consistent signal across N trajectories (default N=10), stratified by agent_version + task_type
- **Validation gate** — no patch applied without passing its ValidationPlan
- **Rollback tracking** — applied patches tracked with status, can be rolled back

---

## 8. Storage Layer

Default zero-config (SQLite), production switches to PostgreSQL. Same interface.

### Data Model

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│ trajectories │────▶│    spans     │────▶│ canonical_steps  │
├─────────────┤     ├──────────────┤     ├──────────────────┤
│ trace_id PK │     │ span_id PK   │     │ step_id PK       │
│ framework   │     │ trace_id FK  │     │ trace_id FK      │
│ agent_name  │     │ parent_id    │     │ node_id          │
│ agent_ver   │     │ span_kind    │     │ tool_name        │
│ task_type   │     │ name         │     │ step_type        │
│ outcome     │     │ input/output │     │ side_effect      │
│ created_at  │     │ tokens/cost  │     │ attempt_index    │
│ metadata    │     │ start/end    │     │ input_hash       │
└─────────────┘     │ raw_attrs    │     │ output_hash      │
                    │ semconv_ver  │     │ mapping_conf     │
                    └──────────────┘     │ tokens/cost      │
                                         └──────────────────┘

┌──────────────────┐     ┌──────────────────┐
│ attribution_runs │     │ step_attributions │
├──────────────────┤     ├──────────────────┤
│ run_id PK        │     │ run_id FK        │
│ trace_id FK      │     │ step_id FK       │
│ config_hash      │     │ layer            │
│ code_version     │     │ quality_delta    │
│ model_version    │     │ cost_delta       │
│ layers_used      │     │ latency_delta    │
│ created_at       │     │ confidence_lo    │
│                  │     │ confidence_hi    │
└──────────────────┘     │ verdict          │
                         │ evidence         │
                         │ calibration      │
                         └──────────────────┘

┌──────────────────┐     ┌──────────────────┐
│ prune_candidates │     │ markov_models    │
├──────────────────┤     ├──────────────────┤
│ candidate_id PK  │     │ model_id PK      │
│ target_type      │     │ agent_name       │
│ target_id        │     │ framework        │
│ savings          │     │ model_blob       │
│ validation_plan  │     │ training_count   │
│ decision_status  │     │ held_out_metrics │
│ validated_by     │     │ updated_at       │
│ created_at       │     └──────────────────┘
│ updated_at       │
└──────────────────┘

┌──────────────────┐
│ cohort_stats     │
├──────────────────┤
│ step_type        │
│ agent_name       │
│ agent_version    │
│ task_type        │
│ success_count    │
│ total_count      │
│ lift_score       │
│ lift_ci_lo       │
│ lift_ci_hi       │
│ updated_at       │
└──────────────────┘
```

### Key Changes from v1

- **attribution_runs** table: every run is versioned (config_hash, code_version, model_version) for reproducibility
- **canonical_steps** table: attribution operates on steps, not raw spans; spans preserved for audit
- **prune_candidates** with decision_status lifecycle: `CANDIDATE → VALIDATED → APPLIED → ROLLED_BACK` (or `REJECTED`)
- **cohort_stats** replaces bayesian_stats: stratified by agent_version + task_type, includes confidence intervals
- **raw_attrs** preserved on spans for source compatibility debugging

### Retention Policy

| Data | Write Frequency | Retention |
|------|----------------|-----------|
| trajectories + spans + steps | Per trace completion | 30 days default, configurable |
| attribution_runs + step_attributions | Per attribution run | Follows trajectory lifecycle |
| prune_candidates | Per cross-trajectory analysis | Permanent |
| markov_models | Layer 2 incremental training | Latest + 1 backup |
| cohort_stats | Per trajectory update | Permanent (aggregate) |

### Interface

```python
class StorageBackend(ABC):
    @abstractmethod
    async def save_trajectory(self, t: Trajectory) -> None: ...
    @abstractmethod
    async def save_attribution_run(self, run: AttributionRun, attrs: list[StepAttribution]) -> None: ...
    @abstractmethod
    async def query_trajectories(self, filters: QueryFilter) -> list[Trajectory]: ...
    @abstractmethod
    async def get_cohort_stats(self, agent_name: str, cohort: CohortFilter) -> list[CohortStat]: ...
    @abstractmethod
    async def update_candidate_status(self, candidate_id: str, status: DecisionStatus) -> None: ...

class SQLiteBackend(StorageBackend): ...
class PostgresBackend(StorageBackend): ...
```

---

## 9. Output Layer

### CLI

```bash
traceshap serve --source langfuse --langfuse-host=... --port 8080
traceshap analyze <trace_id> --layers 0,1,2,3,4
traceshap report <trace_id>
traceshap prune-report --agent myagent --min-trajectories 50
traceshap export <trace_id> --format json|csv|otel

# Analysis jobs (async)
traceshap analyze <trace_id> --async    # returns job_id
traceshap job <job_id>                  # check status
```

### REST API

```
GET  /api/traces                              # list with filters
GET  /api/traces/<trace_id>                   # single trajectory detail (steps + spans)
GET  /api/traces/<trace_id>/attribution       # attribution results

POST /api/analysis-jobs                       # create analysis job, returns job_id
GET  /api/analysis-jobs/<job_id>              # job status + results when complete

GET  /api/agents/<name>/stats                 # agent-level aggregate stats
GET  /api/agents/<name>/prune-candidates      # cross-trajectory prune candidates
PATCH /api/prune-candidates/<id>/status       # update decision status (validate/reject/apply)

WS   /ws/live                                 # real-time attribution push
```

### Web Dashboard

FastAPI + React (lightweight SPA). Four core pages:

| Page | Content |
|------|---------|
| **Overview** | Agent list, trajectory counts, per-dimension score trends, PRUNE_CANDIDATE rate |
| **Trajectory Detail** | Step tree visualization (colored by verdict: green=KEEP, yellow=REVIEW, red=PRUNE_CANDIDATE, gray=INSUFFICIENT_EVIDENCE), click to expand per-dimension deltas + confidence intervals |
| **Ablation View** | Select a step, see per-dimension removal impact with CIs, causal hypothesis graph (Layer 4) |
| **Prune Dashboard** | Cross-trajectory candidates with decision lifecycle, savings trend, validation plan status |

---

## 10. Configuration

```yaml
# traceshap.yaml
source:
  type: langfuse                    # langfuse | otlp | hermes_atropos | direct
  langfuse_host: https://cloud.langfuse.com
  langfuse_public_key: pk-...
  langfuse_secret_key: sk-...
  poll_interval_seconds: 10

adapters:
  langgraph:
    enabled: true
    mode: native                    # in-process instrument()
  openclaw:
    enabled: false
    mode: bridge                    # consume from ClawTrace/OTLP
    otlp_endpoint: http://clawtrace:4318
  hermes:
    enabled: false
    mode: bridge                    # consume from Atropos API
    atropos_api: http://hermes:8080/trajectories

attribution:
  layers: [0, 1, 2]                # production default
  layer_1:
    stratify_by: [agent_version, task_type, model_version]
    min_support: 50
    confidence_level: 0.95
  layer_2:
    embedding_model: all-MiniLM-L6-v2
    embedding_cache: true
    num_samples: 200
    intervention_types: [prefix, contiguous_segment, retry_collapse, tool_alternative]
    calibration_holdout: 0.1        # 10% held out for calibration metrics
  layer_3:
    enabled: false                  # experimental, offline only
    replay_mode: recorded_io_replay
    replay_budget_per_trace: 40
    replay_concurrency: 4
  layer_4:
    enabled: false                  # experimental
    requires_layer_3: true
    claim_type: associational       # "associational" | "causal" (only if replay/randomization data)

pruning:
  prune_epsilon: 0.05              # removal_delta lower_bound >= -epsilon to be candidate
  keep_threshold: 0.10
  min_trajectories: 10
  protected_step_types: [validation, guardrail, auth, initialization, finalization]
  protect_first_last: true
  validation_gate: true             # require validation before auto-prune

outcome:
  source: langfuse_score
  score_name: task_success
  normalization_baseline: historical_p50  # historical_p50 | historical_p95 | fixed
  # composite weights (optional, multi-objective by default)
  # composite_weights: {quality: 0.7, cost: 0.2, latency: 0.1}

storage:
  backend: sqlite
  sqlite_path: ./traceshap.db
  retention_days: 30

server:
  host: 0.0.0.0
  port: 8080
  workers: 4

performance:
  embedding_cache_size: 10000       # max cached embeddings
  max_concurrent_analyses: 8
  layer_2_timeout_seconds: 10       # per-trajectory timeout
```

---

## 11. Deployment

| Method | Use Case | Command |
|--------|----------|---------|
| Local | Dev/debug | `traceshap serve` |
| Docker | Small team production | `docker run traceshap/traceshap` |
| Docker Compose | With PostgreSQL | `docker compose up` |

### Minimal Start

```bash
pip install traceshap
traceshap init          # generate traceshap.yaml template
traceshap serve         # SQLite + local dashboard at http://localhost:8080
```

---

## 12. Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Async framework | asyncio + FastAPI |
| OTel | opentelemetry-sdk, opentelemetry-semantic-conventions |
| Langfuse | langfuse Python SDK |
| Attribution compute | numpy, scipy (Shapley + statistics), scikit-learn (sequence models) |
| Embedding | sentence-transformers (all-MiniLM-L6-v2, local, cached) |
| Storage | SQLAlchemy (SQLite/PG unified ORM) |
| CLI | click |
| Web frontend | React + Vite + Recharts (charts) + react-d3-tree (step tree) |
| Packaging | pyproject.toml, hatchling |
| Containers | Dockerfile, docker-compose.yaml |

---

## 13. Attribution Methodology Background

### Why Layered Attribution

The core problem — "which steps in an agent trajectory matter?" — can be addressed at multiple levels of rigor and cost:

```
Layer 0: Expert Rules                        (instant, rule engine)
Layer 1: Statistical Lift                    (milliseconds, cohort-stratified association)
Layer 2: Sequence-Aware Counterfactual Est.  (seconds, legal interventions only)
Layer 3: Replay SHAP [experimental]          (minutes, real replay in sandbox)
Layer 4: Causal Hypothesis [experimental]    (minutes-hours, requires intervention data)
```

### Theoretical Grounding

| Method | Role in TraceSHAP |
|--------|------------------|
| **SHAP (Shapley values)** | Core attribution: each step is a "player" in a cooperative game, outcome metric is the value function. Satisfies efficiency, symmetry, dummy, and additivity axioms. |
| **Statistical lift** | Layer 1: cohort-stratified association analysis. Explicitly not causal — controls confounding via stratification but cannot establish causation. |
| **Sequence models** | Layer 2: trajectory is a state transition sequence. Learned model serves as counterfactual estimator. Only legal interventions sampled (not arbitrary subsets). |
| **Causal inference** | Layer 4: SCM construction, do-intervention, counterfactual reasoning. Outputs hypotheses unless backed by randomized/intervention data. |
| **Expert heuristics** | Layer 0: known anti-patterns flagged instantly, also serves as baseline for validating statistical results. |

---

## 14. Acceptance Criteria

| Criterion | Definition |
|-----------|-----------|
| **Attribution reproducibility** | Same trace + same config + same model/evaluator version → score difference within configured epsilon across repeated runs |
| **False-prune guardrail** | On eval dataset, high-confidence PRUNE_CANDIDATEs must have quality regression rate < 1% (configurable) when validated via replay |
| **Coverage honesty** | Trajectories lacking outcome, OOD, or with too few cohort peers must output `INSUFFICIENT_EVIDENCE`, never silent PRUNE/KEEP |
| **Source compatibility** | Pass integration tests with: Langfuse OTLP, standard OTel GenAI semconv, OpenInference-style spans |
| **Calibration** | Layer 2 held-out metrics (AUC, RMSE) reported per attribution run; model not used if below configured threshold |

---

## 15. Review Response Log

Changes made based on `2026-05-21-traceshap-design-review.md`:

| Review Point | Action | Section |
|-------------|--------|---------|
| P0: Span ≠ interventable unit | **Adopted**: Added CanonicalStep, Step Normalizer module | §4, §6 |
| P0: Replay safety | **Adopted**: Added SideEffect classification, ReplayCapsule, replay eligibility rules | §4, §6.3 |
| P0: ablation_impact sign ambiguity | **Adopted**: Defined `removal_delta = score_without - score_with`, CI-based thresholds | §7 |
| P1: Value function batch leakage | **Adopted**: Multi-objective deltas, historical P50 baseline normalization | §5 |
| P1: Layer 2 sequence violation | **Adopted**: Legal interventions only, removed random_subset | §6.2 |
| P1: Causal SHAP overclaiming | **Adopted**: Renamed to "Causal Hypothesis", outputs associational unless evidence | §6.4 |
| P1: Framework scope | **Adopted**: LangGraph native, OpenClaw/Hermes via bridge, adapter capability matrix | §3 |
| P2: Storage versioning | **Adopted**: attribution_runs table with config_hash/version | §8 |
| P2: Bayesian confounding | **Adopted**: Cohort-stratified, renamed to "Statistical Lift" | §6.1 |
| P2: Async analysis jobs | **Adopted**: POST /api/analysis-jobs with job lifecycle | §9 |
| P2: Performance budget | **Partially adopted**: Added performance config, embedding cache | §10 |
| Review: Dashboard to v0.2 | **Not adopted**: User explicitly requested Web Dashboard in v0.1 | §9 |
| Review: Delete AutoPruner | **Not adopted**: User requested reserved interface; kept with validation gate | §7 |
| Review: Multi-tenant/PII | **Not adopted**: Enterprise features, not needed for open-source v0.1 | — |
| Review: K8s/Helm to v0.2 | **Adopted**: Removed from v0.1 deployment | §11 |
