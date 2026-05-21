# TraceSHAP Design Spec

> Date: 2026-05-21
> Status: Draft
> Author: star + Claude

## 1. Vision & Goals

TraceSHAP is a production-grade attribution and ablation analysis tool for LLM agent trajectories. It consumes OpenTelemetry spans (at Langfuse trace granularity), computes layered Shapley-value attribution for each span, and produces actionable pruning recommendations — enabling agents to self-evolve by shedding low-value nodes (negative entropy maintenance).

### Core Questions TraceSHAP Answers

- Which spans in an agent trajectory contribute most (or least) to the outcome?
- Which nodes/tools/steps can be pruned without degrading quality?
- How much cost/latency/tokens can be saved by pruning?

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary use case | Production monitoring (with debug & batch as secondary) | User's core need is continuous attribution pipeline |
| Architecture | Pipeline-first with instrument() convenience | Matches continuous monitoring; sidecar only sends spans |
| Attribution granularity | OTel span level (Langfuse trace granularity) | Matches existing observability tooling |
| Attribution method | 5-layer (rules → Bayesian → Markov → SHAP → Causal) | Progressive depth: fast for prod, deep for analysis |
| Value function | Composite: quality − α × normalized_cost | Captures both effectiveness and efficiency |
| Replay strategy | Hybrid: simulated (prod) + real replay (deep analysis) | Balances cost vs accuracy |
| Pruning automation | Report + suggestions in v0.1, auto-prune interface reserved | Safe incremental approach |
| Frameworks | LangGraph + OpenClaw + Hermes (all three) | User's active stack |
| Distribution | Library + CLI + Web Dashboard | Full-stack developer experience |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        TraceSHAP                            │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │Instrument│──▶│  Ingestion   │──▶│  Attribution Engine │  │
│  │  Layer   │   │   Layer      │   │  (Layer 0-4)        │  │
│  └──────────┘   └──────────────┘   └─────────┬──────────┘  │
│                                               │             │
│                                     ┌─────────▼──────────┐  │
│                                     │  Pruning Advisor   │  │
│                                     └─────────┬──────────┘  │
│                                               │             │
│                    ┌──────────────┬────────────┼──────────┐  │
│                    ▼              ▼            ▼          │  │
│               ┌────────┐   ┌──────────┐  ┌──────────┐   │  │
│               │  CLI   │   │Web Dash  │  │  API     │   │  │
│               └────────┘   └──────────┘  └──────────┘   │  │
│                                                          │  │
│  ┌───────────────────────────────────────────────────┐   │  │
│  │              Storage Layer (SQLite/PG)             │   │  │
│  └───────────────────────────────────────────────────┘   │  │
└──────────────────────────────────────────────────────────────┘

External:
  Langfuse ──(OTel/API)──▶ Ingestion Layer
  LangGraph ──(instrument)──▶ Instrument Layer
  OpenClaw  ──(instrument)──▶ Instrument Layer
  Hermes    ──(instrument)──▶ Instrument Layer
```

### 6 Core Modules

| Module | Responsibility |
|--------|---------------|
| **Instrument Layer** | Framework adapters; one-line `instrument()` for LangGraph/OpenClaw/Hermes; span collection and forwarding |
| **Ingestion Layer** | Consume OTel spans (from Langfuse API or instrument direct), reconstruct trajectories (span tree → ordered step sequence) |
| **Attribution Engine** | Core computation; Layer 0-4 layered attribution; output Shapley values + ablation impact per span |
| **Pruning Advisor** | Convert attribution scores to actionable pruning suggestions with confidence and risk assessment |
| **Storage** | Persist trajectories, attributions, models, stats (default SQLite, production PostgreSQL) |
| **Output Layer** | CLI tools + Web Dashboard + REST API |

---

## 3. Instrument Layer

Only collects spans. No analysis. Analysis happens entirely in the pipeline backend.

### Usage

```python
# LangGraph
from traceshap import instrument
app = instrument(langgraph_app, framework="langgraph")

# OpenClaw
agent = instrument(openclaw_agent, framework="openclaw")

# Hermes
agent = instrument(hermes_agent, framework="hermes")

# Auto-detect framework type
app = instrument(any_agent)
```

### Each Framework Adapter Does Three Things

1. **Hook span lifecycle** — intercept each step start/end event
2. **Normalize span attributes** — unify to TraceSHAP internal format
3. **Send to Ingestion** — two modes:
   - **Direct mode:** in-process write to pipeline (single-machine deployment)
   - **OTel export mode:** via OTel exporter to remote collector (distributed deployment)

### Standardized Span Model

```python
@dataclass
class TraceSHAPSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    span_kind: SpanKind    # LLM / TOOL / RETRIEVAL / AGENT / CUSTOM
    name: str
    input: dict
    output: dict
    start_time: datetime
    end_time: datetime
    tokens: TokenUsage | None
    cost: float | None
    metadata: dict         # framework-specific attributes preserved here
```

### Langfuse-Only Mode (No Instrument)

```python
from traceshap import TraceSHAPPipeline
pipeline = TraceSHAPPipeline(source="langfuse", langfuse_host="...", langfuse_api_key="...")
pipeline.run()  # continuously consume traces from Langfuse
```

---

## 4. Ingestion Layer

Reconstructs structured trajectories from a stream of spans.

### Data Models

```python
@dataclass
class Trajectory:
    trace_id: str
    spans: list[TraceSHAPSpan]       # time-ordered
    span_tree: SpanNode              # tree structure (parent-child)
    outcome: Outcome | None          # may arrive late
    metadata: TrajectoryMeta

@dataclass
class Outcome:
    success: bool | None
    quality_score: float | None      # 0-1
    token_cost: int
    latency_ms: int
    custom_metrics: dict

    def composite_score(self, alpha: float = 0.3) -> float:
        """Composite metric: quality - α × normalized_cost
        
        normalized_cost = token_cost / max_token_cost_in_batch (0-1 range)
        If quality_score is None, falls back to float(success) (1.0 or 0.0)
        """
        ...
```

### Three Processing Stages

| Stage | What | Trigger |
|-------|------|---------|
| **Span Buffering** | Buffer spans by trace_id, handle out-of-order arrival | Each span arrival |
| **Tree Assembly** | Rebuild span tree from parent_span_id, detect missing spans | Root span ends or timeout |
| **Outcome Binding** | Bind task result to trajectory | After trajectory assembly |

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

## 5. Attribution Engine

Layered architecture. Each layer independent, can be enabled/disabled.

```python
pipeline = TraceSHAPPipeline(layers=[0, 1, 2])         # production
pipeline = TraceSHAPPipeline(layers=[0, 1, 2, 3, 4])   # deep analysis
```

### Layer 0: Expert Rules Engine

```python
rules = [
    RepetitionRule(threshold=3, similarity=0.9),   # repeated tool calls
    LoopDetectionRule(max_cycle=2),                 # loop patterns
    NoOpRule(similarity_threshold=0.95),            # no-op spans
    CustomRule(fn=my_rule_function),                # user-defined
]
```

Output: `RuleVerdict(span_id, rule_name, severity, recommendation)`

### Layer 1: Bayesian Pre-filter

Cross-trajectory statistics per span pattern:

```
P(success | span_type=X) vs P(success | span_type ∉ trajectory)
```

Output: **lift score** (likelihood ratio) per span type. Requires minimum N historical trajectories (default N=50).

### Layer 2: Markov-approximated SHAP

Production workhorse:

1. Learn state transition model from historical trajectories: `P(stateₜ₊₁ | stateₜ, actionₜ)`
2. State = span embedding (lightweight model: all-MiniLM-L6-v2)
3. Ablation via transition model estimation (no real replay)
4. Monte Carlo Shapley sampling

```python
def markov_shap(trajectory: Trajectory, model: MarkovModel) -> dict[str, float]:
    shapley_values = {}
    for span in trajectory.spans:
        marginal_contributions = []
        for _ in range(num_samples):
            coalition = random_subset(trajectory.spans, exclude=span)
            v_with = model.estimate_outcome(coalition + [span])
            v_without = model.estimate_outcome(coalition)
            marginal_contributions.append(v_with - v_without)
        shapley_values[span.span_id] = mean(marginal_contributions)
    return shapley_values
```

Latency target: < 3 seconds for a single trajectory (~20 spans).

### Layer 3: Full Monte Carlo SHAP

Real agent replay with ablation:

1. **Ablation replay** — modify agent graph to skip target node, execute for real
2. Requires framework adapter support for `replay_without(span_ids)`
3. Async task queue execution, results written back to Storage

```python
class ReplayEngine:
    def replay_without(self, trajectory: Trajectory, ablate_spans: list[str]) -> Outcome:
        # LangGraph: modify graph, disable node
        # OpenClaw: modify action space, remove action
        # Hermes: modify tool list, remove tool
        ...
```

Cost control: sampling budget — max K replays per trace (default K=2n, n=span count).

### Layer 4: Causal SHAP

Built on Layer 3 replay data:

1. Construct SCM (Structural Causal Model), infer causal relations from span tree + temporal order
2. Compute Asymmetric Shapley Values (direction-aware, asymmetric contributions)
3. do-intervention: `do(span=skip)` vs `do(span=alternative_action)`

Extra output: **causal direction** — not just "this span matters" but "it matters because it affects these downstream spans."

### Unified Output

```python
@dataclass
class SpanAttribution:
    span_id: str
    span_name: str
    shapley_value: float              # weighted across layers
    layer_scores: dict[int, float]    # per-layer scores
    ablation_impact: float            # composite_score change on ablation
    confidence: float                 # data volume + layer count
    verdict: Verdict                  # KEEP / REVIEW / PRUNE
    causal_downstream: list[str]      # Layer 4: affected downstream spans
    evidence: list[str]               # supporting evidence descriptions
```

---

## 6. Pruning Advisor

### Decision Logic

```python
def classify_span(attr: SpanAttribution, config: PruningConfig) -> Verdict:
    if attr.ablation_impact < config.prune_threshold and attr.confidence > 0.8:
        return Verdict.PRUNE
    if attr.shapley_value > config.keep_threshold:
        return Verdict.KEEP
    return Verdict.REVIEW
```

### Output

```python
@dataclass
class PruningReport:
    trace_id: str
    timestamp: datetime
    total_spans: int
    verdicts: dict[Verdict, list[SpanAttribution]]
    estimated_savings: Savings
    risk_assessment: RiskLevel       # LOW / MEDIUM / HIGH

@dataclass
class Savings:
    token_reduction: int
    cost_reduction: float
    latency_reduction_ms: int
    quality_impact: float            # negative = degradation
```

### Auto-Pruner Interface (v0.1: interface only, no execution)

```python
class AutoPruner(ABC):
    @abstractmethod
    def apply(self, report: PruningReport, graph_definition: Any) -> Any:
        """Accept pruning report, return modified graph definition"""
        ...

class LangGraphPruner(AutoPruner):
    def apply(self, report, graph):
        for span_attr in report.verdicts[Verdict.PRUNE]:
            graph.remove_node(span_attr.span_name)
        return graph
```

### Safety Constraints

- **Never prune first/last spans** — agent entry and final output always KEEP
- **Causal chain protection** — if span A is PRUNE but is causal upstream of a KEEP span (Layer 4 data), downgrade to REVIEW
- **Batch confidence** — single-trajectory PRUNE is advisory only; span must be consistently PRUNE across N trajectories for high-confidence recommendation (default N=10)

---

## 7. Storage Layer

Default zero-config (SQLite), production switches to PostgreSQL. Same interface.

### Data Model

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│ trajectories │────▶│    spans     │────▶│  attributions    │
├─────────────┤     ├──────────────┤     ├──────────────────┤
│ trace_id PK │     │ span_id PK   │     │ span_id FK       │
│ framework   │     │ trace_id FK  │     │ trace_id FK      │
│ agent_name  │     │ parent_id    │     │ layer            │
│ outcome     │     │ span_kind    │     │ shapley_value    │
│ composite   │     │ name         │     │ ablation_impact  │
│ created_at  │     │ input/output │     │ confidence       │
│ metadata    │     │ tokens/cost  │     │ verdict          │
└─────────────┘     │ start/end    │     │ evidence         │
                    └──────────────┘     │ created_at       │
                                         └──────────────────┘

┌──────────────────┐     ┌──────────────────┐
│ pruning_reports  │     │ markov_models    │
├──────────────────┤     ├──────────────────┤
│ report_id PK     │     │ model_id PK      │
│ trace_id FK      │     │ agent_name       │
│ total_spans      │     │ framework        │
│ prune_count      │     │ model_blob       │
│ savings          │     │ training_count   │
│ risk_level       │     │ updated_at       │
│ created_at       │     └──────────────────┘
└──────────────────┘

┌──────────────────┐
│ bayesian_stats   │
├──────────────────┤
│ span_type        │
│ agent_name       │
│ success_count    │
│ total_count      │
│ lift_score       │
│ updated_at       │
└──────────────────┘
```

### Retention Policy

| Data | Write Frequency | Retention |
|------|----------------|-----------|
| trajectories + spans | Per trace completion | 30 days default, configurable |
| attributions | Per attribution run | Follows trajectory lifecycle |
| pruning_reports | Per trajectory attribution | Permanent (small volume) |
| markov_models | Layer 2 incremental training | Latest + 1 backup only |
| bayesian_stats | Layer 1 per trajectory update | Permanent (aggregate data) |

### Interface

```python
class StorageBackend(ABC):
    @abstractmethod
    async def save_trajectory(self, t: Trajectory) -> None: ...
    @abstractmethod
    async def save_attributions(self, attrs: list[SpanAttribution]) -> None: ...
    @abstractmethod
    async def query_trajectories(self, filters: QueryFilter) -> list[Trajectory]: ...
    @abstractmethod
    async def get_bayesian_stats(self, agent_name: str) -> list[BayesianStat]: ...

class SQLiteBackend(StorageBackend): ...
class PostgresBackend(StorageBackend): ...
```

---

## 8. Output Layer

### CLI

```bash
traceshap serve --source langfuse --langfuse-host=... --port 8080
traceshap analyze <trace_id> --layers 0,1,2,3,4
traceshap report <trace_id>
traceshap prune-report --agent myagent --min-trajectories 50
traceshap export <trace_id> --format json|csv|otel
```

### REST API

```
GET  /api/traces                          # list with filters
GET  /api/traces/<trace_id>               # single trajectory detail
GET  /api/traces/<trace_id>/attribution   # attribution results
GET  /api/traces/<trace_id>/pruning       # pruning suggestions
GET  /api/agents/<name>/stats             # agent-level aggregate stats
GET  /api/agents/<name>/prune-candidates  # cross-trajectory prune candidates
POST /api/analyze                         # manually trigger analysis
WS   /ws/live                             # WebSocket real-time attribution push
```

### Web Dashboard

FastAPI + React (lightweight SPA). Four core pages:

| Page | Content |
|------|---------|
| **Overview** | Agent list, trajectory counts, composite score trends, overall PRUNE rate |
| **Trajectory Detail** | Span tree visualization (tree graph), each span colored by Shapley value (green=KEEP, yellow=REVIEW, red=PRUNE), click to expand attribution detail |
| **Ablation View** | Ablation comparison: select a span, side-by-side "with/without" estimated result diff, causal chain graph (Layer 4) |
| **Prune Dashboard** | Cross-trajectory aggregate view, pruning suggestion leaderboard by agent/span_type, savings trend chart |

---

## 9. Configuration

```yaml
# traceshap.yaml
source:
  type: langfuse                    # langfuse | otel_collector | direct
  langfuse_host: https://cloud.langfuse.com
  langfuse_public_key: pk-...
  langfuse_secret_key: sk-...
  poll_interval_seconds: 10

frameworks:
  - langgraph
  - openclaw
  - hermes

attribution:
  layers: [0, 1, 2]
  layer_2:
    embedding_model: all-MiniLM-L6-v2
    num_samples: 200
  layer_3:
    enabled: false
    replay_budget_per_trace: 40
    replay_concurrency: 4
  layer_4:
    enabled: false
    requires_layer_3: true

pruning:
  prune_threshold: 0.05
  keep_threshold: 0.10
  min_trajectories: 10
  protect_first_last: true

outcome:
  source: langfuse_score
  score_name: task_success
  composite_alpha: 0.3

storage:
  backend: sqlite
  sqlite_path: ./traceshap.db
  retention_days: 30

server:
  host: 0.0.0.0
  port: 8080
  workers: 4
```

---

## 10. Deployment

| Method | Use Case | Command |
|--------|----------|---------|
| Local | Dev/debug | `traceshap serve` |
| Docker | Small team production | `docker run traceshap/traceshap` |
| Docker Compose | With PostgreSQL | `docker compose up` |
| Kubernetes | Large-scale production | Helm chart (pipeline + dashboard separated) |

### Minimal Start

```bash
pip install traceshap
traceshap init          # generate traceshap.yaml template
traceshap serve         # SQLite + local dashboard at http://localhost:8080
```

---

## 11. Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Async framework | asyncio + FastAPI |
| OTel | opentelemetry-sdk, opentelemetry-semantic-conventions |
| Langfuse | langfuse Python SDK |
| Attribution compute | numpy, scipy (Shapley), scikit-learn (Markov models) |
| Embedding | sentence-transformers (all-MiniLM-L6-v2, local) |
| Storage | SQLAlchemy (SQLite/PG unified ORM) |
| CLI | click |
| Web frontend | React + Vite + Recharts (charts) + react-d3-tree (span tree) |
| Packaging | pyproject.toml, hatchling |
| Containers | Dockerfile, docker-compose.yaml, Helm chart |

---

## 12. Attribution Methodology Background

### Why Layered Attribution

The core problem — "which steps in an agent trajectory matter?" — can be addressed at multiple levels of rigor and cost. TraceSHAP provides a progressive stack from expert heuristics through full causal inference:

```
Layer 0: Expert Rules        (instant, rule engine)
Layer 1: Bayesian Pre-filter (milliseconds, statistical patterns)
Layer 2: Markov-approx SHAP  (seconds, state transition model approximates counterfactuals)
Layer 3: Full MC SHAP        (minutes, real subset sampling + agent replay)
Layer 4: Causal SHAP         (minutes-hours, SCM + do-intervention)
```

### Theoretical Grounding

| Method | Role in TraceSHAP |
|--------|------------------|
| **SHAP (Shapley values)** | Core attribution: each span is a "player" in a cooperative game, outcome metric is the value function. Satisfies efficiency, symmetry, dummy, and additivity axioms. |
| **Naive Bayes** | Layer 1 fast screening: compute lift scores across trajectories. Conditional independence assumption is too strong for single-trajectory analysis but works for cross-trajectory pattern discovery. |
| **Markov chains** | Layer 2 core: trajectory is naturally a state transition sequence. Learned transition model serves as SHAP value function approximator — avoids expensive real replay. |
| **Causal inference** | Layer 4 deep analysis: SCM construction, do-intervention, counterfactual reasoning. Answers "if agent had NOT executed step X, what would have happened?" — the theoretically optimal attribution question. |
| **Expert heuristics** | Layer 0 prior knowledge: known anti-patterns flagged instantly, also serves as baseline for validating SHAP results. |
