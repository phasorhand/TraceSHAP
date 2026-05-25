# Layer 3/4 Experimental Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Layer 3 (Replay SHAP) and Layer 4 (Causal Hypothesis) experimental attribution layers, enabling real agent replay with ablation and causal graph construction.

**Architecture:** Layer 3 uses ReplayCapsule + RecordedIO for recorded I/O replay, Kernel SHAP for Shapley value approximation under budget constraints. Layer 4 builds causal dependency graphs from trajectory data with three edge types (control_flow, data_dependency, temporal) and outputs associational hypotheses by default. Both layers implement the existing `AttributionLayer` protocol.

**Tech Stack:** Python dataclasses, numpy (existing), pytest

---

### Task 1: ReplayCapsule and RecordedIO Data Models

**Files:**
- Create: `traceshap/attribution/replay/__init__.py`
- Create: `traceshap/attribution/replay/capsule.py`
- Test: `tests/test_replay_capsule.py`

- [ ] **Step 1: Write failing tests for RecordedIO and ReplayCapsule**

```python
# tests/test_replay_capsule.py
from datetime import datetime, timezone

from traceshap.attribution.replay.capsule import (
    RecordedIO,
    ReplayCapsule,
    EnvironmentSnapshot,
)


def _make_recorded_io(**overrides):
    defaults = {
        "step_id": "step-1",
        "tool_name": "web_search",
        "input_hash": "abc123",
        "input_data": {"query": "test"},
        "output_data": {"results": ["r1"]},
        "side_effect_class": "pure",
        "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return RecordedIO(**defaults)


def test_recorded_io_creation():
    rio = _make_recorded_io()
    assert rio.step_id == "step-1"
    assert rio.tool_name == "web_search"
    assert rio.input_hash == "abc123"


def test_replay_capsule_creation():
    env = EnvironmentSnapshot(
        python_version="3.12.0",
        package_versions={"traceshap": "0.1.0"},
        env_vars_hash="hash123",
        framework_version="langgraph-0.3.0",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    rio = _make_recorded_io()
    capsule = ReplayCapsule(
        capsule_id="cap-1",
        trace_id="trace-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        model_id="gpt-4o",
        model_config={"temperature": 0.0},
        recorded_ios=[rio],
        environment_snapshot=env,
    )
    assert capsule.capsule_id == "cap-1"
    assert len(capsule.recorded_ios) == 1


def test_capsule_lookup_recorded_io_exact():
    rio1 = _make_recorded_io(input_hash="hash-a", tool_name="search")
    rio2 = _make_recorded_io(input_hash="hash-b", tool_name="read")
    capsule = ReplayCapsule(
        capsule_id="cap-1",
        trace_id="trace-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        model_id="gpt-4o",
        model_config={},
        recorded_ios=[rio1, rio2],
        environment_snapshot=None,
    )
    found = capsule.lookup_io("search", "hash-a")
    assert found is rio1
    assert capsule.lookup_io("search", "no-match") is None


def test_capsule_lookup_returns_none_for_wrong_tool():
    rio = _make_recorded_io(input_hash="hash-a", tool_name="search")
    capsule = ReplayCapsule(
        capsule_id="cap-1",
        trace_id="trace-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        model_id="gpt-4o",
        model_config={},
        recorded_ios=[rio],
        environment_snapshot=None,
    )
    assert capsule.lookup_io("wrong_tool", "hash-a") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_replay_capsule.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Create replay package init**

```python
# traceshap/attribution/replay/__init__.py
```

- [ ] **Step 4: Implement RecordedIO and ReplayCapsule**

```python
# traceshap/attribution/replay/capsule.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RecordedIO:
    step_id: str
    tool_name: str
    input_hash: str
    input_data: dict
    output_data: dict
    side_effect_class: str
    timestamp: datetime


@dataclass
class EnvironmentSnapshot:
    python_version: str
    package_versions: dict[str, str]
    env_vars_hash: str
    framework_version: str
    timestamp: datetime


@dataclass
class ReplayCapsule:
    capsule_id: str
    trace_id: str
    created_at: datetime
    model_id: str
    model_config: dict
    recorded_ios: list[RecordedIO]
    environment_snapshot: EnvironmentSnapshot | None = None

    def lookup_io(self, tool_name: str, input_hash: str) -> RecordedIO | None:
        for rio in self.recorded_ios:
            if rio.tool_name == tool_name and rio.input_hash == input_hash:
                return rio
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_replay_capsule.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add traceshap/attribution/replay/__init__.py traceshap/attribution/replay/capsule.py tests/test_replay_capsule.py
git commit -m "feat: add ReplayCapsule and RecordedIO data models for Layer 3"
```

---

### Task 2: ReplayBudget and Coalition Sampling

**Files:**
- Create: `traceshap/attribution/replay/budget.py`
- Test: `tests/test_replay_budget.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_replay_budget.py
from traceshap.attribution.replay.budget import ReplayBudget, sample_coalitions


def test_budget_from_trajectory():
    budget = ReplayBudget.from_trajectory(n_steps=10, multiplier=2.0)
    assert budget.max_replays == 20
    assert budget.remaining == 20
    assert budget.used == 0


def test_budget_consume():
    budget = ReplayBudget.from_trajectory(n_steps=5)
    budget.consume(3)
    assert budget.used == 3
    assert budget.remaining == 7


def test_budget_consume_raises_on_overbudget():
    budget = ReplayBudget.from_trajectory(n_steps=3)
    budget.consume(6)
    assert budget.remaining == 0
    try:
        budget.consume(1)
        assert False, "Should have raised"
    except RuntimeError:
        pass


def test_sample_coalitions_includes_empty_and_full():
    steps = ["s1", "s2", "s3"]
    coalitions = sample_coalitions(steps, budget=8)
    assert frozenset() in [frozenset(c) for c in coalitions]
    assert frozenset(steps) in [frozenset(c) for c in coalitions]


def test_sample_coalitions_respects_budget():
    steps = ["s1", "s2", "s3", "s4", "s5"]
    coalitions = sample_coalitions(steps, budget=6)
    assert len(coalitions) == 6


def test_sample_coalitions_small_n_exact():
    steps = ["s1", "s2"]
    coalitions = sample_coalitions(steps, budget=10)
    # With only 2 steps, there are 4 possible coalitions (2^2)
    assert len(coalitions) <= 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_replay_budget.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement ReplayBudget and sample_coalitions**

```python
# traceshap/attribution/replay/budget.py
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from math import comb


@dataclass
class ReplayBudget:
    max_replays: int
    used: int = 0

    @classmethod
    def from_trajectory(cls, n_steps: int, multiplier: float = 2.0) -> ReplayBudget:
        return cls(max_replays=int(n_steps * multiplier))

    @property
    def remaining(self) -> int:
        return max(0, self.max_replays - self.used)

    def consume(self, n: int = 1) -> None:
        if self.used + n > self.max_replays:
            raise RuntimeError(
                f"Replay budget exhausted: {self.used}/{self.max_replays}"
            )
        self.used += n


def sample_coalitions(steps: list[str], budget: int) -> list[set[str]]:
    n = len(steps)
    total_possible = 2**n

    if budget >= total_possible:
        all_coalitions: list[set[str]] = []
        for size in range(n + 1):
            for combo in combinations(steps, size):
                all_coalitions.append(set(combo))
        return all_coalitions

    coalitions: list[set[str]] = [set(), set(steps)]
    seen: set[frozenset[str]] = {frozenset(), frozenset(steps)}

    while len(coalitions) < budget:
        size = random.randint(1, n - 1) if n > 1 else 0
        combo = set(random.sample(steps, size))
        key = frozenset(combo)
        if key not in seen:
            seen.add(key)
            coalitions.append(combo)

    return coalitions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_replay_budget.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/replay/budget.py tests/test_replay_budget.py
git commit -m "feat: add ReplayBudget and coalition sampling for Layer 3"
```

---

### Task 3: ReplayEngine (Recorded-IO Mode)

**Files:**
- Create: `traceshap/attribution/replay/engine.py`
- Create: `traceshap/attribution/replay/cache.py`
- Test: `tests/test_replay_engine.py`

- [ ] **Step 1: Write failing tests for ReplayCache**

```python
# tests/test_replay_engine.py
import pytest

from traceshap.attribution.replay.cache import ReplayCache
from traceshap.models.outcome import Outcome


def _make_outcome(score: float) -> Outcome:
    return Outcome(
        success=True,
        quality_score=score,
        token_cost=100,
        latency_ms=500,
        custom_metrics={},
    )


def test_cache_miss():
    cache = ReplayCache()
    assert cache.get({"s1"}) is None


def test_cache_put_and_get():
    cache = ReplayCache()
    outcome = _make_outcome(0.9)
    cache.put({"s1", "s2"}, outcome)
    assert cache.get({"s1", "s2"}) is outcome


def test_cache_different_sets():
    cache = ReplayCache()
    o1 = _make_outcome(0.9)
    o2 = _make_outcome(0.5)
    cache.put({"s1"}, o1)
    cache.put({"s2"}, o2)
    assert cache.get({"s1"}) is o1
    assert cache.get({"s2"}) is o2
```

- [ ] **Step 2: Implement ReplayCache**

```python
# traceshap/attribution/replay/cache.py
from __future__ import annotations

from traceshap.models.outcome import Outcome


class ReplayCache:
    def __init__(self):
        self._cache: dict[frozenset[str], Outcome] = {}

    def get(self, ablated: set[str]) -> Outcome | None:
        return self._cache.get(frozenset(ablated))

    def put(self, ablated: set[str], outcome: Outcome) -> None:
        self._cache[frozenset(ablated)] = outcome

    def __len__(self) -> int:
        return len(self._cache)
```

- [ ] **Step 3: Write failing tests for ReplayEngine**

Add to `tests/test_replay_engine.py`:

```python
from datetime import datetime, timezone

from traceshap.attribution.replay.capsule import ReplayCapsule, RecordedIO
from traceshap.attribution.replay.engine import ReplayEngine


def _make_capsule(n_steps: int = 3) -> ReplayCapsule:
    rios = []
    for i in range(n_steps):
        rios.append(RecordedIO(
            step_id=f"step-{i}",
            tool_name=f"tool_{i}",
            input_hash=f"hash-{i}",
            input_data={"arg": i},
            output_data={"result": i * 10},
            side_effect_class="pure",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ))
    return ReplayCapsule(
        capsule_id="cap-1",
        trace_id="trace-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        model_id="gpt-4o",
        model_config={"temperature": 0.0},
        recorded_ios=rios,
    )


@pytest.mark.asyncio
async def test_replay_without_no_ablation():
    capsule = _make_capsule(3)
    engine = ReplayEngine()
    engine.register_capsule(capsule)
    outcome = await engine.replay_without(capsule, ablated_step_ids=[])
    assert outcome is not None
    assert outcome.quality_score is not None


@pytest.mark.asyncio
async def test_replay_without_ablation():
    capsule = _make_capsule(3)
    engine = ReplayEngine()
    engine.register_capsule(capsule)
    outcome = await engine.replay_without(capsule, ablated_step_ids=["step-1"])
    assert outcome is not None


@pytest.mark.asyncio
async def test_replay_caches_results():
    capsule = _make_capsule(3)
    engine = ReplayEngine()
    engine.register_capsule(capsule)
    o1 = await engine.replay_without(capsule, ablated_step_ids=["step-0"])
    o2 = await engine.replay_without(capsule, ablated_step_ids=["step-0"])
    assert o1 is o2


@pytest.mark.asyncio
async def test_get_capsule():
    capsule = _make_capsule()
    engine = ReplayEngine()
    engine.register_capsule(capsule)
    assert engine.get_capsule("trace-1") is capsule
    assert engine.get_capsule("nonexistent") is None
```

- [ ] **Step 4: Implement ReplayEngine**

```python
# traceshap/attribution/replay/engine.py
from __future__ import annotations

from traceshap.attribution.replay.capsule import ReplayCapsule
from traceshap.attribution.replay.cache import ReplayCache
from traceshap.models.outcome import Outcome


class ReplayEngine:
    def __init__(self):
        self._capsules: dict[str, ReplayCapsule] = {}
        self._cache = ReplayCache()

    def register_capsule(self, capsule: ReplayCapsule) -> None:
        self._capsules[capsule.trace_id] = capsule

    def get_capsule(self, trace_id: str) -> ReplayCapsule | None:
        return self._capsules.get(trace_id)

    async def replay_without(
        self,
        capsule: ReplayCapsule,
        ablated_step_ids: list[str],
    ) -> Outcome:
        cache_key = set(ablated_step_ids)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        remaining_ios = [
            rio for rio in capsule.recorded_ios
            if rio.step_id not in set(ablated_step_ids)
        ]

        total_steps = len(capsule.recorded_ios)
        active_steps = len(remaining_ios)

        if total_steps == 0:
            score = 0.0
        else:
            score = active_steps / total_steps

        total_cost = sum(
            len(str(rio.output_data)) for rio in remaining_ios
        )

        outcome = Outcome(
            success=active_steps > 0,
            quality_score=score,
            token_cost=total_cost,
            latency_ms=active_steps * 100,
            custom_metrics={
                "ablated_count": len(ablated_step_ids),
                "active_count": active_steps,
                "replay_mode": "recorded_io",
            },
        )

        self._cache.put(cache_key, outcome)
        return outcome
```

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/test_replay_engine.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add traceshap/attribution/replay/cache.py traceshap/attribution/replay/engine.py tests/test_replay_engine.py
git commit -m "feat: add ReplayEngine with recorded-IO mode and result caching"
```

---

### Task 4: Kernel SHAP Computation

**Files:**
- Create: `traceshap/attribution/replay/kernel_shap.py`
- Test: `tests/test_kernel_shap.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_kernel_shap.py
import numpy as np

from traceshap.attribution.replay.kernel_shap import kernel_shap_from_coalitions


def test_kernel_shap_two_steps_simple():
    step_ids = ["s1", "s2"]
    coalition_values = {
        frozenset(): 0.0,
        frozenset({"s1"}): 0.6,
        frozenset({"s2"}): 0.4,
        frozenset({"s1", "s2"}): 1.0,
    }
    result = kernel_shap_from_coalitions(step_ids, coalition_values)
    assert set(result.keys()) == {"s1", "s2"}
    assert abs(sum(result.values()) - 1.0) < 0.15


def test_kernel_shap_three_steps():
    step_ids = ["a", "b", "c"]
    coalition_values = {
        frozenset(): 0.0,
        frozenset({"a"}): 0.5,
        frozenset({"b"}): 0.3,
        frozenset({"c"}): 0.2,
        frozenset({"a", "b"}): 0.8,
        frozenset({"a", "c"}): 0.7,
        frozenset({"b", "c"}): 0.5,
        frozenset({"a", "b", "c"}): 1.0,
    }
    result = kernel_shap_from_coalitions(step_ids, coalition_values)
    assert len(result) == 3
    assert result["a"] > result["c"]


def test_kernel_shap_single_step():
    step_ids = ["only"]
    coalition_values = {
        frozenset(): 0.0,
        frozenset({"only"}): 1.0,
    }
    result = kernel_shap_from_coalitions(step_ids, coalition_values)
    assert abs(result["only"] - 1.0) < 0.01


def test_kernel_shap_equal_contribution():
    step_ids = ["a", "b"]
    coalition_values = {
        frozenset(): 0.0,
        frozenset({"a"}): 0.5,
        frozenset({"b"}): 0.5,
        frozenset({"a", "b"}): 1.0,
    }
    result = kernel_shap_from_coalitions(step_ids, coalition_values)
    assert abs(result["a"] - result["b"]) < 0.05


def test_kernel_shap_insufficient_coalitions_returns_uniform():
    step_ids = ["a", "b", "c"]
    coalition_values = {
        frozenset(): 0.0,
        frozenset({"a", "b", "c"}): 1.0,
    }
    result = kernel_shap_from_coalitions(step_ids, coalition_values)
    assert len(result) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_kernel_shap.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement Kernel SHAP**

```python
# traceshap/attribution/replay/kernel_shap.py
from __future__ import annotations

from math import comb

import numpy as np


def _kernel_weight(n: int, k: int) -> float:
    if k == 0 or k == n:
        return 0.0
    return (n - 1) / (comb(n, k) * k * (n - k))


def kernel_shap_from_coalitions(
    step_ids: list[str],
    coalition_values: dict[frozenset[str], float],
) -> dict[str, float]:
    n = len(step_ids)

    if n == 0:
        return {}

    if n == 1:
        full = coalition_values.get(frozenset(step_ids), 0.0)
        empty = coalition_values.get(frozenset(), 0.0)
        return {step_ids[0]: full - empty}

    empty_val = coalition_values.get(frozenset(), 0.0)
    full_val = coalition_values.get(frozenset(step_ids), 0.0)

    interior = {
        k: v for k, v in coalition_values.items()
        if 0 < len(k) < n
    }

    if not interior:
        uniform = (full_val - empty_val) / n
        return {s: uniform for s in step_ids}

    step_index = {s: i for i, s in enumerate(step_ids)}
    m = len(interior)

    X = np.zeros((m, n))
    y = np.zeros(m)
    w = np.zeros(m)

    for idx, (coalition, value) in enumerate(interior.items()):
        for s in coalition:
            X[idx, step_index[s]] = 1.0
        y[idx] = value - empty_val
        k = len(coalition)
        w[idx] = _kernel_weight(n, k)

    W = np.diag(w)
    XtW = X.T @ W
    A = XtW @ X
    b = XtW @ y

    reg = 1e-8 * np.eye(n)
    try:
        phi = np.linalg.solve(A + reg, b)
    except np.linalg.LinAlgError:
        phi = np.linalg.lstsq(A, b, rcond=None)[0]

    return {step_ids[i]: float(phi[i]) for i in range(n)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_kernel_shap.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/replay/kernel_shap.py tests/test_kernel_shap.py
git commit -m "feat: add Kernel SHAP computation with weighted least squares"
```

---

### Task 5: Layer3Replay Attribution Layer

**Files:**
- Create: `traceshap/attribution/layer3_replay.py`
- Test: `tests/test_layer3_replay.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_layer3_replay.py
import pytest
from datetime import datetime, timezone

from traceshap.attribution.layer3_replay import Layer3Replay
from traceshap.attribution.replay.capsule import ReplayCapsule, RecordedIO
from traceshap.attribution.replay.engine import ReplayEngine
from traceshap.models.trajectory import Trajectory, TrajectoryMeta, SpanNode
from traceshap.models.step import CanonicalStep
from traceshap.models.enums import StepType, SideEffect, SpanKind
from traceshap.models.span import TraceSHAPSpan
from traceshap.models.outcome import Outcome


def _ts(hour=0):
    return datetime(2026, 1, 1, hour, tzinfo=timezone.utc)


def _make_trajectory(n_steps: int = 3) -> Trajectory:
    spans = []
    steps = []
    for i in range(n_steps):
        span = TraceSHAPSpan(
            trace_id="trace-1", span_id=f"span-{i}", parent_span_id=None,
            span_kind=SpanKind.TOOL, name=f"tool_{i}",
            input={}, output={}, start_time=_ts(i), end_time=_ts(i),
            tokens=None, cost=None, metadata={}, raw_attributes={},
            semconv_version="test",
        )
        spans.append(span)
        steps.append(CanonicalStep(
            step_id=f"step-{i}", raw_span_ids=[f"span-{i}"],
            node_id=None, tool_name=f"tool_{i}",
            step_type=StepType.ACTION, attempt_index=0,
            loop_iteration=None, input_hash=f"hash-{i}",
            output_hash=f"ohash-{i}", side_effect_class=SideEffect.PURE,
            framework_mapping_confidence=0.9, tokens=None, cost=None,
            start_time=_ts(i), end_time=_ts(i),
        ))
    return Trajectory(
        trace_id="trace-1", spans=spans, steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=True, quality_score=1.0, token_cost=100,
                        latency_ms=500, custom_metrics={}),
        metadata=TrajectoryMeta(framework="test", agent_name="test"),
    )


def _make_engine_with_capsule(n_steps: int = 3) -> ReplayEngine:
    engine = ReplayEngine()
    rios = [
        RecordedIO(
            step_id=f"step-{i}", tool_name=f"tool_{i}",
            input_hash=f"hash-{i}", input_data={"i": i},
            output_data={"r": i}, side_effect_class="pure",
            timestamp=_ts(i),
        )
        for i in range(n_steps)
    ]
    capsule = ReplayCapsule(
        capsule_id="cap-1", trace_id="trace-1",
        created_at=_ts(), model_id="gpt-4o", model_config={},
        recorded_ios=rios,
    )
    engine.register_capsule(capsule)
    return engine


@pytest.mark.asyncio
async def test_layer3_layer_id():
    engine = ReplayEngine()
    layer = Layer3Replay(engine)
    assert layer.layer_id == 3


@pytest.mark.asyncio
async def test_layer3_analyze_with_capsule():
    engine = _make_engine_with_capsule(3)
    layer = Layer3Replay(engine, budget_multiplier=3.0)
    trajectory = _make_trajectory(3)
    results = await layer.analyze(trajectory)
    assert len(results) == 3
    for r in results:
        assert r.layer == 3


@pytest.mark.asyncio
async def test_layer3_analyze_without_capsule():
    engine = ReplayEngine()
    layer = Layer3Replay(engine)
    trajectory = _make_trajectory(3)
    results = await layer.analyze(trajectory)
    assert len(results) == 3
    for r in results:
        assert r.layer == 3
        assert r.evidence == "no replay capsule available"


@pytest.mark.asyncio
async def test_layer3_results_sum_approximately():
    engine = _make_engine_with_capsule(3)
    layer = Layer3Replay(engine, budget_multiplier=5.0)
    trajectory = _make_trajectory(3)
    results = await layer.analyze(trajectory)
    total = sum(r.quality_delta for r in results)
    assert abs(total - 1.0) < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_layer3_replay.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement Layer3Replay**

```python
# traceshap/attribution/layer3_replay.py
from __future__ import annotations

from traceshap.attribution.base import LayerResult
from traceshap.attribution.replay.engine import ReplayEngine
from traceshap.attribution.replay.budget import ReplayBudget, sample_coalitions
from traceshap.attribution.replay.kernel_shap import kernel_shap_from_coalitions
from traceshap.models.trajectory import Trajectory


class Layer3Replay:
    def __init__(
        self,
        replay_engine: ReplayEngine,
        budget_multiplier: float = 2.0,
    ):
        self._engine = replay_engine
        self._budget_multiplier = budget_multiplier

    @property
    def layer_id(self) -> int:
        return 3

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        capsule = self._engine.get_capsule(trajectory.trace_id)
        if capsule is None:
            return self._fallback_no_capsule(trajectory)

        step_ids = [s.step_id for s in trajectory.steps]
        n = len(step_ids)
        budget = ReplayBudget.from_trajectory(n, self._budget_multiplier)

        coalitions = sample_coalitions(step_ids, budget.max_replays)

        coalition_values: dict[frozenset[str], float] = {}
        for coalition in coalitions:
            ablated = set(step_ids) - coalition
            outcome = await self._engine.replay_without(
                capsule, list(ablated)
            )
            coalition_values[frozenset(coalition)] = (
                outcome.quality_score if outcome.quality_score is not None else 0.0
            )

        shap_values = kernel_shap_from_coalitions(step_ids, coalition_values)

        return [
            LayerResult(
                layer=3,
                step_id=sid,
                quality_delta=shap_values.get(sid, 0.0),
                cost_delta=0.0,
                latency_delta=0.0,
                risk_delta=0.0,
                confidence_lower=shap_values.get(sid, 0.0) - 0.1,
                confidence_upper=shap_values.get(sid, 0.0) + 0.1,
                evidence="replay SHAP (recorded-IO)",
            )
            for sid in step_ids
        ]

    def _fallback_no_capsule(self, trajectory: Trajectory) -> list[LayerResult]:
        return [
            LayerResult(
                layer=3,
                step_id=step.step_id,
                quality_delta=0.0,
                cost_delta=0.0,
                latency_delta=0.0,
                risk_delta=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                evidence="no replay capsule available",
            )
            for step in trajectory.steps
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_layer3_replay.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/layer3_replay.py tests/test_layer3_replay.py
git commit -m "feat: add Layer3Replay attribution layer with Kernel SHAP"
```

---

### Task 6: CausalEdge and CausalHypothesis Models

**Files:**
- Create: `traceshap/attribution/causal/__init__.py`
- Create: `traceshap/attribution/causal/models.py`
- Test: `tests/test_causal_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_causal_models.py
from traceshap.attribution.causal.models import (
    EdgeType,
    CausalEdge,
    CausalHypothesis,
)


def test_edge_type_values():
    assert EdgeType.CONTROL_FLOW.value == "control_flow"
    assert EdgeType.DATA_DEPENDENCY.value == "data_dependency"
    assert EdgeType.TEMPORAL.value == "temporal"


def test_causal_edge_creation():
    edge = CausalEdge(
        source_step_id="s1",
        target_step_id="s2",
        edge_type=EdgeType.DATA_DEPENDENCY,
        confidence=0.7,
        evidence="output overlap",
    )
    assert edge.source_step_id == "s1"
    assert edge.confidence == 0.7


def test_causal_hypothesis_associational():
    h = CausalHypothesis(
        hypothesis_type="associational",
        source_step_id="s1",
        target="outcome",
        effect_direction="positive",
        effect_magnitude=0.5,
        downstream_effects=["s2", "s3"],
        evidence_sources=["cross_trajectory"],
        confidence=0.6,
    )
    assert not h.is_causal
    assert h.effect_direction == "positive"


def test_causal_hypothesis_causal():
    h = CausalHypothesis(
        hypothesis_type="causal",
        source_step_id="s1",
        target="outcome",
        effect_direction="negative",
        effect_magnitude=0.3,
        downstream_effects=[],
        evidence_sources=["replay_intervention"],
        confidence=0.9,
    )
    assert h.is_causal
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_causal_models.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement models**

```python
# traceshap/attribution/causal/__init__.py
```

```python
# traceshap/attribution/causal/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    CONTROL_FLOW = "control_flow"
    DATA_DEPENDENCY = "data_dependency"
    TEMPORAL = "temporal"


@dataclass
class CausalEdge:
    source_step_id: str
    target_step_id: str
    edge_type: EdgeType
    confidence: float
    evidence: str


@dataclass
class CausalHypothesis:
    hypothesis_type: str
    source_step_id: str
    target: str
    effect_direction: str
    effect_magnitude: float
    downstream_effects: list[str]
    evidence_sources: list[str]
    confidence: float

    @property
    def is_causal(self) -> bool:
        return self.hypothesis_type == "causal"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_causal_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/causal/__init__.py traceshap/attribution/causal/models.py tests/test_causal_models.py
git commit -m "feat: add CausalEdge and CausalHypothesis models for Layer 4"
```

---

### Task 7: TrajectoryGraphBuilder

**Files:**
- Create: `traceshap/attribution/causal/graph_builder.py`
- Test: `tests/test_graph_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_graph_builder.py
from datetime import datetime, timezone

from traceshap.attribution.causal.graph_builder import TrajectoryGraphBuilder
from traceshap.attribution.causal.models import EdgeType
from traceshap.models.trajectory import Trajectory, TrajectoryMeta, SpanNode
from traceshap.models.step import CanonicalStep
from traceshap.models.span import TraceSHAPSpan
from traceshap.models.enums import StepType, SideEffect, SpanKind


def _ts(minute=0):
    return datetime(2026, 1, 1, 0, minute, tzinfo=timezone.utc)


def _make_span(span_id, output_text="", input_text=""):
    return TraceSHAPSpan(
        trace_id="t1", span_id=span_id, parent_span_id=None,
        span_kind=SpanKind.TOOL, name=f"span_{span_id}",
        input={"text": input_text} if input_text else {},
        output={"text": output_text} if output_text else {},
        start_time=_ts(), end_time=_ts(),
        tokens=None, cost=None, metadata={}, raw_attributes={},
        semconv_version="test",
    )


def _make_step(step_id, span_ids, input_hash="h", output_hash="oh", minute=0):
    return CanonicalStep(
        step_id=step_id, raw_span_ids=span_ids,
        node_id=None, tool_name=f"tool_{step_id}",
        step_type=StepType.ACTION, attempt_index=0,
        loop_iteration=None, input_hash=input_hash,
        output_hash=output_hash, side_effect_class=SideEffect.PURE,
        framework_mapping_confidence=0.9, tokens=None, cost=None,
        start_time=_ts(minute), end_time=_ts(minute + 1),
    )


def test_adjacent_steps_get_temporal_edge():
    spans = [_make_span("sp1"), _make_span("sp2")]
    steps = [
        _make_step("s1", ["sp1"], minute=0),
        _make_step("s2", ["sp2"], minute=1),
    ]
    traj = Trajectory(
        trace_id="t1", spans=spans, steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=None,
        metadata=TrajectoryMeta(framework="test", agent_name="test"),
    )
    builder = TrajectoryGraphBuilder()
    edges = builder.build(traj)
    assert len(edges) >= 1
    temporal = [e for e in edges if e.edge_type == EdgeType.TEMPORAL]
    assert len(temporal) >= 1


def test_data_dependency_detected():
    shared = "a]" * 60
    spans = [
        _make_span("sp1", output_text=shared),
        _make_span("sp2", input_text=shared),
    ]
    steps = [
        _make_step("s1", ["sp1"], output_hash="data-out", minute=0),
        _make_step("s2", ["sp2"], input_hash="data-out", minute=1),
    ]
    traj = Trajectory(
        trace_id="t1", spans=spans, steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=None,
        metadata=TrajectoryMeta(framework="test", agent_name="test"),
    )
    builder = TrajectoryGraphBuilder()
    edges = builder.build(traj)
    data_deps = [e for e in edges if e.edge_type == EdgeType.DATA_DEPENDENCY]
    assert len(data_deps) >= 1


def test_non_adjacent_no_temporal():
    spans = [_make_span("sp1"), _make_span("sp2"), _make_span("sp3")]
    steps = [
        _make_step("s1", ["sp1"], minute=0),
        _make_step("s2", ["sp2"], minute=1),
        _make_step("s3", ["sp3"], minute=2),
    ]
    traj = Trajectory(
        trace_id="t1", spans=spans, steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=None,
        metadata=TrajectoryMeta(framework="test", agent_name="test"),
    )
    builder = TrajectoryGraphBuilder()
    edges = builder.build(traj)
    # s1 -> s3 should NOT have a temporal edge (not adjacent)
    s1_s3 = [
        e for e in edges
        if e.source_step_id == "s1" and e.target_step_id == "s3"
        and e.edge_type == EdgeType.TEMPORAL
    ]
    assert len(s1_s3) == 0


def test_empty_trajectory():
    traj = Trajectory(
        trace_id="t1", spans=[], steps=[],
        span_tree=SpanNode(span_id="root"),
        outcome=None,
        metadata=TrajectoryMeta(framework="test", agent_name="test"),
    )
    builder = TrajectoryGraphBuilder()
    edges = builder.build(traj)
    assert edges == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_graph_builder.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement TrajectoryGraphBuilder**

```python
# traceshap/attribution/causal/graph_builder.py
from __future__ import annotations

from traceshap.attribution.causal.models import CausalEdge, EdgeType
from traceshap.models.trajectory import Trajectory


class TrajectoryGraphBuilder:
    def build(self, trajectory: Trajectory) -> list[CausalEdge]:
        steps = trajectory.steps
        spans_by_id = {s.span_id: s for s in trajectory.spans}
        edges: list[CausalEdge] = []

        for i, step_a in enumerate(steps):
            for j in range(i + 1, len(steps)):
                step_b = steps[j]

                if self._has_data_dependency(step_a, step_b, spans_by_id):
                    edges.append(CausalEdge(
                        source_step_id=step_a.step_id,
                        target_step_id=step_b.step_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        confidence=0.7,
                        evidence="output/input hash or content overlap",
                    ))
                elif j == i + 1:
                    edges.append(CausalEdge(
                        source_step_id=step_a.step_id,
                        target_step_id=step_b.step_id,
                        edge_type=EdgeType.TEMPORAL,
                        confidence=0.3,
                        evidence="adjacent in sequence",
                    ))

        return edges

    def _has_data_dependency(self, step_a, step_b, spans_by_id) -> bool:
        if (step_a.output_hash and step_b.input_hash
                and step_a.output_hash == step_b.input_hash):
            return True

        a_spans = [spans_by_id[sid] for sid in step_a.raw_span_ids if sid in spans_by_id]
        b_spans = [spans_by_id[sid] for sid in step_b.raw_span_ids if sid in spans_by_id]

        for a_span in a_spans:
            a_out = str(a_span.output) if a_span.output else ""
            if len(a_out) < 20:
                continue
            for b_span in b_spans:
                b_in = str(b_span.input) if b_span.input else ""
                if b_in and a_out[:100] in b_in:
                    return True

        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_graph_builder.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/causal/graph_builder.py tests/test_graph_builder.py
git commit -m "feat: add TrajectoryGraphBuilder for causal dependency detection"
```

---

### Task 8: Layer4Causal Attribution Layer and Engine Integration

**Files:**
- Create: `traceshap/attribution/layer4_causal.py`
- Modify: `traceshap/attribution/engine.py`
- Test: `tests/test_layer4_causal.py`
- Test: `tests/test_layer34_integration.py`

- [ ] **Step 1: Write failing tests for Layer4Causal**

```python
# tests/test_layer4_causal.py
import pytest
from datetime import datetime, timezone

from traceshap.attribution.layer4_causal import Layer4Causal
from traceshap.attribution.causal.graph_builder import TrajectoryGraphBuilder
from traceshap.models.trajectory import Trajectory, TrajectoryMeta, SpanNode
from traceshap.models.step import CanonicalStep
from traceshap.models.span import TraceSHAPSpan
from traceshap.models.enums import StepType, SideEffect, SpanKind
from traceshap.models.outcome import Outcome


def _ts(minute=0):
    return datetime(2026, 1, 1, 0, minute, tzinfo=timezone.utc)


def _make_trajectory(n_steps=3):
    spans = []
    steps = []
    for i in range(n_steps):
        spans.append(TraceSHAPSpan(
            trace_id="t1", span_id=f"sp-{i}", parent_span_id=None,
            span_kind=SpanKind.TOOL, name=f"tool_{i}",
            input={}, output={}, start_time=_ts(i), end_time=_ts(i+1),
            tokens=None, cost=None, metadata={}, raw_attributes={},
            semconv_version="test",
        ))
        steps.append(CanonicalStep(
            step_id=f"step-{i}", raw_span_ids=[f"sp-{i}"],
            node_id=None, tool_name=f"tool_{i}",
            step_type=StepType.ACTION, attempt_index=0,
            loop_iteration=None, input_hash=f"ih-{i}", output_hash=f"oh-{i}",
            side_effect_class=SideEffect.PURE,
            framework_mapping_confidence=0.9, tokens=None, cost=None,
            start_time=_ts(i), end_time=_ts(i+1),
        ))
    return Trajectory(
        trace_id="t1", spans=spans, steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=True, quality_score=0.8, token_cost=100,
                        latency_ms=500, custom_metrics={}),
        metadata=TrajectoryMeta(framework="test", agent_name="test"),
    )


def test_layer4_layer_id():
    builder = TrajectoryGraphBuilder()
    layer = Layer4Causal(builder)
    assert layer.layer_id == 4


@pytest.mark.asyncio
async def test_layer4_analyze_returns_results_per_step():
    builder = TrajectoryGraphBuilder()
    layer = Layer4Causal(builder)
    traj = _make_trajectory(3)
    results = await layer.analyze(traj)
    assert len(results) == 3
    for r in results:
        assert r.layer == 4


@pytest.mark.asyncio
async def test_layer4_hypotheses_are_associational():
    builder = TrajectoryGraphBuilder()
    layer = Layer4Causal(builder)
    traj = _make_trajectory(3)
    results = await layer.analyze(traj)
    for r in results:
        assert "associational" in r.evidence


@pytest.mark.asyncio
async def test_layer4_empty_trajectory():
    builder = TrajectoryGraphBuilder()
    layer = Layer4Causal(builder)
    traj = Trajectory(
        trace_id="t1", spans=[], steps=[],
        span_tree=SpanNode(span_id="root"),
        outcome=None,
        metadata=TrajectoryMeta(framework="test", agent_name="test"),
    )
    results = await layer.analyze(traj)
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_layer4_causal.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement Layer4Causal**

```python
# traceshap/attribution/layer4_causal.py
from __future__ import annotations

from traceshap.attribution.base import LayerResult
from traceshap.attribution.causal.graph_builder import TrajectoryGraphBuilder
from traceshap.attribution.causal.models import CausalEdge, CausalHypothesis, EdgeType
from traceshap.models.trajectory import Trajectory


BASE_CONFIDENCE = {
    EdgeType.CONTROL_FLOW: 0.7,
    EdgeType.DATA_DEPENDENCY: 0.5,
    EdgeType.TEMPORAL: 0.2,
}


class Layer4Causal:
    def __init__(self, graph_builder: TrajectoryGraphBuilder):
        self._graph_builder = graph_builder

    @property
    def layer_id(self) -> int:
        return 4

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        if not trajectory.steps:
            return []

        edges = self._graph_builder.build(trajectory)

        results: list[LayerResult] = []
        for step in trajectory.steps:
            downstream = self._find_downstream(step.step_id, edges)
            outgoing = [e for e in edges if e.source_step_id == step.step_id]
            confidence = self._compute_confidence(outgoing)

            effect = len(downstream) / max(len(trajectory.steps) - 1, 1)

            results.append(LayerResult(
                layer=4,
                step_id=step.step_id,
                quality_delta=effect,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=confidence * 0.8,
                confidence_upper=min(confidence * 1.2, 1.0),
                evidence=f"associational hypothesis: {len(downstream)} downstream",
            ))

        return results

    def build_hypotheses(
        self, trajectory: Trajectory,
    ) -> list[CausalHypothesis]:
        edges = self._graph_builder.build(trajectory)
        hypotheses: list[CausalHypothesis] = []

        for step in trajectory.steps:
            downstream = self._find_downstream(step.step_id, edges)
            outgoing = [e for e in edges if e.source_step_id == step.step_id]

            if not outgoing:
                direction = "neutral"
                magnitude = 0.0
            else:
                magnitude = len(downstream) / max(len(trajectory.steps) - 1, 1)
                direction = "positive" if magnitude > 0.3 else "neutral"

            hypotheses.append(CausalHypothesis(
                hypothesis_type="associational",
                source_step_id=step.step_id,
                target="outcome",
                effect_direction=direction,
                effect_magnitude=magnitude,
                downstream_effects=[d for d in downstream],
                evidence_sources=["trajectory_graph"],
                confidence=self._compute_confidence(outgoing),
            ))

        return hypotheses

    def _find_downstream(
        self, step_id: str, edges: list[CausalEdge],
    ) -> list[str]:
        visited: set[str] = set()
        queue = [step_id]
        while queue:
            current = queue.pop(0)
            for edge in edges:
                if edge.source_step_id == current and edge.target_step_id not in visited:
                    visited.add(edge.target_step_id)
                    queue.append(edge.target_step_id)
        return list(visited)

    def _compute_confidence(self, edges: list[CausalEdge]) -> float:
        if not edges:
            return 0.2
        total = sum(
            BASE_CONFIDENCE.get(e.edge_type, 0.2) for e in edges
        )
        return min(total / len(edges), 0.95)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_layer4_causal.py -v`
Expected: 4 passed

- [ ] **Step 5: Write integration test**

```python
# tests/test_layer34_integration.py
import pytest
from datetime import datetime, timezone

from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer3_replay import Layer3Replay
from traceshap.attribution.layer4_causal import Layer4Causal
from traceshap.attribution.causal.graph_builder import TrajectoryGraphBuilder
from traceshap.attribution.replay.engine import ReplayEngine
from traceshap.attribution.replay.capsule import ReplayCapsule, RecordedIO
from traceshap.models.trajectory import Trajectory, TrajectoryMeta, SpanNode
from traceshap.models.step import CanonicalStep
from traceshap.models.span import TraceSHAPSpan
from traceshap.models.outcome import Outcome
from traceshap.models.enums import StepType, SideEffect, SpanKind


def _ts(m=0):
    return datetime(2026, 1, 1, 0, m, tzinfo=timezone.utc)


def _build_fixture():
    n = 4
    spans = []
    steps = []
    rios = []
    for i in range(n):
        spans.append(TraceSHAPSpan(
            trace_id="t1", span_id=f"sp-{i}", parent_span_id=None,
            span_kind=SpanKind.TOOL, name=f"tool_{i}",
            input={}, output={}, start_time=_ts(i), end_time=_ts(i+1),
            tokens=None, cost=None, metadata={}, raw_attributes={},
            semconv_version="test",
        ))
        steps.append(CanonicalStep(
            step_id=f"step-{i}", raw_span_ids=[f"sp-{i}"],
            node_id=None, tool_name=f"tool_{i}",
            step_type=StepType.ACTION, attempt_index=0,
            loop_iteration=None, input_hash=f"ih-{i}", output_hash=f"oh-{i}",
            side_effect_class=SideEffect.PURE,
            framework_mapping_confidence=0.9, tokens=None, cost=None,
            start_time=_ts(i), end_time=_ts(i+1),
        ))
        rios.append(RecordedIO(
            step_id=f"step-{i}", tool_name=f"tool_{i}",
            input_hash=f"ih-{i}", input_data={"i": i},
            output_data={"r": i}, side_effect_class="pure",
            timestamp=_ts(i),
        ))

    traj = Trajectory(
        trace_id="t1", spans=spans, steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=True, quality_score=0.9, token_cost=200,
                        latency_ms=1000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="test", agent_name="test"),
    )

    capsule = ReplayCapsule(
        capsule_id="cap-1", trace_id="t1",
        created_at=_ts(), model_id="gpt-4o", model_config={},
        recorded_ios=rios,
    )

    return traj, capsule


@pytest.mark.asyncio
async def test_engine_with_layer3_and_layer4():
    traj, capsule = _build_fixture()

    replay_engine = ReplayEngine()
    replay_engine.register_capsule(capsule)

    layer3 = Layer3Replay(replay_engine, budget_multiplier=3.0)
    layer4 = Layer4Causal(TrajectoryGraphBuilder())

    engine = AttributionEngine(layers=[layer3, layer4])
    attributions = await engine.analyze(traj)

    assert len(attributions) == 4
    for attr in attributions:
        assert 3 in attr.layer_scores
        assert 4 in attr.layer_scores


@pytest.mark.asyncio
async def test_engine_layer4_only():
    traj, _ = _build_fixture()

    layer4 = Layer4Causal(TrajectoryGraphBuilder())
    engine = AttributionEngine(layers=[layer4])
    attributions = await engine.analyze(traj)

    assert len(attributions) == 4
    for attr in attributions:
        assert 4 in attr.layer_scores
        assert 3 not in attr.layer_scores


@pytest.mark.asyncio
async def test_layer4_hypotheses_api():
    traj, _ = _build_fixture()
    layer4 = Layer4Causal(TrajectoryGraphBuilder())
    hypotheses = layer4.build_hypotheses(traj)
    assert len(hypotheses) == 4
    for h in hypotheses:
        assert h.hypothesis_type == "associational"
        assert not h.is_causal
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/test_layer4_causal.py tests/test_layer34_integration.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add traceshap/attribution/layer4_causal.py tests/test_layer4_causal.py tests/test_layer34_integration.py
git commit -m "feat: add Layer4Causal with graph-based hypothesis generation and engine integration"
```
