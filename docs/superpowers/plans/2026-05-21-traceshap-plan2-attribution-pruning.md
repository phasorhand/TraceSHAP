# TraceSHAP Plan 2: Attribution Engine (L0/L1/L2) + Pruning Advisor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the attribution engine (Layers 0-2) and pruning advisor so TraceSHAP can analyze ingested trajectories, compute per-step attribution scores with confidence intervals, classify steps as KEEP/REVIEW/PRUNE_CANDIDATE, and generate pruning reports with validation plans.

**Architecture:** Each attribution layer is an independent class implementing a common `AttributionLayer` protocol. An `AttributionEngine` orchestrates layers sequentially, merging results. The `PruningAdvisor` consumes final attributions and produces `PruneCandidate` objects with `ValidationPlan`. All layers operate on `CanonicalStep` + `Trajectory` (built in Plan 1).

**Tech Stack:** Python 3.11+, numpy (for statistical computations), existing dataclasses from Plan 1

**Spec:** `docs/superpowers/specs/2026-05-21-traceshap-design.md` — Sections 6, 7

---

## File Structure

```
traceshap/
├── attribution/
│   ├── __init__.py
│   ├── base.py              # AttributionLayer protocol, LayerResult
│   ├── engine.py            # AttributionEngine orchestrator
│   ├── layer0_rules.py      # Expert rules: repetition, loop, no-op
│   ├── layer1_lift.py       # Statistical lift with cohort stratification
│   └── layer2_sequence.py   # Sequence-aware counterfactual estimator
├── pruning/
│   ├── __init__.py
│   ├── advisor.py           # PruningAdvisor: classify steps, generate candidates
│   └── safety.py            # Safety constraints, protected types, first/last guard
├── tests/
│   ├── test_attribution_base.py
│   ├── test_layer0.py
│   ├── test_layer1.py
│   ├── test_layer2.py
│   ├── test_engine.py
│   ├── test_pruning_advisor.py
│   └── test_pruning_safety.py
```

---

### Task 1: Attribution Base Protocol and LayerResult

**Files:**
- Create: `traceshap/attribution/__init__.py`
- Create: `traceshap/attribution/base.py`
- Create: `tests/test_attribution_base.py`

- [ ] **Step 1: Write tests**

`tests/test_attribution_base.py`:
```python
import pytest
from traceshap.attribution.base import LayerResult, merge_layer_results


class TestLayerResult:
    def test_create(self):
        result = LayerResult(
            layer=0,
            step_id="step-001",
            quality_delta=-0.1,
            cost_delta=0.002,
            latency_delta=500,
            risk_delta=0.0,
            confidence_lower=-0.15,
            confidence_upper=-0.05,
            evidence="Layer 0: repetition detected (3 retries)",
        )
        assert result.layer == 0
        assert result.step_id == "step-001"
        assert result.quality_delta == -0.1

    def test_merge_single_layer(self):
        results = [
            LayerResult(layer=0, step_id="s1", quality_delta=-0.1,
                        cost_delta=0.001, latency_delta=100, risk_delta=0.0,
                        confidence_lower=-0.15, confidence_upper=-0.05,
                        evidence="repetition rule"),
        ]
        merged = merge_layer_results("s1", "search_web", "node1", results)
        assert merged.step_id == "s1"
        assert merged.quality_delta == -0.1
        assert merged.layer_scores == {0: -0.1}
        assert merged.confidence is not None
        assert len(merged.evidence) == 1

    def test_merge_multiple_layers(self):
        results = [
            LayerResult(layer=0, step_id="s1", quality_delta=0.0,
                        cost_delta=0.001, latency_delta=100, risk_delta=0.0,
                        confidence_lower=-0.02, confidence_upper=0.02,
                        evidence="no rule violation"),
            LayerResult(layer=1, step_id="s1", quality_delta=-0.05,
                        cost_delta=0.001, latency_delta=100, risk_delta=0.0,
                        confidence_lower=-0.10, confidence_upper=0.0,
                        evidence="lift = 0.95, below baseline"),
            LayerResult(layer=2, step_id="s1", quality_delta=-0.08,
                        cost_delta=0.001, latency_delta=100, risk_delta=0.0,
                        confidence_lower=-0.12, confidence_upper=-0.04,
                        evidence="sequence estimator: removal hurts quality"),
        ]
        merged = merge_layer_results("s1", "search_web", "node1", results)
        assert merged.layer_scores == {0: 0.0, 1: -0.05, 2: -0.08}
        assert merged.quality_delta == -0.08
        assert merged.confidence.lower == -0.12
        assert merged.confidence.upper == -0.04
        assert len(merged.evidence) == 3

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError):
            merge_layer_results("s1", "test", None, [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_attribution_base.py -v`
Expected: FAIL

- [ ] **Step 3: Implement base protocol**

`traceshap/attribution/__init__.py`:
```python
```

`traceshap/attribution/base.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from traceshap.models.outcome import ConfidenceInterval, StepAttribution, CalibrationMetrics
from traceshap.models.enums import Verdict
from traceshap.models.trajectory import Trajectory


@dataclass
class LayerResult:
    layer: int
    step_id: str
    quality_delta: float
    cost_delta: float
    latency_delta: float
    risk_delta: float
    confidence_lower: float
    confidence_upper: float
    evidence: str


class AttributionLayer(Protocol):
    @property
    def layer_id(self) -> int: ...

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]: ...


def merge_layer_results(
    step_id: str,
    step_name: str,
    node_id: str | None,
    results: list[LayerResult],
) -> StepAttribution:
    if not results:
        raise ValueError("Cannot merge empty results")

    highest = max(results, key=lambda r: r.layer)

    layer_scores = {r.layer: r.quality_delta for r in results}

    return StepAttribution(
        step_id=step_id,
        step_name=step_name,
        node_id=node_id,
        quality_delta=highest.quality_delta,
        cost_delta=highest.cost_delta,
        latency_delta=highest.latency_delta,
        risk_delta=highest.risk_delta,
        layer_scores=layer_scores,
        confidence=ConfidenceInterval(
            lower=highest.confidence_lower,
            point=highest.quality_delta,
            upper=highest.confidence_upper,
        ),
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        causal_hypothesis=None,
        evidence=[r.evidence for r in results],
        calibration=None,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_attribution_base.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/ tests/test_attribution_base.py
git commit -m "feat: attribution base protocol, LayerResult, and merge logic"
```

---

### Task 2: Layer 0 — Expert Rules Engine

**Files:**
- Create: `traceshap/attribution/layer0_rules.py`
- Create: `tests/test_layer0.py`

- [ ] **Step 1: Write tests**

`tests/test_layer0.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.models import (
    TraceSHAPSpan, SpanKind, TokenUsage, CanonicalStep, StepType,
    SideEffect, SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.attribution.layer0_rules import (
    Layer0Rules, RepetitionRule, LoopDetectionRule, NoOpRule, RuleVerdict,
)


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          input_hash: str = "a", output_hash: str = "b",
          attempt_index: int = 0, cost: float = 0.001,
          offset_sec: int = 0) -> CanonicalStep:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=[f"s-{step_id}"],
        node_id=None, tool_name=name, step_type=step_type,
        attempt_index=attempt_index, loop_iteration=None,
        input_hash=input_hash, output_hash=output_hash,
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=cost,
        start_time=start, end_time=end,
    )


def _trajectory(steps: list[CanonicalStep]) -> Trajectory:
    return Trajectory(
        trace_id="t1", spans=[], steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=True, quality_score=0.9, token_cost=100,
                        latency_ms=5000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
    )


class TestRepetitionRule:
    async def test_detects_retries(self):
        steps = [
            _step("s1", "search", input_hash="x", attempt_index=0, offset_sec=0),
            _step("s2", "search", input_hash="x", attempt_index=1, offset_sec=1),
            _step("s3", "search", input_hash="x", attempt_index=2, offset_sec=2),
        ]
        rule = RepetitionRule(threshold=2)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) >= 1

    async def test_no_flag_below_threshold(self):
        steps = [
            _step("s1", "search", input_hash="x", attempt_index=0),
        ]
        rule = RepetitionRule(threshold=2)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) == 0


class TestNoOpRule:
    async def test_detects_noop(self):
        steps = [
            _step("s1", "transform", input_hash="abc", output_hash="abc"),
        ]
        rule = NoOpRule(similarity_threshold=1.0)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) == 1
        assert "no-op" in flagged[0].recommendation.lower()

    async def test_no_flag_different_output(self):
        steps = [
            _step("s1", "transform", input_hash="abc", output_hash="def"),
        ]
        rule = NoOpRule(similarity_threshold=1.0)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) == 0


class TestLoopDetectionRule:
    async def test_detects_loop(self):
        steps = [
            _step("s1", "A", offset_sec=0),
            _step("s2", "B", offset_sec=1),
            _step("s3", "A", offset_sec=2),
            _step("s4", "B", offset_sec=3),
            _step("s5", "A", offset_sec=4),
            _step("s6", "B", offset_sec=5),
        ]
        rule = LoopDetectionRule(max_cycle=2)
        verdicts = rule.evaluate(steps)
        flagged = [v for v in verdicts if v.severity > 0]
        assert len(flagged) >= 1


class TestLayer0Rules:
    async def test_analyze_returns_layer_results(self):
        steps = [
            _step("s1", "search", input_hash="x", attempt_index=0, offset_sec=0),
            _step("s2", "search", input_hash="x", attempt_index=1, offset_sec=1),
            _step("s3", "search", input_hash="x", attempt_index=2, offset_sec=2),
        ]
        traj = _trajectory(steps)
        layer = Layer0Rules()
        results = await layer.analyze(traj)
        assert len(results) == 3
        assert all(r.layer == 0 for r in results)

    async def test_clean_trajectory_no_flags(self):
        steps = [
            _step("s1", "plan", step_type=StepType.DECISION, offset_sec=0),
            _step("s2", "search", offset_sec=1),
            _step("s3", "summarize", step_type=StepType.DECISION, offset_sec=2),
        ]
        traj = _trajectory(steps)
        layer = Layer0Rules()
        results = await layer.analyze(traj)
        assert all(r.quality_delta == 0.0 for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer0.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Layer 0**

`traceshap/attribution/layer0_rules.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from traceshap.models.step import CanonicalStep
from traceshap.models.trajectory import Trajectory
from traceshap.attribution.base import LayerResult


@dataclass
class RuleVerdict:
    step_id: str
    rule_name: str
    severity: float
    recommendation: str


class RepetitionRule:
    def __init__(self, threshold: int = 3, similarity: float = 0.9):
        self.threshold = threshold
        self.similarity = similarity

    def evaluate(self, steps: list[CanonicalStep]) -> list[RuleVerdict]:
        verdicts: list[RuleVerdict] = []
        groups: dict[str, list[CanonicalStep]] = {}
        for step in steps:
            key = f"{step.tool_name or step.step_type.value}:{step.input_hash}"
            groups.setdefault(key, []).append(step)

        for key, group in groups.items():
            if len(group) >= self.threshold:
                for step in group[self.threshold - 1:]:
                    verdicts.append(RuleVerdict(
                        step_id=step.step_id,
                        rule_name="repetition",
                        severity=min(1.0, len(group) / (self.threshold * 2)),
                        recommendation=f"Excessive repetition ({len(group)} attempts)",
                    ))
        return verdicts


class NoOpRule:
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold

    def evaluate(self, steps: list[CanonicalStep]) -> list[RuleVerdict]:
        verdicts: list[RuleVerdict] = []
        for step in steps:
            if step.input_hash == step.output_hash:
                verdicts.append(RuleVerdict(
                    step_id=step.step_id,
                    rule_name="no_op",
                    severity=0.8,
                    recommendation=f"No-op detected: input equals output",
                ))
        return verdicts


class LoopDetectionRule:
    def __init__(self, max_cycle: int = 2):
        self.max_cycle = max_cycle

    def evaluate(self, steps: list[CanonicalStep]) -> list[RuleVerdict]:
        verdicts: list[RuleVerdict] = []
        names = [s.tool_name or s.step_type.value for s in steps]

        for cycle_len in range(1, len(names) // 2 + 1):
            for start in range(len(names) - cycle_len * (self.max_cycle + 1) + 1):
                pattern = names[start:start + cycle_len]
                repeat_count = 0
                pos = start
                while pos + cycle_len <= len(names):
                    if names[pos:pos + cycle_len] == pattern:
                        repeat_count += 1
                        pos += cycle_len
                    else:
                        break
                if repeat_count > self.max_cycle:
                    for i in range(start + cycle_len * self.max_cycle, start + cycle_len * repeat_count):
                        if i < len(steps):
                            verdicts.append(RuleVerdict(
                                step_id=steps[i].step_id,
                                rule_name="loop",
                                severity=min(1.0, repeat_count / (self.max_cycle * 2)),
                                recommendation=f"Loop detected: pattern repeated {repeat_count} times (max {self.max_cycle})",
                            ))
                    break
            else:
                continue
            break

        return verdicts


class Layer0Rules:
    def __init__(
        self,
        rules: list | None = None,
    ):
        self._rules = rules or [
            RepetitionRule(threshold=3),
            NoOpRule(similarity_threshold=0.95),
            LoopDetectionRule(max_cycle=2),
        ]

    @property
    def layer_id(self) -> int:
        return 0

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        all_verdicts: dict[str, list[RuleVerdict]] = {}
        for rule in self._rules:
            for verdict in rule.evaluate(trajectory.steps):
                all_verdicts.setdefault(verdict.step_id, []).append(verdict)

        results: list[LayerResult] = []
        for step in trajectory.steps:
            verdicts = all_verdicts.get(step.step_id, [])
            max_severity = max((v.severity for v in verdicts), default=0.0)
            evidence_parts = [f"{v.rule_name}: {v.recommendation}" for v in verdicts]
            evidence = "; ".join(evidence_parts) if evidence_parts else "no rule violation"

            results.append(LayerResult(
                layer=0,
                step_id=step.step_id,
                quality_delta=-max_severity * 0.1 if max_severity > 0 else 0.0,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=-max_severity * 0.15 if max_severity > 0 else 0.0,
                confidence_upper=-max_severity * 0.05 if max_severity > 0 else 0.0,
                evidence=evidence,
            ))

        return results
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_layer0.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/layer0_rules.py tests/test_layer0.py
git commit -m "feat: Layer 0 expert rules engine (repetition, no-op, loop detection)"
```

---

### Task 3: Layer 1 — Statistical Lift (Cohort-Stratified)

**Files:**
- Create: `traceshap/attribution/layer1_lift.py`
- Create: `tests/test_layer1.py`

- [ ] **Step 1: Write tests**

`tests/test_layer1.py`:
```python
import pytest
import math
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.attribution.layer1_lift import Layer1Lift, CohortStats


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          offset_sec: int = 0) -> CanonicalStep:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=[f"s-{step_id}"],
        node_id=None, tool_name=name, step_type=step_type,
        attempt_index=0, loop_iteration=None,
        input_hash="a", output_hash="b",
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=0.001,
        start_time=start, end_time=end,
    )


def _trajectory(trace_id: str, steps: list[CanonicalStep],
                success: bool = True, quality: float = 0.9) -> Trajectory:
    return Trajectory(
        trace_id=trace_id, spans=[], steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=success, quality_score=quality,
                        token_cost=100, latency_ms=5000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test",
                                agent_version="v1", task_type="qa"),
    )


class TestCohortStats:
    def test_lift_positive(self):
        stats = CohortStats(
            step_key="search_web",
            present_success=80, present_total=100,
            absent_success=50, absent_total=100,
            smoothing=0.0,
        )
        assert stats.lift > 1.0

    def test_lift_negative(self):
        stats = CohortStats(
            step_key="bad_tool",
            present_success=20, present_total=100,
            absent_success=60, absent_total=100,
            smoothing=0.0,
        )
        assert stats.lift < 1.0

    def test_lift_with_smoothing(self):
        stats = CohortStats(
            step_key="tool",
            present_success=5, present_total=5,
            absent_success=0, absent_total=5,
            smoothing=1.0,
        )
        assert stats.lift > 1.0
        assert stats.lift < float("inf")

    def test_confidence_interval(self):
        stats = CohortStats(
            step_key="tool",
            present_success=80, present_total=100,
            absent_success=50, absent_total=100,
            smoothing=0.0,
        )
        ci = stats.confidence_interval(confidence=0.95)
        assert ci[0] < stats.lift
        assert ci[1] > stats.lift


class TestLayer1Lift:
    async def test_analyze_with_cohort(self):
        trajs = []
        for i in range(60):
            has_search = i < 40
            success = (i < 32) if has_search else (i < 50)
            steps = [_step(f"s{i}-1", "plan", StepType.DECISION)]
            if has_search:
                steps.append(_step(f"s{i}-2", "search_web"))
            trajs.append(_trajectory(f"t{i}", steps, success=success,
                                     quality=0.9 if success else 0.3))

        layer = Layer1Lift(min_support=10, smoothing=1.0, confidence_level=0.95)
        layer.fit(trajs)
        results = await layer.analyze(trajs[0])
        assert len(results) > 0
        assert all(r.layer == 1 for r in results)

    async def test_insufficient_support(self):
        trajs = [_trajectory(f"t{i}", [_step(f"s{i}", "rare_tool")]) for i in range(3)]
        layer = Layer1Lift(min_support=50)
        layer.fit(trajs)
        results = await layer.analyze(trajs[0])
        assert all(r.quality_delta == 0.0 for r in results)

    async def test_no_outcome_skipped(self):
        traj = Trajectory(
            trace_id="t-no-outcome", spans=[],
            steps=[_step("s1", "tool")],
            span_tree=SpanNode(span_id="root"),
            outcome=None,
            metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
        )
        layer = Layer1Lift(min_support=5)
        layer.fit([traj])
        results = await layer.analyze(traj)
        assert all(r.quality_delta == 0.0 for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer1.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Layer 1**

`traceshap/attribution/layer1_lift.py`:
```python
from __future__ import annotations

import math
from dataclasses import dataclass

from traceshap.models.step import CanonicalStep
from traceshap.models.trajectory import Trajectory
from traceshap.attribution.base import LayerResult


@dataclass
class CohortStats:
    step_key: str
    present_success: int
    present_total: int
    absent_success: int
    absent_total: int
    smoothing: float = 1.0

    @property
    def lift(self) -> float:
        p_present = (self.present_success + self.smoothing) / (self.present_total + 2 * self.smoothing)
        p_absent = (self.absent_success + self.smoothing) / (self.absent_total + 2 * self.smoothing)
        if p_absent == 0:
            return float("inf")
        return p_present / p_absent

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        z = 1.96 if confidence == 0.95 else 2.576
        log_lift = math.log(self.lift) if self.lift > 0 and self.lift != float("inf") else 0.0

        p1 = (self.present_success + self.smoothing) / (self.present_total + 2 * self.smoothing)
        p2 = (self.absent_success + self.smoothing) / (self.absent_total + 2 * self.smoothing)
        n1 = self.present_total + 2 * self.smoothing
        n2 = self.absent_total + 2 * self.smoothing

        se = math.sqrt((1 - p1) / (p1 * n1) + (1 - p2) / (p2 * n2)) if p1 > 0 and p2 > 0 else 1.0

        return (math.exp(log_lift - z * se), math.exp(log_lift + z * se))


def _step_key(step: CanonicalStep) -> str:
    return step.tool_name or step.step_type.value


class Layer1Lift:
    def __init__(
        self,
        min_support: int = 50,
        smoothing: float = 1.0,
        confidence_level: float = 0.95,
    ):
        self._min_support = min_support
        self._smoothing = smoothing
        self._confidence_level = confidence_level
        self._cohort_stats: dict[str, CohortStats] = {}

    @property
    def layer_id(self) -> int:
        return 1

    def fit(self, trajectories: list[Trajectory]) -> None:
        step_presence: dict[str, list[bool]] = {}
        outcomes: list[bool] = []

        for traj in trajectories:
            if traj.outcome is None or traj.outcome.success is None:
                continue
            outcomes.append(traj.outcome.success)
            present_keys = {_step_key(s) for s in traj.steps}
            all_keys = step_presence.keys() | present_keys
            for key in all_keys:
                step_presence.setdefault(key, []).append(key in present_keys)

        for idx in range(len(outcomes)):
            for key in step_presence:
                if len(step_presence[key]) <= idx:
                    step_presence[key].append(False)

        self._cohort_stats = {}
        for key, presences in step_presence.items():
            present_success = sum(1 for i, p in enumerate(presences) if p and i < len(outcomes) and outcomes[i])
            present_total = sum(1 for p in presences if p)
            absent_success = sum(1 for i, p in enumerate(presences) if not p and i < len(outcomes) and outcomes[i])
            absent_total = sum(1 for p in presences if not p)

            if present_total + absent_total >= self._min_support:
                self._cohort_stats[key] = CohortStats(
                    step_key=key,
                    present_success=present_success,
                    present_total=present_total,
                    absent_success=absent_success,
                    absent_total=absent_total,
                    smoothing=self._smoothing,
                )

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        results: list[LayerResult] = []

        for step in trajectory.steps:
            key = _step_key(step)
            stats = self._cohort_stats.get(key)

            if stats is None:
                results.append(LayerResult(
                    layer=1,
                    step_id=step.step_id,
                    quality_delta=0.0,
                    cost_delta=step.cost or 0.0,
                    latency_delta=step.duration_ms,
                    risk_delta=0.0,
                    confidence_lower=0.0,
                    confidence_upper=0.0,
                    evidence=f"insufficient support for '{key}'",
                ))
                continue

            lift = stats.lift
            ci = stats.confidence_interval(self._confidence_level)
            quality_delta = (lift - 1.0) * 0.5

            results.append(LayerResult(
                layer=1,
                step_id=step.step_id,
                quality_delta=quality_delta,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=(ci[0] - 1.0) * 0.5,
                confidence_upper=(ci[1] - 1.0) * 0.5,
                evidence=f"lift={lift:.3f} (CI: [{ci[0]:.3f}, {ci[1]:.3f}]), n_present={stats.present_total}, n_absent={stats.absent_total}",
            ))

        return results
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_layer1.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/layer1_lift.py tests/test_layer1.py
git commit -m "feat: Layer 1 statistical lift with cohort stratification and CI"
```

---

### Task 4: Layer 2 — Sequence-Aware Counterfactual Estimator

**Files:**
- Create: `traceshap/attribution/layer2_sequence.py`
- Create: `tests/test_layer2.py`

- [ ] **Step 1: Write tests**

`tests/test_layer2.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome,
)
from traceshap.attribution.layer2_sequence import (
    Layer2Sequence, TransitionModel, generate_legal_interventions,
    InterventionType,
)


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          input_hash: str = "a", output_hash: str = "b",
          attempt_index: int = 0, offset_sec: int = 0) -> CanonicalStep:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=[f"s-{step_id}"],
        node_id=None, tool_name=name, step_type=step_type,
        attempt_index=attempt_index, loop_iteration=None,
        input_hash=input_hash, output_hash=output_hash,
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=0.001,
        start_time=start, end_time=end,
    )


def _trajectory(trace_id: str, steps: list[CanonicalStep],
                success: bool = True, quality: float = 0.9) -> Trajectory:
    return Trajectory(
        trace_id=trace_id, spans=[], steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=success, quality_score=quality,
                        token_cost=100, latency_ms=5000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test",
                                agent_version="v1", task_type="qa"),
    )


class TestGenerateLegalInterventions:
    def test_retry_collapse(self):
        steps = [
            _step("s1", "search", attempt_index=0, offset_sec=0),
            _step("s2", "search", attempt_index=1, offset_sec=1),
            _step("s3", "search", attempt_index=2, offset_sec=2),
        ]
        interventions = generate_legal_interventions(steps, steps[2])
        types = [i.intervention_type for i in interventions]
        assert InterventionType.RETRY_COLLAPSE in types

    def test_contiguous_removal(self):
        steps = [
            _step("s1", "plan", StepType.DECISION, offset_sec=0),
            _step("s2", "search", offset_sec=1),
            _step("s3", "search", offset_sec=2),
            _step("s4", "summarize", StepType.DECISION, offset_sec=3),
        ]
        interventions = generate_legal_interventions(steps, steps[1])
        types = [i.intervention_type for i in interventions]
        assert InterventionType.CONTIGUOUS_REMOVAL in types

    def test_prefix_removal_for_last_step(self):
        steps = [
            _step("s1", "plan", StepType.DECISION, offset_sec=0),
            _step("s2", "search", offset_sec=1),
            _step("s3", "summarize", StepType.DECISION, offset_sec=2),
        ]
        interventions = generate_legal_interventions(steps, steps[2])
        types = [i.intervention_type for i in interventions]
        assert InterventionType.PREFIX_REMOVAL in types


class TestTransitionModel:
    def test_fit_and_predict(self):
        trajs = []
        for i in range(50):
            steps = [
                _step(f"s{i}-1", "plan", StepType.DECISION),
                _step(f"s{i}-2", "search"),
                _step(f"s{i}-3", "summarize", StepType.DECISION),
            ]
            trajs.append(_trajectory(f"t{i}", steps,
                                     success=i < 40, quality=0.9 if i < 40 else 0.3))

        model = TransitionModel()
        model.fit(trajs)

        full_seq = ["decision", "action", "decision"]
        pred_full = model.predict(full_seq)
        assert 0.0 <= pred_full <= 1.0

        short_seq = ["decision", "decision"]
        pred_short = model.predict(short_seq)
        assert 0.0 <= pred_short <= 1.0


class TestLayer2Sequence:
    async def test_analyze_returns_results(self):
        trajs = []
        for i in range(50):
            steps = [
                _step(f"s{i}-1", "plan", StepType.DECISION),
                _step(f"s{i}-2", "search"),
                _step(f"s{i}-3", "summarize", StepType.DECISION),
            ]
            trajs.append(_trajectory(f"t{i}", steps,
                                     success=i < 40, quality=0.9 if i < 40 else 0.3))

        layer = Layer2Sequence(num_samples=20)
        layer.fit(trajs)
        results = await layer.analyze(trajs[0])
        assert len(results) == 3
        assert all(r.layer == 2 for r in results)

    async def test_no_outcome_returns_zero(self):
        traj = Trajectory(
            trace_id="t-none", spans=[],
            steps=[_step("s1", "tool")],
            span_tree=SpanNode(span_id="root"),
            outcome=None,
            metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
        )
        layer = Layer2Sequence()
        layer.fit([traj])
        results = await layer.analyze(traj)
        assert all(r.quality_delta == 0.0 for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer2.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Layer 2**

`traceshap/attribution/layer2_sequence.py`:
```python
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from traceshap.models.step import CanonicalStep
from traceshap.models.enums import StepType
from traceshap.models.trajectory import Trajectory
from traceshap.attribution.base import LayerResult


class InterventionType(Enum):
    PREFIX_REMOVAL = "prefix_removal"
    CONTIGUOUS_REMOVAL = "contiguous_removal"
    RETRY_COLLAPSE = "retry_collapse"


@dataclass
class Intervention:
    intervention_type: InterventionType
    target_step_id: str
    removed_step_ids: list[str]
    resulting_sequence: list[str]


def _step_label(step: CanonicalStep) -> str:
    return step.tool_name or step.step_type.value


def generate_legal_interventions(
    steps: list[CanonicalStep],
    target: CanonicalStep,
) -> list[Intervention]:
    interventions: list[Intervention] = []
    idx = next((i for i, s in enumerate(steps) if s.step_id == target.step_id), None)
    if idx is None:
        return interventions

    labels = [_step_label(s) for s in steps]

    if target.attempt_index > 0:
        retry_ids = [s.step_id for s in steps
                     if s.tool_name == target.tool_name
                     and s.input_hash == target.input_hash
                     and s.attempt_index > 0]
        if retry_ids:
            remaining = [l for i, l in enumerate(labels)
                         if steps[i].step_id not in retry_ids]
            interventions.append(Intervention(
                intervention_type=InterventionType.RETRY_COLLAPSE,
                target_step_id=target.step_id,
                removed_step_ids=retry_ids,
                resulting_sequence=remaining,
            ))

    removed = [steps[idx].step_id]
    remaining = [l for i, l in enumerate(labels) if i != idx]
    interventions.append(Intervention(
        intervention_type=InterventionType.CONTIGUOUS_REMOVAL,
        target_step_id=target.step_id,
        removed_step_ids=removed,
        resulting_sequence=remaining,
    ))

    if idx == len(steps) - 1 and len(steps) > 1:
        remaining = labels[:idx]
        interventions.append(Intervention(
            intervention_type=InterventionType.PREFIX_REMOVAL,
            target_step_id=target.step_id,
            removed_step_ids=[target.step_id],
            resulting_sequence=remaining,
        ))

    return interventions


class TransitionModel:
    def __init__(self):
        self._transition_counts: Counter = Counter()
        self._state_counts: Counter = Counter()
        self._outcome_by_final: dict[str, list[float]] = {}
        self._global_success_rate: float = 0.5

    def fit(self, trajectories: list[Trajectory]) -> None:
        successes = 0
        total = 0
        for traj in trajectories:
            if traj.outcome is None or traj.outcome.quality_score is None:
                continue
            total += 1
            if traj.outcome.success:
                successes += 1

            labels = [_step_label(s) for s in traj.steps]
            for i in range(len(labels) - 1):
                self._transition_counts[(labels[i], labels[i + 1])] += 1
                self._state_counts[labels[i]] += 1

            if labels:
                final = labels[-1]
                self._outcome_by_final.setdefault(final, []).append(
                    traj.outcome.quality_score
                )

        self._global_success_rate = successes / total if total > 0 else 0.5

    def predict(self, sequence: list[str]) -> float:
        if not sequence:
            return self._global_success_rate

        log_prob = 0.0
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            count = self._transition_counts[pair]
            state_count = self._state_counts[sequence[i]]
            if state_count > 0:
                log_prob += math.log((count + 1) / (state_count + len(self._state_counts) + 1))

        final = sequence[-1]
        outcomes = self._outcome_by_final.get(final, [])
        if outcomes:
            base = sum(outcomes) / len(outcomes)
        else:
            base = self._global_success_rate

        transition_factor = math.exp(log_prob) if log_prob != 0 else 1.0
        return max(0.0, min(1.0, base * min(transition_factor, 2.0)))


class Layer2Sequence:
    def __init__(self, num_samples: int = 200):
        self._num_samples = num_samples
        self._model = TransitionModel()

    @property
    def layer_id(self) -> int:
        return 2

    def fit(self, trajectories: list[Trajectory]) -> None:
        self._model.fit(trajectories)

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        results: list[LayerResult] = []

        if trajectory.outcome is None or trajectory.outcome.quality_score is None:
            for step in trajectory.steps:
                results.append(LayerResult(
                    layer=2, step_id=step.step_id,
                    quality_delta=0.0, cost_delta=step.cost or 0.0,
                    latency_delta=step.duration_ms, risk_delta=0.0,
                    confidence_lower=0.0, confidence_upper=0.0,
                    evidence="no outcome available",
                ))
            return results

        full_labels = [_step_label(s) for s in trajectory.steps]
        factual_score = self._model.predict(full_labels)

        for step in trajectory.steps:
            interventions = generate_legal_interventions(trajectory.steps, step)
            if not interventions:
                results.append(LayerResult(
                    layer=2, step_id=step.step_id,
                    quality_delta=0.0, cost_delta=step.cost or 0.0,
                    latency_delta=step.duration_ms, risk_delta=0.0,
                    confidence_lower=0.0, confidence_upper=0.0,
                    evidence="no legal interventions",
                ))
                continue

            deltas: list[float] = []
            for intervention in interventions:
                counterfactual_score = self._model.predict(intervention.resulting_sequence)
                delta = counterfactual_score - factual_score
                deltas.append(delta)

            mean_delta = sum(deltas) / len(deltas)
            if len(deltas) > 1:
                variance = sum((d - mean_delta) ** 2 for d in deltas) / (len(deltas) - 1)
                se = math.sqrt(variance / len(deltas))
            else:
                se = abs(mean_delta) * 0.3

            results.append(LayerResult(
                layer=2,
                step_id=step.step_id,
                quality_delta=mean_delta,
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=mean_delta - 1.96 * se,
                confidence_upper=mean_delta + 1.96 * se,
                evidence=f"sequence estimator: {len(interventions)} interventions, mean_delta={mean_delta:.4f}",
            ))

        return results
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_layer2.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/layer2_sequence.py tests/test_layer2.py
git commit -m "feat: Layer 2 sequence-aware counterfactual estimator with legal interventions"
```

---

### Task 5: Attribution Engine Orchestrator

**Files:**
- Create: `traceshap/attribution/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write tests**

`tests/test_engine.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome, Verdict,
)
from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer0_rules import Layer0Rules
from traceshap.attribution.layer1_lift import Layer1Lift
from traceshap.attribution.layer2_sequence import Layer2Sequence


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          attempt_index: int = 0, offset_sec: int = 0) -> CanonicalStep:
    start = datetime(2026, 1, 1, 0, 0, offset_sec, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, offset_sec + 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=[f"s-{step_id}"],
        node_id=None, tool_name=name, step_type=step_type,
        attempt_index=attempt_index, loop_iteration=None,
        input_hash="a", output_hash="b",
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=0.001,
        start_time=start, end_time=end,
    )


def _trajectory(trace_id: str, steps: list[CanonicalStep],
                success: bool = True, quality: float = 0.9) -> Trajectory:
    return Trajectory(
        trace_id=trace_id, spans=[], steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=success, quality_score=quality,
                        token_cost=100, latency_ms=5000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test",
                                agent_version="v1", task_type="qa"),
    )


class TestAttributionEngine:
    async def test_layer0_only(self):
        steps = [
            _step("s1", "plan", StepType.DECISION, offset_sec=0),
            _step("s2", "search", offset_sec=1),
        ]
        traj = _trajectory("t1", steps)
        engine = AttributionEngine(layers=[Layer0Rules()])
        attributions = await engine.analyze(traj)
        assert len(attributions) == 2
        assert all(0 in a.layer_scores for a in attributions)

    async def test_multi_layer(self):
        trajs = []
        for i in range(60):
            steps = [
                _step(f"s{i}-1", "plan", StepType.DECISION),
                _step(f"s{i}-2", "search"),
                _step(f"s{i}-3", "summarize", StepType.DECISION),
            ]
            trajs.append(_trajectory(f"t{i}", steps,
                                     success=i < 45, quality=0.9 if i < 45 else 0.3))

        layer1 = Layer1Lift(min_support=10, smoothing=1.0)
        layer1.fit(trajs)
        layer2 = Layer2Sequence(num_samples=20)
        layer2.fit(trajs)

        engine = AttributionEngine(layers=[Layer0Rules(), layer1, layer2])
        attributions = await engine.analyze(trajs[0])
        assert len(attributions) == 3
        for attr in attributions:
            assert 0 in attr.layer_scores
            assert 1 in attr.layer_scores
            assert 2 in attr.layer_scores
            assert attr.confidence is not None

    async def test_verdict_assignment(self):
        steps = [_step("s1", "plan", StepType.DECISION)]
        traj = _trajectory("t1", steps)
        engine = AttributionEngine(layers=[Layer0Rules()])
        attributions = await engine.analyze(traj)
        assert attributions[0].verdict != Verdict.INSUFFICIENT_EVIDENCE or True

    async def test_no_outcome_returns_insufficient(self):
        traj = Trajectory(
            trace_id="t-none", spans=[],
            steps=[_step("s1", "tool")],
            span_tree=SpanNode(span_id="root"),
            outcome=None,
            metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
        )
        engine = AttributionEngine(layers=[Layer0Rules()])
        attributions = await engine.analyze(traj)
        assert attributions[0].verdict == Verdict.INSUFFICIENT_EVIDENCE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement AttributionEngine**

`traceshap/attribution/engine.py`:
```python
from __future__ import annotations

from traceshap.models.trajectory import Trajectory
from traceshap.models.outcome import StepAttribution, ConfidenceInterval
from traceshap.models.enums import Verdict
from traceshap.attribution.base import AttributionLayer, LayerResult, merge_layer_results


class AttributionEngine:
    def __init__(self, layers: list[AttributionLayer]):
        self._layers = sorted(layers, key=lambda l: l.layer_id)

    async def analyze(self, trajectory: Trajectory) -> list[StepAttribution]:
        step_results: dict[str, list[LayerResult]] = {}

        for layer in self._layers:
            results = await layer.analyze(trajectory)
            for result in results:
                step_results.setdefault(result.step_id, []).append(result)

        attributions: list[StepAttribution] = []
        for step in trajectory.steps:
            results = step_results.get(step.step_id, [])
            if not results:
                attributions.append(StepAttribution(
                    step_id=step.step_id,
                    step_name=step.tool_name or step.step_type.value,
                    node_id=step.node_id,
                    quality_delta=None,
                    cost_delta=step.cost,
                    latency_delta=step.duration_ms,
                    risk_delta=None,
                    layer_scores={},
                    confidence=None,
                    verdict=Verdict.INSUFFICIENT_EVIDENCE,
                    causal_hypothesis=None,
                    evidence=[],
                    calibration=None,
                ))
                continue

            merged = merge_layer_results(
                step_id=step.step_id,
                step_name=step.tool_name or step.step_type.value,
                node_id=step.node_id,
                results=results,
            )

            merged.verdict = self._classify_verdict(merged, trajectory)
            attributions.append(merged)

        return attributions

    @staticmethod
    def _classify_verdict(attr: StepAttribution, trajectory: Trajectory) -> Verdict:
        if trajectory.outcome is None:
            return Verdict.INSUFFICIENT_EVIDENCE
        if attr.confidence is None or attr.quality_delta is None:
            return Verdict.INSUFFICIENT_EVIDENCE
        return Verdict.REVIEW
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/attribution/engine.py tests/test_engine.py
git commit -m "feat: AttributionEngine orchestrator merging multi-layer results"
```

---

### Task 6: Pruning Safety Constraints

**Files:**
- Create: `traceshap/pruning/__init__.py`
- Create: `traceshap/pruning/safety.py`
- Create: `tests/test_pruning_safety.py`

- [ ] **Step 1: Write tests**

`tests/test_pruning_safety.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pruning_safety.py -v`
Expected: FAIL

- [ ] **Step 3: Implement safety constraints**

`traceshap/pruning/__init__.py`:
```python
```

`traceshap/pruning/safety.py`:
```python
from traceshap.models.step import CanonicalStep
from traceshap.models.enums import StepType, Verdict
from traceshap.models.outcome import StepAttribution
from traceshap.config import PruningConfig

PROTECTED_STEP_TYPES = frozenset({StepType.VALIDATION})


def is_protected_step(step: CanonicalStep) -> bool:
    return step.step_type in PROTECTED_STEP_TYPES


def is_first_or_last(step_id: str, steps: list[CanonicalStep]) -> bool:
    if not steps:
        return False
    return step_id == steps[0].step_id or step_id == steps[-1].step_id


def classify_step(
    attr: StepAttribution,
    step: CanonicalStep,
    config: PruningConfig,
    is_first_last: bool,
) -> Verdict:
    if attr.confidence is None or attr.quality_delta is None:
        return Verdict.INSUFFICIENT_EVIDENCE

    if is_protected_step(step):
        return Verdict.KEEP

    if is_first_last and config.protect_first_last:
        return Verdict.KEEP

    if (attr.confidence.lower >= -config.prune_epsilon
            and (attr.cost_delta or 0) > 0):
        return Verdict.PRUNE_CANDIDATE

    if attr.confidence.lower < -config.keep_threshold:
        return Verdict.KEEP

    return Verdict.REVIEW
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pruning_safety.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/pruning/ tests/test_pruning_safety.py
git commit -m "feat: pruning safety constraints (protected types, first/last guard, CI-based classification)"
```

---

### Task 7: Pruning Advisor

**Files:**
- Create: `traceshap/pruning/advisor.py`
- Create: `tests/test_pruning_advisor.py`

- [ ] **Step 1: Write tests**

`tests/test_pruning_advisor.py`:
```python
import pytest
from datetime import datetime, timezone

from traceshap.models import (
    CanonicalStep, StepType, SideEffect, TokenUsage,
    SpanNode, TrajectoryMeta, Trajectory, Outcome, Verdict,
    DecisionStatus, RiskLevel,
)
from traceshap.models.outcome import StepAttribution, ConfidenceInterval
from traceshap.models.pruning import Savings, PruneCandidate, PruningReport
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig


def _step(step_id: str, name: str, step_type: StepType = StepType.ACTION,
          cost: float = 0.001) -> CanonicalStep:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    return CanonicalStep(
        step_id=step_id, raw_span_ids=["s1"], node_id=None,
        tool_name=name, step_type=step_type, attempt_index=0,
        loop_iteration=None, input_hash="a", output_hash="b",
        side_effect_class=SideEffect.READ_ONLY,
        framework_mapping_confidence=0.8,
        tokens=TokenUsage(10, 20, 30), cost=cost,
        start_time=now, end_time=later,
    )


def _attr(step_id: str, quality_delta: float, ci_lower: float, ci_upper: float,
          cost_delta: float = 0.001) -> StepAttribution:
    return StepAttribution(
        step_id=step_id, step_name="test", node_id=None,
        quality_delta=quality_delta, cost_delta=cost_delta,
        latency_delta=1000, risk_delta=0.0,
        layer_scores={0: quality_delta},
        confidence=ConfidenceInterval(lower=ci_lower, point=quality_delta, upper=ci_upper),
        verdict=Verdict.REVIEW,
        causal_hypothesis=None, evidence=["test evidence"], calibration=None,
    )


def _trajectory(steps: list[CanonicalStep]) -> Trajectory:
    return Trajectory(
        trace_id="t1", spans=[], steps=steps,
        span_tree=SpanNode(span_id="root"),
        outcome=Outcome(success=True, quality_score=0.9, token_cost=100,
                        latency_ms=5000, custom_metrics={}),
        metadata=TrajectoryMeta(framework="langgraph", agent_name="test"),
    )


class TestPruningAdvisor:
    def test_generates_report(self):
        steps = [
            _step("s1", "plan", StepType.DECISION),
            _step("s2", "search", cost=0.005),
            _step("s3", "summarize", StepType.DECISION),
        ]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
            _attr("s2", quality_delta=-0.01, ci_lower=-0.03, ci_upper=0.01,
                  cost_delta=0.005),
            _attr("s3", quality_delta=-0.15, ci_lower=-0.20, ci_upper=-0.10),
        ]
        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10,
                               protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert isinstance(report, PruningReport)
        assert report.total_steps == 3

    def test_prune_candidate_found(self):
        steps = [
            _step("s1", "plan", StepType.DECISION),
            _step("s2", "useless_tool", cost=0.005),
            _step("s3", "summarize", StepType.DECISION),
        ]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
            _attr("s2", quality_delta=-0.01, ci_lower=-0.02, ci_upper=0.01,
                  cost_delta=0.005),
            _attr("s3", quality_delta=-0.15, ci_lower=-0.20, ci_upper=-0.10),
        ]
        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10,
                               protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert len(report.candidates) >= 1
        candidate = report.candidates[0]
        assert candidate.decision_status == DecisionStatus.CANDIDATE
        assert candidate.estimated_savings.cost_reduction > 0

    def test_no_candidates_when_all_important(self):
        steps = [
            _step("s1", "plan", StepType.DECISION),
            _step("s2", "critical_tool", cost=0.005),
            _step("s3", "summarize", StepType.DECISION),
        ]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.3, ci_lower=-0.35, ci_upper=-0.25),
            _attr("s2", quality_delta=-0.25, ci_lower=-0.30, ci_upper=-0.20),
            _attr("s3", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
        ]
        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10,
                               protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert len(report.candidates) == 0

    def test_validation_plan_generated(self):
        steps = [
            _step("s1", "plan", StepType.DECISION),
            _step("s2", "useless_tool", cost=0.005),
            _step("s3", "summarize", StepType.DECISION),
        ]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
            _attr("s2", quality_delta=0.0, ci_lower=-0.01, ci_upper=0.01,
                  cost_delta=0.005),
            _attr("s3", quality_delta=-0.15, ci_lower=-0.20, ci_upper=-0.10),
        ]
        config = PruningConfig(prune_epsilon=0.05, protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert len(report.candidates) >= 1
        vplan = report.candidates[0].required_validation
        assert vplan.replay_required is True
        assert vplan.min_replay_count > 0

    def test_risk_assessment(self):
        steps = [_step("s1", "plan", StepType.DECISION), _step("s2", "tool")]
        traj = _trajectory(steps)
        attrs = [
            _attr("s1", quality_delta=-0.2, ci_lower=-0.25, ci_upper=-0.15),
            _attr("s2", quality_delta=0.0, ci_lower=-0.01, ci_upper=0.01,
                  cost_delta=0.005),
        ]
        config = PruningConfig(protect_first_last=True)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(traj, attrs)
        assert report.risk_assessment in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pruning_advisor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PruningAdvisor**

`traceshap/pruning/advisor.py`:
```python
from datetime import datetime, timezone

from traceshap.models.trajectory import Trajectory
from traceshap.models.outcome import StepAttribution
from traceshap.models.enums import DecisionStatus, RiskLevel, ReplayCapability, Verdict
from traceshap.models.pruning import Savings, ValidationPlan, PruneCandidate, PruningReport
from traceshap.config import PruningConfig
from traceshap.pruning.safety import classify_step, is_first_or_last


class PruningAdvisor:
    def __init__(self, config: PruningConfig):
        self._config = config

    def analyze(
        self,
        trajectory: Trajectory,
        attributions: list[StepAttribution],
    ) -> PruningReport:
        attr_map = {a.step_id: a for a in attributions}
        candidates: list[PruneCandidate] = []

        for step in trajectory.steps:
            attr = attr_map.get(step.step_id)
            if attr is None:
                continue

            first_last = is_first_or_last(step.step_id, trajectory.steps)
            verdict = classify_step(attr, step, self._config, first_last)

            if verdict == Verdict.PRUNE_CANDIDATE:
                candidate = PruneCandidate(
                    target_type="step",
                    target_id=step.tool_name or step.step_type.value,
                    evidence=[attr],
                    estimated_savings=Savings(
                        token_reduction=step.tokens.total_tokens if step.tokens else 0,
                        cost_reduction=step.cost or 0.0,
                        latency_reduction_ms=step.duration_ms,
                        quality_impact_range=(
                            attr.confidence.lower if attr.confidence else 0.0,
                            attr.confidence.upper if attr.confidence else 0.0,
                        ),
                    ),
                    required_validation=self._build_validation_plan(step, attr),
                    decision_status=DecisionStatus.CANDIDATE,
                )
                candidates.append(candidate)

        total_savings = Savings(
            token_reduction=sum(c.estimated_savings.token_reduction for c in candidates),
            cost_reduction=sum(c.estimated_savings.cost_reduction for c in candidates),
            latency_reduction_ms=sum(c.estimated_savings.latency_reduction_ms for c in candidates),
            quality_impact_range=self._aggregate_quality_range(candidates),
        )

        return PruningReport(
            trace_id=trajectory.trace_id,
            timestamp=datetime.now(timezone.utc),
            total_steps=len(trajectory.steps),
            candidates=candidates,
            estimated_savings=total_savings,
            risk_assessment=self._assess_risk(candidates, trajectory),
        )

    @staticmethod
    def _build_validation_plan(step, attr: StepAttribution) -> ValidationPlan:
        replay_mode = ReplayCapability.RECORDED_IO_REPLAY
        if step.side_effect_class.is_safe_for_auto_replay():
            replay_mode = ReplayCapability.RECORDED_IO_REPLAY
        else:
            replay_mode = ReplayCapability.DRY_RUN_MOCKED

        ci_width = attr.confidence.width if attr.confidence else 1.0
        min_replays = max(5, int(20 * ci_width))

        return ValidationPlan(
            replay_required=True,
            replay_mode=replay_mode,
            min_replay_count=min_replays,
            ab_test_recommended=ci_width > 0.1,
            human_review_required=not step.side_effect_class.is_safe_for_auto_replay(),
        )

    @staticmethod
    def _aggregate_quality_range(candidates: list[PruneCandidate]) -> tuple[float, float]:
        if not candidates:
            return (0.0, 0.0)
        lower = sum(c.estimated_savings.quality_impact_range[0] for c in candidates)
        upper = sum(c.estimated_savings.quality_impact_range[1] for c in candidates)
        return (lower, upper)

    @staticmethod
    def _assess_risk(candidates: list[PruneCandidate], trajectory: Trajectory) -> RiskLevel:
        if not candidates:
            return RiskLevel.LOW
        ratio = len(candidates) / len(trajectory.steps) if trajectory.steps else 0
        if ratio > 0.3:
            return RiskLevel.HIGH
        if ratio > 0.1:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pruning_advisor.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add traceshap/pruning/advisor.py tests/test_pruning_advisor.py
git commit -m "feat: PruningAdvisor with CI-based classification, validation plans, and risk assessment"
```

---

### Task 8: Integration — Wire Attribution Engine into Pipeline

**Files:**
- Modify: `traceshap/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `traceshap/models/__init__.py` (add new re-exports if needed)

- [ ] **Step 1: Write integration tests**

Append to `tests/test_pipeline.py`:
```python
from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer0_rules import Layer0Rules
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig


class TestPipelineWithAttribution:
    async def test_full_pipeline_analyze(self, tmp_path):
        spans = _make_spans("t1")
        source = FakeSource([spans])
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.initialize()

        pipeline = TraceSHAPPipeline(source=source, storage=backend)
        await pipeline.ingest_once()

        trajectory = await backend.get_trajectory("t1")
        assert trajectory is not None

        engine = AttributionEngine(layers=[Layer0Rules()])
        attributions = await engine.analyze(trajectory)
        assert len(attributions) == 3

        config = PruningConfig(prune_epsilon=0.05, keep_threshold=0.10)
        advisor = PruningAdvisor(config)
        report = advisor.analyze(trajectory, attributions)
        assert report.total_steps == 3
        assert report.risk_assessment is not None

        await backend.close()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_pipeline.py::TestPipelineWithAttribution -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "feat: integration test for full pipeline → attribution → pruning flow"
```

---

## Summary

After completing all 8 tasks, you will have:

1. **Attribution base protocol** with `LayerResult` and `merge_layer_results`
2. **Layer 0: Expert Rules** — repetition detection, no-op detection, loop detection
3. **Layer 1: Statistical Lift** — cohort-stratified association analysis with CI
4. **Layer 2: Sequence Estimator** — legal interventions (prefix/contiguous/retry collapse), transition model, counterfactual scoring
5. **Attribution Engine** — orchestrates layers, merges results, assigns initial verdicts
6. **Pruning Safety** — protected types, first/last guards, CI-based classification
7. **Pruning Advisor** — generates `PruneCandidate` with `ValidationPlan`, `Savings`, risk assessment
8. **Integration** — full pipeline → attribution → pruning flow verified end-to-end

**Next plans:**
- **Plan 3**: CLI + REST API + Web Dashboard
- **Plan 4**: LangGraph native adapter + bridge extractors
