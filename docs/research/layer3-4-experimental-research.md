# Layer 3/4 实验层调研报告

> 调研日期：2026-05-22
> 基于 TraceSHAP v0.1.0 现有 Layer 0-2 实现

## 1. 概述

TraceSHAP 当前实现了三层归因分析：
- **Layer 0**：基于规则的模式检测（重复、循环、无效操作）
- **Layer 1**：统计提升（跨轨迹关联分析）
- **Layer 2**：序列感知反事实估计（基于 `TransitionModel` 的合法干预）

Layer 2 的核心限制是：它只通过 transition model 预测反事实结果，而不是真正重放 agent 来观测真实结果。Layer 3 和 Layer 4 旨在弥补这一差距：

| 层级 | 方法 | 信号强度 | 成本 | 适用场景 |
|------|------|----------|------|----------|
| Layer 3 (Replay SHAP) | 真实 agent 重放 + 步骤消融 | 高 | 高（API 调用 + 沙箱） | 离线 eval harness |
| Layer 4 (Causal Hypothesis) | 基于 Layer 3 数据的因果推断 | 最高（有条件） | 中（计算密集） | 离线分析报告 |

**核心设计原则：**
1. 实验层永远不在生产环境运行
2. 安全模型：`side_effect_class` 决定可用的 replay 模式
3. 预算约束：每条 trace 最多 K 次重放（默认 K=2n）
4. Layer 4 默认输出关联性假设（associational），只有在满足严格条件时才输出因果假设（causal）

---

## 2. Replay 基础设施调研

### 2.1 框架原生重放支持

#### LangGraph

LangGraph 提供了最完善的原生重放能力：

- **State Checkpointing**：`StateGraph` 支持通过 `MemorySaver` 或 `SqliteSaver` 持久化每个节点执行后的状态快照
- **Time Travel**：可以从任意 checkpoint 恢复执行，修改输入后重新运行
- **Replay API**：`graph.stream(None, config={"configurable": {"thread_id": ..., "checkpoint_id": ...}})` 可以从指定 checkpoint 重放

```python
# LangGraph 原生 replay 示例
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 获取历史状态
states = list(graph.get_state_history(config))

# 从某个 checkpoint 重放（修改状态后）
graph.update_state(target_config, {"messages": modified_messages})
result = graph.invoke(None, target_config)
```

**TraceSHAP 集成点：** 当前 `LangGraphSpanCollector` 只收集 span 数据，不捕获 checkpoint 信息。Layer 3 需要扩展 adapter 以同时记录 checkpointer 状态。

#### CrewAI

- **无原生 replay**：CrewAI v0.x 没有内置的执行重放机制
- **Task-level 粒度**：可以通过重新构建 `Crew` 并修改 `tasks` 列表来模拟消融
- **Memory System**：有 short-term/long-term memory，但不支持状态快照恢复
- **实现方案**：需要在 `CrewBase` 级别 hook，记录每个 task 的输入/输出用于 recorded-IO replay

#### AutoGen (v0.4+)

- **AgentChat Runtime**：支持通过 message history 恢复对话状态
- **GroupChat replay**：可以通过修改 `messages` 历史来模拟不同的对话路径
- **无原生 checkpoint**：需要自行实现状态序列化
- **实现方案**：拦截 `on_messages` 回调，记录所有 agent 间消息用于 replay

#### 综合评估

| 框架 | Checkpoint 支持 | Replay 原生支持 | 消融难度 | 推荐 Replay 模式 |
|------|----------------|----------------|----------|-----------------|
| LangGraph | 内置 | Time Travel API | 低 | LIVE_SANDBOX_REPLAY |
| CrewAI | 无 | 无 | 中 | RECORDED_IO_REPLAY |
| AutoGen | 手动 | 消息历史恢复 | 中 | RECORDED_IO_REPLAY |
| 通用 OTel | 无 | 无 | 高 | DRY_RUN_MOCKED |

### 2.2 工具调用模拟策略

工具调用模拟是 replay 安全性的核心。有三种主要策略：

#### 策略 1：Recorded I/O Replay（推荐的 MVP 方案）

在原始执行时记录所有工具调用的输入/输出对，replay 时用录制的输出替代真实执行：

```python
@dataclass
class RecordedIO:
    step_id: str
    tool_name: str
    input_hash: str          # 用于匹配
    input_data: dict         # 完整输入
    output_data: dict        # 录制的输出
    side_effect_class: SideEffect
    timestamp: datetime

class RecordedIOStore:
    """按 input_hash 索引的录制数据存储"""

    def lookup(self, tool_name: str, input_hash: str) -> RecordedIO | None:
        """精确匹配：相同工具 + 相同输入 hash"""
        ...

    def lookup_fuzzy(self, tool_name: str, input_data: dict,
                     similarity_threshold: float = 0.9) -> RecordedIO | None:
        """模糊匹配：用于输入略有变化的场景"""
        ...
```

**优点：** 零外部副作用，成本仅为 LLM API 调用费用
**缺点：** 当消融改变了上游步骤的输出时，下游步骤的输入可能与录制数据不匹配
**缓解方案：** 使用 fuzzy matching + 标记 confidence 降低

#### 策略 2：DRY_RUN_MOCKED

完全模拟所有外部调用，包括 LLM 调用：

```python
class MockedToolRegistry:
    """为每个工具注册 mock 函数"""

    def register_mock(self, tool_name: str,
                      mock_fn: Callable[[dict], dict]) -> None: ...

    def register_default_mock(self, tool_name: str,
                              default_output: dict) -> None: ...
```

**适用场景：** `IRREVERSIBLE_WRITE` 类步骤的安全评估
**限制：** 模拟的保真度直接影响归因质量

#### 策略 3：LIVE_SANDBOX_REPLAY

在隔离沙箱中执行真实工具调用：

- **Docker 容器沙箱**：通过 `docker-py` 启动临时容器，挂载工具运行时
- **网络隔离**：iptables 规则限制出站连接到 allow-list（如 LLM API endpoint）
- **文件系统快照**：使用 overlay filesystem 或 tmpfs
- **数据库沙箱**：SQLite in-memory 或 PostgreSQL schema clone

```python
class SandboxConfig:
    image: str = "traceshap-sandbox:latest"
    network_mode: str = "none"              # 默认无网络
    allowed_endpoints: list[str] = []       # 白名单
    timeout_seconds: int = 300
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
```

**适用场景：** 最高保真度需求，离线 eval 批量运行
**成本：** 高（需要 Docker、计算资源、LLM API 费用）

### 2.3 ReplayCapsule 数据采集

`ReplayCapsule` 是 replay 的核心数据结构，需要捕获完整重现执行所需的所有信息：

```python
@dataclass
class ReplayCapsule:
    # 标识
    capsule_id: str
    trace_id: str
    created_at: datetime

    # 轨迹数据
    trajectory: Trajectory                    # 完整轨迹（复用现有模型）
    outcome: Outcome                          # 原始结果

    # 模型快照
    model_id: str                             # e.g., "gpt-4o-2024-08-06"
    model_config: dict                        # temperature, top_p, etc.
    prompt_hashes: dict[str, str]             # step_id -> hash of system/user prompt
    system_prompts: dict[str, str]            # step_id -> 完整 system prompt 文本

    # 工具配置
    tool_schemas: dict[str, dict]             # tool_name -> JSON schema
    tool_versions: dict[str, str]             # tool_name -> version

    # 录制的 I/O
    recorded_ios: list[RecordedIO]            # 所有工具调用的录制数据

    # 环境快照
    environment_snapshot: EnvironmentSnapshot  # 环境变量、API keys（脱敏）、依赖版本

@dataclass
class EnvironmentSnapshot:
    python_version: str
    package_versions: dict[str, str]          # 关键包的版本
    env_vars_hash: str                        # 环境变量的 hash（不存储明文）
    framework_version: str                    # LangGraph/CrewAI 版本
    timestamp: datetime
```

**数据采集时机：**

1. **在线采集（推荐）**：扩展现有 `LangGraphSpanCollector` 等 adapter，在 `on_llm_end` / `on_tool_end` 时同步记录 I/O
2. **离线重建**：从已有的 OTel span 数据中提取（信息可能不完整）
3. **混合模式**：在线记录关键数据，离线补充环境快照

**存储方案：**

```python
# 扩展现有 SQLAlchemy 模型
class ReplayCapsuleORM(Base):
    __tablename__ = "replay_capsules"

    capsule_id = Column(String, primary_key=True)
    trace_id = Column(String, ForeignKey("traces.trace_id"), index=True)
    model_id = Column(String)
    model_config_json = Column(Text)          # JSON 序列化
    recorded_ios_json = Column(Text)          # JSON 序列化（大字段）
    environment_json = Column(Text)
    created_at = Column(DateTime)

    # 预计单条 capsule 大小：10KB ~ 500KB（取决于工具 I/O 数据量）
```

### 2.4 沙箱方案

#### 方案对比

| 方案 | 隔离级别 | 启动延迟 | 复杂度 | 适用场景 |
|------|----------|----------|--------|----------|
| `subprocess` + 环境变量覆盖 | 低 | <100ms | 低 | PURE/READ_ONLY 步骤 |
| Docker 容器 | 高 | 1-5s | 中 | IDEMPOTENT_WRITE 步骤 |
| Firecracker microVM | 最高 | 100-300ms | 高 | 安全性要求极高场景 |
| 进程内 mock（无沙箱） | 无 | 0ms | 最低 | DRY_RUN_MOCKED 模式 |

#### 推荐方案：分层沙箱

```python
class SandboxManager:
    """根据 side_effect_class 选择沙箱级别"""

    async def create_sandbox(
        self,
        capsule: ReplayCapsule,
        steps_to_ablate: list[str],
        mode: ReplayCapability,
    ) -> ReplayContext:
        max_side_effect = max(
            (s.side_effect_class for s in capsule.trajectory.steps
             if s.step_id not in steps_to_ablate),
            default=SideEffect.PURE,
        )

        if mode == ReplayCapability.DRY_RUN_MOCKED:
            return InProcessMockContext(capsule)
        elif mode == ReplayCapability.RECORDED_IO_REPLAY:
            return RecordedIOContext(capsule)
        elif mode == ReplayCapability.LIVE_SANDBOX_REPLAY:
            if max_side_effect == SideEffect.IRREVERSIBLE_WRITE:
                raise ReplayForbiddenError(
                    "IRREVERSIBLE_WRITE steps require DRY_RUN_MOCKED mode"
                )
            return DockerSandboxContext(capsule, self._sandbox_config)
```

---

## 3. Shapley 值计算策略

### 3.1 精确计算 vs 近似算法

#### 精确计算的不可行性

Shapley 值的精确计算需要遍历所有 2^n 个子集（n = 步骤数）。对于典型 agent 轨迹：

| 步骤数 | 子集数 | 以 K=2n 预算的可行性 |
|--------|--------|---------------------|
| 5 | 32 | 可行（预算 10，精确需 32） |
| 10 | 1,024 | 不可行（预算 20） |
| 15 | 32,768 | 不可行（预算 30） |
| 20 | 1,048,576 | 不可行（预算 40） |

结论：对于 n > 7 的轨迹，必须使用近似算法。

#### 近似算法对比

**1. Permutation Sampling（排列采样）**

随机采样 M 个排列，对每个排列计算每个步骤的边际贡献：

```python
import random

def permutation_shapley(
    steps: list[str],
    value_fn: Callable[[set[str]], float],  # replay 函数
    num_samples: int,
) -> dict[str, float]:
    n = len(steps)
    phi = {s: 0.0 for s in steps}

    for _ in range(num_samples):
        perm = random.sample(steps, n)
        prev_value = value_fn(set())
        for i, step in enumerate(perm):
            coalition = set(perm[:i + 1])
            curr_value = value_fn(coalition)
            phi[step] += (curr_value - prev_value)
            prev_value = curr_value

    return {s: v / num_samples for s, v in phi.items()}
```

**优点：** 无偏估计，实现简单
**缺点：** 每个 sample 需要 n 次 value function 评估（= n 次 replay）
**总 replay 次数：** M * n（远超预算）

**优化：** 利用结果缓存 —— 相同 coalition 的 replay 结果可复用。

**2. Kernel SHAP（加权最小二乘）**

将 Shapley 值计算转化为加权线性回归问题，只需采样 O(n log n) 个 coalitions：

```python
def kernel_shap(
    steps: list[str],
    value_fn: Callable[[set[str]], float],
    budget: int,                              # 最大 replay 次数
) -> dict[str, float]:
    n = len(steps)
    # 采样 budget 个随机 coalitions
    coalitions = sample_coalitions(steps, budget)

    # 构建设计矩阵 X (budget x n) 和响应向量 y (budget,)
    X = np.zeros((budget, n))
    y = np.zeros(budget)
    weights = np.zeros(budget)

    for i, coalition in enumerate(coalitions):
        for j, step in enumerate(steps):
            X[i, j] = 1.0 if step in coalition else 0.0
        y[i] = value_fn(coalition)
        # Kernel SHAP 权重
        k = len(coalition)
        if 0 < k < n:
            weights[i] = (n - 1) / (comb(n, k) * k * (n - k))

    # 加权最小二乘
    W = np.diag(weights)
    phi = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
    return {steps[j]: phi[j] for j in range(n)}
```

**优点：** replay 次数 = budget（精确控制成本）
**缺点：** 有偏估计（尤其在 budget 极小时）
**推荐：** 作为 Layer 3 的默认算法

**3. Stratified Sampling（分层采样）**

按 coalition 大小分层采样，确保不同大小的 coalition 都有代表性：

```python
def stratified_shapley(steps, value_fn, budget):
    n = len(steps)
    samples_per_size = max(1, budget // n)
    # 对每个 coalition 大小 k=1..n-1，采样 samples_per_size 个 coalitions
    ...
```

**优点：** 比纯随机采样更稳定
**缺点：** 实现稍复杂

### 3.2 预算约束下的采样策略

#### 自适应预算分配

Layer 3 的默认预算为 K = 2n。建议的分配策略：

```python
@dataclass
class ReplayBudget:
    max_replays: int                          # K = 2n
    used: int = 0

    # 预算分配策略
    mandatory_coalitions: int = 2             # 空集 + 全集（基准值）
    per_step_singleton: int = 0              # 每个步骤的单步消融（可选）

    @classmethod
    def from_trajectory(cls, n_steps: int, multiplier: float = 2.0) -> "ReplayBudget":
        budget = int(n_steps * multiplier)
        return cls(max_replays=budget)

    @property
    def remaining(self) -> int:
        return self.max_replays - self.used
```

#### 优先采样策略

并非所有 coalition 的信息量相等。建议优先采样以下类型：

1. **单步消融（leave-one-out）**：每次移除一个步骤，直接衡量单步影响。需要 n 次 replay。
2. **高不确定性区域**：根据 Layer 2 的 confidence interval 宽度，优先 replay 不确定性最高的步骤组合。
3. **Layer 0/1 标记的可疑步骤**：对已被标记为 `REVIEW` 或 `PRUNE_CANDIDATE` 的步骤优先消融。

```python
def prioritized_sampling(
    steps: list[CanonicalStep],
    layer2_results: list[LayerResult],
    budget: ReplayBudget,
) -> list[set[str]]:
    """生成优先级排序的 coalition 采样计划"""
    coalitions = []

    # 1. 空集和全集（基准）
    coalitions.append(set())
    coalitions.append({s.step_id for s in steps})

    # 2. Leave-one-out（按 Layer 2 不确定性排序）
    by_uncertainty = sorted(
        zip(steps, layer2_results),
        key=lambda x: x[1].confidence_upper - x[1].confidence_lower,
        reverse=True,
    )
    all_ids = {s.step_id for s in steps}
    for step, _ in by_uncertainty:
        if budget.remaining <= len(coalitions):
            break
        coalitions.append(all_ids - {step.step_id})

    # 3. 随机补充（Kernel SHAP 权重采样）
    while len(coalitions) < budget.max_replays:
        coalitions.append(random_coalition_with_kernel_weight(steps))

    return coalitions[:budget.max_replays]
```

#### 缓存与去重

```python
class ReplayCache:
    """基于 frozenset(ablated_step_ids) 的结果缓存"""

    def __init__(self):
        self._cache: dict[frozenset[str], Outcome] = {}

    def get(self, ablated: set[str]) -> Outcome | None:
        return self._cache.get(frozenset(ablated))

    def put(self, ablated: set[str], outcome: Outcome) -> None:
        self._cache[frozenset(ablated)] = outcome
```

### 3.3 现有库支持

#### `shap` 库 (v0.45+)

`shap` 是最成熟的 Shapley 值计算库，但其设计面向 ML 模型解释：

- **KernelExplainer**：适用于任意黑盒模型。可以将 replay 函数包装为 `model` 参数：

```python
import shap
import numpy as np

def replay_value_function(coalition_matrix: np.ndarray) -> np.ndarray:
    """将 shap 的 coalition 矩阵转换为 replay 调用"""
    results = []
    for row in coalition_matrix:
        ablated = {steps[i].step_id for i, v in enumerate(row) if v == 0}
        outcome = replay_engine.replay_without(capsule, ablated, mode)
        results.append(outcome.quality_score)
    return np.array(results)

explainer = shap.KernelExplainer(
    replay_value_function,
    data=np.ones((1, n_steps)),    # background: 全集
    nsamples=budget,
)
shap_values = explainer.shap_values(np.ones((1, n_steps)))
```

**问题：**
- `shap` 假设 value function 评估很快，没有内置的 budget 控制（`nsamples` 参数可用但不精确）
- 异步支持不好 —— `shap` 是同步的，需要在 `asyncio.to_thread` 中运行
- 不支持自定义采样策略

**结论：** 可以用于 MVP，但长期应自行实现 Kernel SHAP 以获得更好的控制。

#### 其他相关库

| 库 | 适用性 | 备注 |
|---|--------|------|
| `shap` | 中 | Kernel SHAP 可用，但缺乏异步和预算控制 |
| `sage` (SAGE) | 低 | 面向特征重要性，非 coalition game |
| `captum` (PyTorch) | 低 | 面向深度学习模型，API 不匹配 |
| 自行实现 Kernel SHAP | 高 | ~200 行代码，完全控制预算和异步 |

---

## 4. 因果推断方法论

### 4.1 因果图构建

Agent 轨迹天然包含三种依赖关系，可构建 DAG（有向无环图）：

#### 边类型定义

```python
class EdgeType(Enum):
    CONTROL_FLOW = "control_flow"       # 步骤 A 的结果决定是否执行步骤 B
    DATA_DEPENDENCY = "data_dependency" # 步骤 A 的输出是步骤 B 的输入
    TEMPORAL = "temporal"               # 仅时间顺序关联

@dataclass
class CausalEdge:
    source_step_id: str
    target_step_id: str
    edge_type: EdgeType
    confidence: float                   # 0.0 ~ 1.0
    evidence: str
```

#### 图构建算法

```python
class TrajectoryGraphBuilder:
    """从轨迹数据自动构建因果假设图"""

    def build(self, trajectory: Trajectory) -> list[CausalEdge]:
        edges = []

        for i, step_a in enumerate(trajectory.steps):
            for j, step_b in enumerate(trajectory.steps):
                if j <= i:
                    continue

                # 1. 数据依赖检测
                if self._has_data_dependency(step_a, step_b, trajectory):
                    edges.append(CausalEdge(
                        source_step_id=step_a.step_id,
                        target_step_id=step_b.step_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        confidence=0.7,
                        evidence="output_hash overlap detected",
                    ))

                # 2. 控制流检测
                elif self._has_control_flow(step_a, step_b, trajectory):
                    edges.append(CausalEdge(
                        source_step_id=step_a.step_id,
                        target_step_id=step_b.step_id,
                        edge_type=EdgeType.CONTROL_FLOW,
                        confidence=0.9,
                        evidence="conditional branching observed",
                    ))

                # 3. 时间依赖（fallback）
                elif j == i + 1:
                    edges.append(CausalEdge(
                        source_step_id=step_a.step_id,
                        target_step_id=step_b.step_id,
                        edge_type=EdgeType.TEMPORAL,
                        confidence=0.3,
                        evidence="adjacent in sequence",
                    ))

        return edges

    def _has_data_dependency(self, a: CanonicalStep, b: CanonicalStep,
                             trajectory: Trajectory) -> bool:
        """检测 A 的输出是否出现在 B 的输入中（基于 span 数据）"""
        a_spans = [s for s in trajectory.spans if s.span_id in a.raw_span_ids]
        b_spans = [s for s in trajectory.spans if s.span_id in b.raw_span_ids]

        for a_span in a_spans:
            for b_span in b_spans:
                if a_span.output and b_span.input:
                    # 简单策略：检查输出文本是否出现在输入中
                    a_out = str(a_span.output)
                    b_in = str(b_span.input)
                    if len(a_out) > 20 and a_out[:100] in b_in:
                        return True
        return False
```

#### 跨轨迹图聚合

单条轨迹的因果图可能不可靠。通过聚合多条相同任务的轨迹，可以提高图的鲁棒性：

```python
def aggregate_graphs(
    graphs: list[list[CausalEdge]],
    min_frequency: float = 0.5,       # 至少在 50% 的轨迹中出现
) -> list[CausalEdge]:
    """聚合多条轨迹的因果图，保留高频边"""
    edge_counts: Counter = Counter()
    edge_examples: dict = {}

    for graph in graphs:
        for edge in graph:
            key = (edge.source_step_id, edge.target_step_id, edge.edge_type)
            edge_counts[key] += 1
            edge_examples[key] = edge

    total = len(graphs)
    return [
        CausalEdge(
            source_step_id=edge_examples[key].source_step_id,
            target_step_id=edge_examples[key].target_step_id,
            edge_type=edge_examples[key].edge_type,
            confidence=count / total,
            evidence=f"observed in {count}/{total} trajectories",
        )
        for key, count in edge_counts.items()
        if count / total >= min_frequency
    ]
```

### 4.2 观测 vs 干预估计

这是 Layer 4 设计中最关键的区分：

| 估计类型 | 数据来源 | 可以声称 | 不可以声称 |
|----------|----------|----------|------------|
| 观测（observational） | 跨轨迹的统计关联 | "步骤 A 与成功相关" | "步骤 A 导致成功" |
| 干预（interventional） | Layer 3 replay 消融 | "移除步骤 A 会改变结果" | "步骤 A 是充分原因" |
| 反事实（counterfactual） | 多次随机化 replay | "如果 A 不发生，B 不会发生" | 需要强假设 |

#### 干预估计的条件

只有满足以下条件时，Layer 4 才将假设标记为 `"causal"`：

1. **replay 干预数据**：至少有 1 次成功的 Layer 3 replay，移除目标步骤后观察到结果变化
2. **多次 replay 一致性**：如果有多次 replay，结果方向一致（同向变化 > 80%）
3. **无混杂路径**：因果图中不存在从被消融步骤到结果的替代路径

```python
@dataclass
class CausalHypothesis:
    hypothesis_type: str                      # "causal" | "associational"
    source_step_id: str
    target: str                               # "outcome" | step_id
    effect_direction: str                     # "positive" | "negative" | "neutral"
    effect_magnitude: float                   # 效应量
    downstream_effects: list[str]             # 受影响的下游步骤
    evidence_sources: list[str]               # ["replay_intervention", "cross_trajectory", ...]
    confidence_by_edge_type: dict[str, float] # {"control_flow": 0.9, ...}

    @property
    def is_causal(self) -> bool:
        return self.hypothesis_type == "causal"
```

### 4.3 置信度评估

#### 按边类型的默认置信度

基于因果推断文献和 agent 轨迹的特点：

| 边类型 | 基础置信度 | 提升条件 | 上限 |
|--------|-----------|----------|------|
| CONTROL_FLOW | 0.7 | + replay 验证 → 0.9 | 0.95 |
| DATA_DEPENDENCY | 0.5 | + 精确 hash 匹配 → 0.7; + replay → 0.85 | 0.90 |
| TEMPORAL | 0.2 | + 跨轨迹高频 → 0.4; + replay → 0.6 | 0.70 |

#### 置信度计算框架

```python
def compute_confidence(
    edge: CausalEdge,
    replay_results: list[ReplayResult] | None,
    cross_trajectory_freq: float | None,
) -> float:
    base = {
        EdgeType.CONTROL_FLOW: 0.7,
        EdgeType.DATA_DEPENDENCY: 0.5,
        EdgeType.TEMPORAL: 0.2,
    }[edge.edge_type]

    adjustments = 0.0

    # replay 干预证据
    if replay_results:
        consistent = sum(1 for r in replay_results if r.direction_consistent)
        consistency_rate = consistent / len(replay_results)
        adjustments += 0.2 * consistency_rate

    # 跨轨迹频率
    if cross_trajectory_freq is not None:
        adjustments += 0.1 * cross_trajectory_freq

    cap = {
        EdgeType.CONTROL_FLOW: 0.95,
        EdgeType.DATA_DEPENDENCY: 0.90,
        EdgeType.TEMPORAL: 0.70,
    }[edge.edge_type]

    return min(base + adjustments, cap)
```

### 4.4 现有框架对比

#### DoWhy

- **核心能力**：图-based 因果推断，支持识别（identification）和估计（estimation）
- **适用性**：高 —— 可以接受用户定义的因果图，提供多种估计器
- **集成方式**：将轨迹数据转换为 `pandas.DataFrame`，步骤是否存在作为 treatment variable

```python
import dowhy

# 将轨迹数据转为 DoWhy 格式
data = trajectories_to_dataframe(trajectories)
model = dowhy.CausalModel(
    data=data,
    treatment="step_A_present",
    outcome="quality_score",
    graph=trajectory_dag_to_dot(causal_graph),
)
identified = model.identify_effect()
estimate = model.estimate_effect(
    identified,
    method_name="backdoor.linear_regression",
)
```

**限制：** DoWhy 假设大样本（数百条轨迹），单条轨迹不适用。

#### EconML

- **核心能力**：异质性处理效应估计（CATE）
- **适用性**：中 —— 当有足够轨迹数据时，可以估计 "在特定条件下移除步骤 X 的效应"
- **典型用法**：`CausalForestDML` 或 `LinearDML`

#### CausalML (Uber)

- **核心能力**：uplift modeling，处理效应估计
- **适用性**：中 —— 更适合 A/B 测试场景
- **与 TraceSHAP 的差异**：CausalML 面向用户级别处理效应，非步骤级别

#### 综合建议

| 框架 | 推荐度 | 理由 |
|------|--------|------|
| DoWhy | 高 | 图-based，与 Layer 4 的因果图设计天然契合 |
| 自行实现 | 高 | 轨迹数据的特殊性（少样本、序列结构）需要定制 |
| EconML | 中 | 在大规模 eval 场景下有价值 |
| CausalML | 低 | 设计目标不匹配 |

**MVP 推荐：** Layer 4 MVP 应自行实现简单的关联分析 + 图构建，不依赖外部因果推断库。当用户积累了足够的 replay 数据后，可选集成 DoWhy 进行更严格的因果估计。

---

## 5. 实现方案建议

### 5.1 最小可行 Layer 3 (MVP)

**范围：** 仅支持 RECORDED_IO_REPLAY 模式，使用 Kernel SHAP 近似

**文件结构：**

```
traceshap/attribution/
├── layer3_replay.py          # Layer 3 主逻辑
├── replay/
│   ├── __init__.py
│   ├── capsule.py            # ReplayCapsule 数据结构
│   ├── engine.py             # ReplayEngine（核心）
│   ├── recorded_io.py        # RecordedIO 存储和匹配
│   ├── budget.py             # 预算管理
│   └── cache.py              # 结果缓存
```

**核心流程：**

```
ReplayCapsule → 预算分配 → Coalition 采样 → Replay 执行 → Kernel SHAP 计算 → LayerResult
```

**MVP 实现清单：**

| 组件 | 代码量估计 | 复杂度 | 依赖 |
|------|-----------|--------|------|
| `ReplayCapsule` dataclass | ~80 行 | 低 | 无新依赖 |
| `RecordedIOStore` | ~120 行 | 低 | 无新依赖 |
| `ReplayEngine` (recorded-IO only) | ~200 行 | 中 | 无新依赖 |
| `KernelSHAP` (自行实现) | ~150 行 | 中 | `numpy`（已有） |
| `ReplayBudget` | ~50 行 | 低 | 无新依赖 |
| `Layer3Replay` (AttributionLayer) | ~150 行 | 中 | 以上所有 |
| `ReplayCapsuleCollector` (adapter 扩展) | ~100 行 | 中 | 框架相关 |
| 测试 | ~300 行 | 中 | `pytest` |
| **总计** | **~1150 行** | | |

**关键设计决策：**

1. **Replay 粒度**：消融单位为 `CanonicalStep`，与 Layer 2 一致
2. **Coalition 表示**：`frozenset[str]` (step_ids to ablate)，而非 "保留哪些步骤"
3. **Recorded-IO 匹配**：先精确匹配 `input_hash`，失败则标记为 `LOW_CONFIDENCE`
4. **异步执行**：`ReplayEngine.replay_without()` 为 `async` 方法，支持并发 replay

**MVP 代码骨架：**

```python
# traceshap/attribution/layer3_replay.py

class Layer3Replay:
    """Replay SHAP attribution layer (experimental)"""

    def __init__(
        self,
        replay_engine: ReplayEngine,
        budget_multiplier: float = 2.0,
        default_mode: ReplayCapability = ReplayCapability.RECORDED_IO_REPLAY,
    ):
        self._engine = replay_engine
        self._budget_multiplier = budget_multiplier
        self._default_mode = default_mode

    @property
    def layer_id(self) -> int:
        return 3

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        capsule = self._engine.get_capsule(trajectory.trace_id)
        if capsule is None:
            return self._fallback_no_capsule(trajectory)

        n = len(trajectory.steps)
        budget = ReplayBudget.from_trajectory(n, self._budget_multiplier)

        # 生成采样计划
        coalitions = self._sample_coalitions(trajectory, budget)

        # 执行 replay
        coalition_values = {}
        for ablated_set in coalitions:
            mode = self._select_mode(trajectory, ablated_set)
            outcome = await self._engine.replay_without(
                capsule, list(ablated_set), mode
            )
            coalition_values[frozenset(ablated_set)] = (
                outcome.quality_score if outcome else 0.0
            )

        # Kernel SHAP 计算
        shap_values = kernel_shap_from_coalitions(
            step_ids=[s.step_id for s in trajectory.steps],
            coalition_values=coalition_values,
        )

        # 转换为 LayerResult
        return self._to_layer_results(trajectory, shap_values)
```

### 5.2 最小可行 Layer 4 (MVP)

**范围：** 仅输出 associational 假设，基于跨轨迹统计 + Layer 2/3 证据聚合

**文件结构：**

```
traceshap/attribution/
├── layer4_causal.py          # Layer 4 主逻辑
├── causal/
│   ├── __init__.py
│   ├── graph_builder.py      # 因果图构建
│   ├── hypothesis.py         # CausalHypothesis 数据结构
│   └── confidence.py         # 置信度计算
```

**MVP 实现清单：**

| 组件 | 代码量估计 | 复杂度 | 依赖 |
|------|-----------|--------|------|
| `CausalHypothesis` dataclass | ~60 行 | 低 | 无新依赖 |
| `CausalEdge` + `EdgeType` | ~40 行 | 低 | 无新依赖 |
| `TrajectoryGraphBuilder` | ~200 行 | 中 | 无新依赖 |
| `ConfidenceCalculator` | ~100 行 | 低 | 无新依赖 |
| `Layer4Causal` (AttributionLayer) | ~200 行 | 中 | Layer 3 结果（可选） |
| 测试 | ~250 行 | 中 | `pytest` |
| **总计** | **~850 行** | | |

**关键设计决策：**

1. **默认模式**：MVP 阶段所有假设标记为 `"associational"`
2. **升级为 causal 的条件**：需要 Layer 3 replay 数据，且满足 4.2 节中的三个条件
3. **图可视化**：复用现有的 `plotly` 依赖，生成有向图
4. **与现有 `StepAttribution.causal_hypothesis` 字段的集成**：Layer 4 的结果填充此字段

**MVP 代码骨架：**

```python
# traceshap/attribution/layer4_causal.py

class Layer4Causal:
    """Causal Hypothesis attribution layer (experimental)"""

    def __init__(
        self,
        graph_builder: TrajectoryGraphBuilder,
        layer3_results: dict[str, list[LayerResult]] | None = None,
    ):
        self._graph_builder = graph_builder
        self._layer3_results = layer3_results or {}

    @property
    def layer_id(self) -> int:
        return 4

    async def analyze(self, trajectory: Trajectory) -> list[LayerResult]:
        # 1. 构建因果图
        edges = self._graph_builder.build(trajectory)

        # 2. 为每个步骤生成假设
        results = []
        for step in trajectory.steps:
            downstream = self._find_downstream(step.step_id, edges)
            hypothesis_type = self._classify_hypothesis(
                step.step_id, edges, trajectory
            )

            confidence = self._compute_step_confidence(step.step_id, edges)

            results.append(LayerResult(
                layer=4,
                step_id=step.step_id,
                quality_delta=self._aggregate_effect(step.step_id, edges),
                cost_delta=step.cost or 0.0,
                latency_delta=step.duration_ms,
                risk_delta=0.0,
                confidence_lower=confidence * 0.8,
                confidence_upper=min(confidence * 1.2, 1.0),
                evidence=f"causal graph: {len(downstream)} downstream effects, "
                         f"type={hypothesis_type}",
            ))

        return results
```

### 5.3 依赖项清单

#### MVP（无新依赖）

Layer 3 和 Layer 4 的 MVP 可以不引入任何新依赖：

| 现有依赖 | 用途 |
|----------|------|
| `numpy` | Kernel SHAP 矩阵运算 |
| `sqlalchemy` | ReplayCapsule 持久化 |
| `plotly` | 因果图可视化 |

#### 完整版（可选依赖）

```toml
[project.optional-dependencies]
experimental = [
    "shap>=0.45",             # KernelExplainer 备选
    "dowhy>=0.11",            # 正式因果推断
    "networkx>=3.2",          # 图算法（路径搜索、环检测）
    "docker>=7.0",            # 沙箱 replay
]
```

### 5.4 工作量评估

| 阶段 | 内容 | 估计工时 | 前置依赖 |
|------|------|----------|----------|
| **Phase 1: 数据采集** | ReplayCapsule + RecordedIO + adapter 扩展 | 3-4 天 | 无 |
| **Phase 2: Replay 引擎** | ReplayEngine (recorded-IO) + 缓存 + 预算 | 4-5 天 | Phase 1 |
| **Phase 3: Kernel SHAP** | 自行实现 + 采样策略 | 2-3 天 | Phase 2 |
| **Phase 4: Layer 3 集成** | `Layer3Replay` + 引擎注册 + 测试 | 2-3 天 | Phase 3 |
| **Phase 5: 因果图** | GraphBuilder + EdgeType + 置信度 | 3-4 天 | 无（可并行） |
| **Phase 6: Layer 4 集成** | `Layer4Causal` + 假设分类 + 测试 | 2-3 天 | Phase 5 |
| **Phase 7: 端到端测试** | 集成测试 + 文档 + 示例 | 2-3 天 | Phase 4, 6 |
| **总计** | | **18-25 天** | |

**建议开发顺序：**

```
Phase 1 → Phase 2 → Phase 3 → Phase 4
                                  ↘
Phase 5 ──────────────────→ Phase 6 → Phase 7
```

Phase 1-4（Layer 3）和 Phase 5-6（Layer 4 图构建）可以部分并行。

---

## 6. 学术参考

### Shapley 值与序列决策过程

1. **SVERL: Shapley Values for Explaining Reinforcement Learning** (2025)
   - arXiv: 2505.07797
   - 核心贡献：将 Shapley 值应用于 RL 序列决策，定义了 state-action 级别的 coalition game
   - 与 TraceSHAP 的关系：直接适用于 agent trajectory 的步骤级归因

2. **SHARP: Shapley Value-based Multi-Agent Credit Assignment** (2026)
   - arXiv: 2602.08335
   - 核心贡献：多 agent 系统中通过反事实 masking 隔离每个 agent 的因果影响
   - 与 TraceSHAP 的关系：其 masking 策略可用于 replay 消融设计

3. **A Unified Approach to Shapley Value Computation** (Jia et al., 2019)
   - arXiv: 1904.02868
   - 核心贡献：分析了 Kernel SHAP、Sampling SHAP 等近似算法的收敛性
   - 与 TraceSHAP 的关系：指导预算分配策略

4. **Data Shapley: Equitable Valuation of Data for Machine Learning** (Ghorbani & Zou, 2019)
   - ICML 2019
   - 核心贡献：将 Shapley 值用于数据估值，提出 TMC-Shapley 截断采样
   - 与 TraceSHAP 的关系：TMC 截断策略可直接用于 replay 预算节省

### Agent 轨迹分析

5. **AgentSHAP: Interpreting LLM-Based Agent Tool Usage via SHAP** (2025)
   - arXiv: 2512.12597
   - 核心贡献：Monte Carlo Shapley 计算 tool 重要性
   - 局限：只做 tool 级别，不做 step 级别

6. **AgentDiagnose: Diagnosing LLM Agent Capabilities** (EMNLP 2025)
   - ACL Anthology: 2025.emnlp-demos.15
   - 核心贡献：5 维度 agent 能力诊断

7. **TrajAD: Trajectory Anomaly Detection for LLM Agents** (2026)
   - arXiv: 2602.06443
   - 核心贡献：轨迹异常检测 + step-level error localization
   - 与 TraceSHAP 的关系：异常检测信号可辅助 Layer 3 的采样优先级

### 因果推断方法论

8. **Causal Inference in Statistics: A Primer** (Pearl et al., 2016)
   - 基础教材，介绍 do-calculus 和因果图
   - 与 TraceSHAP 的关系：Layer 4 因果图的理论基础

9. **Elements of Causal Inference** (Peters et al., 2017)
   - MIT Press, 开放获取
   - 核心贡献：因果发现算法（PC、FCI）在有限数据下的表现分析
   - 与 TraceSHAP 的关系：指导 Layer 4 在少样本场景下的保守策略

10. **DoWhy: An End-to-End Library for Causal Inference** (Sharma & Kiciman, 2020)
    - arXiv: 2011.04216
    - 核心贡献：四步因果推断框架（建模 → 识别 → 估计 → 反驳）
    - 与 TraceSHAP 的关系：Layer 4 完整版的候选框架

### Replay 与评估

11. **Offline Evaluation of LLM Agents** (各大厂实践总结)
    - OpenAI Codex 的 "measure → improve → ship" 循环
    - Anthropic 的 eval-driven development
    - 与 TraceSHAP 的关系：Layer 3 的 eval harness 设计参考

12. **VCR: Virtual Conversation Replay for LLM Testing** (2025)
    - 核心思路：录制 LLM 对话的 I/O，replay 时使用录制数据替代真实 API 调用
    - 与 TraceSHAP 的关系：直接适用于 RECORDED_IO_REPLAY 模式

---

## 7. 风险与开放问题

### 高风险项

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| **Replay 不保真** | 消融后的轨迹与真实反事实差异过大，导致 Shapley 值不准确 | 引入 replay fidelity score，当 recorded-IO 匹配率 < 70% 时降级为 Layer 2 估计 |
| **LLM 非确定性** | 相同输入 + temperature=0 仍可能产生不同输出 | 多次 replay 取均值；报告方差作为 confidence interval 的组成部分 |
| **预算不足** | K=2n 可能不够支撑可靠的 Shapley 估计 | 引入自适应预算：当估计方差过大时，提示用户增加预算 |
| **因果声称过强** | 用户误将 associational 假设当作因果结论 | MVP 强制所有假设标记为 associational；UI 中明确标注 |

### 开放问题

1. **消融粒度选择**：当一个 `CanonicalStep` 包含多个 sub-span（如 LLM 调用 + tool 调用）时，应该消融整个 step 还是单个 sub-span？
   - 建议：MVP 以 step 为单位，后续版本支持 sub-step 消融

2. **跨步骤依赖处理**：消融步骤 A 时，依赖 A 输出的步骤 B 应该如何处理？
   - 方案 A：同时消融 B（保守，但丢失信息）
   - 方案 B：用 default 值替代 A 的输出（需要 per-tool default 配置）
   - 方案 C：让 replay 自然处理（LLM 会适应缺失的上下文）
   - 建议：MVP 使用方案 C（最简单），辅以 confidence 降低标记

3. **成本模型**：Layer 3 的 replay 涉及真实 LLM API 调用，如何向用户展示预估成本？
   - 建议：基于 `TokenUsage` 数据估算成本，在执行前展示 "预计消耗 X tokens / $Y"

4. **与 Layer 2 的融合**：Layer 2 的 `TransitionModel` 预测和 Layer 3 的 replay 结果是否应该加权融合？
   - 建议：当 Layer 3 可用时，以 Layer 3 为准（`merge_layer_results` 已实现 highest-layer-wins 策略）

5. **隐私与安全**：`ReplayCapsule` 中存储了完整的 prompt 和工具 I/O，可能包含敏感信息
   - 建议：增加 `capsule.redact()` 方法，支持正则表达式脱敏；默认不持久化 system prompt 的原文

6. **图的方向性**：agent 轨迹中的因果关系并非总是单向的（如 retry 循环），如何处理循环依赖？
   - 建议：将 retry 循环折叠为单个节点（复用 Layer 0 的 loop 检测），确保图为 DAG
