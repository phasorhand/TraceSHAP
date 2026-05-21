# TraceSHAP 调研报告：Agent Trajectory 归因分析全景

> 调研日期：2026-05-20

## 一、核心定位

TraceSHAP 的目标是对 LLM Agent 的 trajectory（执行轨迹）做**步骤级归因分析**，回答一个核心问题：

> **在一次 agent 执行中，哪些步骤/决策对最终结果的贡献最大（或最具破坏性）？**

遵循 OpenTelemetry 协议，无痛嵌入 LangGraph、OpenClaw、Hermes 等主流 agent 框架。

---

## 二、现有开源项目

### 2.1 直接相关项目

| 项目 | 时间 | 核心做法 | 与 TraceSHAP 的差异 |
|------|------|----------|-------------------|
| **AgentSHAP** | 2025.12 | Monte Carlo Shapley 计算 tool 重要性，agent 作为黑盒，通过不同 tool 子集测试 | 只做 **tool 级别**归因，不做 trajectory step 级别 |
| **llmSHAP** | 2025.11 | 对 LLM 决策支持场景做 Shapley 解释 | 面向单次 LLM 调用，非 agent 多步轨迹 |
| **AgentDiagnose** | EMNLP 2025 | 量化 5 种 agent 核心能力（回溯、分解、观察、自验证、目标质量），t-SNE 可视化 | 做能力诊断，不做 Shapley 归因 |
| **AgentTrace** | 2026.02 | 三层分类（认知/操作/上下文）的结构化日志，集成 OTel | 侧重安全/可审计，无归因分析 |
| **TrajAD** | 2026.02 | 轨迹异常检测，含 TrajBench 数据集，step-level error localization | 做异常检测，非贡献度归因 |

**关键论文：**
- AgentSHAP: https://arxiv.org/abs/2512.12597
- llmSHAP: https://arxiv.org/abs/2511.01311 (代码: https://github.com/filipnaudot/llmSHAP)
- AgentDiagnose: https://aclanthology.org/2025.emnlp-demos.15/
- AgentTrace: https://arxiv.org/abs/2602.10133
- TrajAD: https://arxiv.org/abs/2602.06443
- Agentic Transparency Survey (Vector Institute): https://github.com/VectorInstitute/Agentic-Transparency

### 2.2 理论基础项目

| 项目 | 核心思路 |
|------|----------|
| **SVERL** (Shapley Values for RL) | 用 Shapley 值解释 RL 序列决策，直接适用于 agent trajectory |
| **SHARP** | 多 agent 系统的 Shapley 信用分配，通过反事实 masking 隔离每个 agent 的因果影响 |

- SVERL: https://arxiv.org/abs/2505.07797
- SHARP: https://arxiv.org/abs/2602.08335

### 2.3 TraceSHAP 的差异化空间

**目前没有任何项目同时满足：**
1. 对 trajectory 做 **step 级别** Shapley 归因（AgentSHAP 只到 tool 级别）
2. **OTel 原生**集成（AgentDiagnose 无 OTel）
3. **跨框架**支持 LangGraph + OpenClaw + Hermes（LangSmith 只支持 LangGraph）

---

## 三、各大厂/项目的做法

### 3.1 OpenAI Codex

- 基于 codex-1（o3 变体），每个 worktree 有临时可观测栈
- 可用 LogQL 查询日志、PromQL 查询指标
- 采用 **"measure → improve → ship"** 循环：生产 trace 中失败的会被转为 eval case
- GitHub Action 在每个 PR 上跑 eval，低于质量阈值阻止合并
- 原生框架 adapter 将 span/attribute/event 归一化为统一 schema，OTel 作为 fallback
- 来源: https://openai.com/index/introducing-codex/

### 3.2 Claude Code (Anthropic)

- 遵循 ReAct 模式（reasoning + tool invocation → harness 执行 → 结果反馈）
- 核心理念是 **eval-driven development**：先定义 eval 再让 agent 去满足
- Agent SDK 支持 MLflow GenAI tracing
- 内部对几乎每个 PR 运行同一套 code review 系统
- 架构分析论文: https://arxiv.org/abs/2604.14228
- Eval 指南: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

### 3.3 Hermes (NousResearch)

- **Atropos RL Environments**：通过多样环境收集和评估 LLM trajectory，作为 service 与 trajectory API 交互
- **批量轨迹生成**：并行运行数千条 tool-calling trajectory，自动 checkpoint，输出 ShareGPT 格式用于微调
- **轨迹压缩**：减少 trajectory 数据体积用于高效存储/训练
- 11 种 tool-call parser，覆盖主流模型架构
- Hermes 4 支持 hybrid 模式：直接响应或显式推理链
- 文档: https://hermes-agent.nousresearch.com/docs/
- 代码: https://github.com/NousResearch/hermes-agent

### 3.4 OpenClaw

- 2026.01 上线，247K GitHub stars，模型无关，认知决策与工具执行分离
- **Opik-openclaw 插件**：全栈可观测，记录每次 LLM 调用、工具执行、memory recall、context assembly、agent delegation
- **ClawTrace**：专用可观测平台，交互式 trace tree 展示完整 payload
- **OpenClaw-RL** (2026.02)：异步 RL 框架，从自然对话反馈训练个性化 agent
- ClawTrace: https://www.clawtrace.ai/
- 安全审计论文: https://arxiv.org/abs/2602.14364

### 3.5 OpenHands (原 OpenDevin)

- 将 agent trajectory 存为 execution trace
- 评估 harness 支持 15+ benchmark（SWE-bench, HumanEvalFix, WebArena, GPQA 等）
- 研究发现：失败 trajectory 更长且方差更大；72-81% 的失败 trajectory 能正确定位问题文件，但无法精确修改代码
- 论文: https://arxiv.org/abs/2407.16741

---

## 四、OTel 在 LLM/Agent 领域的生态

### 4.1 OTel GenAI 语义约定（官方）

OTel 已有**官方 Agent span 语义约定**：
- `gen_ai.operation.name` = `"invoke_agent"`
- Span name = `"invoke_agent {gen_ai.agent.name}"`
- Issue #2664 提案定义了 task/action/agent/team/artifact/memory 的属性
- 规范: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/

### 4.2 主要开源 OTel 扩展

| 项目 | 维护方 | 特点 |
|------|--------|------|
| **OpenLLMetry** | Traceloop | OTel LLM 扩展，2025.02 已提交捐赠给 OTel 官方；主导 LLM 语义约定工作组 |
| **OpenInference** | Arize (Phoenix) | 定义 10 种 span kind：CHAIN/LLM/TOOL/RETRIEVER/EMBEDDING/**AGENT**/RERANKER/GUARDRAIL/EVALUATOR/PROMPT |
| **AgentSight** | eBPF 研究 | 用 eBPF 做边界 tracing，2.9% overhead，解决高层意图与底层系统调用的语义鸿沟 |

- OpenLLMetry: https://github.com/traceloop/openllmetry
- OpenInference: https://github.com/Arize-ai/openinference
- Phoenix: https://github.com/Arize-ai/phoenix
- AgentSight: https://arxiv.org/abs/2508.02736

### 4.3 商业平台 OTel 支持

- **Braintrust**: 自动 tracing OpenAI Agents SDK / LangGraph / Mastra / Pydantic AI 等，免费 1M spans/月
- **Datadog**: LLM Observability 原生支持 OTel GenAI 语义约定 (v1.37+)
- **LiteLLM**: OTel 集成跨 provider tracing

---

## 五、Agent 评估框架与轨迹分析

### 5.1 Benchmark 一览

| Benchmark | 关注点 | 轨迹分析能力 |
|-----------|--------|------------|
| AgentBench | 8 种交互环境 | 逐步打分、失败模式分类 |
| SWE-bench | GitHub issue 解决 | 成功/失败轨迹长度、方差对比 |
| ATBench (2026.04) | 安全评估 | 程序化异常检测 |
| TRAJECT-Bench (2025.10) | Agentic tool use | Trajectory-aware 评估 |
| AgentRewardBench | Web agent 轨迹 | 评估"评估器"本身的质量 |
| TrajBench | TrajAD 配套 | 扰动+补全合成异常 |

### 5.2 评估方法论

- **TRACE Framework**: 多维评估，含"证据银行"累积推理步骤知识 (https://arxiv.org/abs/2510.02837)
- **Agent-as-a-Judge**: 多 LLM agent 辩论式评估
- **Trajectory Reward Modeling**: 轨迹级 reward model 做 agent 对齐 (https://arxiv.org/abs/2604.08178)
- **Understanding Code Agent Behaviour** (2025.11): 跨 OpenHands/SWE-agent/Prometheus 的成功/失败轨迹实证研究 (https://arxiv.org/abs/2511.00197)

### 5.3 LangGraph / LangSmith

- **LangSmith**: 线程级 tracing，自动 instrument LangGraph
- **Trajectory evaluators**: 对中间决策打分，不只看最终输出
- **Multi-turn evals**: 度量语义意图、语义结果、agent trajectory
- **agentevals 库** (开源): `graph_trajectory_strict_match`、LLM-as-judge trajectory evaluator
  - https://github.com/langchain-ai/agentevals

---

## 六、归因方法论：从传统 ML 到 Agent Trajectory

### 6.1 核心问题：归因的本质

归因（Attribution）的本质是回答：**"在一个复杂系统的输出中，每个输入/步骤/特征贡献了多少？"**

这个问题从传统 ML 到 agent trajectory，方法论是一脉相承的，只是"特征"和"模型"的定义在演变：

```
特征重要性（传统ML）→ token 归因（NLP）→ step 归因（Agent Trajectory）
```

### 6.2 SHAP（Shapley Additive Explanations）

**在传统 ML 中的作用：**
- 基于博弈论 Shapley 值，计算每个特征对预测的"公平"贡献
- 满足四个公理：效率性、对称性、虚拟性、可加性
- 是目前最具理论保证的特征归因方法

**在 Agent Trajectory 中的适用性：⭐⭐⭐⭐⭐**

| 传统 ML | Agent Trajectory | 映射关系 |
|---------|-----------------|----------|
| 特征（feature） | 轨迹中的步骤（step/action） | 每个 step 是一个 "player" |
| 模型预测 | 任务完成度/reward | 联盟博弈的 "value function" |
| 特征子集 | 步骤子集（去掉某些步骤后重跑/模拟） | 反事实评估 |

**关键挑战：**
- **组合爆炸**：n 个步骤有 2^n 个子集，需要 Monte Carlo 近似
- **步骤间依赖**：传统 SHAP 假设特征独立，但 trajectory 步骤有**强序列依赖**
- **反事实模拟成本**：去掉某个步骤后重跑 agent 代价高昂

**解决方向：**
- Monte Carlo Shapley（AgentSHAP 已验证可行）
- 基于 LLM 的轨迹模拟（不真正重跑，用 LLM 估计"如果没有这步会怎样"）
- 分层 Shapley：先对 phase/stage 归因，再在 stage 内对 step 归因

### 6.3 朴素贝叶斯（Naive Bayes）

**传统 ML 中的作用：**
- 基于条件独立假设，计算 P(Y|X₁,X₂,...,Xₙ)
- 简单高效，常用于文本分类

**在 Agent Trajectory 中的适用性：⭐⭐⭐**

可以构建一个**步骤-结果**的贝叶斯模型：
```
P(成功 | step₁=tool_call, step₂=search, step₃=edit, ...) 
≈ P(成功) × ∏ P(stepᵢ | 成功) / P(stepᵢ)
```

**优势：**
- 计算极快，适合大规模轨迹数据的快速筛选
- 天然给出每个 step type 的"似然比"，可作为归因的粗粒度估计

**局限：**
- **条件独立假设在 trajectory 中严重不成立**（步骤高度依赖上下文）
- 只能做 step type 级别的统计归因，无法做单次 trajectory 的实例级归因

**适用场景：**
- 作为 TraceSHAP 的**快速预筛选**：先用贝叶斯找出统计上重要的 step pattern，再用 SHAP 做精细归因
- 跨大量 trajectory 的**模式发现**（哪类 action 整体上与成功/失败关联最强）

### 6.4 马尔科夫链 / 隐马尔科夫模型（HMM）

**传统 ML 中的作用：**
- 建模序列数据的状态转移概率
- 假设当前状态只依赖前一状态（马尔科夫性）

**在 Agent Trajectory 中的适用性：⭐⭐⭐⭐**

Agent trajectory 天然是一个**状态转移序列**：
```
State₀ →(action₁)→ State₁ →(action₂)→ State₂ → ... → Stateₙ (terminal)
```

**可以做什么：**
- **转移概率建模**：学习 P(stateₜ₊₁ | stateₜ, actionₜ)，识别异常转移
- **Viterbi 解码**：找到最可能的"正常"轨迹路径，偏离该路径的步骤即为归因候选
- **状态价值估计**：类似 RL 中的 V(s)，每个状态的"通往成功的概率"
- **Credit Assignment**：通过 TD(λ) 或类似方法将最终 reward 分配到每个步骤

**与 SHAP 的结合：**
- 马尔科夫模型可以作为 SHAP 的**value function 近似器**
- 不用真正重跑 agent，而是用学到的转移模型估计"去掉某步后的期望结果"
- 这正是 TraceSHAP 可以创新的地方：**Markov-approximated SHAP**

**局限：**
- 一阶马尔科夫假设对 LLM agent 可能太强（agent 有长程记忆）
- 可考虑高阶马尔科夫或 HMM 来缓解

### 6.5 因果推断（Causal Inference）

**传统 ML/统计中的作用：**
- 从观测数据中推断因果关系（而非仅仅相关性）
- 核心工具：do-calculus、SCM（结构因果模型）、反事实推理

**在 Agent Trajectory 中的适用性：⭐⭐⭐⭐⭐**

因果推断是 trajectory 归因的**理论最优解**，因为我们真正想问的是：

> "如果 agent **没有**执行 step X，结果会怎样？" （反事实问题）

**具体方法：**

1. **结构因果模型 (SCM)**
   ```
   构建 trajectory 的因果图：
   Context → Step₁ → Observation₁ → Step₂ → ... → Outcome
   
   每个节点有结构方程：
   Stepₜ = f(Contextₜ, Observationₜ₋₁, Agent_Policy)
   ```

2. **do-intervention**
   - do(Stepₜ = skip)：强制跳过某步，观察对 Outcome 的影响
   - 可以通过 trajectory replay 实现（从 Stepₜ₊₁ 开始重跑）

3. **反事实推理**
   - 对已完成的 trajectory：如果当时选了另一个 action，结果会如何？
   - 需要 world model（可以用 LLM 本身作为 world model 来估计）

4. **Granger 因果**
   - 跨多条 trajectory：某类 step 是否 Granger-causes 特定结果？
   - 时序数据天然适用

**与 SHAP 的关系：**
- SHAP 本质上是一种**边际贡献**度量，不是严格的因果度量
- 因果 SHAP（Causal SHAP / Asymmetric Shapley Values）是近年研究热点
- TraceSHAP 可以提供两个层次：快速的 marginal SHAP + 深度的 causal SHAP

### 6.6 专家经验（Expert Heuristics）

**作用：⭐⭐⭐**

在实际场景中，专家经验常作为归因的**先验知识**：
- 已知某些 action pattern 是 anti-pattern（如反复搜索相同关键词）
- 已知某些 tool 调用顺序是最优路径

**在 TraceSHAP 中的定位：**
- 作为 SHAP 计算的**先验权重**：专家认为重要的步骤先评估
- 作为**基线（baseline）对比**：SHAP 归因 vs 专家判断的一致性指标
- 作为**规则引擎层**：快速标记已知模式，只对未知模式启动 SHAP 分析

### 6.7 方法论总结：分层归因架构

```
┌─────────────────────────────────────────────────────┐
│                   TraceSHAP 分层归因                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Layer 0: 专家规则 (Expert Rules)                     │
│  ├── 已知 anti-pattern 直接标记                        │
│  ├── 速度：即时                                       │
│  └── 适用：已知模式                                    │
│                                                     │
│  Layer 1: 贝叶斯快筛 (Bayesian Pre-filter)            │
│  ├── 跨 trajectory 统计，识别高影响 step type           │
│  ├── 速度：毫秒级                                     │
│  └── 适用：大批量轨迹的模式发现                          │
│                                                     │
│  Layer 2: 马尔科夫近似 (Markov-approximated SHAP)     │
│  ├── 学习状态转移模型，估计反事实                        │
│  ├── 速度：秒级                                       │
│  └── 适用：在线/准实时归因                              │
│                                                     │
│  Layer 3: 完整 SHAP (Full Monte Carlo Shapley)       │
│  ├── 真正的子集采样 + agent replay                     │
│  ├── 速度：分钟级                                     │
│  └── 适用：关键 trajectory 的深度分析                   │
│                                                     │
│  Layer 4: 因果推断 (Causal SHAP)                      │
│  ├── SCM 建模 + do-intervention + 反事实               │
│  ├── 速度：分钟~小时级                                 │
│  └── 适用：需要严格因果解释的场景                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**由粗到精、由快到慢，用户可按需选择归因深度。**

---

## 七、TraceSHAP 的战略定位

### 7.1 市场空白

| 维度 | 现有工具 | TraceSHAP 机会 |
|------|---------|---------------|
| 归因方法 | AgentSHAP（仅 tool 级） | **Step 级** Shapley 归因 |
| OTel 集成 | OpenLLMetry, OpenInference, AgentTrace | 原生 OTel span + GenAI 语义约定 |
| 框架支持 | LangSmith（仅 LangGraph）, ClawTrace（仅 OpenClaw） | 跨框架：LangGraph + OpenClaw + Hermes |
| 轨迹分析 | AgentDiagnose（能力评分）, TrajAD（异常检测） | 因果归因：哪些步骤驱动了结果 |
| 理论基础 | SVERL, SHARP（RL/多 agent） | 将 RL credit assignment 应用于 LLM agent trace |
| 归因层次 | 单一方法 | **分层归因**：规则→贝叶斯→马尔科夫→SHAP→因果 |

### 7.2 建议技术路线

1. **数据层**：消费 OTel trace（GenAI 语义约定），为 LangGraph / OpenClaw / Hermes 各写一个 extractor
2. **建模层**：将 trajectory 建模为合作博弈，step 是 player，outcome metric 是 value function
3. **计算层**：实现分层归因（Layer 0-4），默认用 Layer 2（Markov-approximated SHAP）平衡速度与精度
4. **输出层**：每个 span/step 附加归因分数，可在任何 OTel 后端（Phoenix, Jaeger 等）可视化
5. **评估层**：用 AgentBench / SWE-bench 的公开 trajectory 数据验证归因质量

---

## 八、参考文献索引

### 核心论文
- AgentSHAP: https://arxiv.org/abs/2512.12597
- llmSHAP: https://arxiv.org/abs/2511.01311
- SVERL: https://arxiv.org/abs/2505.07797
- SHARP: https://arxiv.org/abs/2602.08335
- AgentDiagnose: https://aclanthology.org/2025.emnlp-demos.15/
- AgentTrace: https://arxiv.org/abs/2602.10133
- TrajAD: https://arxiv.org/abs/2602.06443
- TRACE Framework: https://arxiv.org/abs/2510.02837
- Understanding Code Agent Behaviour: https://arxiv.org/abs/2511.00197
- From Features to Actions (XAI→Agentic): https://arxiv.org/abs/2602.06841
- Trajectory Reward Modeling: https://arxiv.org/abs/2604.08178
- Dive into Claude Code: https://arxiv.org/abs/2604.14228

### OTel 与可观测
- OTel GenAI Agent Spans: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
- OpenLLMetry: https://github.com/traceloop/openllmetry
- OpenInference: https://github.com/Arize-ai/openinference
- Phoenix: https://github.com/Arize-ai/phoenix
- AgentSight: https://arxiv.org/abs/2508.02736

### 框架与平台
- Hermes Agent: https://github.com/NousResearch/hermes-agent
- OpenHands: https://arxiv.org/abs/2407.16741
- agentevals: https://github.com/langchain-ai/agentevals
- Agentic Transparency: https://github.com/VectorInstitute/Agentic-Transparency
