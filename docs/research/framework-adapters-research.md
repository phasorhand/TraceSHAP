# 框架适配器调研报告

> 调研日期: 2026-05-22
> 适用项目: TraceSHAP -- LLM Agent 轨迹 Shapley 归因分析库
> 当前已有适配: LangGraph (native), Langfuse (API), OTLP JSON (file)

---

## 1. 概述

本报告对设计规格中提及的 OpenClaw 和 Hermes 两个框架进行了实际生态调研, 同时评估了 OTLP/OTel 作为通用桥接层的可行性, 并对 CrewAI、AutoGen/AG2、LlamaIndex、Semantic Kernel 等主流框架的可观测性集成成熟度进行了横向对比, 最终给出适配器实现的优先级建议和路线图。

**核心发现:**
- OpenClaw 和 Hermes 均为真实存在的开源框架, 且都已具备 OpenTelemetry 导出能力。
- OpenClaw 通过 `diagnostics-otel` 插件原生导出符合 GenAI 语义规范的 OTLP 数据。
- Hermes 有两条数据路径: ShareGPT JSONL 轨迹导出 (用于 RL/训练) 和 `hermes-otel` 插件 (用于可观测性)。
- OTel GenAI 语义规范 (v1.40.0) 仍处于 Development 状态, 但主流框架已广泛采用。
- 建议采用 "OTLP 通用层 + source_hint 特化" 的分层策略, 而非为每个框架编写独立适配器。

---

## 2. OpenClaw 调研

### 2.1 框架概况

OpenClaw 是一个开源的自托管 AI Agent 运行时, 使用 TypeScript 编写, 运行在 Node.js 之上。其核心是一个长期运行的 Gateway 守护进程, 处理 WebSocket 连接、会话管理、认证和路由, 作为整个 Agent 系统的控制平面。

**关键特征:**
- 语言: TypeScript / Node.js (非 Python, 因此不适合原生 in-process 插桩)
- 架构: 单 Gateway 进程, 插件化扩展
- 文档: https://docs.openclaw.ai/
- GitHub: https://github.com/openclaw/openclaw
- 生态: 活跃的插件生态, 包括可观测性、追踪、自定义技能等

### 2.2 追踪数据格式

OpenClaw 有两个主要的追踪数据来源:

#### 2.2.1 diagnostics-otel 插件 (推荐)

OpenClaw 内置 `@openclaw/diagnostics-otel` 插件, 导出符合 OTel GenAI 语义规范 (`gen_ai_latest_experimental`) 的 OTLP 数据。

**Span 层级结构:**
```
openclaw.agent.turn          (根 span, 每轮 agent 交互)
  +-- chat gpt-5.4-mini      (模型推理)
  +-- execute_tool web_search (工具执行)
  +-- execute_tool read_file  (工具执行)
  +-- chat gpt-5.4-mini      (第二次推理)
  +-- subagent.run            (嵌套 agent)
  +-- hook.execute            (钩子执行)
  +-- cron.job                (定时任务)
```

**Span 属性:**
- 遵循 `gen_ai.*` 语义规范 (gen_ai.system, gen_ai.operation.name, gen_ai.usage.input_tokens 等)
- 包含有界标识符: channel, provider, model, error category, hash-only request ids
- 默认不包含 prompt/response 文本 (隐私保护)
- 可通过 `diagnostics.otel.captureContent` 选项启用内容捕获:
  - `inputMessages`, `outputMessages`, `toolInputs`, `toolOutputs`, `systemPrompt`

**Metrics:**
- 计数器和直方图: token 用量、成本、上下文大小、运行时长、消息流计数器、队列深度、会话状态

**OTLP 导出配置:**
```yaml
# OpenClaw 配置示例
diagnostics:
  otel:
    traces:
      endpoint: "http://localhost:4318/v1/traces"
      protocol: "http/protobuf"
    metrics:
      endpoint: "http://localhost:4318/v1/metrics"
    captureContent:
      inputMessages: true
      outputMessages: true
      toolInputs: true
      toolOutputs: true
```

#### 2.2.2 ClawTrace (Epsilla)

ClawTrace 是 Epsilla 提供的 OpenClaw 可观测性控制平面:
- 安装: `openclaw plugins install @epsilla/clawtrace`
- 功能: 执行路径、调用图、时间线可视化
- 内置 AI 分析师 "Tracy" 使用 Cypher 查询 (PuppyGraph 后端)
- PyPI 包 `clawtrace==0.1.11` 可用于本地审查和评分 agent 对话轨迹
- 提供 `/v1/evolve/ask` 端点供 agent 自省

**对 TraceSHAP 的意义:** ClawTrace 是一个独立的可观测性平台, 而非数据格式。TraceSHAP 应通过 OTLP 层消费 OpenClaw 数据, 而非直接对接 ClawTrace API。

### 2.3 集成方案

**推荐方案: OTLP + source_hint="openclaw"**

由于 OpenClaw 已原生支持 OTLP 导出且遵循 GenAI 语义规范, 无需编写独立适配器:

1. 用户配置 OpenClaw 的 `diagnostics-otel` 插件指向 TraceSHAP 的 OTLP 端点
2. TraceSHAP 使用 `OTLPJsonSource` (文件) 或未来的 `OTLPLiveSource` (实时端点) 接收数据
3. `source_hint="openclaw"` 传递给 `StepNormalizer`, 用于:
   - 将 `openclaw.agent.turn` span 识别为 AGENT 类型
   - 将 `execute_tool` span 识别为 TOOL 类型
   - 将 `chat` span 识别为 LLM 类型
   - 调整 `framework_mapping_confidence` 为中等 (0.65-0.75)

**需要增强的部分:**
- `_infer_span_kind()` 需要识别 OpenClaw 特有的 span 名称前缀
- 需要处理 `captureContent` 开启/关闭两种场景 (关闭时 input/output 为空)
- Outcome binding 需要从 ClawTrace 或外部来源获取

**不需要的部分 (v0.1):**
- Dry replay (OpenClaw 是 Node.js 运行时, 无法在 Python 进程内图变异)
- Live replay (需要 OpenClaw Gateway 支持)
- Prune patch (需要 OpenClaw 插件协议支持)

```python
# 伪代码: OpenClaw source_hint 处理
class OTLPJsonSource(SpanSource):
    def _convert_span(self, raw: dict) -> TraceSHAPSpan | None:
        # ... existing logic ...
        if self._source_hint == "openclaw":
            span_kind = self._infer_openclaw_kind(name, attributes)
        # ...

    def _infer_openclaw_kind(self, name: str, attrs: list[dict]) -> SpanKind:
        if name.startswith("openclaw.agent"):
            return SpanKind.AGENT
        if name.startswith("execute_tool"):
            return SpanKind.TOOL
        if name.startswith("chat ") or _get_attr(attrs, "gen_ai.system"):
            return SpanKind.LLM
        if name.startswith("hook."):
            return SpanKind.CUSTOM
        return SpanKind.CUSTOM
```

---

## 3. Hermes 调研

### 3.1 框架概况

Hermes Agent 是 Nous Research 于 2026 年 2 月发布的开源自主 AI Agent, 定位为 "会成长的 Agent"。除任务自动化外, 它还是一个生成训练数据、运行 RL 实验和导出轨迹用于微调的平台。

**关键特征:**
- 语言: Python (可作为库使用)
- 创建者: Nous Research
- 特色: 持久记忆, 自我改进, 技能学习
- GitHub: https://github.com/nousresearch/hermes-agent
- 文档: https://hermes-agent.nousresearch.com/docs/

### 3.2 Atropos API

Atropos 并非一个独立的 API 端点, 而是 Nous Research 的强化学习框架, 与 Hermes Agent 深度集成:

- Hermes 内置 batch trajectory 生成和 Tinker-Atropos RL 环境
- 轨迹导出 -> Atropos 训练管线 (ShareGPT JSONL 格式)
- 这是 Nous Research 内部使用的同一研究工具链

#### 3.2.1 轨迹导出格式 (ShareGPT JSONL)

Hermes 以 ShareGPT 兼容的 JSONL 格式保存对话轨迹:

```jsonl
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant..."},
    {"from": "human", "value": "Search for X"},
    {"from": "gpt", "value": "<think>reasoning</think>\nI'll search..."},
    {"from": "tool", "value": {"results": [...]}},
    {"from": "gpt", "value": "Based on the results..."}
  ],
  "timestamp": "2026-05-22T10:30:00Z",
  "model": "hermes-3-llama-3.1-70b",
  "completed": true
}
```

**格式特点:**
- 每行一个完整 JSON 对象
- `conversations` 数组使用 ShareGPT 角色约定 (`from`: system/human/gpt/tool)
- 工具内容如果是 JSON 则被解析为对象 (而非字符串)
- 多个工具结果用换行符连接
- Reasoning 统一规范化为 `<think>` 标签
- 批处理变体包含 `prompt_index` 字段
- 工具统计被规范化以包含所有可能工具的零值默认值 (确保 HuggingFace 数据集兼容)

**源代码位置:** `agent/trajectory.py`, `run_agent.py` (`_save_trajectory`), `batch_runner.py`

#### 3.2.2 hermes-otel 插件

Hermes 也有社区开发的 `hermes-otel` 插件, 自动将执行数据导出为 OTel span:

- 捕获: LLM 工具调用、模型调用、API 请求
- 依赖: `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`
- 支持后端: Phoenix, Langfuse, SigNoz, Jaeger
- 支持多后端同时配置 (YAML 配置文件)

### 3.3 集成方案

Hermes 有两条可行的集成路径:

#### 路径 A: ShareGPT JSONL 导入 (专用 Source)

需要编写 `HermesTrajectorySource`, 将 ShareGPT JSONL 转换为 `TraceSHAPSpan`:

```python
class HermesTrajectorySource(SpanSource):
    """从 Hermes ShareGPT JSONL 文件导入轨迹."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._polled = False

    async def connect(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Hermes trajectory not found: {self._path}")

    async def poll(self) -> list[TraceSHAPSpan]:
        if self._polled:
            return []
        self._polled = True
        spans = []
        with open(self._path) as f:
            for line in f:
                entry = json.loads(line.strip())
                spans.extend(self._convert_entry(entry))
        return spans

    def _convert_entry(self, entry: dict) -> list[TraceSHAPSpan]:
        trace_id = hashlib.sha256(
            json.dumps(entry, default=str).encode()
        ).hexdigest()[:32]
        spans = []
        conversations = entry.get("conversations", [])
        for i, msg in enumerate(conversations):
            role = msg["from"]
            span_kind = {
                "system": SpanKind.CUSTOM,
                "human": SpanKind.CUSTOM,
                "gpt": SpanKind.LLM,
                "tool": SpanKind.TOOL,
            }.get(role, SpanKind.CUSTOM)
            # ... 构建 TraceSHAPSpan ...
        return spans
```

**优点:** 直接消费 Hermes 原生格式, 无需额外配置
**缺点:** ShareGPT 格式缺乏时间戳粒度 (只有整条轨迹的 timestamp, 没有每步的 start/end)、缺乏 span 层级、缺乏 token 用量等关键信息

#### 路径 B: OTLP + source_hint="hermes" (推荐)

通过 `hermes-otel` 插件导出 OTLP 数据, TraceSHAP 使用通用 OTLP 层消费:

1. 用户安装 `hermes-otel` 插件并配置 OTLP 端点
2. TraceSHAP 使用 `OTLPJsonSource` 或 `OTLPLiveSource` 接收
3. `source_hint="hermes"` 用于特化步骤映射

**推荐:** 同时支持两条路径。路径 A 作为 "离线导入" (适合已有 JSONL 数据的研究者), 路径 B 作为 "实时接入" (适合生产环境)。

**Replay 支持:**
- Dry replay: 部分支持 (可修改 tool list 配置)
- Live replay: 仅沙盒模式
- Prune patch: 通过修改 tool policy 实现 (Hermes 支持技能启用/禁用)

---

## 4. OTLP 通用桥接层

### 4.1 OTel GenAI 语义规范现状

截至 2026 年 5 月, OTel GenAI 语义规范的状态:

| 项目 | 版本 | 状态 |
|------|------|------|
| 语义规范总体 | v1.40.0 | Development (非 Stable) |
| GenAI client spans | experimental | 主流框架已采用 |
| GenAI agent spans | experimental | 新增, 采用中 |
| GenAI events | experimental | 采用中 |
| GenAI metrics | experimental | 采用中 |

**关键属性 (TraceSHAP 需关注):**

```
# 必需
gen_ai.operation.name       # "chat", "text_completion", "embeddings"
gen_ai.provider.name        # "openai", "anthropic" 等

# 推荐
gen_ai.system               # LLM 提供者
gen_ai.request.model        # 请求的模型
gen_ai.response.model       # 实际使用的模型
gen_ai.usage.input_tokens   # 输入 token 数
gen_ai.usage.output_tokens  # 输出 token 数

# Agent 特有 (新增)
gen_ai.agent.name           # Agent 名称
gen_ai.data_source.id       # 数据源标识

# Span 命名
create_agent {gen_ai.agent.name}
invoke_agent {gen_ai.agent.name}
```

**版本迁移:**
- 使用 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 环境变量
- 现有 v1.36.0 之前的插桩不应默认更改版本

**对 TraceSHAP 的影响:**
- `_infer_span_kind()` 和 `_extract_tokens()` 已能处理 `gen_ai.*` 属性 (现有代码已就绪)
- 需要添加对 `gen_ai.agent.name` 的识别以优化 AGENT span 检测
- Agent span 的 `gen_ai.operation.name` 为 `create_agent` / `invoke_agent`

### 4.2 实时 OTLP 端点消费

当前 `OTLPJsonSource` 仅支持文件导入, 需要新增实时 OTLP 端点接收能力:

**方案对比:**

| 方案 | 协议 | 端口 | 优势 | 劣势 |
|------|------|------|------|------|
| OTLP/HTTP | HTTP/protobuf | 4318 | 简单部署, 防火墙友好 | 非流式, 逐批 |
| OTLP/gRPC | gRPC | 4317 | 高吞吐, 流式 | 需要 gRPC 依赖 |
| OTel Collector 转发 | 多种 | - | 灵活, 可预处理 | 额外组件 |
| 文件导出 + 监听 | 文件系统 | - | 最简单 | 延迟高, 不适合实时 |

**推荐: OTLP/HTTP 端点**

```python
# 概念设计: OTLPLiveSource
class OTLPLiveSource(SpanSource):
    """实时 OTLP HTTP 端点, 接收推送的 trace 数据."""

    def __init__(self, host: str = "0.0.0.0", port: int = 4318,
                 source_hint: str = "otlp"):
        self._host = host
        self._port = port
        self._source_hint = source_hint
        self._buffer: list[TraceSHAPSpan] = []
        self._server = None

    async def connect(self) -> None:
        # 启动 aiohttp/FastAPI 服务器
        # POST /v1/traces 接收 OTLP protobuf/JSON
        ...

    async def poll(self) -> list[TraceSHAPSpan]:
        # 返回并清空缓冲区
        spans = list(self._buffer)
        self._buffer.clear()
        return spans

    async def close(self) -> None:
        if self._server:
            await self._server.shutdown()
```

**实现要点:**
- 使用 `aiohttp` 或 `uvicorn` 实现轻量 HTTP 服务器
- 接受 `application/x-protobuf` 和 `application/json` 两种 Content-Type
- 解析 `ExportTraceServiceRequest` 消息
- 线程安全的 span 缓冲区
- 可选的身份验证 (Bearer token)

### 4.3 框架 OTel 集成成熟度

| 框架 | OTel 原生 | GenAI SemConv | 导出方式 | 成熟度 |
|------|-----------|---------------|----------|--------|
| LangGraph/LangChain | 是 (Q1 2026) | 是 | SDK 内置 | 高 |
| OpenAI Agents SDK | 是 (Q1 2026) | 是 | SDK 内置 | 高 |
| CrewAI | 是 | 是 | 内置 + 插件 | 高 |
| AutoGen/AG2 | 是 (Feb 2026) | 是 | `autogen.opentelemetry` | 高 |
| LlamaIndex | 是 | 是 | `opentelemetry-instrumentation-llamaindex` v0.60 | 高 |
| Semantic Kernel | 是 | 是 (MS 推动) | .NET/Python SDK | 高 |
| OpenClaw | 是 | 是 | `diagnostics-otel` 插件 | 中高 |
| Hermes | 社区插件 | 部分 | `hermes-otel` 插件 | 中 |

**结论:** 到 2026 年中, 几乎所有主流 Agent 框架都已支持 OTel 导出, 且大多遵循 GenAI 语义规范。这极大地降低了 TraceSHAP 需要编写框架特定适配器的必要性。

---

## 5. 主流框架适配优先级

### 5.1 CrewAI

- **OTel 支持:** 原生内置, 遵循 GenAI 语义规范
- **导出:** CrewAI AMP 可直接导出 OTel traces 和 logs 到任意 Collector
- **第三方集成:** MLflow, Langfuse, Deepchecks, SigNoz 等均支持
- **TraceSHAP 集成难度:** 低 -- 使用通用 OTLP Source + `source_hint="crewai"` 即可
- **生态规模:** 大 (GitHub stars 20k+, 活跃社区)
- **特殊需求:** CrewAI 的 crew/task/agent 三层结构需要在 StepNormalizer 中处理

### 5.2 AutoGen / AG2

- **OTel 支持:** `autogen.opentelemetry` 模块, 四个 instrument 函数
- **捕获内容:** 模型名、provider、token 用量、成本、温度、工具调用参数和结果、完整对话消息
- **TraceSHAP 集成难度:** 低 -- OTLP 导出 + `source_hint="autogen"`
- **生态规模:** 大 (Microsoft 支持, 广泛企业采用)
- **特殊需求:** 多 Agent 对话的 span 层级需要正确解析

### 5.3 LlamaIndex

- **OTel 支持:** `opentelemetry-instrumentation-llamaindex` v0.60.0 (2026 年 4 月)
- **捕获内容:** prompts, completions, embeddings (可通过 `TRACELOOP_TRACE_CONTENT=false` 禁用)
- **TraceSHAP 集成难度:** 低 -- OpenLLMetry 提供标准化的 OTel span
- **生态规模:** 大 (RAG 领域标杆)
- **特殊需求:** RAG pipeline 的 retriever/reranker span 需要正确映射到 `SpanKind.RETRIEVER` / `SpanKind.RERANKER`

### 5.4 Semantic Kernel

- **OTel 支持:** 原生, Microsoft 是 OTel GenAI Agent SemConv 的共同推动者
- **捕获内容:** 分布式 trace, agent actions, tool invocations, multi-agent 工作流
- **TraceSHAP 集成难度:** 低 -- 标准 OTLP + `source_hint="semantic_kernel"`
- **生态规模:** 大 (Microsoft 生态, .NET + Python)
- **特殊需求:** Microsoft Agent Framework (AutoGen + Semantic Kernel 融合) 可能在 2026 Q3 后成为单一入口
- **注意:** Semantic Kernel 1.0 GA 目标为 2026 Q1, API 基本稳定

### 5.5 优先级排序

基于生态规模、OTel 成熟度和 TraceSHAP 用户需求, 推荐以下优先级:

| 优先级 | 框架 | 理由 | 实现方式 |
|--------|------|------|----------|
| P0 | 通用 OTLP Live Source | 一次实现, 覆盖所有 OTel 框架 | 新增 `OTLPLiveSource` |
| P1 | LangGraph | 已有原生适配, 增强 source_hint 处理 | 现有代码增强 |
| P1 | CrewAI | 生态大, OTel 成熟, 用户需求多 | OTLP + source_hint |
| P1 | AutoGen/AG2 | Microsoft 支持, 企业采用广泛 | OTLP + source_hint |
| P2 | LlamaIndex | RAG 场景重要, OTel 成熟 | OTLP + source_hint |
| P2 | OpenClaw | 真实框架, 原生 OTel | OTLP + source_hint |
| P2 | Semantic Kernel | Microsoft 生态, 可能与 AutoGen 合并 | OTLP + source_hint |
| P3 | Hermes | 较新, ShareGPT 格式需专用解析 | OTLP + ShareGPT Source |

---

## 6. 实现建议

### 6.1 通用 vs 专用适配器

**核心问题:** 是否需要为 OpenClaw 和 Hermes 编写独立的 `SpanSource` 实现?

**结论: 大多数情况下不需要。**

当前 `OTLPJsonSource` 已经能处理绝大多数场景。真正需要特化的部分不在 Source 层, 而在 Normalizer 层:

```
数据流: Framework -> [OTel Export] -> [OTLP Source] -> [StepNormalizer(source_hint)] -> CanonicalStep
                                        ^                        ^
                                    通用, 无需改                需要 source_hint 特化
```

**需要特化的地方 (在 StepNormalizer 中):**

1. **span 名称到 SpanKind 的映射规则** -- 每个框架的 span 命名模式不同
2. **mapping_confidence 估算** -- 基于框架提供的元数据丰富度
3. **node_id 提取** -- 不同框架在 attributes 中存放 node/step 标识的 key 不同
4. **side_effect_class 推断** -- 框架特定的工具分类

**唯一需要独立 Source 的例外: Hermes ShareGPT JSONL 导入**

ShareGPT JSONL 不是 OTLP 格式, 无法复用现有 Source:
- 缺乏 span 层级 (扁平对话轮次)
- 缺乏每步时间戳 (只有轨迹级 timestamp)
- 缺乏 token 用量细节
- 需要将对话轮次 "合成" 为 TraceSHAPSpan

因此建议新增一个轻量的 `HermesJsonlSource`, 作为研究场景的便捷入口。

### 6.2 实时 OTLP Source

**这是最高优先级的基础设施投资。**

实现 `OTLPLiveSource` 可以一次性覆盖所有支持 OTel 导出的框架, 包括 OpenClaw、Hermes (via hermes-otel)、CrewAI、AutoGen、LlamaIndex、Semantic Kernel 等。

**设计要点:**

```python
# traceshap/ingestion/sources/otlp_live.py

class OTLPLiveSource(SpanSource):
    """
    内嵌 OTLP HTTP 接收端点.
    接收 POST /v1/traces 请求, 解析 ExportTraceServiceRequest,
    将 span 缓冲并在 poll() 时返回.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4318,
        source_hint: str = "otlp",
        auth_token: str | None = None,
        max_buffer_size: int = 10000,
    ):
        ...
```

**关键决策:**
- 协议: 优先支持 OTLP/HTTP JSON (实现简单), 后续增加 protobuf 支持
- 依赖: `aiohttp` (轻量) 或复用项目已有的 HTTP 框架
- 缓冲: 有界环形缓冲区, 防止内存溢出
- 解析: 复用 `OTLPJsonSource._parse_otlp()` 和 `_convert_span()` 逻辑 (提取为共用模块)

**重构建议:**
```
traceshap/ingestion/sources/
    base.py              # SpanSource ABC (现有)
    otlp_common.py       # 提取: OTLP 解析逻辑 (从 otlp_json.py 移出)
    otlp_json.py         # 文件导入 (使用 otlp_common)
    otlp_live.py         # 实时端点 (使用 otlp_common)
    langfuse.py          # Langfuse API (现有)
    hermes_jsonl.py      # Hermes ShareGPT JSONL 导入 (新增)
    factory.py           # 工厂 (扩展)
```

### 6.3 推荐路线图

#### Phase 1: OTLP 通用增强 (2-3 周)

1. **提取 OTLP 解析公共模块** (`otlp_common.py`)
   - 从 `OTLPJsonSource` 提取 `_parse_otlp`, `_convert_span`, `_infer_span_kind`, `_extract_tokens`
   - 增加 `gen_ai.agent.name` 和 `gen_ai.operation.name` 的 AGENT span 检测

2. **实现 `OTLPLiveSource`**
   - OTLP/HTTP JSON 接收端点
   - 有界缓冲, poll() 批量返回
   - 基础认证支持

3. **source_hint 驱动的 span 映射增强**
   - 在 `StepNormalizer` 中添加 per-hint 的 span 名称模式匹配
   - OpenClaw: `openclaw.agent.turn` -> AGENT, `execute_tool` -> TOOL, `chat` -> LLM
   - CrewAI: `crew.*` / `task.*` / `agent.*` 模式
   - AutoGen/AG2: conversation/agent/tool 模式

4. **更新 `factory.py`** -- 新增 `otlp_live` 类型

#### Phase 2: Hermes 专用导入 (1 周)

5. **实现 `HermesJsonlSource`**
   - 解析 ShareGPT JSONL
   - 对话轮次 -> TraceSHAPSpan 转换
   - 处理 `<think>` 标签 (reasoning extraction)

#### Phase 3: 验证与文档 (1 周)

6. **端到端集成测试**
   - OpenClaw diagnostics-otel -> OTLPLiveSource -> StepNormalizer
   - Hermes JSONL -> HermesJsonlSource -> StepNormalizer
   - CrewAI OTel -> OTLPLiveSource -> StepNormalizer

7. **用户文档**
   - 每个框架的配置指南 (如何配置 OTel 导出指向 TraceSHAP)

---

## 7. 风险与开放问题

### 风险

1. **GenAI SemConv 稳定性风险 (中)**
   - 规范仍在 Development 状态, 属性名可能变更
   - 缓解: 使用 `OTEL_SEMCONV_STABILITY_OPT_IN` 机制, 在代码中兼容新旧属性名

2. **OpenClaw span 内容缺失 (低-中)**
   - 默认 `captureContent=false`, span 不含 prompt/response 文本
   - 影响: TraceSHAP 的 `input`/`output` 字段为空, 影响 Layer 2 (语义嵌入) 归因
   - 缓解: 文档中建议用户启用 `captureContent`, 或通过 Langfuse 等第三方获取内容

3. **Hermes ShareGPT 格式信息损失 (中)**
   - 缺乏时间戳粒度和 token 用量, 影响归因精度
   - 缓解: 推荐用户使用 hermes-otel 插件 (OTLP 路径) 获取更丰富的数据

4. **实时 OTLP 端点安全性 (低)**
   - 暴露 HTTP 端点接收外部数据
   - 缓解: Bearer token 认证, 绑定 localhost, 速率限制

5. **Microsoft Agent Framework 融合 (低)**
   - AutoGen + Semantic Kernel 可能合并为统一框架
   - 缓解: 使用通用 OTLP 层, 不依赖特定框架 API

### 开放问题

1. **Outcome binding 标准化**
   - 各框架对 "任务成功/失败" 的表达方式不同
   - OpenClaw: ClawTrace 评分
   - Hermes: `completed` 字段 + Atropos reward
   - 通用 OTLP: 无标准 outcome attribute
   - **需要设计一个 outcome adapter 层**, 或统一通过 Langfuse score / 自定义 attribute 获取

2. **Replay 能力的泛化**
   - 当前只有 LangGraph 支持 dry replay (图变异)
   - OpenClaw/Hermes 的 replay 需要框架级协作
   - **是否需要定义 Replay Protocol?**

3. **OTLPLiveSource 的背压处理**
   - 如果 agent 产生 span 的速度 > TraceSHAP 消费速度, 如何处理?
   - 选项: 丢弃最旧 span, 返回 503, 磁盘溢出

4. **source_hint 自动检测**
   - 是否可以从 OTLP resource attributes 自动推断框架类型?
   - 例如 `service.name="openclaw"` 或 `telemetry.sdk.name` 包含框架信息
   - 可以作为 `source_hint` 的 fallback

5. **ClawTrace API 直接消费**
   - ClawTrace 有 FastAPI 后端和 REST API
   - 是否值得实现 `ClawTraceSource` 直接从 ClawTrace 拉取?
   - 目前建议不做 -- OTLP 路径已足够, ClawTrace 是上层平台

---

## 附录: 参考链接

### OpenClaw
- [OpenClaw 文档](https://docs.openclaw.ai/)
- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [OpenClaw OTel 导出文档](https://docs.openclaw.ai/gateway/opentelemetry)
- [ClawTrace GitHub](https://github.com/epsilla-cloud/clawtrace)
- [ClawTrace 发布博客](https://www.epsilla.com/blogs/clawtrace-launch-openclaw-agent-observability)
- [diagnostics-otel GenAI SemConv PR](https://github.com/openclaw/openclaw/pull/11100)

### Hermes
- [Hermes Agent 文档](https://hermes-agent.nousresearch.com/docs/)
- [Hermes Agent GitHub](https://github.com/nousresearch/hermes-agent)
- [轨迹格式文档](https://hermes-agent.nousresearch.com/docs/developer-guide/trajectory-format)
- [Python 库使用](https://hermes-agent.nousresearch.com/docs/guides/python-library)
- [hermes-otel 插件](https://hermesatlas.com/projects/briancaffey/hermes-otel)

### OTel GenAI
- [GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [GenAI Agent Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/)
- [GenAI Events](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/)
- [OTLP 规范 1.10.0](https://opentelemetry.io/docs/specs/otlp/)
- [AI Agent 可观测性博客](https://opentelemetry.io/blog/2025/ai-agent-observability/)

### 框架 OTel 集成
- [CrewAI OTel 导出](https://docs.crewai.com/en/enterprise/guides/capture_telemetry_logs)
- [AG2 OTel 博客](https://docs.ag2.ai/latest/docs/blog/2026/02/08/AG2-OpenTelemetry-Tracing/)
- [LlamaIndex OTel PyPI](https://pypi.org/project/opentelemetry-instrumentation-llamaindex/)
- [Semantic Kernel 可观测性](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/)
- [Microsoft Agent Framework OTel](https://www.devleader.ca/2026/04/02/opentelemetry-and-observability-in-microsoft-agent-framework)
