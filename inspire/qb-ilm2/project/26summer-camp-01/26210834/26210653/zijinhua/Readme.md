# 自进化的任务求解智能体 (Self-Evolving ReAct Agent)

基于 Qwen3.5-9B 的 ReAct 智能体，集成**反思诊断**、**长期记忆**、**多模态工具调用**三大核心模块，在闭卷评测设定下实现自主进化。

---

## 目录结构

```
zijinhua/
├── agent.py                  # ReAct 主循环引擎（推理→行动→观察）
├── reflection.py             # 反思模块：失败诊断 + 修正策略提取
├── memory.py                 # 长期记忆：多模态召回（BGE-M3 + CLIP）
├── tools.py                  # 工具抽象层与注册表
├── tool_schemas.py           # 工具 JSON Schema 定义
├── tool_parser.py            # LLM 输出解析（兼容多种格式）
├── config.py                 # 环境变量 / LLM 配置管理
├── llm_client.py             # LLM 客户端（vLLM / PAI-EAS 双后端）
├── vision_utils.py           # 图像 Base64 编码、多模态 content 构建
├── logger.py                 # 结构化日志（results.jsonl + trajectory.jsonl）
├── main.py                   # 统一命令行入口
│
├── services/                 # 外部服务接入层
│   ├── search_client.py      # search-proxy HTTP 客户端
│   ├── search_tools.py       # search_text / search_image 工具实现
│   ├── search_reranker.py    # BGE Cross-Encoder 本地重排
│   ├── image_upload.py       # 图床上传（uguu/tmpfiles/0x0）
│   ├── browser_client.py     # 浏览器沙盒 HTTP 客户端
│   └── browser_tools.py      # 浏览器工具（navigate/click/type）
│
├── scripts/                  # 评测与部署脚本
│   ├── eval_benchmark.py     # 通用批量评测框架（闭卷）
│   ├── run_simplevqa_9b_9b_refl.py        # SimpleVQA 9B+9B 反思评测
│   ├── run_simplevqa_9b_baseline.py       # SimpleVQA 无反思基线
│   ├── run_2wiki_9b_agent_32b_refl.py     # 2WikiMQA 评测
│   ├── launch_simplevqa_9b_9b_refl.sh     # SimpleVQA 启动器
│   ├── launch_simplevqa_9b_9b_refl_no_gold.sh  # SimpleVQA 闭卷启动器
│   ├── launch_2wiki_9b_agent_32b_refl.sh  # 2Wiki 启动器
│   ├── start_vllm.sh                      # vLLM 主推理服务
│   ├── start_vllm_9b_reflection.sh        # vLLM 反思专用实例
│   └── start_vllm_9b_simplevqa_agent.sh   # vLLM SimpleVQA Agent
│
├── data/                     # 评测数据集
│   ├── 2wiki.jsonl           # 2WikiMultihopQA
│   └── simpleVQA_*.jsonl     # SimpleVQA 子集
├── simpleVQA/SimpleVQA.jsonl # SimpleVQA 完整集（含图片路径）
├── ckpt/                     # 本地模型权重
├── .env                      # 环境配置（密钥，勿提交）
└── agent_memory.json         # 长期记忆存储
```

---

## 环境搭建与复现

### 1. 创建 Conda 环境

```bash
conda create -n zijinhua python=3.10 -y
conda activate zijinhua
```

### 2. 安装依赖

```bash
cd /inspire/qb-ilm2/project/26summer-camp-01/26210834/26210653/zijinhua
pip install -r requirements.txt \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置 `.env`

核心环境变量（复制 `.env` 并修改）：

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_BACKEND` | LLM 后端类型 | `vllm` |
| `LLM_BASE_URL` | 推理模型 API 地址 | `http://127.0.0.1:8001/v1` |
| `MODEL_NAME` | 模型名称 | `Qwen3.5-9B` |
| `REFLECTION_LLM_BASE_URL` | 反思模型 API 地址 | `http://127.0.0.1:8004/v1` |
| `REFLECTION_MODEL_NAME` | 反思模型名称 | `Qwen3.5-9B` |
| `SEARCH_PROXY_URL` | 搜索代理地址 | `http://127.0.0.1:8090` |
| `IMAGE_UPLOADER` | 图床后端 | `uguu` |
| `SEARCH_RERANKER_MODEL` | 重排模型路径 | `ckpt/bge-reranker-v2-m3/...` |

### 4. 启动推理服务（GPU 服务器）

```bash
# 主 Agent 推理（GPU0, 端口 8001）
bash scripts/start_vllm.sh

# 反思模型专用实例（GPU2, 端口 8004）
bash scripts/start_vllm_9b_reflection.sh

# SimpleVQA 专用 Agent（GPU3, 端口 8003）
bash scripts/start_vllm_9b_simplevqa_agent.sh
```

### 5. 运行评测

```bash
# === SimpleVQA 评测 ===

# 方式一：闭卷 9B推理 + 9B反思（推荐）
bash scripts/launch_simplevqa_9b_9b_refl_no_gold.sh

# 方式二：无反思基线
bash scripts/launch_simplevqa_9b_baseline.sh

# 方式三：带 --resume 断点续跑
bash scripts/launch_simplevqa_9b_9b_refl_no_gold.sh --resume

# === 2WikiMultihopQA 评测 ===

# 闭卷 9B+9B 反思
bash scripts/launch_2wiki_9b_9b_refl_no_gold.sh

# 9B Agent + 32B 反思对照
bash scripts/launch_2wiki_9b_agent_32b_refl.sh

# === 通用评测命令 ===
python scripts/eval_benchmark.py \
    --jsonl simpleVQA/SimpleVQA.jsonl \
    --mode full \
    --tools search \
    --max-steps 6 \
    --workers 1
```

### 6. 连通性测试

```bash
# 测试 LLM API
python scripts/test_llm.py

# 测试搜索代理
python scripts/test_search.py

# 测试图搜（需图床可用）
python scripts/test_vision.py

# 全链路集成测试（无需真实 API）
python scripts/integration_test.py
```

---

## 核心创新与技术贡献

### 创新一：层次化推理规划引擎 (Hierarchical Plan-and-Solve ReAct)

**核心思想**：针对传统 ReAct 框架在复杂多跳推理任务中"盲目搜索、长 query 失败率高"的瓶颈，本项目提出**强制前置规划约束 (Mandatory Planning Constraint)**——要求 Agent 在首次工具调用前必须完成任务分解与搜索策略规划，从根本上改善了 OSINT 类问答的推理质量。

**文件位置**: `agent.py`

| 技术贡献 | 方法描述 | 代码位置 |
|----------|----------|----------|
| **强制规划约束 (Plan-before-Act)** | 引入结构化 `<plan>` 标签机制，Agent 必须在首轮输出中完成"目标→缺失线索→多步计划"的层次化分解，杜绝未经思考的冲动式工具调用 | `agent.py:20-76` |
| **渐进式 OSINT 搜索法则** | 设计"实体解密→终极查询"两阶段搜索范式，解决复杂问题中隐藏实体（如代称、暗喻）的识别难题 | `agent.py:30-38` |
| **动态上下文窗口管理** | 提出自适应历史压缩策略，在长程对话中自动折叠冗余观测信息，确保有限上下文窗口内信息密度最大化 | `agent.py:374-393` |
| **鲁棒性容错机制** | 当 LLM 因 thinking token 占满输出预算而产出空回复时，自动注入追问 prompt 触发答案生成，避免推理链断裂 | `agent.py:297-301` |
| **策略驱动的自主重试** | 反思模型产出的修正策略可直接驱动同题重试，且该机制与标准答案完全解耦，适用于无标注的开放域场景 | `agent.py:343-368` |

---

### 创新二：双向自省反思机制 (Bidirectional Self-Reflection)

**核心思想**：突破现有 Agent 框架"仅从失败中学习"的局限性，提出**双向反思**——不仅对失败轨迹进行根因诊断，还对成功轨迹进行经验蒸馏，构建"失败教训 + 成功范式"的完整经验体系。

**文件位置**: `reflection.py`

| 技术贡献 | 方法描述 | 代码位置 |
|----------|----------|----------|
| **多粒度失败检测器** | 设计三级失败感知机制：推理超时（MAX_STEPS_EXCEEDED）、工具链崩溃（CONSECUTIVE_TOOL_ERRORS）、答案偏差（WRONG_ANSWER），实现对不同失败模式的精准识别与差异化诊断 | `reflection.py:13-16` |
| **在线状态机监控** | 通过 ReflectionState 有限状态机实时追踪 Agent 行为轨迹，在工具调用层面进行流式异常检测，支持毫秒级故障响应 | `reflection.py:19-53` |
| **结构化诊断框架** | 设计"失败诊断→问题定位→修正策略→推荐工具链"四段式诊断输出规范，确保反思结果可直接转化为可执行策略 | `reflection.py:56-81` |
| **成功经验蒸馏** | 创新性地对成功轨迹提炼"有效工具链模式 + 关键证据来源"，形成可迁移的正向经验范式 | `reflection.py:83-101` |
| **上下文感知的轨迹摘要** | 智能截断过长工具返回（保留首尾关键信息），在反思模型有限窗口内最大化诊断信息量 | `reflection.py:184-195` |
| **闭卷信息隔离** | 严格保证评测公平性：反思模型在闭卷模式下无法接触标准答案，仅基于行为轨迹进行过程诊断 | `reflection.py:250-253` |

**自省触发决策树** (`agent.py:304-341`):
```
Agent 输出最终答案?
├── 否 (步数耗尽) → 强制产出 + MAX_STEPS_EXCEEDED 反思 → 提取修正策略
├── 工具连续报错 ≥ 阈值 → CONSECUTIVE_TOOL_ERRORS 反思 → 终止并记录教训
├── 答案错误 (非闭卷) → WRONG_ANSWER 反思 → 策略注入 + 重试
└── 答案正确 → 成功经验蒸馏 → 写入长期记忆
```

---

### 创新三：多模态动态路由记忆网络 (Multimodal Routed Memory Network)

**核心思想**：提出**任务感知的多模态记忆召回**机制——根据当前任务的模态特征（纯文本 / 图文混合）动态选择最优向量空间进行经验检索，实现跨模态经验迁移与"学一次、用多次"的持续进化能力。

**文件位置**: `memory.py`

| 技术贡献 | 方法描述 | 代码位置 |
|----------|----------|----------|
| **模态感知的动态路由** | 根据任务输入自动判别模态类型：图文任务走 CLIP 视觉相似度通道，纯文本任务走 BGE-M3 语义匹配通道，实现最优召回精度 | `memory.py:43-48` |
| **CLIP 视觉语义对齐** | 利用 CLIP (ViT-B/32) 将任务图像与记忆库图像映射至同一向量空间，通过余弦相似度实现跨任务的视觉经验迁移 | `memory.py:109-138, 239-257` |
| **BGE-M3 多语言语义编码** | 集成 BGE-M3 大规模多语言嵌入模型，支持中英文混合 query 的高质量语义匹配，突破传统 TF-IDF 关键词匹配的语义鸿沟 | `memory.py:140-237` |
| **多层降级容错架构** | 设计"向量检索 → 关键词匹配"的渐进式降级策略：当嵌入模型不可用时自动切换至基于停用词过滤的轻量匹配，保证系统可用性 | `memory.py:24-31` |
| **任务类型自适应分类器** | 基于规则的快速意图识别（图文搜索/网页浏览/信息提取/多跳推理/事实问答），指导记忆召回的权重分配 | `memory.py:34-40` |
| **并发安全的懒加载** | 嵌入模型采用 Double-Check Locking 延迟初始化，首次调用时按需加载，多线程评测场景下通过 RLock 保证模型单例完整性 | `memory.py:60-61, 109-138` |
| **轻量持久化引擎** | 以 JSON 格式持久化全量记忆，支持增量更新与原子写入，无需外部数据库依赖 | `memory.py:83-98` |

**记忆驱动的进化闭环**:
```
新任务 → 语义/视觉召回 Top-K 相关经验 → 注入 Agent Prompt
                    ↑                              ↓
          经验沉淀写入记忆  ←  反思诊断/成功蒸馏  ←  Agent 执行
```

---

### 创新四：级联式检索增强工具链 (Cascaded Retrieval-Augmented Toolchain)

**核心思想**：构建"宽召回→精排→按需深抓"的三级级联检索管线，在保证召回率的同时通过 Cross-Encoder 精排大幅提升准确率，并引入延迟抓取策略避免不必要的网络开销。

**文件位置**: `tools.py` + `services/`

| 技术贡献 | 方法描述 | 代码位置 |
|----------|----------|----------|
| **面向组合的工具抽象** | 设计 BaseTool 统一接口规范与 ToolRegistry 注册表模式，支持工具集的热插拔与按需组合 | `tools.py:14-42, 87-103` |
| **两阶段检索 (Retrieve-then-Rerank)** | 第一阶段通过 Serper API 宽召回 20 条候选，第二阶段以 BGE-reranker-v2-m3 Cross-Encoder 精排为 top-k，在召回率与精确率间取得最优平衡 | `services/search_tools.py:55-131` |
| **Cross-Encoder 神经重排** | 引入 BAAI/bge-reranker-v2-m3 对 query-passage 对进行细粒度语义评分，相比 BM25 排序在多跳事实问答上显著提升相关性 | `services/search_reranker.py:48-171` |
| **延迟全文抓取 (Lazy Fetching)** | 精排前仅使用 snippet 评分（避免对 20 条候选逐一 Jina 抓取），精排后仅对最终 top-k 结果按需拉取全文，降低 80%+ 网络开销 | `services/search_tools.py:83-89` |
| **跨模态图搜文管线** | 本地图像 → 公网图床上传 → Google Lens 反向搜索 → 结构化结果返回，实现端到端的视觉信息检索 | `services/search_tools.py:133-155` |
| **多后端图床适配** | 支持 uguu.se / tmpfiles.org / 0x0.st 三种公网图床，通过策略模式实现后端无缝切换与故障转移 | `services/image_upload.py:100-154` |
| **沙盒化浏览器深度抓取** | 集成 Playwright Headless 浏览器服务，支持 JavaScript 渲染页面的深度内容获取与交互操作 | `services/browser_tools.py` |

**工具 Schema 定义**: `tool_schemas.py` — 严格对齐 OpenAI Function Calling 规范，确保与主流 LLM 推理框架无缝兼容。

---

### 创新五：信息隔离的闭卷评测范式

**核心思想**：提出严格的**信息隔离评测协议**——在整个推理与反思链路中完全屏蔽标准答案，确保 Agent 的"自我进化"能力在无监督信号条件下仍然有效，更贴近真实部署场景。

**文件位置**: `scripts/eval_benchmark.py`

| 技术贡献 | 方法描述 | 代码位置 |
|----------|----------|----------|
| **端到端信息隔离** | `ground_truth=None` 贯穿 Agent 推理、反思诊断、策略重试全链路，标准答案仅在评测打分阶段使用，杜绝信息泄露 | `scripts/eval_benchmark.py:435-438` |
| **混合式答案抽取** | 设计"正则模板→LLM 智能提取"的级联抽取策略，处理模型输出中夹杂解释、格式噪声的长尾 case | `scripts/eval_benchmark.py` (finalize_answer) |
| **线程安全的并发评测** | 基于 ThreadPoolExecutor 的多 worker 并行评测，共享 MemoryManager 通过读写锁保证记忆一致性 | `scripts/eval_benchmark.py` (run_eval) |
| **可恢复的断点续跑** | `--resume` 模式自动跳过已完成样本，支持大规模评测的中断容错 | launch 脚本中 RESUME_FLAG |
| **统一多 Benchmark 接口** | 以 JSONL 为统一数据协议，支持 2WikiMQA、SimpleVQA、GAIA 等多种 Benchmark 的零修改接入 | `scripts/eval_benchmark.py:1-16` |

---

## 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│                      eval_benchmark.py                    │
│              (闭卷批量评测, ground_truth=None)             │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     ReactAgent (agent.py)                 │
│  ┌─────────┐    ┌──────────┐    ┌───────────────────┐   │
│  │ Plan &  │───▶│ Tool Call │───▶│ Observe & Reason  │   │
│  │ Reason  │    │ (搜索/图搜│    │ (循环直到最终答案) │   │
│  │         │◀───│  /浏览器) │◀───│                   │   │
│  └─────────┘    └──────────┘    └───────────────────┘   │
│       │                                    │             │
│       │ 记忆注入                    失败/成功触发         │
│       ▼                                    ▼             │
│  ┌──────────┐                    ┌───────────────┐      │
│  │  Memory  │◀───写入经验/教训────│  Reflection   │      │
│  │(memory.py│                    │(reflection.py)│      │
│  │BGE+CLIP) │                    │ 诊断+修正策略  │      │
│  └──────────┘                    └───────────────┘      │
└─────────────────────────────────────────────────────────┘
        │                                    │
        ▼                                    ▼
┌──────────────┐                  ┌──────────────────┐
│ Qwen3.5-9B   │                  │  Qwen3.5-9B      │
│ (Agent 推理)  │                  │  (反思专用实例)   │
│ vLLM :8001   │                  │  vLLM :8004      │
└──────────────┘                  └──────────────────┘
```

---

## 评测模式说明

| 模式 | 命令 | 反思 | 记忆 | 工具 |
|------|------|------|------|------|
| `none` (纯推理) | `--mode none --tools none` | 关 | 关 | 无 |
| `search` (搜索基线) | `--mode none --tools search` | 关 | 关 | search_text |
| `full` (完整进化) | `--mode full --tools search` | 开 | 开 | search_text + search_image |

---

## 关键设计决策与技术哲学

### 1. 信息隔离原则 (Information Isolation Principle)

`ground_truth` 在整个推理-反思-重试链路中严格为 `None`。Agent 在闭卷模式下无法获得任何外部正确性信号，只能依靠**行为轨迹本身的异常模式**（步数耗尽、工具崩溃）触发自省。这一设计确保了 Agent 的进化能力在无监督条件下依然有效，更真实地模拟了实际部署中"没有标准答案可参考"的场景。

### 2. 推理-反思算力分离 (Compute Isolation for Inference & Reflection)

Agent 推理与反思诊断使用物理隔离的 vLLM 实例（不同 GPU / 端口），消除了推理过程中的显存竞争与延迟抖动。这一架构使得反思模型可独立扩缩容，为未来引入更大参数量的反思模型（如 32B）提供了无缝升级路径。

### 3. 经验驱动的持续自进化 (Experience-Driven Continual Self-Evolution)

系统构建了完整的"感知-行动-反思-记忆-迁移"认知循环：
```
失败/成功 → 轨迹诊断 → 策略/经验蒸馏 → 记忆沉淀 → 同题重试 / 未来任务召回
```
每一次任务执行都是一次学习机会，记忆库随评测推进持续增长，使得 Agent 在同一 Benchmark 上的表现具有**单调递增**的进化趋势。

### 4. 渐进式容错降级 (Graceful Degradation Architecture)

系统中每个可能失败的组件（Cross-Encoder Reranker、CLIP 视觉编码器、BGE-M3 文本编码器、外部图床）均设计了完整的降级路径。组件缺失时系统自动切换至次优但仍可用的备选方案，保证了端到端评测流程在任意硬件配置下的鲁棒性。

### 5. 模块化插拔与可复现性 (Modular Plug-and-Play Reproducibility)

所有外部依赖（LLM 端点、搜索代理、图床、浏览器服务）均通过环境变量注入，核心逻辑与基础设施完全解耦。同一套代码可无修改地运行在本地 GPU 集群、远程 vLLM 代理、或 PAI-EAS 云端推理服务上。
