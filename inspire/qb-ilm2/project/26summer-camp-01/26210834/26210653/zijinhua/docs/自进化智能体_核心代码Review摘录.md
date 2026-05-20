# 自进化智能体 — 核心代码摘录（供架构 Review）

**项目路径**：`/data/jcy/zzs/zijinhua`  
**文档版本**：2026-05-20（Plan-and-Solve + finalize_answer + 9B/32B 消融矩阵 + 同题重试）

## 模型与服务分工

| 角色 | 默认端点 | 模型 | 用途 |
|------|----------|------|------|
| ReAct 基座（2Wiki / 打榜 9B） | `:8001` | Qwen3.5-9B | 主推理、工具调用 |
| 9B 反思（专用实例） | `:8004` | Qwen3.5-9B | `run_2wiki_9b_agent_32b_refl.py`（Agent :8001 + 反思 :8004，分卡） |
| ReAct 基座（32B） | `:8003` | Qwen3-32B | `run_2wiki_32b_eval.py` / benchmark 32B 消融 |
| 反思 / 诊断（32B） | `:8000` | Qwen3-32B | `mode=full`：失败诊断、成功总结、**同题重试策略提取** |
| 打榜 9B（可选分卡） | `:8002` | Qwen3.5-9B | `launch_benchmark_no_gold.sh` |
| 打榜 32B 反思（可选） | `:8003` | Qwen3-32B | 与 2Wiki 抢卡时需分时 |
| 文搜 / 图搜 | `:8090` | search-proxy | Serper + Jina；本地 BGE Reranker 精排 |
| 浏览器沙盒 | `:8080` | browser-service | Playwright；`DISABLE_BROWSER=1` 可关 |

**多卡参考**（`launch_benchmark_no_gold.sh`）：GPU0=:8001，GPU1=:8002，GPU2=:8000（32B 反思），GPU3=:8003；9B 反思实例另起 `:8004`（`start_vllm_9b_reflection.sh`）。

> **命名说明**：无 `WebSearchTool` / `SandboxBrowserTool` 类；HTTP 在 `services/search_client.py`、`services/browser_client.py`，经 `tools.HarnessFunctionTool` 注册为 `search_text` / `browser_*`。  
> `run_2wiki_9b_agent_32b_refl.py` 文件名保留历史，**当前实现为 9B Agent + 9B 反思（:8004）**，非 32B。

## 近期评测快照

### 2Wiki `data/2wiki.jsonl`（100 题，已完成）

| 配置 | EM | 均耗时 | 日志 / summary |
|------|-----|--------|----------------|
| 9B baseline + search | **57%** | 21.6s | `logs/eval/runs/2wiki_9b_baseline.log` |
| 9B + 32B full + search（raw pred） | **70%** | 45.9s | `2wiki_9b_agent_32b_refl`（旧跑次，32B 反思 :8000） |
| 9B + 32B full + 记忆（raw pred） | **69%** | 67.3s | `2wiki_9b_full_refl_mem` |
| 9B + 32B full + 记忆 + **`finalize_answer`** | **85%** | 78.4s | `2wiki_9b_full_refl_mem_no_gold_answer` |
| 段落 QA `2wiki_train_100`（tools=none） | **78%** | 12.2s | `2wiki_train_100.log` |

### 2Wiki 进行中 / 实验跑次（`finalize_answer`，截至日志快照）

| run_name | 进度 | 暂估 EM | 说明 |
|----------|------|---------|------|
| `2wiki_9b_judge_refl` | 73/100 | ~62% (45✓/28✗) | 日志头写「32B 裁判」；**当前主干 `agent.py` 无步内 judge API**，属实验/历史配置 |
| `2wiki_32b_judge_refl` | 57/100 | ~65% (37✓/20✗) | 32B Agent :8003 + 32B :8000 |
| `2wiki_9b_9b_refl` | 10/100 | ~90% (9✓/1✗) | 9B :8001 + 9B 反思 :8004，样本过少仅供参考 |

### 其他

| 任务 | 配置 | 结果 | 日志 |
|------|------|------|------|
| benchmark 100 | full+search，无标答 + `finalize_answer` | 已提交 `group_004.*` | `logs/benchmark/runs/group_004_no_gold.log` |
| benchmark 100（有 GT） | 历史 full+OSINT+Reranker | **~20%** EM | `ground_truth.json` 对照 |
| SimpleVQA 99 | baseline+search / full 无搜索 | 33.3% / 40.4% | `simpleVQA_99_*` |

> **判分**：`scripts/answer_extract.normalize_answer` + 子串包含。`finalize_answer` **不使用 gold**；`results.jsonl` 含 `pred` / `pred_raw` / `extract`（`rule` / `llm` / `raw` 等）。

---

## 1. 主控大脑 `agent.py` — `ReactAgent`

**运行状态**：ReAct、Plan-and-Solve、工具调用、记忆注入、失败反思（含修正策略）、答对成功反思、**同题重试**均已跑通。JSON 净化在 `Message.from_dict` **之前**执行。

### 1.1 OSINT + Plan-and-Solve Prompt

```python
REACT_SYSTEM_PROMPT = (
    "You are an expert OSINT ReAct agent...\n\n"
    "## 强制前置计划法则 (Plan-and-Solve) — 不可跳过\n"
    "在第一次调用任何工具之前，必须先输出 <plan>...</plan>\n"
    + OSINT_SEARCH_WORKFLOW  # 短 query、渐进式 OSINT、领域关键词等
)
```

`scripts/run_benchmark.py` 的 `BENCHMARK_SYSTEM` / `BENCHMARK_FULL_SYSTEM` 已拼接 `OSINT_SEARCH_WORKFLOW`。

### 1.2 `run` — ReAct 主循环 + 记忆 + 失败/成功 + 重试

```python
def run(self, instruction, image_path=None, ground_truth=None,
        task_index=None, system_prompt=None, retry_on_wrong=False) -> str:
    if self.enable_memory:
        self._memory_user_suffix = self.memory_manager.build_memory_user_suffix(
            instruction, top_k=3, image_path=image_path)

    for step in range(self.max_steps):
        self._sanitize_llm_response_tool_calls_dict(llm_response)  # 先于 Message.from_dict
        ...
        if assistant_msg.tool_calls:
            # 执行工具；连续 3 次报错 → _run_reflection(CONSECUTIVE_TOOL_ERRORS) 并 abort
            continue

        if assistant_msg.content.strip():
            final_answer = assistant_msg.content.strip()
            break

    if not final_answer:
        self._run_reflection(..., MAX_STEPS_EXCEEDED, final_pred=...)
    elif (enable_reflection and enable_memory and ground_truth
          and not _answers_match(final_answer, ground_truth)):
        self._run_reflection(..., WRONG_ANSWER, ...)  # 仅 full+记忆+有标答评测
    elif _should_store_success_memory() and _answers_match(...):
        self._run_success_reflection(...)

    # full 模式评测默认开启：从反思文本抽「修正策略」后同题重试（与 ground_truth 解耦）
    if retry_on_wrong and self.last_reflection:
        strategy = MemoryManager.extract_correction_strategy(self.last_reflection)
        if strategy:
            final_answer = self._retry_with_correction(...) or final_answer

    return final_answer
```

**反思 / 记忆触发小结**：

| 类型 | 时机 | 写入记忆 |
|------|------|----------|
| `CONSECUTIVE_TOOL_ERRORS` | 循环内连续 3 次 tool 报错 | 失败反思后 `add_memory_from_reflection`（含修正策略→good） |
| `MAX_STEPS_EXCEEDED` | 步数用尽 | 同上 |
| `WRONG_ANSWER` | `enable_memory` + 有 `ground_truth` 且 EM 错 | 同上 |
| 成功记忆 | `MEMORY_STORE_SUCCESS` + 答对 | `add_memory_from_success_reflection` |
| 同题重试 | `retry_on_wrong` + 有 `last_reflection` | 不单独入库；`results.retried=true` |

`eval_benchmark.py`：`retry_on_wrong=(mode == "full")`。

### 1.3 JSON 净化（防 vLLM 400）

```python
@staticmethod
def _sanitize_llm_response_tool_calls_dict(llm_response: dict) -> None:
    # 非法 arguments → "{}"，避免 Unterminated string → 400

@classmethod
def _compact_messages(cls, messages, max_chars=12000) -> list[dict]:
    # 折叠中间轮次；折叠时 arguments 写 "{}"，禁止截断加 "..."
```

### 1.4 `_call_llm` / 反思客户端

```python
def _call_llm(self, messages, *, use_tools=True) -> dict:
    return self.llm_client.chat_completion(api_messages, tools=tools or None)

# 基座: LLM_BASE_URL / MODEL_NAME
# 反思: REFLECTION_LLM_BASE_URL / REFLECTION_MODEL_NAME（默认 :8000 Qwen3-32B）
```

---

## 2. 免疫系统 `reflection.py` & `memory.py`

### 2.1 反思触发 `ReflectionState`

```python
def record_tool_result(self, result: str) -> None:
    if s.lower().startswith("error"): ...
    if json.loads(s).get("ok") is False: ...  # 浏览器等结构化失败
    else: self.consecutive_tool_errors = 0

def should_trigger(self) -> FailureReason | None:
    if self.got_final_answer:
        return None
    ...
```

### 2.2 失败 / 成功反思 Prompt

```python
REFLECTION_SYSTEM_PROMPT = """
## 失败诊断
## 问题步骤
## 修正策略          ← 供 extract_correction_strategy / 同题重试
## 建议的工具调用序列
"""

SUCCESS_REFLECTION_SYSTEM_PROMPT = """
## 成功要点
## 有效工具链
## 答案依据
"""
```

### 2.3 记忆 JSON + 多模态召回

```python
def add_memory_from_reflection(self, instruction, reflection_text, ...):
    parsed_bad, parsed_good = self.parse_reflection(reflection_text)
    # 失败记忆：bad + good（修正策略段落）

def add_memory_from_success_reflection(...):
    # 仅 good_experience，outcome=success

def get_relevant_memories(self, instruction, top_k=3, image_path=None):
    # 有图 → CLIP；纯文本 → BGE-M3（/data/jcy/ckpt/bge-m3）；降级 → 关键词

@staticmethod
def extract_correction_strategy(reflection_text: str) -> str:
    _, good = parse_reflection(reflection_text)
    return good[:800] or reflection_text[:500]
```

**记忆单条 schema**（节选）：

```json
{
  "task_id": "mem_0001",
  "task_type": "多跳推理",
  "bad_experience": "...",
  "good_experience": "...",
  "instruction": "...",
  "text_embedding": [...],
  "image_embedding": [...],
  "outcome": "failure | success"
}
```

`threading.RLock`：多 worker 共享 `agent_memory.json` 时读写加锁。

---

## 3. 真实工具接入 `tools.py` + `services/*`

**运行状态**：`search_text` → **8090 召回（默认 20 条）+ BGE Reranker top_k**；`SEARCH_RERANKER_ENABLED=false` 可关。打榜默认 `fetch=false`。

```python
def create_production_registry(include_mock=False) -> ToolRegistry:
    if os.getenv("DISABLE_BROWSER", "0") in ("1", "true", "yes"):
        return create_search_only_registry(include_mock)
    return create_harness_registry(include_mock)
```

---

## 4. 答案后处理 `scripts/answer_extract.py`

```python
def finalize_answer(raw_pred, question, *, llm_client=None) -> tuple[str, str]:
    """
    规则抽取 → 超长则可选 9B LLM 精简。
    绝不使用 gold。
    """
```

| 环境变量 | 默认 |
|----------|------|
| `EVAL_ANSWER_LLM_EXTRACT` | `auto` |
| `EVAL_ANSWER_LLM_THRESHOLD` | `20` |

---

## 5. 埋点与日志 `logger.py`

| 字段 | `trajectory.jsonl` | `results.jsonl` |
|------|-------------------|-----------------|
| `task_index` / `tool_calls` / `tool_call_id` | ✓ | — |
| `index` / `instruction` / `answer` / `pred` | — | ✓ |
| `pred_raw` / `extract` | — | ✓ 评测 |
| `retried` | — | ✓ 同题重试 |

`read_trajectory(task_index=...)` 按题过滤，避免多 worker 交错。

---

## 6. 评测与打榜入口

| 脚本 | 用途 |
|------|------|
| `scripts/run_benchmark.py` | 打榜 → `group_XXX.json/csv/zip` |
| `scripts/eval_benchmark.py` | 通用评测 + `finalize_answer` |
| `scripts/run_2wiki_baseline.py` | 2Wiki baseline |
| `scripts/run_2wiki_9b_agent_32b_refl.py` | **9B :8001 + 9B 反思 :8004** |
| `scripts/run_2wiki_9b_ablation.py` | 9B+9B 同端口 :8001（`2wiki_train_100`） |
| `scripts/run_2wiki_32b_eval.py` | 32B Agent :8003 + 32B 反思 :8000 |
| `scripts/run_benchmark_32b_ablation.py` | benchmark 32B 消融 |
| `scripts/answer_extract.py` | 答案规整 |

| Launch | 说明 |
|--------|------|
| `launch_2wiki_9b_baseline.sh` | baseline |
| `launch_2wiki_9b_agent_32b_refl.sh` | 9B+9B（:8004 反思） |
| `launch_2wiki_9b_ablation.sh` | 9B 同端口消融 |
| `launch_benchmark_no_gold.sh` | 打榜无标答 |

| 参数 | 效果 |
|------|------|
| `--mode baseline` | 无反思、无记忆、无重试 |
| `--mode full` | 反思 + 记忆 + `retry_on_wrong` |
| `--tools search` | `search_text` + `search_image` |
| `--max-steps 6` | 评测常用 |

---

## 7. 待 Review / 已知问题

1. **EM 与 pred 格式**：同一轨迹 `finalize_answer` 可从 ~70% 提到 ~85%；需与赛方口径对齐。  
2. **打榜无标答**：不触发 `WRONG_ANSWER` / 成功记忆，但 **仍可做同题重试**（`retry_on_wrong` 不依赖 gold）。  
3. **共享记忆库**：多题 `agent_memory.json` 可能交叉污染；失败记忆的「修正策略」会进入 `good_experience` 被召回。  
4. **上下文 8K**：多模态 + 多轮 search 仍可能超 `max_model_len=8192`。  
5. **9B 反思质量**：`:8004` 9B 做诊断/成功总结弱于 32B，消融待跑完。  
6. **并发轨迹**：`workers>1` 须按 `task_index` 过滤 `trajectory.jsonl`。  
7. **双模型显存**：32B Agent + 32B 反思需多卡；9B+9B 需 :8001 + :8004 两实例。  
8. **实验 run 名 `*_judge_refl`**：日志与当前主干代码可能不一致，以 `agent.py` 为准。

---

## 8. 关键文件索引

```
agent.py                      # ReactAgent.run, Plan-and-Solve, 重试, JSON 净化
reflection.py                 # REFLECTION / SUCCESS prompts, FailureReason
memory.py                     # 召回, extract_correction_strategy
scripts/answer_extract.py     # finalize_answer
tools.py + services/search_*  # 搜索 + Reranker
logger.py
scripts/eval_benchmark.py
scripts/run_2wiki_9b_agent_32b_refl.py   # 实际 9B+9B
scripts/run_2wiki_32b_eval.py
scripts/run_benchmark.py
scripts/launch_2wiki_9b_agent_32b_refl.sh
scripts/launch_benchmark_no_gold.sh
```

---

*维护：代码变更后请同步更新「模型分工」「近期评测」「已知问题」三节。*
