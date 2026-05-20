"""
全链路闭环联调：模拟「失败 → 反思 → 记忆 → 进化成功」。

运行：python scripts/integration_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from _bootstrap import project_root

ROOT = project_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from agent import REACT_SYSTEM_PROMPT, ReactAgent  # noqa: E402
from logger import AgentLogger  # noqa: E402
from memory import MemoryManager  # noqa: E402
from reflection import ReflectionManager  # noqa: E402
from tools import create_default_registry  # noqa: E402

TASK_INSTRUCTION = "识别图片 ./123.jpg 中的城市并查询首都"
IMAGE_PATH = "./123.jpg"
GROUND_TRUTH = "阿布扎比"
LOG_DIR = Path("logs")
MEMORY_PATH = Path("agent_memory.json")

MOCK_REFLECTION_REPORT = """## 失败诊断
Agent 在有图片输入时反复调用不存在的工具 `ghost_search`，且从未使用 `mock_search` 进行图文检索，导致连续工具报错并被迫中止。

## 问题步骤
- Step 1: 调用了 ghost_search（工具不存在）
- Step 2: 再次调用 ghost_search，未处理 image_path
- Step 3: 第三次调用 ghost_search，触发连续错误阈值

## 修正策略
1. 有图片路径时，应使用 mock_search 并传入 query 与图片相关关键词。
2. 禁止调用未在工具列表中注册的工具名称。
3. 拿到搜索结果后，直接给出首都名称，不要继续无效循环。

## 建议的工具调用序列
先 mock_search(query="图片中城市 首都") → 根据返回确认城市 → 输出最终首都名称。
"""


class IntegrationMockLLM:
    def __init__(self, mode: str) -> None:
        if mode not in ("fail", "success"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self._turn = 0

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._is_reflection_call(messages):
            print("    [MockLLM] 生成反思诊断报告 ...")
            return {"role": "assistant", "content": MOCK_REFLECTION_REPORT}

        self._turn += 1
        if self.mode == "fail":
            print(f"    [MockLLM] 第 {self._turn} 轮 -> 调用不存在工具 ghost_search")
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_fail_{self._turn}",
                        "type": "function",
                        "function": {
                            "name": "ghost_search",
                            "arguments": json.dumps(
                                {"query": "city in image capital"},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }

        if self._turn == 1:
            print(f"    [MockLLM] 第 {self._turn} 轮 -> 正确调用 mock_search")
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_ok_001",
                        "type": "function",
                        "function": {
                            "name": "mock_search",
                            "arguments": json.dumps(
                                {"query": "图片中城市是哪里 首都是什么"},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }

        print(f"    [MockLLM] 第 {self._turn} 轮 -> 输出最终答案")
        return {"role": "assistant", "content": GROUND_TRUTH}

    @staticmethod
    def _is_reflection_call(messages: list[dict[str, Any]]) -> bool:
        if not messages:
            return False
        first = messages[0]
        return first.get("role") == "system" and "诊断专家" in str(
            first.get("content", "")
        )


def log_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def preview_file(path: Path, max_lines: int = 3) -> None:
    if not path.is_file():
        print(f"  [缺失] {path}")
        return
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    print(f"  {path} ({len(lines)} 行)")
    for line in lines[:max_lines]:
        print(f"    {line[:120]}{'...' if len(line) > 120 else ''}")
    if len(lines) > max_lines:
        print(f"    ... 共 {len(lines)} 行")


def reset_workspace() -> None:
    if LOG_DIR.exists():
        shutil.rmtree(LOG_DIR)
    if MEMORY_PATH.exists():
        MEMORY_PATH.unlink()


def build_agent(mock_llm: IntegrationMockLLM, max_steps: int = 5) -> ReactAgent:
    return ReactAgent(
        tool_registry=create_default_registry(),
        logger=AgentLogger(log_dir=LOG_DIR),
        llm_client=mock_llm,  # type: ignore[arg-type]
        reflection_manager=ReflectionManager(tool_error_threshold=3),
        memory_manager=MemoryManager(storage_path=MEMORY_PATH),
        max_steps=max_steps,
        enable_reflection=True,
        enable_memory=True,
    )


def main() -> None:
    log_section("全链路闭环联调启动")
    reset_workspace()

    log_section("模块初始化")
    memory = MemoryManager(storage_path=MEMORY_PATH)
    print(f"  项目根目录: {ROOT}")
    print(f"  Tools: {[t.name for t in create_default_registry().list_tools()]}")

    log_section("=== 第一次尝试（预期失败）===")
    agent_fail = build_agent(IntegrationMockLLM(mode="fail"))
    print(f"启动前记忆: {len(memory.list_memories())} 条")
    answer1 = agent_fail.run(
        instruction=TASK_INSTRUCTION,
        image_path=IMAGE_PATH,
        ground_truth=GROUND_TRUTH,
        task_index=0,
    )
    print(f"输出: {answer1[:120]}...")
    print(f"失败原因: {agent_fail.last_failure_reason}")

    log_section("=== 第二次尝试（预期成功 · 带记忆）===")
    agent_ok = build_agent(IntegrationMockLLM(mode="success"))
    recalled = agent_ok.memory_manager.get_relevant_memories(TASK_INSTRUCTION, top_k=3)
    print(f"召回记忆: {len(recalled)} 条")
    answer2 = agent_ok.run(
        instruction=TASK_INSTRUCTION,
        image_path=IMAGE_PATH,
        ground_truth=GROUND_TRUTH,
        task_index=1,
    )
    print(f"输出: {answer2}")
    print(f"成功: {answer2.strip() == GROUND_TRUTH}")

    log_section("落盘检查")
    logger = AgentLogger(log_dir=LOG_DIR)
    preview_file(logger.results_path)
    preview_file(logger.trajectory_path)
    preview_file(MEMORY_PATH, max_lines=6)
    print("\n请检查 results.jsonl 和 trajectory.jsonl 是否正确生成。")


if __name__ == "__main__":
    main()
