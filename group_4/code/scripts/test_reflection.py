#!/usr/bin/env python3
"""测试本机 Qwen3-32B 反思模型连通性。"""
from __future__ import annotations

import os
import sys

from _bootstrap import project_root

ROOT = project_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from config import LLMConfig, load_dotenv  # noqa: E402
from llm_client import create_reflection_llm_client  # noqa: E402
from reflection import FailureReason, ReflectionManager  # noqa: E402


def main() -> None:
    load_dotenv()
    cfg = LLMConfig.from_env()
    assert cfg.reflection is not None
    client = create_reflection_llm_client(cfg)
    rm = ReflectionManager()

    fake_trajectory = [
        {
            "step_id": 0,
            "role": "user",
            "content": "请查询某城市所在国家的首都",
            "tool_call_id": None,
        },
        {
            "step_id": 1,
            "role": "assistant",
            "content": "调用 ghost_search",
            "tool_call_id": None,
        },
        {
            "step_id": 1,
            "role": "tool",
            "content": "Error: unknown tool 'ghost_search'",
            "tool_call_id": "call_0",
        },
    ]

    request = rm.build_diagnostic_request(
        trajectory=fake_trajectory,
        failure_reason=FailureReason.CONSECUTIVE_TOOL_ERRORS,
        instruction="请查询某城市所在国家的首都",
    )
    print(f"Reflection endpoint: {cfg.reflection.base_url}")
    print(f"Model: {cfg.reflection.model_name}\n")
    reply = rm.call_diagnostic_model(client, request)
    print("--- 反思输出（节选）---")
    print(reply[:1200])


if __name__ == "__main__":
    main()
