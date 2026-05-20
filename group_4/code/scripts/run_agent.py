"""使用真实 LLM + harness 工具（search-proxy:8090, browser:8080）运行 ReAct Agent。"""

from __future__ import annotations

import os
import sys

from _bootstrap import project_root

ROOT = project_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from agent import ReactAgent  # noqa: E402
from config import LLMConfig, load_dotenv  # noqa: E402
from llm_client import create_llm_client, create_reflection_llm_client  # noqa: E402
from logger import AgentLogger  # noqa: E402
from tools import create_production_registry  # noqa: E402


def main() -> None:
    load_dotenv()
    llm_cfg = LLMConfig.from_env()
    llm_client = create_llm_client(llm_cfg)
    reflection_client = create_reflection_llm_client(llm_cfg)

    logger = AgentLogger(log_dir="logs")
    registry = create_production_registry(include_mock=False)

    agent = ReactAgent(
        tool_registry=registry,
        logger=logger,
        llm_client=llm_client,
        reflection_llm_client=reflection_client,
        max_steps=int(os.getenv("MAX_STEPS", "10")),
    )

    instruction = os.getenv(
        "AGENT_INSTRUCTION",
        "什么是 ReAct 智能体？请用 search_text 搜索并简要总结。",
    )
    backend = llm_cfg.backend
    if backend == "sglang" and llm_cfg.sglang:
        print(f"LLM (Harness 基座): {llm_cfg.sglang.base_url} model={llm_cfg.sglang.model_name}")
    elif llm_cfg.eas:
        print(f"LLM (EAS): {llm_cfg.eas.api_url}")
    if llm_cfg.reflection:
        print(
            f"Reflection (辅助): {llm_cfg.reflection.base_url} "
            f"model={llm_cfg.reflection.model_name}"
        )
    print(f"Tools: {[s['function']['name'] for s in agent.get_tool_schemas()]}\n")

    image_path = os.getenv("AGENT_IMAGE_PATH") or None
    if image_path:
        print(f"Image: {image_path}")

    answer = agent.run(
        instruction=instruction,
        image_path=image_path,
        ground_truth=None,
        task_index=0,
    )

    print("=" * 60)
    print(answer)
    print("=" * 60)
    print(f"\n日志: {logger.results_path} | {logger.trajectory_path}")


if __name__ == "__main__":
    main()
