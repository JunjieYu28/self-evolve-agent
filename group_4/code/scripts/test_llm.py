"""测试 LLM 连通性（EAS 或本地 vLLM/SGLang）。"""

from __future__ import annotations

import os
import sys

from _bootstrap import project_root

ROOT = project_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from config import LLMConfig, load_dotenv  # noqa: E402
from llm_client import EASLLMClient, create_llm_client  # noqa: E402


def main() -> None:
    load_dotenv()
    cfg = LLMConfig.from_env()
    if cfg.backend == "sglang":
        assert cfg.sglang is not None
        print(f"Backend: vllm/sglang (OpenAI)")
        print(f"  URL:   {cfg.sglang.base_url}")
        print(f"  Model: {cfg.sglang.model_name}")
        client = create_llm_client(cfg)
    else:
        assert cfg.eas is not None
        client = EASLLMClient(cfg.eas)
        print(f"Backend: EAS")
        print(f"  URL:   {client.chat_url}")

    reply = client.chat_completion(
        [{"role": "user", "content": "用一句话介绍 ReAct 智能体。"}]
    )
    print("Reply:", reply.get("content", reply))


if __name__ == "__main__":
    main()
