"""测试 Qwen3.5-9B 多模态（本地 vLLM 需去掉 --language-model-only 后重启）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from _bootstrap import project_root

ROOT = project_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from config import LLMConfig, load_dotenv  # noqa: E402
from llm_client import create_llm_client  # noqa: E402
from vision_utils import build_vision_user_content  # noqa: E402

DEFAULT_IMAGE = ROOT / "data" / "SimpleVQA" / "images" / "0.webp"


def main() -> None:
    load_dotenv()
    image = Path(os.getenv("TEST_IMAGE", str(DEFAULT_IMAGE)))
    if not image.is_file():
        print(f"测试图片不存在: {image}")
        print("请先: python main.py download-vqa")
        sys.exit(1)

    cfg = LLMConfig.from_env()
    if cfg.backend != "sglang" or cfg.sglang is None:
        print("需要 LLM_BACKEND=vllm")
        sys.exit(1)

    client = create_llm_client(cfg)
    content = build_vision_user_content(
        "请用一句话描述这张图片中的主要内容。",
        image,
    )
    messages = [
        {"role": "system", "content": "你是视觉问答助手，结合图片简洁回答。"},
        {"role": "user", "content": content},
    ]

    print(f"LLM: {cfg.sglang.base_url} model={cfg.sglang.model_name}")
    print(f"Image: {image}")
    print("请求中...")

    try:
        resp = client.chat_completion(messages, tools=None)
    except Exception as exc:
        print(f"\n失败: {exc}")
        print(
            "\n若提示不支持 image / multimodal，请重启 vLLM：\n"
            "  kill $(cat logs/vllm.pid) 2>/dev/null; bash scripts/start_vllm.sh"
        )
        sys.exit(1)

    print("\n回复:", resp.get("content", resp))


if __name__ == "__main__":
    main()
