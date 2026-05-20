#!/usr/bin/env python3
"""
2Wiki 子集评测：32B ReAct +（可选）32B 反思，不改动 .env 与打榜 group_004。

数据默认: data/2wiki_train_100.jsonl（每行含 question / context / answer）

前置:
  1. 32B Agent（带 tool call）: bash scripts/start_vllm_32b_agent.sh   # :8003
  2. full 模式且 --reflection-on-32b: 本机 :8000 上 Qwen3-32B 反思服务

用法:
  python scripts/run_2wiki_32b_eval.py --limit 10
  python scripts/run_2wiki_32b_eval.py --mode full --tools search --workers 1
  python scripts/run_2wiki_32b_eval.py --tools none --mode baseline   # 仅用段落，不搜索

日志: logs/eval/2wiki_train_100_<mode>_<tools>/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _bootstrap import project_root

ROOT = project_root()
DEFAULT_JSONL = ROOT / "data" / "2wiki_train_100.jsonl"


def _apply_llm_env(args: argparse.Namespace) -> None:
    os.environ["LLM_BACKEND"] = "sglang"
    os.environ["LLM_BASE_URL"] = (
        args.llm_base_url or os.getenv("WIKI32_LLM_BASE_URL", "http://127.0.0.1:8003/v1")
    ).strip()
    os.environ["MODEL_NAME"] = (
        args.model_name or os.getenv("WIKI32_MODEL_NAME", "Qwen3-32B")
    ).strip()
    os.environ["LLM_API_KEY"] = os.getenv("LLM_API_KEY", "EMPTY")
    os.environ["LLM_ENABLE_THINKING"] = "false"
    os.environ.setdefault("EVAL_TEMPERATURE", "0.3")
    os.environ.setdefault("EVAL_MAX_TOKENS", "512")
    os.environ.setdefault("EVAL_LLM_TIMEOUT", "300")

    if args.mode == "full":
        if args.reflection_on_32b:
            os.environ["REFLECTION_LLM_BASE_URL"] = (
                args.reflection_base_url
                or os.getenv("WIKI32_REFLECTION_BASE_URL", "http://127.0.0.1:8000/v1")
            ).strip()
            os.environ["REFLECTION_MODEL_NAME"] = (
                args.reflection_model
                or os.getenv("WIKI32_REFLECTION_MODEL", "Qwen3-32B")
            ).strip()
        else:
            os.environ["REFLECTION_LLM_BASE_URL"] = (
                args.reflection_base_url
                or os.getenv("REFLECTION_LLM_BASE_URL", "http://127.0.0.1:8001/v1")
            ).strip()
            os.environ["REFLECTION_MODEL_NAME"] = (
                args.reflection_model
                or os.getenv("REFLECTION_MODEL_NAME", "qwen-3.5")
            ).strip()
        os.environ["REFLECTION_ENABLE_THINKING"] = "false"


def _patch_eval_llm_client() -> None:
    import scripts.eval_benchmark as eb

    eb._eval_llm_client = None  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser(description="2Wiki JSONL 评测（32B Agent + 可选 32B 反思）")
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(DEFAULT_JSONL),
        help=f"2Wiki JSONL 路径（默认 {DEFAULT_JSONL}）",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "full"),
        default="full",
        help="baseline=无反思记忆; full=反思+记忆（默认）",
    )
    parser.add_argument(
        "--tools",
        choices=("none", "search", "all", "full"),
        default="search",
        help="none=仅段落; search=search_text（默认）",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6, dest="max_steps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="32B Agent API（默认 http://127.0.0.1:8003/v1）",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--reflection-on-32b",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="full 模式反思走 :8000 Qwen3-32B（默认开启）",
    )
    parser.add_argument("--reflection-base-url", default=None)
    parser.add_argument("--reflection-model", default=None)
    args = parser.parse_args()

    jsonl = Path(args.jsonl)
    if not jsonl.is_file():
        raise FileNotFoundError(f"找不到数据文件: {jsonl}")

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from config import load_dotenv

    load_dotenv()
    for _k in ("all_proxy", "ALL_PROXY"):
        os.environ.pop(_k, None)

    _apply_llm_env(args)
    _patch_eval_llm_client()

    print("=" * 60)
    print("2Wiki 32B 评测")
    print(f"  数据: {jsonl}")
    print(f"  Agent: {os.environ['LLM_BASE_URL']}  model={os.environ['MODEL_NAME']}")
    print(f"  mode={args.mode}  tools={args.tools}  max_steps={args.max_steps}  workers={args.workers}")
    if args.mode == "full":
        print(
            f"  反思: {os.environ.get('REFLECTION_LLM_BASE_URL')}  "
            f"model={os.environ.get('REFLECTION_MODEL_NAME')}"
        )
    print("=" * 60)

    from scripts.eval_benchmark import run_eval

    eval_args = argparse.Namespace(
        dataset="2wiki",
        jsonl=str(jsonl),
        split="validation",
        limit=args.limit,
        offset=args.offset,
        mode=args.mode,
        tools=args.tools,
        max_steps=args.max_steps,
        resume=args.resume,
        workers=args.workers,
    )
    run_eval(eval_args)


if __name__ == "__main__":
    main()
