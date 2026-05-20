#!/usr/bin/env python3
"""
2Wiki 纯 ReAct Baseline（9B）：无反思、无记忆、无跨题经验注入。

与 full 模式区别:
  - 不写入 agent_memory.json，不调用 32B/9B 反思
  - 不重试（retry_on_wrong=False）
  - ReAct 主循环仍会把每轮 assistant + tool 结果留在 messages 里，
    作为下一轮的上下文（标准 OpenAI 多轮对话），即「工具调用后直接进下一轮」

数据: data/2wiki.jsonl（100 题）

前置:
  bash scripts/start_vllm.sh   # :8001 9B

用法:
  python scripts/run_2wiki_baseline.py
  python scripts/run_2wiki_baseline.py --limit 10
  python scripts/run_2wiki_baseline.py --resume

日志:
  logs/eval/2wiki_9b_baseline_full_search/
  logs/eval/runs/2wiki_9b_baseline.log

答案后处理:
  规则抽取 + 过长时可选 9B 精简（EVAL_ANSWER_LLM_EXTRACT=auto，见 scripts/answer_extract.py）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _bootstrap import project_root

ROOT = project_root()
DEFAULT_JSONL = ROOT / "data" / "2wiki.jsonl"
DEFAULT_RUN_NAME = "2wiki_9b_baseline"
DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "qwen-3.5"


def _apply_baseline_env(args: argparse.Namespace) -> None:
    base = (args.llm_base_url or os.getenv("WIKI9_LLM_BASE_URL", DEFAULT_BASE_URL)).strip()
    model = (args.model_name or os.getenv("WIKI9_MODEL_NAME", DEFAULT_MODEL)).strip()
    os.environ["LLM_BACKEND"] = "sglang"
    os.environ["LLM_BASE_URL"] = base
    os.environ["MODEL_NAME"] = model
    os.environ["LLM_API_KEY"] = os.getenv("LLM_API_KEY", "EMPTY")
    os.environ["LLM_ENABLE_THINKING"] = "false"
    os.environ.setdefault("EVAL_TEMPERATURE", "0.3")
    os.environ.setdefault("EVAL_MAX_TOKENS", "512")
    os.environ.setdefault("EVAL_LLM_TIMEOUT", "300")
    os.environ.setdefault("EVAL_ANSWER_LLM_EXTRACT", "auto")
    os.environ.setdefault("EVAL_ANSWER_LLM_THRESHOLD", "20")


def _patch_eval_llm_client() -> None:
    import scripts.eval_benchmark as eb

    eb._eval_llm_client = None  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2Wiki Baseline：9B ReAct，无反思/记忆"
    )
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--tools",
        choices=("none", "search", "all", "full"),
        default="search",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6, dest="max_steps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--no-llm-extract",
        action="store_true",
        help="关闭长答案 LLM 精简，仅规则抽取",
    )
    args = parser.parse_args()

    jsonl = Path(args.jsonl)
    if not jsonl.is_file():
        raise FileNotFoundError(f"找不到数据: {jsonl}")

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from config import load_dotenv

    load_dotenv()
    for _k in ("all_proxy", "ALL_PROXY"):
        os.environ.pop(_k, None)

    _apply_baseline_env(args)
    if args.no_llm_extract:
        os.environ["EVAL_ANSWER_LLM_EXTRACT"] = "0"
    _patch_eval_llm_client()

    print("=" * 60)
    print("2Wiki Baseline（9B ReAct，无反思/记忆）")
    print(f"  数据: {jsonl}")
    print(f"  Agent: {os.environ['LLM_BASE_URL']}  model={os.environ['MODEL_NAME']}")
    print(f"  tools={args.tools}  max_steps={args.max_steps}  workers={args.workers}")
    print(f"  答案精简: EVAL_ANSWER_LLM_EXTRACT={os.environ.get('EVAL_ANSWER_LLM_EXTRACT')}")
    print(f"  run_name={args.run_name}")
    print("=" * 60)

    from scripts.eval_benchmark import run_eval

    run_eval(
        argparse.Namespace(
            dataset="2wiki",
            jsonl=str(jsonl),
            split="validation",
            limit=args.limit,
            offset=args.offset,
            mode="baseline",
            tools=args.tools,
            max_steps=args.max_steps,
            resume=args.resume,
            workers=args.workers,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
