#!/usr/bin/env python3
"""
2Wiki 消融：9B 推理 + 9B 反思（同一 vLLM :8001），与 32B+32B 对照。

与历史跑次区别:
  - logs/eval/2wiki_train_100_full_search/（2wiki_train_100.log）
    旧消融：Agent 与反思同 :8001；新默认反思走专用 :8004
  - 本脚本 Agent / 反思均走 http://127.0.0.1:8001/v1  qwen-3.5

前置:
  bash scripts/start_vllm.sh   # 或 restart_vllm.sh，确保 :8001 9B 已起

用法:
  python scripts/run_2wiki_9b_ablation.py --limit 10
  python scripts/run_2wiki_9b_ablation.py --mode full --tools search --workers 1
  python scripts/run_2wiki_9b_ablation.py --resume

日志（独立目录，不覆盖 32B / 旧 9B 结果）:
  logs/eval/2wiki_train_100_9b_ablation_full_search/
  控制台可重定向: logs/eval/runs/2wiki_train_100_9b_ablation.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _bootstrap import project_root

ROOT = project_root()
DEFAULT_JSONL = ROOT / "data" / "2wiki_train_100.jsonl"
DEFAULT_RUN_NAME = "2wiki_train_100_9b_ablation"
DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "qwen-3.5"


def _apply_9b_env(args: argparse.Namespace) -> None:
    """Agent 与反思均使用本机 9B vLLM，不改动 .env 文件。"""
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

    if args.mode == "full":
        refl_base = (args.reflection_base_url or os.getenv("WIKI9_REFLECTION_BASE_URL", base)).strip()
        refl_model = (args.reflection_model or os.getenv("WIKI9_REFLECTION_MODEL", model)).strip()
        os.environ["REFLECTION_LLM_BASE_URL"] = refl_base
        os.environ["REFLECTION_MODEL_NAME"] = refl_model
        os.environ["REFLECTION_ENABLE_THINKING"] = "false"


def _patch_eval_llm_client() -> None:
    import scripts.eval_benchmark as eb

    eb._eval_llm_client = None  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2Wiki 消融：9B Agent + 9B 反思（:8001）"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(DEFAULT_JSONL),
        help=f"2Wiki JSONL（默认 {DEFAULT_JSONL}）",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=f"日志目录前缀（默认 {DEFAULT_RUN_NAME}）",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "full"),
        default="full",
        help="baseline=无反思记忆; full=9B反思+记忆（默认，与 32B 消融对齐）",
    )
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
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help=f"9B API（默认 {DEFAULT_BASE_URL}）",
    )
    parser.add_argument("--model-name", default=None, help=f"默认 {DEFAULT_MODEL}")
    parser.add_argument(
        "--reflection-base-url",
        default=None,
        help="默认与 Agent 相同（:8001）",
    )
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

    _apply_9b_env(args)
    _patch_eval_llm_client()

    agent_url = os.environ["LLM_BASE_URL"]
    agent_model = os.environ["MODEL_NAME"]
    print("=" * 60)
    print("2Wiki 9B 消融（推理 + 反思均 9B）")
    print(f"  数据: {jsonl}")
    print(f"  Agent: {agent_url}  model={agent_model}")
    print(
        f"  mode={args.mode}  tools={args.tools}  max_steps={args.max_steps}  "
        f"workers={args.workers}"
    )
    if args.mode == "full":
        print(
            f"  反思: {os.environ.get('REFLECTION_LLM_BASE_URL')}  "
            f"model={os.environ.get('REFLECTION_MODEL_NAME')}"
        )
    print(f"  日志目录名: {args.run_name}_<mode>_<tools>")
    print("  对照: run_2wiki_32b_eval.py（32B+32B）")
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
        run_name=args.run_name,
    )
    run_eval(eval_args)


if __name__ == "__main__":
    main()
