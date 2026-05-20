#!/usr/bin/env python3
"""
32B 推理消融：benchmark 前 N 题，独立组号/日志，不修改 group_004 与 run_benchmark.py。

默认将 ReAct 基座指向本机 32B Agent vLLM（:8003，需先运行 start_vllm_32b_agent.sh）。
与正在跑的 9B group_004（:8001）互不干扰。

用法:
  python scripts/run_benchmark_32b_ablation.py
  python scripts/run_benchmark_32b_ablation.py --limit 50 --mode baseline --tools search
  python scripts/run_benchmark_32b_ablation.py --mode full --reflection-port 8001

输出:
  group_032.json / group_032.csv / group_032.zip
  logs/benchmark/group_032/
  logs/benchmark/runs/group_032_32b.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _bootstrap import project_root

ROOT = project_root()


def _apply_ablation_env(args: argparse.Namespace) -> None:
    """在加载 .env 之后覆盖 LLM 相关变量（不改动 .env 文件）。"""
    os.environ["LLM_BACKEND"] = "sglang"
    os.environ["LLM_BASE_URL"] = (
        args.llm_base_url or os.getenv("ABLATION_LLM_BASE_URL", "http://127.0.0.1:8003/v1")
    ).strip()
    os.environ["MODEL_NAME"] = (
        args.model_name or os.getenv("ABLATION_MODEL_NAME", "Qwen3-32B")
    ).strip()
    os.environ["LLM_ENABLE_THINKING"] = "false"
    os.environ.setdefault("EVAL_TEMPERATURE", "0.3")
    os.environ.setdefault("EVAL_MAX_TOKENS", "512")
    os.environ.setdefault("EVAL_LLM_TIMEOUT", "300")

    if args.mode == "full" and args.reflection_on_9b:
        # full 模式：反思仍走 9B，避免与 32B Agent 抢 :8000/:8003
        os.environ["REFLECTION_LLM_BASE_URL"] = (
            args.reflection_base_url
            or os.getenv("ABLATION_REFLECTION_BASE_URL", "http://127.0.0.1:8001/v1")
        ).strip()
        os.environ["REFLECTION_MODEL_NAME"] = (
            args.reflection_model or os.getenv("ABLATION_REFLECTION_MODEL", "qwen-3.5")
        ).strip()


def _patch_eval_llm_client() -> None:
    """重置 eval 共享 client，确保使用上面覆盖后的环境变量。"""
    import scripts.eval_benchmark as eb

    eb._eval_llm_client = None  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser(
        description="32B 消融 benchmark（独立组号，默认前 50 题）"
    )
    parser.add_argument(
        "--group",
        default="032",
        help="组号（默认 032 -> group_032.*；勿用 004）",
    )
    parser.add_argument("--limit", type=int, default=50, help="题目数量（默认 50）")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("baseline", "full"),
        default="baseline",
        help="baseline=仅 32B ReAct；full=带记忆/反思（默认 baseline，对比更干净）",
    )
    parser.add_argument(
        "--tools",
        choices=("none", "search", "all", "full"),
        default="search",
    )
    parser.add_argument("--max-steps", type=int, default=6, dest="max_steps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="32B Agent API（默认 http://127.0.0.1:8003/v1）",
    )
    parser.add_argument("--model-name", default=None, help="默认 Qwen3-32B")
    parser.add_argument(
        "--reflection-on-9b",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="full 模式时反思走 9B :8001（默认开启，避免抢 32B）",
    )
    parser.add_argument("--reflection-base-url", default=None)
    parser.add_argument("--reflection-model", default=None)
    parser.add_argument(
        "--no-fast-search",
        action="store_true",
        help="关闭打榜搜索加速",
    )
    ablation_args = parser.parse_args()

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from config import load_dotenv

    load_dotenv()
    _apply_ablation_env(ablation_args)
    _patch_eval_llm_client()

    from scripts import run_benchmark as rb

    rb_args = argparse.Namespace(
        group=ablation_args.group,
        mode=ablation_args.mode,
        tools=ablation_args.tools,
        limit=ablation_args.limit,
        offset=ablation_args.offset,
        max_steps=ablation_args.max_steps,
        resume=ablation_args.resume,
        workers=ablation_args.workers,
        no_fast_search=ablation_args.no_fast_search,
    )

    log_path = ROOT / "logs" / "benchmark" / "runs" / "group_032_32b.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("32B 消融实验")
    print(f"  Agent LLM: {os.environ['LLM_BASE_URL']}  model={os.environ['MODEL_NAME']}")
    print(f"  组号: {ablation_args.group}  题数: {ablation_args.limit}  mode={ablation_args.mode}")
    if ablation_args.mode == "full" and ablation_args.reflection_on_9b:
        print(
            f"  反思 LLM: {os.environ.get('REFLECTION_LLM_BASE_URL')}  "
            f"model={os.environ.get('REFLECTION_MODEL_NAME')}"
        )
    print(f"  日志: {log_path}")
    print("  说明: 不影响 group_004 / :8001 9B 正在运行的任务")
    print("=" * 60)

    rb.run_benchmark(rb_args)


if __name__ == "__main__":
    main()
