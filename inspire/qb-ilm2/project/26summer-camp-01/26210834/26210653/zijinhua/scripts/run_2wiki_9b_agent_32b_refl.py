#!/usr/bin/env python3
"""
2Wiki 评测：9B Agent（:8001）+ 9B 反思（:8004 专用）。

与 run_2wiki_32b_eval.py（32B+32B）、run_2wiki_9b_ablation.py（旧版同端口 9B+9B）对照。

前置:
  - 9B Agent: bash scripts/start_vllm.sh              # :8001
  - 9B 反思: bash scripts/start_vllm_9b_reflection.sh # :8004 GPU2

用法:
  python scripts/run_2wiki_9b_agent_32b_refl.py --jsonl data/2wiki.jsonl
  python scripts/run_2wiki_9b_agent_32b_refl.py --limit 10 --resume
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _bootstrap import project_root

ROOT = project_root()
DEFAULT_JSONL = ROOT / "data" / "2wiki.jsonl"
DEFAULT_RUN_NAME = "2wiki_9b_agent_32b_refl"
AGENT_URL = "http://127.0.0.1:8001/v1"
AGENT_MODEL = "qwen-3.5"
REFL_URL = "http://127.0.0.1:8004/v1"
REFL_MODEL = AGENT_MODEL


def _apply_env(args: argparse.Namespace) -> None:
    agent_base = (args.llm_base_url or os.getenv("WIKI9_AGENT_BASE_URL", AGENT_URL)).strip()
    agent_model = (args.model_name or os.getenv("WIKI9_AGENT_MODEL", AGENT_MODEL)).strip()
    refl_base = (args.reflection_base_url or os.getenv("WIKI9_REFLECTION_BASE_URL", REFL_URL)).strip()
    refl_model = (args.reflection_model or os.getenv("WIKI9_REFLECTION_MODEL", REFL_MODEL)).strip()

    os.environ["LLM_BACKEND"] = "sglang"
    os.environ["LLM_BASE_URL"] = agent_base
    os.environ["MODEL_NAME"] = agent_model
    os.environ["LLM_API_KEY"] = os.getenv("LLM_API_KEY", "EMPTY")
    os.environ["LLM_ENABLE_THINKING"] = "false"
    os.environ.setdefault("EVAL_TEMPERATURE", "0.3")
    os.environ.setdefault("EVAL_MAX_TOKENS", "512")
    os.environ.setdefault("EVAL_LLM_TIMEOUT", "300")

    if args.mode == "full":
        os.environ["REFLECTION_LLM_BASE_URL"] = refl_base
        os.environ["REFLECTION_MODEL_NAME"] = refl_model
        os.environ["REFLECTION_ENABLE_THINKING"] = "false"


def _patch_eval_llm_client() -> None:
    import scripts.eval_benchmark as eb

    eb._eval_llm_client = None  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser(description="2Wiki：9B 推理 + 9B 反思")
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--mode", choices=("baseline", "full"), default="full")
    parser.add_argument("--tools", choices=("none", "search", "all", "full"), default="search")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6, dest="max_steps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--reflection-base-url", default=None)
    parser.add_argument("--reflection-model", default=None)
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

    _apply_env(args)
    _patch_eval_llm_client()

    print("=" * 60)
    print("2Wiki 评测：9B Agent + 9B 反思")
    print(f"  数据: {jsonl}")
    print(f"  Agent: {os.environ['LLM_BASE_URL']}  model={os.environ['MODEL_NAME']}")
    print(
        f"  mode={args.mode}  tools={args.tools}  max_steps={args.max_steps}  "
        f"workers={args.workers}  resume={args.resume}"
    )
    if args.mode == "full":
        print(
            f"  反思: {os.environ.get('REFLECTION_LLM_BASE_URL')}  "
            f"model={os.environ.get('REFLECTION_MODEL_NAME')}"
        )
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
            mode=args.mode,
            tools=args.tools,
            max_steps=args.max_steps,
            resume=args.resume,
            workers=args.workers,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
