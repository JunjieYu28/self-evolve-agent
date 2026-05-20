"""
Pro Max Batch Runner — 批量评测 + 跨任务进化
=============================================

关键设计：MemoryStore 在所有任务间共享，实现"进化"效果。
Task 1 无记忆 → Task N 拥有 N-1 条历史经验。

用法:
    python batch_runner.py --dataset 2wiki --max-cases 5
    python batch_runner.py --dataset simpleVQA --output-dir results/pro_simpleVQA
    python batch_runner.py --eval-only --output-dir results/pro_2wiki
"""
import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

import config
from agent import Agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("harness.batch_runner")

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
SIMPLEVQA_JSONL = config.DATA_DIR / "simpleVQA_99.jsonl"
SIMPLEVQA_IMG_DIR = config.DATA_DIR
WIKI2_TRAIN_JSONL = config.DATA_DIR / "2wiki_train_100.jsonl"
WIKI2_TEST_JSONL = config.DATA_DIR / "2wiki_test_100.jsonl"


def load_simplevqa() -> list[dict]:
    tasks = []
    with open(SIMPLEVQA_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            image_path = SIMPLEVQA_IMG_DIR / item.get("image", "")
            image_b64 = None
            if image_path.exists():
                with open(image_path, "rb") as img_f:
                    image_b64 = base64.b64encode(img_f.read()).decode()

            tasks.append({
                "id": f"svqa_{item['data_id']}",
                "instruction": item["question"],
                "image_b64": image_b64,
                "image_url": item.get("image_url", ""),
                "ground_truth": item.get("answer", ""),
                "dataset": "simpleVQA",
                "index": item["data_id"],
            })
    return tasks


def load_2wiki(use_train: bool = True) -> list[dict]:
    jsonl_path = WIKI2_TRAIN_JSONL if use_train else WIKI2_TEST_JSONL
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            tasks.append({
                "id": f"wiki_{idx}",
                "instruction": item["question"],
                "image_b64": None,
                "image_url": None,
                "ground_truth": item.get("answer", ""),
                "dataset": "2wiki",
                "index": idx,
            })
    return tasks


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------
def run_batch(
    tasks: list[dict],
    output_dir: Path,
    max_steps: int = config.MAX_STEPS,
    llm_base_url: str = config.LLM_BASE_URL,
    model_name: str = config.MODEL_NAME,
    resume: bool = True,
    enable_memory: bool = True,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"
    traj_dir = str(output_dir / "trajectories")

    # 共享 MemoryStore — 进化的关键
    memory_store = None
    if enable_memory:
        from modules.memory import MemoryStore
        memory_dir = str(output_dir / "memory_data")
        memory_store = MemoryStore(memory_dir)
        logger.info("MemoryStore initialized at: %s", memory_dir)

    # Resume support
    completed_ids = set()
    if resume and results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed_ids.add(r["task_id"])
        logger.info("Resuming: %d tasks already completed", len(completed_ids))

    agent = Agent(
        llm_base_url=llm_base_url,
        model_name=model_name,
        max_steps=max_steps,
        memory_store=memory_store,
    )

    results = []
    total = len(tasks)

    for i, task in enumerate(tasks):
        task_id = task["id"]
        if task_id in completed_ids:
            continue

        logger.info("[%d/%d] Running %s ...", i + 1, total, task_id)
        start_time = time.time()

        try:
            result = agent.run_task(task, trajectory_dir=traj_dir)
            elapsed = time.time() - start_time

            record = {
                "task_id": task_id,
                "dataset": task["dataset"],
                "index": task["index"],
                "instruction": task["instruction"],
                "image_url": task.get("image_url", ""),
                "ground_truth": task.get("ground_truth", ""),
                "pred": result["answer"],
                "steps": result["steps"],
                "total_tokens": result["total_tokens"],
                "tool_calls": result["tool_calls"],
                "task_type": result["task_type"],
                "trajectory_path": result["trajectory_path"],
                "elapsed_seconds": round(elapsed, 2),
            }
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error("Task %s failed: %s", task_id, exc, exc_info=True)
            record = {
                "task_id": task_id,
                "dataset": task["dataset"],
                "index": task["index"],
                "instruction": task["instruction"],
                "image_url": task.get("image_url", ""),
                "ground_truth": task.get("ground_truth", ""),
                "pred": f"[ERROR] {exc}",
                "steps": 0,
                "total_tokens": 0,
                "tool_calls": 0,
                "task_type": "error",
                "trajectory_path": "",
                "elapsed_seconds": round(elapsed, 2),
            }

        results.append(record)
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "[%d/%d] Done %s in %.1fs | steps=%d tokens=%d type=%s",
            i + 1, total, task_id, elapsed,
            record["steps"], record["total_tokens"], record["task_type"],
        )

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_results(results_file: Path):
    records = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("No results.")
        return

    by_dataset = {}
    for r in records:
        by_dataset.setdefault(r.get("dataset", "unknown"), []).append(r)

    for ds, items in by_dataset.items():
        has_gt = [r for r in items if r.get("ground_truth", "").strip()]
        if not has_gt:
            print(f"\n[{ds}] {len(items)} tasks, no ground truth for auto-eval")
            continue

        correct = 0
        for r in has_gt:
            gt = r["ground_truth"].strip().lower()
            pred = r.get("pred", "").strip().lower()
            if gt in pred or pred in gt:
                correct += 1

        acc = correct / len(has_gt) * 100
        print(f"\n[{ds}] Accuracy: {correct}/{len(has_gt)} = {acc:.1f}%")

        total_steps = sum(r["steps"] for r in items)
        total_tokens = sum(r.get("total_tokens", 0) for r in items)
        total_tools = sum(r.get("tool_calls", 0) for r in items)
        total_time = sum(r["elapsed_seconds"] for r in items)
        n = len(items)
        print(f"  Avg steps: {total_steps/n:.1f} | Avg tokens: {total_tokens/n:.0f} | Avg tools: {total_tools/n:.1f} | Avg time: {total_time/n:.1f}s")

        # 进化曲线: 前半 vs 后半
        mid = len(has_gt) // 2
        first_half = has_gt[:mid]
        second_half = has_gt[mid:]

        def half_acc(items):
            c = sum(1 for r in items if r["ground_truth"].strip().lower() in r.get("pred", "").strip().lower()
                    or r.get("pred", "").strip().lower() in r["ground_truth"].strip().lower())
            return c / len(items) * 100 if items else 0

        print(f"  Evolution: first half={half_acc(first_half):.1f}%, second half={half_acc(second_half):.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pro Max Batch Runner")
    parser.add_argument("--dataset", choices=["simpleVQA", "2wiki", "all"], default="all")
    parser.add_argument("--use-test", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--output-dir", default="results/pro")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--llm-url", default=config.LLM_BASE_URL)
    parser.add_argument("--model", default=config.MODEL_NAME)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.eval_only:
        results_file = output_dir / "results.jsonl"
        if not results_file.exists():
            print(f"No results: {results_file}")
            sys.exit(1)
        evaluate_results(results_file)
        return

    tasks = []
    if args.dataset in ("simpleVQA", "all"):
        tasks.extend(load_simplevqa())
    if args.dataset in ("2wiki", "all"):
        tasks.extend(load_2wiki(use_train=not args.use_test))

    if args.max_cases:
        tasks = tasks[:args.max_cases]

    logger.info("Total tasks: %d | Memory: %s", len(tasks), not args.no_memory)

    run_batch(
        tasks,
        output_dir=output_dir,
        max_steps=args.max_steps,
        llm_base_url=args.llm_url,
        model_name=args.model,
        resume=not args.no_resume,
        enable_memory=not args.no_memory,
    )

    print("\n" + "=" * 60)
    evaluate_results(output_dir / "results.jsonl")


if __name__ == "__main__":
    main()
