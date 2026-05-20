"""
Benchmark Runner for harness-pro (with strategy memory evolution)
================================================================
Uses agent.py with shared MemoryStore across all tasks.
Output format matches instructor requirements.
"""
import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(Path(__file__).parent))

import config
from agent import Agent
from modules.memory import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("harness.benchmark")

BENCHMARK_CSV = Path("/inspire/qb-ilm2/project/26summer-camp-01/public/benchmark.csv")
DEFAULT_OUTPUT_DIR = Path("/inspire/qb-ilm2/project/26summer-camp-01/26210834/26210834/results")


def build_proxy_url(port: int) -> str:
    base = (
        "https://notebook-inspire.sii.edu.cn/"
        "ws-7c23bd1d-9bae-4238-803a-737a35480e18/"
        "project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/"
        "user-5b1a894f-4b65-4bb2-afcd-8691a9eec556/"
        "vscode/22680b47-5984-4188-97d7-b37d50c64593/"
        "02f9bedd-7c26-495c-9881-33b246833f17/"
        f"proxy/{port}/v1"
    )
    return base


def load_benchmark():
    tasks = []
    with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            raw_image = row.get("image", "").strip()
            if raw_image and not raw_image.startswith(("http://", "https://")):
                image_b64 = raw_image
                image_url = None
            elif raw_image:
                image_b64 = None
                image_url = raw_image
            else:
                image_b64 = None
                image_url = None
            tasks.append({
                "id": f"bench_{idx}",
                "instruction": row["problem"],
                "image_b64": image_b64,
                "image_url": image_url,
                "ground_truth": row.get("answer", ""),
                "index": idx,
            })
    return tasks


def run_single(agent: Agent, task: dict, traj_dir: str) -> dict:
    start = time.time()
    try:
        result = agent.run_task(task, trajectory_dir=traj_dir)
        elapsed = time.time() - start
        return {
            "index": task["index"],
            "task_id": task["id"],
            "instruction": task["instruction"][:200],
            "pred": result["answer"],
            "steps": result["steps"],
            "total_tokens": result.get("total_tokens", 0),
            "tool_calls": result.get("tool_calls", 0),
            "task_type": result.get("task_type", ""),
            "elapsed_seconds": round(elapsed, 2),
        }
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("Task %s failed: %s", task["id"], exc)
        return {
            "index": task["index"],
            "task_id": task["id"],
            "instruction": task["instruction"][:200],
            "pred": f"[ERROR] {exc}",
            "steps": 0,
            "total_tokens": 0,
            "tool_calls": 0,
            "task_type": "error",
            "elapsed_seconds": round(elapsed, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Runner (harness-pro)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0, help="Start from this question index")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--group-id", default="4")
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-increment run directory: run1-pro/, run2-pro/, ...
    suffix = "-pro"
    existing_runs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("run") and d.name.endswith(suffix)]
    run_nums = []
    for d in existing_runs:
        try:
            num_part = d.name[3:].replace(suffix, "")
            run_nums.append(int(num_part))
        except ValueError:
            pass
    next_run = max(run_nums, default=0) + 1 if args.no_resume else max(run_nums, default=1)

    if args.no_resume:
        output_dir = base_output_dir / f"run{next_run}{suffix}"
    else:
        output_dir = base_output_dir / f"run{max(run_nums, default=1)}{suffix}"

    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"
    traj_dir = str(output_dir / "trajectories")

    logger.info("Output dir: %s", output_dir)

    # Clean up when --no-resume
    if args.no_resume:
        if results_file.exists():
            results_file.unlink()
        traj_path = Path(traj_dir)
        if traj_path.exists():
            shutil.rmtree(traj_path)
        logger.info("Cleaned old results (--no-resume)")

    tasks = load_benchmark()
    if args.start_index > 0:
        tasks = [t for t in tasks if t["index"] >= args.start_index]
    if args.max_cases:
        tasks = tasks[:args.max_cases]

    # Resume support
    completed_ids = set()
    if not args.no_resume and results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                if line.strip():
                    completed_ids.add(json.loads(line)["task_id"])
        logger.info("Resuming: %d done", len(completed_ids))

    pending = [t for t in tasks if t["id"] not in completed_ids]
    if not pending:
        logger.info("All done!")
        return

    ports = list(range(args.base_port, args.base_port + args.workers))
    # All workers share same port (SGLang handles batching internally)
    ports = [args.base_port] * args.workers
    logger.info("Running %d tasks with %d workers (ports %s)", len(pending), args.workers, ports)

    # Shared strategy memory
    memory_dir = str(output_dir / "memory_data")
    shared_memory = MemoryStore(memory_dir)
    logger.info("Strategy memory initialized: %s", shared_memory.stats())

    write_lock = Lock()
    results = []

    def worker(item):
        idx, task = item
        port = ports[idx % len(ports)]
        url = build_proxy_url(port)
        agent = Agent(
            llm_base_url=url,
            model_name=config.MODEL_NAME,
            max_steps=config.MAX_STEPS,
            memory_store=shared_memory,
        )
        record = run_single(agent, task, traj_dir)
        with write_lock:
            results.append(record)
            with open(results_file, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("[%d/%d] Done %s in %.1fs steps=%d",
                    len(results), len(pending), task["id"], record["elapsed_seconds"], record["steps"])
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, (i, t)): t for i, t in enumerate(pending)}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error("Worker error: %s", e)

    # Export results
    all_results = []
    with open(results_file, "r") as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))
    all_results.sort(key=lambda r: r["index"])

    preds = {r["index"]: r["pred"] for r in all_results}

    # Write CSV in benchmark.csv format (problem, image, answer)
    csv_path = output_dir / f"group_{args.group_id}.csv"
    with open(BENCHMARK_CSV, "r", encoding="utf-8") as fin, \
         open(csv_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["problem", "image", "answer"])
        writer.writeheader()
        for idx, row in enumerate(reader):
            writer.writerow({
                "problem": row["problem"],
                "image": row.get("image", ""),
                "answer": preds.get(idx, ""),
            })

    # Write trajectory JSON
    json_path = output_dir / f"group_{args.group_id}.json"
    all_traj = []
    for r in all_results:
        traj_file = Path(traj_dir) / f"{r['task_id']}.jsonl"
        entries = []
        if traj_file.exists():
            with open(traj_file) as f:
                entries = [json.loads(l) for l in f if l.strip()]
        all_traj.append({"task_id": r["task_id"], "index": r["index"], "trajectory": entries})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_traj, f, ensure_ascii=False, indent=2)

    # Create zip
    zip_path = output_dir / f"group_{args.group_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, f"group_{args.group_id}.csv")
        zf.write(json_path, f"group_{args.group_id}.json")

    # Copy latest to base dir for easy submission
    shutil.copy2(csv_path, base_output_dir / f"group_{args.group_id}.csv")
    shutil.copy2(json_path, base_output_dir / f"group_{args.group_id}.json")
    shutil.copy2(zip_path, base_output_dir / f"group_{args.group_id}.zip")

    print(f"\nRun dir: {output_dir}")
    print(f"Exported: {csv_path}")
    print(f"Exported: {json_path}")
    print(f"Exported: {zip_path}")
    print(f"Latest submission: {base_output_dir / f'group_{args.group_id}.zip'}")


if __name__ == "__main__":
    main()
