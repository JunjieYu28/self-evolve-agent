"""Quick test on specific indices to validate architecture changes."""
import json
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

import config
from agent import Agent
from modules.memory import MemoryStore

BENCHMARK_PATH = Path("/inspire/qb-ilm2/project/26summer-camp-01/public/benchmark.csv")
TEST_INDICES = [13, 15, 17, 20, 21, 26, 34, 37, 39, 40]


def load_tasks():
    import csv
    csv.field_size_limit(10 * 1024 * 1024)
    tasks = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i in TEST_INDICES:
                task = {
                    "id": str(i),
                    "instruction": row.get("problem", row.get("question", "")),
                    "ground_truth": row.get("ground_truth", row.get("answer", "")),
                }
                if row.get("image_url"):
                    task["image_url"] = row["image_url"]
                tasks.append(task)
    return tasks


def run_one(task, worker_id):
    memory = MemoryStore()
    agent = Agent(memory_store=memory)
    start = time.time()
    result = agent.run_task(task, trajectory_dir="results/quick_test/trajectories")
    elapsed = time.time() - start
    result["ground_truth"] = task.get("ground_truth", "")
    result["elapsed"] = round(elapsed, 1)
    correct = result["ground_truth"].lower().strip() in result["answer"].lower().strip() or \
              result["answer"].lower().strip() in result["ground_truth"].lower().strip()
    result["correct"] = correct
    return result


def main():
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks: indices {[t['id'] for t in tasks]}")

    output_dir = Path("results/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    correct_count = 0

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(run_one, task, i): task
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "CORRECT" if result["correct"] else "WRONG"
                if result["correct"]:
                    correct_count += 1
                print(f"[{status}] Q{result['task_id']}: "
                      f"answer='{result['answer'][:50]}' "
                      f"truth='{result['ground_truth'][:50]}' "
                      f"steps={result['steps']} time={result['elapsed']}s")
            except Exception as exc:
                print(f"[ERROR] Q{task['id']}: {exc}")

    print(f"\n{'='*60}")
    print(f"Results: {correct_count}/{len(results)} correct ({100*correct_count/max(len(results),1):.0f}%)")
    print(f"{'='*60}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
