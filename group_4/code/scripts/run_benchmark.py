"""
打榜 benchmark.csv：按题序跑 Agent，生成 group_{组号}.json / .csv / .zip。

用法:
  python scripts/run_benchmark.py --group 004
  python scripts/run_benchmark.py --group 004 --mode full --tools search
  python scripts/run_benchmark.py --group 004 --limit 5
  python scripts/run_benchmark.py --group 004 --resume
  python scripts/run_benchmark.py --group 004 --workers 8   # 并发（需 vLLM max-num-seqs>=8）

输出（项目根目录）:
  group_004.json   # 每题轨迹 + 答案
  group_004.csv    # 原 benchmark 结构，answer 列填入预测
  group_004.zip    # 上述两文件打包

中间日志: logs/benchmark/group_004/
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from _bootstrap import project_root

ROOT = project_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from config import load_dotenv  # noqa: E402
from logger import AgentLogger  # noqa: E402
from memory import MemoryManager  # noqa: E402

from agent import OSINT_SEARCH_WORKFLOW  # noqa: E402
from scripts.eval_benchmark import run_one_sample  # noqa: E402

BENCHMARK_CSV = ROOT / "benchmark.csv"
WORK_LOG_ROOT = ROOT / "logs" / "benchmark"

BENCHMARK_SYSTEM = (
    "你是多模态 OSINT 问答 Agent，可使用 search_text 检索事实（勿用浏览器）。\n"
    "结合题目、图片（若有）与检索结果作答；只输出最终答案本身，不要解释、"
    "不要 Markdown、不要长段落。\n"
    + OSINT_SEARCH_WORKFLOW
)

BENCHMARK_FULL_SYSTEM = (
    "你是会自我进化的多模态 OSINT 问答 Agent，可使用 search_text 检索事实。\n"
    "结合题目、图片、检索与系统记忆作答；只输出最短最终答案。\n"
    "若收到 <memory-context> 或 <correction-hint>，优先参考【成功经验】与【策略】，"
    "但 search_text 的 query 仍必须遵守下方短查询法则。\n"
    + OSINT_SEARCH_WORKFLOW
)


def normalize_group_id(raw: str) -> str:
    """004 / 4 -> 004（三位组号，用于文件名）。"""
    digits = re.sub(r"\D", "", raw.strip()) or "0"
    return digits.zfill(3)


def load_benchmark_rows(csv_path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def decode_benchmark_image(b64_data: str, dest: Path) -> str:
    raw = base64.b64decode(b64_data.strip())
    dest.parent.mkdir(parents=True, exist_ok=True)
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        dest = dest.with_suffix(".png")
    elif raw[:3] == b"\xff\xd8\xff":
        dest = dest.with_suffix(".jpg")
    else:
        dest = dest.with_suffix(".png")
    dest.write_bytes(raw)
    return str(dest)


def prepare_samples(
    rows: list[dict[str, str]],
    *,
    group_id: str,
    offset: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    cache_dir = WORK_LOG_ROOT / f"group_{group_id}" / "images"
    samples: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i < offset:
            continue
        if limit is not None and len(samples) >= limit:
            break
        problem = (row.get("problem") or "").strip()
        image_b64 = (row.get("image") or "").strip()
        image_path: str | None = None
        if image_b64:
            image_path = decode_benchmark_image(
                image_b64, cache_dir / f"{i:03d}"
            )
        samples.append(
            {
                "index": i,
                "instruction": problem,
                "answer": (row.get("answer") or "").strip(),
                "image": image_path,
                "raw_row": row,
            }
        )
    return samples


def pick_benchmark_system_prompt(mode: str, tools: str) -> str:
    if mode == "full":
        return BENCHMARK_FULL_SYSTEM
    if tools == "none":
        return (
            "你是多模态问答助手。结合题目与图片（若有）作答；"
            "只输出最终答案本身，不要解释。"
        )
    return BENCHMARK_SYSTEM


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    return {int(it["index"]): it for it in items if "index" in it}


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_submission_csv(
    rows: list[dict[str, str]],
    predictions: dict[int, str],
    out_path: Path,
) -> None:
    csv.field_size_limit(sys.maxsize)
    fieldnames = list(rows[0].keys()) if rows else ["problem", "image", "answer"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            if i in predictions:
                out["answer"] = predictions[i]
            writer.writerow(out)


def write_zip(json_path: Path, csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname=json_path.name)
        zf.write(csv_path, arcname=csv_path.name)


def _process_one_sample(
    sample: dict[str, Any],
    *,
    mode: str,
    log_dir: Path,
    max_steps: int,
    memory_path: Path | None,
    tools: str,
    system_prompt: str,
    file_lock: threading.Lock,
    memory_manager: MemoryManager | None,
) -> dict[str, Any]:
    idx = sample["index"]
    r = run_one_sample(
        sample,
        mode=mode,
        log_dir=log_dir,
        max_steps=max_steps,
        memory_path=memory_path,
        tools=tools,
        system_prompt=system_prompt,
        file_lock=file_lock,
        memory_manager=memory_manager,
    )
    traj = AgentLogger(log_dir=log_dir, task_index=idx).read_trajectory(idx)
    return {
        "index": idx,
        "problem": sample["instruction"],
        "has_image": bool(sample.get("image")),
        "answer": r.pred,
        "raw_answer": r.pred,
        "error": r.error,
        "elapsed_sec": round(r.elapsed_sec, 2),
        "trajectory": traj,
    }


def _checkpoint_payload(
    group_id: str,
    mode: str,
    tools: str,
    workers: int,
    done_items: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "group": group_id,
        "benchmark_csv": str(BENCHMARK_CSV),
        "mode": mode,
        "tools": tools,
        "workers": workers,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items": [done_items[i] for i in sorted(done_items)],
    }


def configure_benchmark_runtime(*, fast_search: bool) -> None:
    """打榜加速：缩短搜索超时、默认不抓全文（避免 8 worker 打满 search-proxy）。"""
    if fast_search:
        # 打榜进程内强制覆盖 .env 中的长超时，避免 8 路并发堆满 proxy
        os.environ["SEARCH_PROXY_TIMEOUT"] = os.getenv(
            "BENCHMARK_SEARCH_PROXY_TIMEOUT", "45"
        )
        os.environ["SEARCH_FETCH_DEFAULT"] = "false"


def run_benchmark(args: argparse.Namespace) -> None:
    load_dotenv()
    for _k in ("all_proxy", "ALL_PROXY"):
        os.environ.pop(_k, None)
    configure_benchmark_runtime(fast_search=not args.no_fast_search)

    group_id = normalize_group_id(args.group)
    prefix = f"group_{group_id}"
    json_out = ROOT / f"{prefix}.json"
    csv_out = ROOT / f"{prefix}.csv"
    zip_out = ROOT / f"{prefix}.zip"

    if not BENCHMARK_CSV.is_file():
        raise FileNotFoundError(f"找不到打榜数据: {BENCHMARK_CSV}")

    all_rows = load_benchmark_rows(BENCHMARK_CSV)

    log_dir = WORK_LOG_ROOT / prefix
    if not args.resume and log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 须在清空 log_dir 之后再解码图片，否则 rmtree 会删掉刚写入的 images/
    samples = prepare_samples(
        all_rows, group_id=group_id, offset=args.offset, limit=args.limit
    )
    if not samples:
        print("没有样本可运行。")
        return

    checkpoint_path = log_dir / "checkpoint.json"
    done_items = load_checkpoint(checkpoint_path) if args.resume else {}
    predictions: dict[int, str] = {
        idx: str(it.get("answer") or "") for idx, it in done_items.items()
    }

    memory_path = log_dir / "agent_memory.json" if args.mode == "full" else None
    system_prompt = pick_benchmark_system_prompt(args.mode, args.tools)
    file_lock = threading.Lock()
    memory_lock = threading.RLock()
    shared_memory = (
        MemoryManager(memory_path, lock=memory_lock)
        if args.mode == "full" and memory_path
        else None
    )

    if args.tools != "none" and args.max_steps <= 2:
        args.max_steps = int(os.getenv("EVAL_MAX_STEPS_WITH_TOOLS", "8"))

    pending = [s for s in samples if s["index"] not in done_items]
    total = len(samples)
    workers = max(1, args.workers)
    checkpoint_lock = threading.Lock()

    print(f"组号: {group_id} -> {prefix}.*")
    print(f"数据: {BENCHMARK_CSV} ({len(all_rows)} 题)")
    print(f"本次运行: {len(pending)}/{total} 题 (resume={args.resume})")
    print(f"模式: {args.mode}, 工具: {args.tools}, max_steps: {args.max_steps}")
    print(f"并发: {workers} workers")
    if args.mode == "full":
        print(f"记忆库: {memory_path}")
        if workers > 1:
            print("注意: full 多 worker 共享记忆库（已加锁），9B 反思可能并发排队")
    if workers > 1:
        print(
            "提示: 9B vLLM 需 max-num-seqs >= workers；"
            "可重启: VLLM_MAX_NUM_SEQS=8 bash scripts/restart_vllm.sh"
        )
    if not args.no_fast_search:
        print(
            "搜索加速: fetch=false, SEARCH_PROXY_TIMEOUT="
            f"{os.getenv('SEARCH_PROXY_TIMEOUT', '45')}s"
        )
    print(f"日志: {log_dir}")
    print(f"提交物: {json_out.name}, {csv_out.name}, {zip_out.name}")
    print("-" * 60)

    t0 = time.time()
    finished = len(done_items)

    def _on_item_done(item: dict[str, Any]) -> None:
        nonlocal finished
        idx = item["index"]
        with checkpoint_lock:
            done_items[idx] = item
            predictions[idx] = str(item.get("answer") or "")
            finished = len(done_items)
            save_checkpoint(
                checkpoint_path,
                _checkpoint_payload(
                    group_id, args.mode, args.tools, workers, done_items
                ),
            )
        err = f" err={item['error']}" if item.get("error") else ""
        line = (
            f"[{finished}/{total}] 完成 index={idx} "
            f"({item['elapsed_sec']:.0f}s) pred={item['answer']!r}{err}"
        )
        print(line, flush=True)
        log_path = WORK_LOG_ROOT / "runs" / f"group_{group_id}_run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    if workers == 1:
        for sample in pending:
            print(f"[开始] index={sample['index']} ...", flush=True)
            item = _process_one_sample(
                sample,
                mode=args.mode,
                log_dir=log_dir,
                max_steps=args.max_steps,
                memory_path=memory_path,
                tools=args.tools,
                system_prompt=system_prompt,
                file_lock=file_lock,
                memory_manager=shared_memory,
            )
            _on_item_done(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_one_sample,
                    sample,
                    mode=args.mode,
                    log_dir=log_dir,
                    max_steps=args.max_steps,
                    memory_path=memory_path,
                    tools=args.tools,
                    system_prompt=system_prompt,
                    file_lock=file_lock,
                    memory_manager=shared_memory,
                ): sample
                for sample in pending
            }
            for fut in as_completed(futures):
                _on_item_done(fut.result())

    items = [done_items[i] for i in sorted(done_items)]
    submission = {
        "group": group_id,
        "benchmark_csv": str(BENCHMARK_CSV),
        "mode": args.mode,
        "tools": args.tools,
        "workers": workers,
        "n_questions": len(all_rows),
        "n_run": len(items),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "items": items,
    }
    json_out.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_submission_csv(all_rows, predictions, csv_out)
    write_zip(json_out, csv_out, zip_out)

    with_img = sum(1 for s in samples if s.get("image"))
    print("-" * 60)
    print(f"完成 {len(items)} 题, 带图 {with_img}/{len(samples)}, 耗时 {time.time() - t0:.0f}s")
    print(f"轨迹 -> {json_out}")
    print(f"答案 -> {csv_out}")
    print(f"压缩包 -> {zip_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="打榜 benchmark.csv 生成提交文件")
    parser.add_argument(
        "--group",
        default="004",
        help="组号，如 004（默认 004）",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "full"),
        default=os.getenv("BENCHMARK_MODE", "full"),
        help="baseline=无反思记忆; full=反思+记忆（默认 full）",
    )
    parser.add_argument(
        "--tools",
        choices=("none", "search", "all", "full"),
        default=os.getenv("BENCHMARK_TOOLS", "search"),
        help="none=纯模型; search=仅搜索(默认); all=搜索+浏览器",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 题（调试）")
    parser.add_argument("--offset", type=int, default=0, help="起始题号偏移")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2,
        dest="max_steps",
        help="ReAct 最大步数；有工具时自动升为 8",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 checkpoint 续跑（logs/benchmark/group_XXX/checkpoint.json）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("BENCHMARK_WORKERS", "4")),
        help="并发 worker 数（默认 4；需 vLLM max-num-seqs 足够）",
    )
    parser.add_argument(
        "--no-fast-search",
        action="store_true",
        help="关闭打榜搜索加速（fetch 全文 + 长超时，8 并发易打满 proxy）",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
