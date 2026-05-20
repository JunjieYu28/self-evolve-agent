#!/usr/bin/env python3
"""从 SimpleVQA test 集抽取 N 条，生成「无答案题库」与「独立标答」两份 JSONL。

用法:
  python scripts/make_simplevqa_split.py --n 50 --seed 42
  python scripts/make_simplevqa_split.py --n 50 --out-prefix simpleVQA_50

输出:
  data/{prefix}_questions.jsonl  — 仅题目+图片路径（评测用，模型不可见答案）
  data/{prefix}_answers.jsonl    — index/data_id + answer（事后打分用）
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
SIMPLEVQA_TEST = DATA_ROOT / "SimpleVQA" / "test.jsonl"
IMAGES_DIR = DATA_ROOT / "SimpleVQA" / "images"


def resolve_image(data_id) -> str | None:
    for ext in (".webp", ".png", ".jpg", ".jpeg"):
        p = IMAGES_DIR / f"{data_id}{ext}"
        if p.is_file():
            return str(p)
    return None


def load_test_rows() -> list[dict]:
    rows: list[dict] = []
    with SIMPLEVQA_TEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="抽取 SimpleVQA 题库/标答分离 JSONL")
    parser.add_argument("--n", type=int, default=50, help="抽取条数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=str, default="simpleVQA_50")
    parser.add_argument(
        "--source",
        type=Path,
        default=SIMPLEVQA_TEST,
        help="源 JSONL（默认 data/SimpleVQA/test.jsonl）",
    )
    args = parser.parse_args()

    rows = []
    with args.source.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    eligible: list[dict] = []
    for r in rows:
        data_id = r.get("data_id")
        img = resolve_image(data_id)
        if img and (r.get("question") or "").strip() and (r.get("answer") or "").strip():
            eligible.append({**r, "_image_path": img})

    if len(eligible) < args.n:
        raise SystemExit(f"可用样本仅 {len(eligible)} 条，少于 --n {args.n}")

    rng = random.Random(args.seed)
    picked = rng.sample(eligible, args.n)
    picked.sort(key=lambda x: x["data_id"])

    q_path = DATA_ROOT / f"{args.out_prefix}_questions.jsonl"
    a_path = DATA_ROOT / f"{args.out_prefix}_answers.jsonl"

    with q_path.open("w", encoding="utf-8") as fq, a_path.open(
        "w", encoding="utf-8"
    ) as fa:
        for i, r in enumerate(picked):
            data_id = r["data_id"]
            q_rec = {
                "index": i,
                "data_id": data_id,
                "question": r["question"].strip(),
                "image": r["_image_path"],
                "language": r.get("language"),
            }
            a_rec = {
                "index": i,
                "data_id": data_id,
                "answer": r["answer"].strip(),
            }
            fq.write(json.dumps(q_rec, ensure_ascii=False) + "\n")
            fa.write(json.dumps(a_rec, ensure_ascii=False) + "\n")

    print(f"源文件: {args.source} ({len(rows)} 条)")
    print(f"抽取: {args.n} 条 (seed={args.seed})")
    print(f"题库（无答案）: {q_path}")
    print(f"标答:           {a_path}")
    print()
    print("评测示例（模型看不到答案）:")
    print(
        f"  python scripts/eval_benchmark.py --jsonl {q_path} "
        f"--dataset simplevqa --mode full --tools search --workers 4"
    )
    print("全部跑完后打分:")
    print(
        f"  python scripts/score_simplevqa_predictions.py "
        f"--pred logs/eval/<run>/results.jsonl --gold {a_path}"
    )


if __name__ == "__main__":
    main()
