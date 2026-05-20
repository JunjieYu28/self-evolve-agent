#!/usr/bin/env python3
"""SimpleVQA 分离评测：用标答文件对 results.jsonl 统一计算准确率（事后打分）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_benchmark import extract_short_answer, is_correct  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleVQA 预测 vs 独立标答打分")
    parser.add_argument("--pred", type=Path, required=True, help="results.jsonl（含 pred）")
    parser.add_argument("--gold", type=Path, required=True, help="*_answers.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="可选：写出 scored.jsonl")
    args = parser.parse_args()

    gold_by_index: dict[int, str] = {}
    gold_by_id: dict = {}
    for row in load_jsonl(args.gold):
        ans = (row.get("answer") or "").strip()
        if "index" in row:
            gold_by_index[int(row["index"])] = ans
        if "data_id" in row:
            gold_by_id[row["data_id"]] = ans

    preds = load_jsonl(args.pred)
    correct = 0
    scored = 0
    details: list[dict] = []

    for row in preds:
        idx = int(row.get("index", -1))
        data_id = row.get("data_id")
        gold = gold_by_index.get(idx, "")
        if not gold and data_id is not None:
            gold = gold_by_id.get(data_id, "")
        if not gold and "answer" in row:
            gold = (row.get("answer") or "").strip()

        raw_pred = row.get("pred") or row.get("prediction") or ""
        pred = extract_short_answer(str(raw_pred))

        if not gold:
            details.append(
                {
                    "index": idx,
                    "data_id": data_id,
                    "gold": "",
                    "pred": pred,
                    "ok": None,
                    "scored": False,
                }
            )
            continue

        scored += 1
        ok = is_correct(pred, gold)
        if ok:
            correct += 1
        details.append(
            {
                "index": idx,
                "data_id": data_id,
                "gold": gold,
                "pred": pred,
                "ok": ok,
                "scored": True,
            }
        )

    acc = (correct / scored * 100) if scored else 0.0
    print(f"预测条数: {len(preds)}")
    print(f"可打分:   {scored}")
    print(f"正确:     {correct}")
    print(f"准确率:   {acc:.2f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"明细:     {args.out}")


if __name__ == "__main__":
    main()
