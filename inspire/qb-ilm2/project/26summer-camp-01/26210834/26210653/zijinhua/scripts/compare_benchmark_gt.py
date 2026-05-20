#!/usr/bin/env python3
"""对比 benchmark 预测与 ground_truth.json（支持 checkpoint / json 提交物）。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from _bootstrap import project_root

ROOT = project_root()


def load_gt(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def preds_from_checkpoint(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for it in data.get("items") or []:
        if "index" in it:
            out[int(it["index"])] = str(it.get("answer") or "")
    return out


def preds_from_log(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"完成 index=(\d+).*pred='(.+)'", line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^the (character|podcast|book) is\s*", "", s)
    s = re.sub(r"[*_]", "", s)
    s = re.sub(r"[^\w\s%,.\-$]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match(pred: str, truth: str) -> bool:
    p, t = norm(pred), norm(truth)
    if not t:
        return False
    if p == t or t in p or p in t:
        return True
    pt, tt = set(p.split()), set(t.split())
    if tt and len(pt & tt) / len(tt) >= 0.8:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=ROOT / "ground_truth.json")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    if args.checkpoint:
        preds = preds_from_checkpoint(args.checkpoint)
    elif args.log:
        preds = preds_from_log(args.log)
    else:
        print("请指定 --checkpoint 或 --log", file=sys.stderr)
        sys.exit(1)

    if not preds:
        print("无预测结果")
        return

    hits = []
    misses = []
    for idx in sorted(preds):
        q = str(idx + 1)
        truth = gt.get(q, "?")
        pred = preds[idx]
        if match(pred, truth):
            hits.append((idx + 1, truth, pred))
        else:
            misses.append((idx + 1, truth, pred))

    n = len(preds)
    print(f"\n=== {args.label} ===")
    print(f"已完成: {n} 题 | 命中: {len(hits)}/{n} ({100*len(hits)/n:.1f}%)")
    if hits:
        print("\n命中:")
        for q, t, p in hits:
            print(f"  Q{q}: {t}")
    if misses:
        print("\n未命中:")
        for q, t, p in misses[:30]:
            print(f"  Q{q}: GT={t!r}  pred={p[:60]!r}")
        if len(misses) > 30:
            print(f"  ... 另有 {len(misses)-30} 题")


if __name__ == "__main__":
    main()
