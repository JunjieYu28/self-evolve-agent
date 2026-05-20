#!/usr/bin/env python3
"""从 ModelScope 下载 BGE Reranker 到 data 盘（供 search_text Cross-Encoder 使用）。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "ckpt" / "bge-reranker-v2-m3"
MODEL_ID = "AI-ModelScope/bge-reranker-v2-m3"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download bge-reranker-v2-m3 via ModelScope")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"下载目录，默认 {DEFAULT_CACHE}",
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MODELSCOPE_CACHE", str(PROJECT_ROOT / ".cache" / "modelscope"))

    from modelscope import snapshot_download

    print(f"Downloading {MODEL_ID} -> {cache_dir}")
    model_dir = snapshot_download(MODEL_ID, cache_dir=str(cache_dir))
    print(f"模型已下载至: {model_dir}")

    model_path = Path(model_dir)
    weights = list(model_path.rglob("model.safetensors")) + list(
        model_path.rglob("pytorch_model.bin")
    )
    if not weights:
        print("警告: 未找到 model.safetensors / pytorch_model.bin，请检查目录是否完整", file=sys.stderr)
        return 1
    for w in weights:
        print(f"  权重文件: {w} ({w.stat().st_size / (1024**3):.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
