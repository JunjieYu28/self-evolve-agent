#!/usr/bin/env python3
"""从 ModelScope 下载 Qwen3.5-9B 到指定目录。"""
from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen/Qwen3.5-9B from ModelScope")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "ckpt" / "Qwen3.5-9B",
        help="模型保存目录",
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading qwen/Qwen3.5-9B -> {cache_dir}")
    model_dir = snapshot_download("qwen/Qwen3.5-9B", cache_dir=str(cache_dir))
    print(f"Done: {model_dir}")


if __name__ == "__main__":
    main()
