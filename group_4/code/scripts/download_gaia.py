#!/usr/bin/env python3
"""下载 GAIA 数据集到本地（需 HF 账号授权 + Token）。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "gaia"
DEFAULT_HF_HOME = ROOT / "huggingface_cache"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download gaia-benchmark/GAIA")
    parser.add_argument(
        "--config",
        default="2023_all",
        help="数据集配置名，如 2023_all、2023_level1、2023_level2、2023_level3",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"保存目录（save_to_disk），默认 {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help=f"HF 缓存根目录，默认 {DEFAULT_HF_HOME}",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="仅 load_dataset，不 save_to_disk（数据在 HF 缓存中）",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "错误：未设置 HF_TOKEN。\n"
            "1. 登录 https://huggingface.co/datasets/gaia-benchmark/GAIA 并同意条款\n"
            "2. 创建 Token：https://huggingface.co/settings/tokens\n"
            "3. export HF_TOKEN=hf_xxx\n"
            "4. 重新运行本脚本",
            file=sys.stderr,
        )
        return 1

    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.hf_home / "datasets"))
    # 使用官方 Hub；镜像站对 gated 数据集常不可用
    os.environ.pop("HF_ENDPOINT", None)

    from datasets import load_dataset

    print(f"Downloading gaia-benchmark/GAIA ({args.config})...")
    ds = load_dataset("gaia-benchmark/GAIA", args.config)
    print(ds)

    if not args.no_save:
        args.out.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(args.out))
        print(f"Saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
