"""下载 2WikiMultihopQA 数据集到 data/2WikiMultihopQA/。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from _bootstrap import project_root

os.chdir(project_root())
sys.path.insert(0, str(project_root()))

from datasets import load_dataset  # noqa: E402


def main() -> None:
    out_dir = Path("data/2WikiMultihopQA")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("下载 framolfese/2WikiMultihopQA ...")
    ds = load_dataset("framolfese/2WikiMultihopQA")

    disk_path = out_dir / "hf_dataset"
    ds.save_to_disk(str(disk_path))

    jsonl_dir = out_dir / "jsonl"
    jsonl_dir.mkdir(exist_ok=True)
    for split in ds:
        path = jsonl_dir / f"{split}.jsonl"
        print(f"  {split}: {len(ds[split])} -> {path}")
        ds[split].to_json(str(path), force_ascii=False)

    print("完成。")


if __name__ == "__main__":
    main()
