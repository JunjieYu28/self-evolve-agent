"""下载 SimpleVQA 数据集到 data/SimpleVQA/。"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

from _bootstrap import project_root

os.chdir(project_root())
sys.path.insert(0, str(project_root()))

from datasets import load_dataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402


def main() -> None:
    out_dir = Path("data/SimpleVQA")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1/3 下载 HF 原始文件 ...")
    snapshot_download(
        repo_id="m-a-p/SimpleVQA",
        repo_type="dataset",
        local_dir=str(out_dir / "raw"),
    )

    print("2/3 保存 datasets 格式 ...")
    ds = load_dataset("m-a-p/SimpleVQA")
    ds.save_to_disk(str(out_dir / "hf_dataset"))

    print("3/3 导出 JSONL + 图片 ...")
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / "test.jsonl"

    split = ds["test"]
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in split:
            record = dict(row)
            b64 = record.pop("image", "")
            image_path = images_dir / f"{row['data_id']}.webp"
            if b64 and not image_path.exists():
                image_path.write_bytes(base64.b64decode(b64))
            record["image_path"] = str(image_path.relative_to(out_dir)).replace(
                "\\", "/"
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"完成: {len(split)} 条 -> {out_dir}")


if __name__ == "__main__":
    main()
