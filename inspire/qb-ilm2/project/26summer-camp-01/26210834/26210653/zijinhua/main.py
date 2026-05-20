"""
项目统一入口。

用法:
  python main.py integration   # 全链路 Mock 联调（失败→反思→记忆→成功）
  python main.py agent         # 真实 Qwen + 生产工具
  python main.py test-llm      # 测试 Harness 基座 LLM
  python main.py test-reflection  # 测试反思模型（默认 Qwen3.5-9B）
  python main.py test-vision      # 测试 9B 多模态看图
  bash scripts/start_vllm.sh   # 启动本地 vLLM（需先执行）
  python main.py test-search   # 测试 search-proxy 搜索
  python main.py download-2wiki
  python main.py download-vqa
  python main.py eval --dataset 2wiki --limit 10 --workers 8  # 并发评测
  python main.py eval --jsonl data/simpleVQA_99.jsonl --mode full --tools search  # 反思+记忆+搜索
  python main.py benchmark --group 004 --workers 8  # 打榜 benchmark（可并发）
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

COMMANDS = {
    "integration": "integration_test.py",
    "agent": "run_agent.py",
    "test-llm": "test_llm.py",
    "test-reflection": "test_reflection.py",
    "test-vision": "test_vision.py",
    "test-search": "test_search.py",
    "download-2wiki": "download_2wiki.py",
    "download-vqa": "download_simplevqa.py",
    "eval": "eval_benchmark.py",
    "benchmark": "run_benchmark.py",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("可用命令:", ", ".join(COMMANDS))
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)

    cmd = sys.argv[1]
    script = COMMANDS.get(cmd)
    if not script:
        print(f"未知命令: {cmd}")
        print("可用:", ", ".join(COMMANDS))
        sys.exit(1)

    path = SCRIPTS / script
    raise SystemExit(
        subprocess.call([sys.executable, str(path), *sys.argv[2:]], cwd=ROOT)
    )


if __name__ == "__main__":
    main()
