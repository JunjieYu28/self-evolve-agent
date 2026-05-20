"""测试 search-proxy / 直连搜索。"""

from __future__ import annotations

import os
import sys

from _bootstrap import project_root

ROOT = project_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from config import load_dotenv  # noqa: E402
from web_search_tool import WebSearchTool  # noqa: E402


def main() -> None:
    load_dotenv()
    tool = WebSearchTool()
    print(f"模式: {'proxy' if tool._use_proxy() else 'direct'}")
    print(f"URL: {tool.search_proxy_url or '(直连 Serper)'}")

    if tool._use_proxy():
        try:
            health = tool.health_check()
            print("health:", health)
        except Exception as exc:
            print("health 失败:", exc)
            return

    print("\n--- 文搜测试 ---")
    print(tool.execute(query="ReAct agent", top_k=2, fetch=False)[:600])


if __name__ == "__main__":
    main()
