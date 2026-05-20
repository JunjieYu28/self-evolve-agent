"""将项目根目录加入 sys.path，供 scripts/ 下脚本导入核心模块。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def project_root() -> Path:
    return ROOT
