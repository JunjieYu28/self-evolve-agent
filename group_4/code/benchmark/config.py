"""
Centralized configuration for the Pro Max Agent Harness.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# LLM (9B — main inference)
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/"
    "project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/"
    "user-5b1a894f-4b65-4bb2-afcd-8691a9eec556/"
    "vscode/22680b47-5984-4188-97d7-b37d50c64593/"
    "02f9bedd-7c26-495c-9881-33b246833f17/proxy/8000/v1",
)
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.5-9B")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "10000"))

# ---------------------------------------------------------------------------
# Teacher LLM (32B — planning, verification, reflection)
# ---------------------------------------------------------------------------
TEACHER_ENABLED = os.getenv("TEACHER_ENABLED", "1") == "1"
TEACHER_BASE_URL = os.getenv(
    "TEACHER_BASE_URL",
    "https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/"
    "project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/"
    "user-5b1a894f-4b65-4bb2-afcd-8691a9eec556/"
    "vscode/22680b47-5984-4188-97d7-b37d50c64593/"
    "02f9bedd-7c26-495c-9881-33b246833f17/proxy/8001/v1",
)
TEACHER_MODEL = os.getenv(
    "TEACHER_MODEL",
    "Qwen3-32B",
)
TEACHER_MAX_TOKENS = int(os.getenv("TEACHER_MAX_TOKENS", "4096"))
TEACHER_TIMEOUT = int(os.getenv("TEACHER_TIMEOUT", "60"))

# Which 32B components to enable (each can be toggled independently)
TEACHER_PLANNER_ENABLED = os.getenv("TEACHER_PLANNER_ENABLED", "1") == "1"
TEACHER_VERIFIER_ENABLED = os.getenv("TEACHER_VERIFIER_ENABLED", "0") == "1"
TEACHER_REFLECTOR_ENABLED = os.getenv("TEACHER_REFLECTOR_ENABLED", "1") == "1"
TEACHER_MAX_VERIFY_ATTEMPTS = int(os.getenv("TEACHER_MAX_VERIFY_ATTEMPTS", "2"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_DIR = PROJECT_ROOT / "memory_data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))
MEMORY_MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "1500"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------
REFLECTION_ENABLED = os.getenv("REFLECTION_ENABLED", "1") == "1"
REFLECTION_ON_SUCCESS = os.getenv("REFLECTION_ON_SUCCESS", "1") == "1"

# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------
COMPACTION_TOKEN_THRESHOLD = int(os.getenv("COMPACTION_TOKEN_THRESHOLD", "20000"))
COMPACTION_PRESERVE_RECENT = int(os.getenv("COMPACTION_PRESERVE_RECENT", "8"))

# ---------------------------------------------------------------------------
# Tool Hooks
# ---------------------------------------------------------------------------
TOOL_RETRY_MAX = int(os.getenv("TOOL_RETRY_MAX", "2"))
TOOL_DEDUP_ENABLED = os.getenv("TOOL_DEDUP_ENABLED", "1") == "1"
MAX_TOOL_CALLS_PER_TASK = int(os.getenv("MAX_TOOL_CALLS_PER_TASK", "8"))
MAX_TOOLS_PER_STEP = int(os.getenv("MAX_TOOLS_PER_STEP", "2"))
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "1500"))

# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------
FORCE_ANSWER_ENABLED = os.getenv("FORCE_ANSWER_ENABLED", "1") == "1"
STUCK_THRESHOLD = int(os.getenv("STUCK_THRESHOLD", "3"))
ADAPTIVE_BUDGET = os.getenv("ADAPTIVE_BUDGET", "0") == "1"
