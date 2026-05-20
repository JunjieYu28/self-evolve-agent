"""从环境变量加载 LLM / 服务配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_EAS_API_URL = (
    "http://gw-bzokqkvr2cblz8ok6y.cn-wulanchabu-acdr-1.pai-eas.aliyuncs.com"
    "/api/predict/qwen_35_9b"
)
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_REFLECTION_LLM_BASE_URL = "http://127.0.0.1:8004/v1"
DEFAULT_SEARCH_PROXY_URL = "http://127.0.0.1:8090"
DEFAULT_SANDBOX_BASE_URL = "http://127.0.0.1:8080"


@dataclass(frozen=True)
class EASConfig:
    api_url: str
    api_token: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: float
    enable_thinking: bool

    @classmethod
    def from_env(cls) -> EASConfig:
        api_url = os.getenv("EAS_API_URL", DEFAULT_EAS_API_URL).strip()
        api_token = os.getenv("EAS_API_TOKEN", "").strip()
        if not api_token:
            raise ValueError(
                "未设置 EAS_API_TOKEN。请在 .env 中配置 PAI-EAS 调用 Token。"
            )
        return cls(
            api_url=api_url,
            api_token=api_token,
            model_name=os.getenv("EAS_MODEL_NAME", "qwen_35_9b").strip(),
            temperature=float(os.getenv("EAS_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("EAS_MAX_TOKENS", "2048")),
            timeout=float(os.getenv("EAS_TIMEOUT", "120")),
            enable_thinking=os.getenv("EAS_ENABLE_THINKING", "false").lower()
            in ("1", "true", "yes"),
        )


@dataclass(frozen=True)
class SGLangConfig:
    """本地 SGLang / vLLM OpenAI 兼容端点（与 harness task_runner 一致）。"""

    base_url: str
    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: float
    enable_thinking: bool

    @classmethod
    def from_env(cls) -> SGLangConfig:
        return cls(
            base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).strip(),
            api_key=os.getenv("LLM_API_KEY", "EMPTY").strip() or "EMPTY",
            model_name=os.getenv("MODEL_NAME", "qwen-3.5").strip(),
            temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "16000")),
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            enable_thinking=os.getenv("LLM_ENABLE_THINKING", "true").lower()
            in ("1", "true", "yes"),
        )


@dataclass(frozen=True)
class ReflectionLLMConfig:
    """
    反思 / 记忆组织用模型（默认专用 Qwen3.5-9B @ :8004，与 Agent :8001 分离）。
    """

    base_url: str
    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: float
    enable_thinking: bool

    @classmethod
    def from_env(cls) -> ReflectionLLMConfig:
        return cls(
            base_url=os.getenv(
                "REFLECTION_LLM_BASE_URL", DEFAULT_REFLECTION_LLM_BASE_URL
            ).strip(),
            api_key=os.getenv("REFLECTION_LLM_API_KEY", "EMPTY").strip() or "EMPTY",
            model_name=os.getenv("REFLECTION_MODEL_NAME", "qwen-3.5").strip(),
            temperature=float(os.getenv("REFLECTION_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("REFLECTION_MAX_TOKENS", "2048")),
            timeout=float(os.getenv("REFLECTION_LLM_TIMEOUT", "300")),
            enable_thinking=os.getenv("REFLECTION_ENABLE_THINKING", "false").lower()
            in ("1", "true", "yes"),
        )


@dataclass(frozen=True)
class LLMConfig:
    """统一 LLM 配置：backend=eas | sglang。"""

    backend: str
    eas: EASConfig | None = None
    sglang: SGLangConfig | None = None
    reflection: ReflectionLLMConfig | None = None

    @classmethod
    def from_env(cls) -> LLMConfig:
        backend = os.getenv("LLM_BACKEND", "eas").strip().lower()
        reflection = ReflectionLLMConfig.from_env()
        if backend in ("sglang", "vllm", "local", "openai"):
            return cls(
                backend="sglang",
                sglang=SGLangConfig.from_env(),
                reflection=reflection,
            )
        if backend not in ("eas", "pai", "cloud"):
            raise ValueError(
                f"未知 LLM_BACKEND={backend!r}，请使用 eas / sglang / vllm"
            )
        return cls(backend="eas", eas=EASConfig.from_env(), reflection=reflection)


def load_dotenv(path: str | Path | None = None) -> None:
    """轻量 .env 加载（无第三方依赖）。"""
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value
