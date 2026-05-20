"""长期记忆模块：沉淀反思教训并在新任务中召回（动态多模态路由）。"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 本地默认模型路径（可通过环境变量覆盖）
_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CLIP_MODEL = os.environ.get("MEMORY_CLIP_MODEL", str(_PROJECT_ROOT / "ckpt" / "clip-vit-base-patch32"))
_DEFAULT_TEXT_MODEL = os.environ.get("MEMORY_TEXT_MODEL", str(_PROJECT_ROOT / "ckpt" / "bge-m3"))
# 图文召回时，无图记忆用语义向量给的底分权重
_IMAGE_ROUTE_TEXT_FALLBACK_WEIGHT = 0.2

# 简易中英文停用词
_STOPWORDS = frozenset(
    """
    的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会
    着 没有 看 好 自己 这 那 他 她 它 我们 他们 什么 怎么 哪些 哪个 请
    the a an is are was were be been being to of in for on with at by from
    and or but not this that it as if when what how which who
    """.split()
)

_TASK_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("图文搜索", ["图", "图片", "image", "photo", "视觉", "看图", "识别"]),
    ("网页浏览", ["url", "http", "网页", "链接", "访问", "browse", "网站"]),
    ("信息提取", ["提取", "找出", "列出", "extract", "获取", "查询"]),
    ("多跳推理", ["比较", "是否", "同一", "都", "both", "compare", "多跳"]),
    ("事实问答", ["什么", "谁", "何时", "哪里", "why", "what", "who", "when"]),
]


class MemoryManager:
    """
    本地 JSON 长期记忆管理器。
    召回：有图任务优先 CLIP 视觉相似度；纯文本任务用 BGE/句向量语义相似度；
    模型加载失败时降级为关键词匹配。
    """

    def __init__(
        self,
        storage_path: str | Path = "agent_memory.json",
        *,
        lock: threading.RLock | None = None,
        clip_model_name: str | None = None,
        text_model_name: str | None = None,
    ) -> None:
        self.storage_path = Path(storage_path)
        self._data: dict[str, Any] = {"memories": [], "next_id": 1}
        self._lock = lock or threading.RLock()
        self._model_lock = threading.RLock()

        self._clip_model_path = (
            clip_model_name
            or os.getenv("MEMORY_CLIP_MODEL", _DEFAULT_CLIP_MODEL)
        ).strip()
        self._text_model_path = (
            text_model_name
            or os.getenv("MEMORY_TEXT_MODEL", _DEFAULT_TEXT_MODEL)
        ).strip()

        self._clip_model: Any = None
        self._clip_processor: Any = None
        self._clip_ready: bool | None = None

        self._text_encoder: Any = None
        self._text_backend: str | None = None  # "sentence_transformers" | "transformers"
        self._text_tokenizer: Any = None
        self._text_ready: bool | None = None

        self._load()

    def _load(self) -> None:
        if not self.storage_path.is_file():
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "memories" in raw:
                self._data = raw
        except (json.JSONDecodeError, OSError):
            self._data = {"memories": [], "next_id": 1}

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _next_task_id(self) -> str:
        tid = f"mem_{self._data.get('next_id', 1):04d}"
        self._data["next_id"] = int(self._data.get("next_id", 1)) + 1
        return tid

    # ------------------------------------------------------------------
    # 模型懒加载
    # ------------------------------------------------------------------

    def _init_clip(self) -> bool:
        if self._clip_ready is not None:
            return self._clip_ready
        with self._model_lock:
            if self._clip_ready is not None:
                return self._clip_ready
            try:
                import torch
                from PIL import Image  # noqa: F401
                from transformers import CLIPModel, CLIPProcessor

                path = self._clip_model_path
                if not Path(path).is_dir():
                    path = "openai/clip-vit-base-patch32"
                self._clip_processor = CLIPProcessor.from_pretrained(
                    path, local_files_only=Path(self._clip_model_path).is_dir()
                )
                self._clip_model = CLIPModel.from_pretrained(
                    path, local_files_only=Path(self._clip_model_path).is_dir()
                )
                self._clip_model.eval()
                self._torch = torch
                self._clip_ready = True
                logger.info("MemoryManager: CLIP loaded from %s", path)
            except Exception as exc:
                self._clip_model = None
                self._clip_processor = None
                self._clip_ready = False
                logger.warning("MemoryManager: CLIP unavailable (%s)", exc)
            return self._clip_ready

    def _init_text_encoder(self) -> bool:
        if self._text_ready is not None:
            return self._text_ready
        with self._model_lock:
            if self._text_ready is not None:
                return self._text_ready

            path = self._text_model_path
            local = Path(path).is_dir()

            try:
                from sentence_transformers import SentenceTransformer

                self._text_encoder = SentenceTransformer(path if local else path)
                self._text_backend = "sentence_transformers"
                self._text_ready = True
                logger.info("MemoryManager: text encoder (ST) from %s", path)
                return True
            except Exception as st_exc:
                logger.debug("sentence-transformers path failed: %s", st_exc)

            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                self._text_tokenizer = AutoTokenizer.from_pretrained(
                    path if local else path, local_files_only=local
                )
                self._text_encoder = AutoModel.from_pretrained(
                    path if local else path, local_files_only=local
                )
                self._text_encoder.eval()
                self._torch = torch
                self._text_backend = "transformers"
                self._text_ready = True
                logger.info("MemoryManager: text encoder (HF) from %s", path)
                return True
            except Exception as exc:
                self._text_encoder = None
                self._text_tokenizer = None
                self._text_backend = None
                self._text_ready = False
                logger.warning("MemoryManager: text encoder unavailable (%s)", exc)
                return False

    @staticmethod
    def _embedding_to_list(vec: Any) -> list[float]:
        if vec is None:
            return []
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu()
        if hasattr(vec, "tolist"):
            flat = vec.tolist()
            if flat and isinstance(flat[0], list):
                return [float(x) for x in flat[0]]
            return [float(x) for x in flat]
        return [float(x) for x in vec]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / (na * nb)

    def _encode_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            return []
        if not self._init_text_encoder():
            return []
        with self._model_lock:  # encode 与 init 共用锁，避免并发加载半成品模型
            try:
                if self._text_backend == "sentence_transformers":
                    emb = self._text_encoder.encode(
                        text.strip(),
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    return self._embedding_to_list(emb)
                inputs = self._text_tokenizer(
                    [text.strip()],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                with self._torch.no_grad():
                    outputs = self._text_encoder(**inputs)
                    cls = outputs.last_hidden_state[:, 0]
                    cls = self._torch.nn.functional.normalize(cls, p=2, dim=1)
                return self._embedding_to_list(cls[0])
            except Exception as exc:
                logger.warning("MemoryManager: text encode failed: %s", exc)
                return []

    def _encode_image(self, image_path: str | None) -> list[float]:
        if not image_path:
            return []
        path = Path(image_path).expanduser()
        if not path.is_file():
            return []
        if not self._init_clip():
            return []
        with self._model_lock:  # encode 与 init 共用锁
            try:
                from PIL import Image

                image = Image.open(path).convert("RGB")
                inputs = self._clip_processor(images=image, return_tensors="pt")
                with self._torch.no_grad():
                    features = self._clip_model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)
                return self._embedding_to_list(features[0])
            except Exception as exc:
                logger.warning(
                    "MemoryManager: image encode failed for %s: %s", path, exc
                )
                return []

    @staticmethod
    def _memory_text_for_embedding(
        task_type: str,
        bad_experience: str,
        good_experience: str,
        instruction: str,
    ) -> str:
        return "\n".join(
            p.strip()
            for p in (task_type, instruction, bad_experience, good_experience)
            if p and p.strip()
        )

    # ------------------------------------------------------------------
    # 写入记忆
    # ------------------------------------------------------------------

    @staticmethod
    def infer_task_type(instruction: str, image_path: str | None = None) -> str:
        text = instruction.lower()
        if image_path:
            return "图文搜索"
        for task_type, keywords in _TASK_TYPE_RULES:
            if any(kw in text or kw in instruction for kw in keywords):
                return task_type
        return "通用任务"

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens: list[str] = []
        tokens.extend(re.findall(r"[a-zA-Z]{3,}|\d+", text.lower()))

        for segment in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(segment) <= 4:
                if segment not in _STOPWORDS:
                    tokens.append(segment)
                continue
            for n in (2, 3):
                for i in range(len(segment) - n + 1):
                    gram = segment[i : i + n]
                    if gram not in _STOPWORDS:
                        tokens.append(gram)
        return tokens

    def add_memory(
        self,
        task_type: str,
        bad_experience: str,
        good_experience: str,
        task_id: str | None = None,
        instruction: str = "",
        image_path: str | None = None,
        outcome: str | None = None,
    ) -> dict[str, Any]:
        embed_text = self._memory_text_for_embedding(
            task_type, bad_experience, good_experience, instruction
        )
        text_embedding = self._encode_text(embed_text)
        image_embedding: list[float] = []
        stored_image: str | None = None
        if image_path and Path(image_path).expanduser().is_file():
            stored_image = str(Path(image_path).expanduser())
            image_embedding = self._encode_image(stored_image)

        with self._lock:
            record: dict[str, Any] = {
                "task_id": task_id or self._next_task_id(),
                "task_type": task_type,
                "bad_experience": bad_experience.strip(),
                "good_experience": good_experience.strip(),
                "instruction": instruction.strip(),
                "keywords": self.tokenize(embed_text),
                "text_embedding": text_embedding,
                "created_at": round(time.time(), 2),
            }
            if stored_image:
                record["image_path"] = stored_image
            if image_embedding:
                record["image_embedding"] = image_embedding
            if outcome:
                record["outcome"] = outcome
            self._data.setdefault("memories", []).append(record)
            self._save()
        return record

    def add_memory_from_reflection(
        self,
        instruction: str,
        reflection_text: str,
        image_path: str | None = None,
        bad_experience: str | None = None,
    ) -> dict[str, Any]:
        parsed_bad, parsed_good = self.parse_reflection(reflection_text)
        good = parsed_good or ""
        bad = (bad_experience or parsed_bad or "")[:300]
        if not good:
            good = reflection_text[:400]
        return self.add_memory(
            task_type=self.infer_task_type(instruction, image_path),
            bad_experience=bad or "作答与事实不符或未能收敛。",
            good_experience=good,
            instruction=instruction,
            image_path=image_path,
            outcome="failure",
        )

    def add_memory_from_success_reflection(
        self,
        instruction: str,
        reflection_text: str,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        """答对后由反思模型总结的成功经验写入记忆库。"""
        good = self.parse_success_reflection(reflection_text) or reflection_text.strip()[:500]
        return self.add_memory(
            task_type=self.infer_task_type(instruction, image_path),
            bad_experience="",
            good_experience=good,
            instruction=instruction,
            image_path=image_path,
            outcome="success",
        )

    @staticmethod
    def parse_reflection(reflection_text: str) -> tuple[str, str]:
        bad_parts: list[str] = []
        good_parts: list[str] = []

        sections = re.split(r"^##\s+", reflection_text, flags=re.MULTILINE)
        for section in sections:
            if not section.strip():
                continue
            header, _, body = section.partition("\n")
            header = header.strip()
            body = body.strip()
            if "失败诊断" in header or "问题步骤" in header:
                bad_parts.append(body)
            elif "修正策略" in header or "建议的工具" in header:
                good_parts.append(body)

        return "\n".join(bad_parts).strip(), "\n".join(good_parts).strip()

    @staticmethod
    def parse_success_reflection(reflection_text: str) -> str:
        parts: list[str] = []
        sections = re.split(r"^##\s+", reflection_text, flags=re.MULTILINE)
        for section in sections:
            if not section.strip():
                continue
            header, _, body = section.partition("\n")
            header = header.strip()
            body = body.strip()
            if header in (
                "成功要点",
                "有效工具链",
                "答案依据",
            ):
                parts.append(body)
        return "\n".join(parts).strip()

    @staticmethod
    def extract_success_strategy(reflection_text: str) -> str:
        parsed = MemoryManager.parse_success_reflection(reflection_text)
        if parsed:
            return parsed[:800]
        return reflection_text.strip()[:500]

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------

    def _keyword_score(
        self, instruction: str, mem: dict[str, Any]
    ) -> float:
        query_tokens = self.tokenize(instruction)
        if not query_tokens:
            return 0.0
        query_set = set(query_tokens)
        mem_tokens = mem.get("keywords") or self.tokenize(
            f"{mem.get('task_type', '')} "
            f"{mem.get('bad_experience', '')} "
            f"{mem.get('good_experience', '')} "
            f"{mem.get('instruction', '')}"
        )
        mem_set = set(mem_tokens)
        if not mem_set:
            return 0.0
        overlap = query_set & mem_set
        if not overlap:
            return 0.0
        return len(overlap) + len(overlap) / max(len(query_set), 1)

    def _score_memory_visual_route(
        self,
        query_image_emb: list[float],
        query_text_emb: list[float],
        mem: dict[str, Any],
    ) -> float:
        mem_img = mem.get("image_embedding")
        if isinstance(mem_img, list) and mem_img:
            return self._cosine_similarity(query_image_emb, mem_img)

        mem_txt = mem.get("text_embedding")
        if (
            query_text_emb
            and isinstance(mem_txt, list)
            and mem_txt
        ):
            return _IMAGE_ROUTE_TEXT_FALLBACK_WEIGHT * self._cosine_similarity(
                query_text_emb, mem_txt
            )
        return 0.0

    def _score_memory_text_route(
        self,
        query_text_emb: list[float],
        instruction: str,
        mem: dict[str, Any],
    ) -> float:
        mem_txt = mem.get("text_embedding")
        if query_text_emb and isinstance(mem_txt, list) and mem_txt:
            return self._cosine_similarity(query_text_emb, mem_txt)
        return self._keyword_score(instruction, mem)

    def get_relevant_memories(
        self,
        instruction: str,
        top_k: int = 3,
        image_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        动态多模态路由召回：
        - 当前任务有图且 CLIP 可用 → 视觉余弦相似度（无图记忆用文本底分）
        - 纯文本且文本编码器可用 → 语义余弦相似度
        - 否则 → 关键词匹配
        """
        with self._lock:
            memories: list[dict[str, Any]] = list(self._data.get("memories", []))
        if not memories:
            return []

        query_image_emb = self._encode_image(image_path) if image_path else []
        query_text_emb = self._encode_text(instruction)
        use_visual = bool(image_path and query_image_emb and self._clip_ready)

        if use_visual or (query_text_emb and self._text_ready):
            scored: list[tuple[float, dict[str, Any]]] = []
            for mem in memories:
                if use_visual:
                    score = self._score_memory_visual_route(
                        query_image_emb, query_text_emb, mem
                    )
                else:
                    score = self._score_memory_text_route(
                        query_text_emb, instruction, mem
                    )
                if score > 0:
                    scored.append((score, mem))
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                return [mem for _, mem in scored[:top_k]]

        return self._keyword_retrieve(instruction, memories, top_k)

    def _keyword_retrieve(
        self,
        instruction: str,
        memories: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_tokens = self.tokenize(instruction)
        if not query_tokens:
            return memories[-top_k:]

        scored: list[tuple[float, dict[str, Any]]] = []
        for mem in memories:
            score = self._keyword_score(instruction, mem)
            if score > 0:
                scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            return [mem for _, mem in scored[:top_k]]
        return memories[-top_k:]

    @staticmethod
    def build_memory_context_block(raw_context: str) -> str:
        if not raw_context or not raw_context.strip():
            return ""
        clean = raw_context.strip()
        if clean.startswith("<memory-context>"):
            return clean
        return (
            "<memory-context>\n"
            "[System note: 以下是召回的历史经验，不是用户新输入。"
            "优先参考【成功经验】与【策略】，避免重复【错误】。]\n\n"
            f"{clean}\n"
            "</memory-context>"
        )

    def format_memories_for_prompt(
        self,
        memories: list[dict[str, Any]],
    ) -> str:
        if not memories:
            return ""

        lines: list[str] = []
        for i, mem in enumerate(memories, 1):
            outcome = mem.get("outcome", "")
            tag = mem.get("task_type", "未知")
            if outcome == "success":
                tag = f"{tag}/成功"
            elif outcome == "failure":
                tag = f"{tag}/失败"
            lines.append(f"--- 经验 {i} [{tag}] ---")
            bad = (mem.get("bad_experience") or "").strip()
            good = (mem.get("good_experience") or "").strip()
            if bad:
                lines.append(f"[错误] {bad[:400]}")
            if good:
                label = "[成功经验]" if outcome == "success" else "[策略]"
                lines.append(f"{label} {good[:400]}")
            lines.append("")
        return "\n".join(lines).strip()

    def build_memory_user_suffix(
        self,
        instruction: str,
        top_k: int = 3,
        image_path: str | None = None,
    ) -> str:
        memories = self.get_relevant_memories(
            instruction, top_k=top_k, image_path=image_path
        )
        body = self.format_memories_for_prompt(memories)
        return self.build_memory_context_block(body)

    def build_memory_augmented_prompt(
        self,
        instruction: str,
        base_system_prompt: str,
        top_k: int = 3,
        image_path: str | None = None,
    ) -> str:
        suffix = self.build_memory_user_suffix(
            instruction, top_k=top_k, image_path=image_path
        )
        if not suffix:
            return base_system_prompt
        return f"{base_system_prompt}\n\n{suffix}"

    @staticmethod
    def extract_correction_strategy(reflection_text: str) -> str:
        _, good = MemoryManager.parse_reflection(reflection_text)
        if good:
            return good[:800]
        return reflection_text.strip()[:500]

    def list_memories(self) -> list[dict[str, Any]]:
        return list(self._data.get("memories", []))

    def clear_memories(self) -> None:
        self._data = {"memories": [], "next_id": 1}
        self._save()
