"""严格 JSONL 格式的 Agent 埋点日志。"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class AgentLogger:
    """
    拦截 ReactAgent 生命周期，落盘两类 JSONL 文件：
    - results.jsonl
    - trajectory.jsonl
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        file_lock: threading.Lock | None = None,
        task_index: int | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.log_dir / "results.jsonl"
        self.trajectory_path = self.log_dir / "trajectory.jsonl"
        self._trajectory_step_id = 0
        self._task_index = 0
        self._file_lock = file_lock or threading.Lock()
        self._bound_task_index = task_index

    def reset_trajectory_step(self) -> None:
        """新任务开始时重置轨迹步序。"""
        self._trajectory_step_id = 0

    def log_trajectory(
        self,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """
        追加轨迹记录，格式：
        {"timestamp", "step_id", "role", "content", "tool_call_id"?, "tool_calls"?}
        tool_call_id / tool_calls 仅在存在时写入。
        """
        record: dict[str, Any] = {
            "timestamp": round(time.time(), 2),
            "step_id": self._trajectory_step_id,
            "role": role,
            "content": content,
        }
        if self._bound_task_index is not None:
            record["task_index"] = self._bound_task_index
        if tool_call_id is not None:
            record["tool_call_id"] = tool_call_id
        if tool_calls is not None:
            record["tool_calls"] = tool_calls

        with self._file_lock:
            with self.trajectory_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._trajectory_step_id += 1

    def read_trajectory(self, task_index: int | None = None) -> list[dict[str, Any]]:
        """
        读取 trajectory.jsonl。
        若构造时绑定了 task_index（或传入 task_index），仅返回该样本的记录，
        避免评测多题共用一个文件时反思 prompt 累积整场轨迹导致 32k 超限。
        """
        if not self.trajectory_path.is_file():
            return []
        filter_idx = (
            task_index if task_index is not None else self._bound_task_index
        )
        records: list[dict[str, Any]] = []
        with self.trajectory_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if filter_idx is not None:
                    if rec.get("task_index") != filter_idx:
                        continue
                records.append(rec)
        return records

    def log_result(
        self,
        instruction: str,
        image: str | None,
        answer: str | None,
        pred: str,
        index: int | None = None,
        retried: bool = False,
        *,
        pred_raw: str | None = None,
        extract: str | None = None,
    ) -> None:
        """
        追加结果记录，格式：
        {"index": ..., "instruction": ..., "image": ..., "answer": ..., "pred": ..., "retried": ...}
        评测可选 pred_raw（Agent 原文）、extract（rule|llm|raw 等）
        """
        if index is None:
            index = self._task_index
            self._task_index += 1

        record: dict[str, Any] = {
            "index": index,
            "instruction": instruction,
            "image": image,
            "answer": answer,
            "pred": pred,
        }
        if pred_raw is not None:
            record["pred_raw"] = pred_raw
        if extract is not None:
            record["extract"] = extract
        if retried:
            record["retried"] = True
        with self._file_lock:
            with self.results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
