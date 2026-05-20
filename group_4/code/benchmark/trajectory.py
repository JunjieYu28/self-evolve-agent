"""
Trajectory recording and replay for the Agent Harness.

Each task produces a JSONL file where every line is one interaction turn,
tagged with role, timestamp, step_id, and optional tool metadata.

V2: In-memory message cache for hot-path reads; disk writes for persistence.
"""

import json
import time
from pathlib import Path
from typing import Optional

from roles import Role


class Trajectory:
    """
    Append-only JSONL trajectory store with in-memory cache.

    File layout (one JSON object per line):
    {
        "timestamp":    float,
        "step_id":      int | None,
        "role":         str,
        "content":      str | dict,
        "tool_call_id": str | None,
        ...extra fields...
    }
    """

    def __init__(self, task_id: str, output_dir: str = "trajectories"):
        self.task_id = task_id
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.path = output_path / f"{task_id}.jsonl"
        self._entries: list[dict] = []
        self._messages_cache: list[dict] = []

    def write(
        self,
        role: Role,
        content,
        step_id: Optional[int] = None,
        tool_call_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "step_id": step_id,
            "role": role.value,
            "content": content,
            "tool_call_id": tool_call_id,
        }
        if extra:
            entry.update(extra)

        self._entries.append(entry)

        msg = self._entry_to_message(entry)
        self._messages_cache.append(msg)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _entry_to_message(self, entry: dict) -> dict:
        role = entry["role"]
        msg: dict = {"role": role, "content": entry["content"] or ""}
        if role == "assistant" and entry.get("tool_calls"):
            msg["tool_calls"] = entry["tool_calls"]
        if entry.get("tool_call_id"):
            msg["tool_call_id"] = entry["tool_call_id"]
        return msg

    def to_messages(self) -> list[dict]:
        """Return OpenAI-compatible messages from in-memory cache (no disk I/O)."""
        return list(self._messages_cache)

    def replace_messages(self, new_messages: list[dict]) -> None:
        """Replace in-memory messages cache (used after compaction)."""
        self._messages_cache = list(new_messages)

    def rewrite_last_assistant(self, new_content: str, step_id: Optional[int] = None) -> None:
        """Replace the most recent assistant entry's content (memory + disk)."""
        # Update in-memory entries
        for i in range(len(self._entries) - 1, -1, -1):
            if self._entries[i]["role"] == "assistant":
                self._entries[i]["content"] = new_content
                self._entries[i]["tool_calls"] = None
                if step_id is not None:
                    self._entries[i]["step_id"] = step_id
                break

        # Update messages cache
        for i in range(len(self._messages_cache) - 1, -1, -1):
            if self._messages_cache[i]["role"] == "assistant":
                self._messages_cache[i]["content"] = new_content
                self._messages_cache[i].pop("tool_calls", None)
                break

        # Rewrite disk file from memory
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        """Return all recorded entries (from memory, not disk)."""
        return list(self._entries)

    def summary(self) -> dict:
        role_counts: dict[str, int] = {}
        for e in self._entries:
            r = e["role"]
            role_counts[r] = role_counts.get(r, 0) + 1
        return {
            "task_id": self.task_id,
            "total_turns": len(self._entries),
            "role_counts": role_counts,
            "path": str(self.path),
        }

    def export_json(self) -> str:
        return json.dumps(self._entries, ensure_ascii=False, indent=2)
