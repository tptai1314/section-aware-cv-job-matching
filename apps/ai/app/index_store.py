"""Kho payload đã trích xuất (JSONL) song song với Chroma.

Chroma chỉ lưu vector; để chấm hybrid (§7) ở endpoint `/internal/recommend` cần
lại cấu trúc skills/experience/budget của từng tài liệu. `DocumentStore` giữ
payload theo `doc_id`, ghi cạnh store Chroma (`<chroma_path>.documents.jsonl`);
khi `chroma_path` None thì chỉ giữ trong bộ nhớ.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_VALID_TYPES = ("cv", "jd")


class DocumentStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._docs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("id"):
                    self._docs[record["id"]] = record

    def _flush(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for record in self._docs.values():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def upsert(self, documents: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        with self._lock:
            for document in documents:
                doc_id = str(document.get("id") or document.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                payload = dict(document.get("payload") or {})
                doc_type = str(document.get("type") or document.get("doc_type") or "")
                if doc_type not in _VALID_TYPES:
                    doc_type = "cv"
                self._docs[doc_id] = {"id": doc_id, "type": doc_type, "payload": payload}
                count += 1
            if count:
                self._flush()
        return count

    def get(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._docs.get(doc_id)
            return None if record is None else dict(record)

    def delete(self, ids: Iterable[str]) -> int:
        wanted = {str(doc_id).strip() for doc_id in ids if str(doc_id).strip()}
        if not wanted:
            return 0
        with self._lock:
            removed = [doc_id for doc_id in wanted if doc_id in self._docs]
            for doc_id in removed:
                del self._docs[doc_id]
            if removed:
                self._flush()
        return len(removed)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._docs)

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)


__all__ = ["DocumentStore"]
