"""Outbox đồng bộ PostgreSQL → ChromaDB (plan §6 / B7).

Event JSONL: mỗi dòng một `OutboxEvent`. Worker (scripts/sync_worker.py) đọc
event sau cursor, apply vào Chroma qua `apply_event`/`process_events`, ghi cursor
(theo số dòng đã xử lý) xuống file `.cursor`. Reconcile: lấy source của sự thật
(manifest docs) rồi sửa Chroma cho khớp — upsert thiếu, delete thừa.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .chroma import FULL_SECTION, delete_document, upsert_document

Cursor = int
ResolveDoc = Callable[[str, str], Optional["object"]]

ACTIONS = ("upsert", "delete")


@dataclass
class OutboxEvent:
    event_id: str
    doc_type: str  # cv | jd
    doc_id: str
    action: str  # upsert | delete
    version: int = 1
    created_at: float = field(default_factory=time.time)
    language: str | None = None
    status: str = "READY"
    owner_id: str | None = None
    visibility: str | None = None

    def validate(self) -> None:
        if self.doc_type not in ("cv", "jd"):
            raise ValueError(f"invalid doc_type: {self.doc_type!r}")
        if self.action not in ACTIONS:
            raise ValueError(f"invalid action: {self.action!r}")
        if not self.event_id or not self.doc_id:
            raise ValueError("event_id and doc_id are required")
        if not (0 < self.version <= 1000):
            raise ValueError(f"invalid version: {self.version}")

    def to_json(self) -> dict:
        self.validate()
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, data: dict) -> OutboxEvent:
        return cls(**{k: data.get(k) for k in ("event_id", "doc_type", "doc_id", "action", "version",
                                               "created_at", "language", "status", "owner_id",
                                               "visibility")})


class Outbox:
    """JSONL outbox với cursor (số dòng) chống idempotency trùng lặp."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, *events: OutboxEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event.to_json(), ensure_ascii=False) + "\n")

    def read_all(self) -> list[tuple[Cursor, OutboxEvent]]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for cursor, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = OutboxEvent.from_json(json.loads(line))
                    event.validate()
                    out.append((cursor, event))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"outbox line {cursor} invalid: {exc}")
        return out

    def read_since(self, after: Cursor) -> list[tuple[Cursor, OutboxEvent]]:
        return [(c, e) for c, e in self.read_all() if c > after]

    @property
    def cursor_file(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".cursor")

    def save_cursor(self, cursor: Cursor) -> None:
        self.cursor_file.write_text(str(cursor), encoding="utf-8")

    def load_cursor(self) -> Cursor:
        if not self.cursor_file.exists():
            return 0
        try:
            return int(self.cursor_file.read_text(encoding="utf-8").strip())
        except ValueError:
            raise ValueError(f"invalid cursor file: {self.cursor_file}")

    def ack(self, cursor: Cursor) -> None:
        if cursor > self.load_cursor():
            self.save_cursor(cursor)


def apply_event(
    client,
    event: OutboxEvent,
    embedder: object,
    embed_model: str,
    resolve_doc: ResolveDoc,
    batch_size: int = 128,
) -> dict:
    """Áp một event lên Chroma. Trả stats {action, upserted, deleted, skipped}."""
    event.validate()
    if event.action == "delete":
        n = delete_document(client, event.doc_type, event.doc_id)
        return {"action": "delete", "upserted": 0, "deleted": n, "skipped": False}

    doc = resolve_doc(event.doc_type, event.doc_id)
    if doc is None:
        return {"action": "upsert", "upserted": 0, "deleted": 0, "skipped": True,
                "reason": "document not found in source"}

    from ..schema import Document

    if not isinstance(doc, Document):
        raise TypeError(f"resolver returned {type(doc).__name__}, expected Document")

    sections = {name: sec.text for name, sec in doc.sections.items()}
    sections[FULL_SECTION] = doc.raw_text
    stats = upsert_document(
        client,
        event.doc_type,
        event.doc_id,
        sections,
        embedder.encode,
        embed_model,
        version=event.version,
        language=event.language or doc.lang,
        status=event.status,
        owner_id=event.owner_id,
        visibility=event.visibility,
        batch_size=batch_size,
    )
    return {"action": "upsert", "upserted": stats["upserted"],
            "deleted": stats["deleted_stale"] + stats["deleted_leftover"], "skipped": False}


def process_events(
    client,
    events: Iterable[tuple[Cursor, OutboxEvent]],
    embedder: object,
    embed_model: str,
    resolve_doc: ResolveDoc,
    max_events: int = 100,
) -> dict:
    """Xử lý batch event theo thứ tự; trả stats tổng + cursor cao nhất đã thành công."""
    summary = {"upserted": 0, "deleted": 0, "skipped": 0, "processed": 0, "cursor": 0}
    for cursor, event in events:
        if summary["processed"] >= max_events:
            break
        result = apply_event(client, event, embedder, embed_model, resolve_doc)
        summary["processed"] += 1
        summary["upserted"] += result["upserted"]
        summary["deleted"] += result["deleted"]
        if result.get("skipped"):
            summary["skipped"] += 1
            continue
        summary["cursor"] = max(summary["cursor"], cursor)
    return summary


def reconcile(
    client,
    source: dict[str, dict[str, object]],
    embedder: object,
    embed_model: str,
    status: str = "READY",
    owner_id: str | None = None,
    visibility: str | None = None,
) -> dict:
    """Sửa Chroma cho khớp source (doc_type -> {doc_id: Document}).

    - Upsert doc có trong source nhưng chưa có trong Chroma (missing).
    - Doc đã có nhưng được reconcile lại => bump version (refreshed).
    - Xoá vector doc có trong Chroma nhưng không còn trong source (orphan).

    Trả {upserted, deleted(orphan), missing, refreshed}.
    """
    from ..schema import Document
    from .chroma import get_document

    result = {"upserted": 0, "deleted": 0, "missing": 0, "refreshed": 0}

    stored: dict[str, set[str]] = {}
    for doc_type, docs in source.items():
        stored[doc_type] = _present_doc_ids(client, doc_type)

    for doc_type, docs in source.items():
        for doc_id, doc in docs.items():
            if not isinstance(doc, Document):
                raise TypeError(f"{doc_id} is not a Document")
            rows = get_document(client, doc_type, doc_id, version=None, include=["metadatas"])
            latest = max((r["metadata"] or {}).get("document_version", 0)
                         for r in rows) if rows else 0
            if latest == 0:
                action = "missing"
                result[action] += 1
            else:
                action = "refreshed"
                result[action] += 1
            event = OutboxEvent(
                event_id=f"reconcile:{doc_type}:{doc_id}",
                doc_type=doc_type,
                doc_id=doc_id,
                action="upsert",
                version=latest + 1,
                status=status,
                owner_id=owner_id,
                visibility=visibility,
            )
            stats = apply_event(client, event, embedder, embed_model,
                                lambda _t, _i, d=doc: d)
            result["upserted"] += stats["upserted"]

    for doc_type, present in stored.items():
        expected = set(source.get(doc_type, {}))
        orphans = present - expected
        if orphans:
            for doc_id in orphans:
                if not doc_id:
                    continue
                n = delete_document(client, doc_type, doc_id)
                result["deleted"] += n
    return result


def _present_doc_ids(client, doc_type: str) -> set[str]:
    """Toàn bộ doc_id đang có (mọi version) trong một collection."""
    from .chroma import collection_name

    collection = client.get_collection(collection_name(doc_type))
    ids = collection.get(include=[])["ids"]
    prefix = f"{doc_type}:"  # "cv:" | "jd:"
    out: set[str] = set()
    for vid in ids:
        rest = vid[len(prefix) :]  # "cv:u-7:v1:skills" -> "u-7:v1:skills"
        if ":v" in rest:
            out.add(rest.split(":v", 1)[0])
    return out


__all__ = [
    "ACTIONS",
    "Outbox",
    "OutboxEvent",
    "apply_event",
    "process_events",
    "reconcile",
]