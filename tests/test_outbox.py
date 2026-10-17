"""Tests outbox → Chroma sync (B7): append, cursor, apply, reconcile."""

import json

from cvjd.models.embedding import MockEmbedder
from cvjd.schema import Document, Section
from cvjd.storage import chroma
from cvjd.storage.outbox import (
    Outbox,
    OutboxEvent,
    apply_event,
    process_events,
    reconcile,
)

EMBED_MODEL = "mock"


def build_client():
    client = chroma.build_client(None)
    chroma.ensure_collections(client)
    return client


def make_doc(doc_id, doc_type="cv", text="React developer"):
    return Document(
        doc_id=doc_id,
        doc_type=doc_type,
        lang="en",
        raw_text=text,
        sections={"skills": Section(name="skills", text=text)},
    )


def test_outbox_roundtrip(tmp_path):
    path = tmp_path / "outbox.jsonl"
    box = Outbox(path)
    box.append(
        OutboxEvent(event_id="e1", doc_type="cv", doc_id="c1", action="upsert", version=1),
        OutboxEvent(event_id="e2", doc_type="jd", doc_id="j1", action="delete"),
    )
    box.append(OutboxEvent(event_id="e3", doc_type="cv", doc_id="c2", action="upsert", version=2))

    assert [e.event_id for _, e in box.read_all()] == ["e1", "e2", "e3"]
    assert [e.event_id for _, e in box.read_since(1)] == ["e2", "e3"]

    box.save_cursor(2)
    assert box.load_cursor() == 2


def test_outbox_rejects_invalid(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"event_id": "x", "doc_type": "cv", "doc_id": "c", "action": "wat"}), encoding="utf-8")
    box = Outbox(path)
    try:
        box.read_all()
        assert False, "should raise"
    except ValueError:
        pass


def test_apply_event_upsert_and_delete():
    client = build_client()
    embedder = MockEmbedder(dim=32)
    source = {"cv": {"c1": make_doc("c1")}}

    def resolve(doc_type, doc_id):
        return source[doc_type][doc_id]

    ev = OutboxEvent(event_id="e1", doc_type="cv", doc_id="c1", action="upsert", version=1)
    stats = apply_event(client, ev, embedder, EMBED_MODEL, resolve)
    assert stats["upserted"] >= 2
    rows = chroma.get_document(client, "cv", "c1", include=["metadatas"])
    assert all(r["metadata"]["document_version"] == 1 for r in rows)

    ev2 = OutboxEvent(event_id="e2", doc_type="cv", doc_id="c1", action="delete")
    deleted = apply_event(client, ev2, embedder, EMBED_MODEL, resolve)["deleted"]
    assert deleted == len(rows)
    assert chroma.get_document(client, "cv", "c1") == []


def test_apply_event_missing_doc_skipped():
    client = build_client()
    embedder = MockEmbedder(dim=32)
    ev = OutboxEvent(event_id="e", doc_type="cv", doc_id="ghost", action="upsert", version=1)
    stats = apply_event(client, ev, embedder, EMBED_MODEL, lambda _t, _i: None)
    assert stats["skipped"] is True


def test_process_events_idempotent_with_cursor(tmp_path):
    client = build_client()
    embedder = MockEmbedder(dim=32)
    box = Outbox(tmp_path / "outbox.jsonl")
    box.append(
        OutboxEvent(event_id="e1", doc_type="cv", doc_id="c1", action="upsert", version=1),
        OutboxEvent(event_id="e2", doc_type="cv", doc_id="c2", action="upsert", version=1),
        OutboxEvent(event_id="e3", doc_type="cv", doc_id="c3", action="upsert", version=1),
    )
    source = {"cv": {f"c{i}": make_doc(f"c{i}") for i in (1, 2, 3)}}

    def resolve(doc_type, doc_id):
        return source[doc_type][doc_id]

    stats = process_events(client, box.read_since(box.load_cursor()), embedder, EMBED_MODEL, resolve, max_events=2)
    assert stats["processed"] == 2
    assert stats["cursor"] == 2
    box.ack(stats["cursor"])
    # xử lý lại từ cursor 2 -> chỉ event e3 còn lại, không nhân đôi c1/c2
    stats2 = process_events(client, box.read_since(box.load_cursor()), embedder, EMBED_MODEL, resolve)
    assert stats2["processed"] == 1
    assert stats2["cursor"] == 3
    assert len(chroma.get_document(client, "cv", "c1")) == 2  # skills + _full, đúng 1 version


def test_reconcile_brings_store_in_line(tmp_path):
    # store persistent trong tmp_path để cách ly khỏi các test khác
    # (EphemeralClient dùng chung store trong tiến trình).
    client = chroma.build_client(str(tmp_path / "chroma"))
    chroma.ensure_collections(client)
    embedder = MockEmbedder(dim=32)
    source = {
        "cv": {"rc1": make_doc("rc1"), "rc2": make_doc("rc2")},
        "jd": {"rj1": make_doc("rj1", "jd", "React Ops")},
    }

    def resolve(doc_type, doc_id):
        return source[doc_type][doc_id]

    apply_event(client, OutboxEvent("x", "cv", "rc1", "upsert", 1), embedder, EMBED_MODEL, resolve)
    apply_event(client, OutboxEvent("y", "cv", "rz9", "upsert", 1), embedder, EMBED_MODEL, lambda _t, _i: make_doc("rz9"))

    rep = reconcile(client, source, embedder, EMBED_MODEL)
    assert rep["missing"] == 2  # rc2, rj1
    assert rep["deleted"] == 2  # orphan rz9 có 2 vector (skills + _full)
    assert rep["refreshed"] == 1  # rc1 bump 1 -> 2

    assert all(v["metadata"]["document_version"] == 2 for v in chroma.get_document(client, "cv", "rc1"))
    assert len(chroma.get_document(client, "cv", "rz9")) == 0
    assert len(chroma.get_document(client, "jd", "rj1")) == 2