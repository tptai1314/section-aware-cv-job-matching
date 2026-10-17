"""Internal hybrid ranking + indexing endpoints (plan §7, §10.2).

Hai chế độ cho `/internal/recommend`:
  - **stateless**: `candidates` được gửi kèm -> chỉ chấm công thức §7.
  - **retrieval**: bỏ trống `candidates` -> truy vấn Chroma (section-aware) rồi
    chấm §7 trên payload đã index qua `/internal/index`.

Chroma + payload store được cache theo `settings.chroma_path`; embedder dùng
BGE-M3 của AI service. Không truy cập DB/quyền ownership — phần đó do NestJS lo.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:  # cho phép import cvjd khi chạy ngoài PYTHONPATH
    sys.path.insert(0, str(_SRC))

from cvjd.data import document_from_payload
from cvjd.service import MatchingService
from cvjd.storage import chroma as chroma_mod
from cvjd.utils.config import load_config

from . import explain as explain_mod
from .index_store import DocumentStore

_lock = threading.Lock()
_services: dict[str, MatchingService] = {}
_stores: dict[str, DocumentStore] = {}
_stateless: MatchingService | None = None


def get_stateless_service() -> MatchingService:
    """MatchingService thuần công thức (không embedder/Chroma) cho chế độ stateless."""
    global _stateless
    with _lock:
        if _stateless is None:
            _stateless = MatchingService(None, None, config=load_config())
        return _stateless


def _key(settings) -> str:
    return settings.chroma_path or "__memory__"


def build_service(settings) -> MatchingService:
    """Dựng MatchingService thật: Chroma (persistent/ephemeral) + embedder BGE-M3."""
    from .embeddings import get_embedder

    client = chroma_mod.build_client(settings.chroma_path)
    chroma_mod.ensure_collections(client)
    embedder = get_embedder(settings.embed_model, settings.embed_device)
    return MatchingService(
        client,
        embedder,
        embed_model=settings.embed_model,
        config=load_config(),
    )


def get_service(settings) -> MatchingService:
    with _lock:
        service = _services.get(_key(settings))
        if service is None:
            service = build_service(settings)
            _services[_key(settings)] = service
        return service


def get_store(settings) -> DocumentStore:
    with _lock:
        store = _stores.get(_key(settings))
        if store is None:
            path = f"{settings.chroma_path}.documents.jsonl" if settings.chroma_path else None
            store = DocumentStore(path)
            _stores[_key(settings)] = store
        return store


def reset_cache() -> None:
    global _stateless
    with _lock:
        _services.clear()
        _stores.clear()
        _stateless = None


def _record(document: Mapping[str, Any]) -> dict[str, Any] | None:
    data = dict(document)
    doc_id = str(data.pop("id", None) or data.pop("doc_id", None) or "").strip()
    if not doc_id:
        return None
    doc_type = str(data.pop("type", None) or data.pop("doc_type", None) or "cv")
    record = {"id": doc_id, "type": doc_type if doc_type in ("cv", "jd") else "cv", "payload": data}
    owner_id = data.pop("owner_id", None)
    visibility = data.pop("visibility", None)
    if owner_id:
        record["owner_id"] = str(owner_id)
    if visibility:
        record["visibility"] = str(visibility)
    return record


def index(documents: Sequence[Mapping[str, Any]], settings) -> dict[str, Any]:
    """Index payload đã trích xuất: upsert vector (Chroma) + lưu payload."""
    records = [record for record in (_record(d) for d in documents) if record]
    if not records:
        return {"indexed": 0, "stats": {}}
    docs = [
        document_from_payload(record["id"], record["type"], record["payload"])
        for record in records
    ]
    meta = {
        record["id"]: {
            "owner_id": record.get("owner_id"),
            "visibility": record.get("visibility"),
        }
        for record in records
        if record.get("owner_id") or record.get("visibility")
    }
    stats = get_service(settings).index(docs, meta=meta)
    stored = get_store(settings).upsert(records)
    return {"indexed": stored, "stats": stats}


def delete(documents: Sequence[Mapping[str, Any]], settings) -> dict[str, Any]:
    """Xoá vector (Chroma) + payload cho từng tài liệu (privacy/H26)."""
    records = [record for record in (_record(d) for d in documents) if record]
    if not records:
        return {"deleted": 0, "stats": {}}
    service = get_service(settings)
    vectors = 0
    for record in records:
        vectors += service.delete(record["id"], record["type"])
    stored = get_store(settings).delete(record["id"] for record in records)
    return {"deleted": stored, "stats": {"vectors": vectors}}


def recommend(
    query: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]] | None,
    direction: str,
    top_k: int,
    keyword_scores: Mapping[str, float] | None,
    keyword_alpha: float | None,
    settings,
    pool_size: int | None = None,
    owner_id: str | None = None,
    visibilities: Sequence[str] | None = None,
    llm_client=None,
    explain: bool = False,
    explain_top_n: int = 3,
) -> list[dict[str, Any]]:
    target_type = "cv" if direction == "jd2cv" else "jd"
    if candidates:
        items = _rank_payloads(
            query, candidates, target_type, top_k, keyword_scores, keyword_alpha, settings
        )
        payload_by_id = {
            str(c["id"]): dict(c) for c in candidates if c.get("id") is not None
        }
    else:
        where = chroma_mod.access_where(owner_id, visibilities or ("public",))
        items, payload_by_id = _retrieve_and_rank(
            query, target_type, top_k, keyword_scores, keyword_alpha, settings, pool_size, where
        )
    if explain and llm_client is not None and explain_top_n > 0:
        _attach_explanations(
            items, query, payload_by_id, direction, llm_client, settings, explain_top_n
        )
    return items


def _rank_payloads(
    query: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    target_type: str,
    top_k: int,
    keyword_scores: Mapping[str, float] | None,
    keyword_alpha: float | None,
    settings,
) -> list[dict[str, Any]]:
    semantic_scores = {
        str(c["id"]): float(c["semantic"])
        for c in candidates
        if c.get("id") is not None and c.get("semantic") is not None
    }
    recommendations = get_stateless_service().rank_structured(
        query,
        target_type,
        candidates,
        semantic_scores=semantic_scores or None,
        top_k=top_k,
        keyword_scores=keyword_scores,
        keyword_alpha=keyword_alpha,
    )
    return _to_items(recommendations)


def _retrieve_and_rank(
    query: Mapping[str, Any],
    target_type: str,
    top_k: int,
    keyword_scores: Mapping[str, float] | None,
    keyword_alpha: float | None,
    settings,
    pool_size: int | None,
    where: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    service = get_service(settings)
    store = get_store(settings)
    query_type = "jd" if target_type == "cv" else "cv"
    query_doc = document_from_payload(
        str(query.get("id") or query.get("doc_id") or "query"), query_type, query
    )
    results = service.retriever.retrieve(
        query_doc,
        target_type,
        pool_size=pool_size or settings.recommend_pool_size,
        top_k=0,
        where=None if where is None else dict(where),
    )
    semantic_scores = {result.doc_id: result.semantic for result in results}
    section_components = {result.doc_id: result.components for result in results}

    payloads = []
    for result in results:
        record = store.get(result.doc_id)
        if record is not None:
            payloads.append({"id": result.doc_id, **record["payload"]})
    payload_by_id = {str(p["id"]): p for p in payloads}
    recommendations = service.rank_structured(
        query,
        target_type,
        payloads,
        semantic_scores=semantic_scores,
        section_components=section_components,
        top_k=top_k,
        keyword_scores=keyword_scores,
        keyword_alpha=keyword_alpha,
    )
    return _to_items(recommendations), payload_by_id


def _attach_explanations(
    items: list[dict[str, Any]],
    query: Mapping[str, Any],
    payload_by_id: Mapping[str, dict[str, Any]],
    direction: str,
    llm_client,
    settings,
    top_n: int,
) -> None:
    """Gắn giải thích LLM (evidence-grounded) cho top-N candidate, tại chỗ.

    JD luôn đóng vai `job`, CV đóng vai `cv` bất kể chiều (jd2cv/cv2jd).
    LLM lỗi -> explanation với ready=false, không làm hỏng kết quả xếp hạng.
    """
    for item in items[:top_n]:
        candidate = payload_by_id.get(str(item["id"]))
        if candidate is None:
            item["explanation"] = None
            continue
        if direction == "jd2cv":
            job, cv = query, candidate
        else:
            job, cv = candidate, query
        item["explanation"] = explain_mod.explain_with_llm(
            llm_client, settings, dict(job), dict(cv), item
        )


def _to_items(recommendations) -> list[dict[str, Any]]:
    return [
        {
            "id": r.doc_id,
            "score": r.score,
            "components": {
                "semantic": r.semantic,
                "skills": r.skills,
                "experience": r.experience,
                "budget": r.budget,
                "preferences": r.preferences,
                "keyword": r.keyword,
            },
            "section_components": dict(r.section_components),
        }
        for r in recommendations
    ]


__all__ = ["delete", "get_service", "get_store", "index", "recommend", "reset_cache"]
