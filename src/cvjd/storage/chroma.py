"""ChromaDB persistence: collections, section vectors, stable IDs, upsert/delete.

Theo plan.cv.md §6: hai collection `frevia_cvs` / `frevia_jobs`, vector ID ổn định
`{type}:{doc_id}:v{version}:{section}`, mỗi section + một vector full-text (`_full`),
upsert thay thế toàn bộ vector cũ của tài liệu khi version tăng.
"""

from __future__ import annotations

import hashlib
import time
from typing import Callable, Iterable, Optional

import numpy as np

FULL_SECTION = "_full"

CV_COLLECTION = "frevia_cvs"
JOB_COLLECTION = "frevia_jobs"
DEFAULT_HNSW_SPACE = "cosine"
DEFAULT_VERSION = 1

EmbedFn = Callable[[list[str]], np.ndarray]

_STATUS_READY = "READY"
_VALID_TYPES = ("cv", "jd")


def build_client(path: Optional[str] = None):
    """PersistentClient khi có path, ngược lại EphemeralClient (test).

    Chroma 1.5 (rust backend) trên Windows: mở nhiều PersistentClient trên
    CÙNG path trong một process khiến các client mở sau đọc catalog rỗng
    (diag: mở lại phải close client trước, hoặc dùng process mới). Tại đây:
    nếu catalog chưa thấy collection dự kiến thì close và mở lại đúng một lần
    (pattern được verify OK); store MỚI (chưa có collection) là hợp lệ, không
    raise — caller gọi ensure_collections() để tạo.
    """
    import chromadb
    from chromadb.errors import NotFoundError

    if not path:
        return chromadb.EphemeralClient()

    expected = {collection_name(t) for t in _VALID_TYPES}

    def _has_catalog(client) -> bool:
        try:
            return expected <= {col.name for col in client.list_collections()}
        except NotFoundError:
            return False

    client = chromadb.PersistentClient(path=str(path))
    if _has_catalog(client):
        return client

    # lần mở đầu (ngay sau khi process khác ghi) có thể chưa thấy catalog;
    # đóng client hiện tại (giải phóng lock) rồi mở lại đúng một lần.
    try:
        client.close()
    except Exception:
        pass
    time.sleep(0.5)
    return chromadb.PersistentClient(path=str(path))


def collection_name(doc_type: str) -> str:
    if doc_type == "cv":
        return CV_COLLECTION
    if doc_type == "jd":
        return JOB_COLLECTION
    raise ValueError(f"doc_type must be one of {_VALID_TYPES}, got {doc_type!r}")


def ensure_collections(client, space: str = DEFAULT_HNSW_SPACE) -> dict[str, str]:
    """Tạo hai collection nếu chưa có. Trả dict doc_type -> name."""
    for doc_type in _VALID_TYPES:
        name = collection_name(doc_type)
        client.get_or_create_collection(name, metadata={"hnsw:space": space})
    return {"cv": CV_COLLECTION, "jd": JOB_COLLECTION}


def vector_id(doc_type: str, doc_id: str, version: int, section: str) -> str:
    return f"{doc_type}:{doc_id}:v{version}:{section}"


def parse_vector_id(vid: str) -> tuple[str, str, int, str]:
    first, mid, version, section = vid.split(":", 3)
    _vers = version[1:]
    if first not in _VALID_TYPES or not _vers.isdigit():
        raise ValueError(f"invalid vector id: {vid!r}")
    return first, mid, int(_vers), section


def section_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def section_vectors(sections: dict[str, object], raw_text: str) -> dict[str, str]:
    """Thiên về tổ hợp section + `_full` full-text. Nhận mapping name -> Section
    (có thuộc tính .text) hoặc name -> str."""
    out: dict[str, str] = {}
    for name, section in sections.items():
        text = getattr(section, "text", section)
        if isinstance(text, str) and text.strip():
            out[str(name)] = text.strip()
    if raw_text and raw_text.strip():
        out[FULL_SECTION] = raw_text.strip()
    return out


def _meta(
    doc_type: str,
    doc_id: str,
    version: int,
    section: str,
    embed_model: str,
    content_hash: str,
    language: Optional[str],
    status: str,
    owner_id: Optional[str],
    visibility: Optional[str],
) -> dict:
    meta: dict = {
        "document_id": doc_id,
        "document_type": doc_type,
        "document_version": int(version),
        "section": section,
        "status": status,
        "embedding_model": embed_model,
        "content_hash": content_hash,
        # luôn ghi owner_id/visibility để metadata filter (`access_where`) khớp được
        "owner_id": owner_id or "",
        "visibility": visibility or "public",
    }
    if language:
        meta["language"] = language
    return meta


def access_where(
    owner_id: Optional[str] = None,
    visibilities: "Iterable[str]" = ("public",),
) -> Optional[dict]:
    """Metadata filter cho retrieval: visibility cho phép HOẶC tài liệu của chính user.

    Trả None khi không cần lọc (mọi visibility, không owner) để `query` bỏ `where`.
    """
    vis = [v for v in (visibilities or ()) if v]
    clauses: list[dict] = []
    if vis:
        clauses.append({"visibility": {"$in": vis}})
    if owner_id:
        clauses.append({"owner_id": {"$eq": owner_id}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}


def _stale_filter(doc_id: str, version: int) -> dict:
    return {
        "$and": [
            {"document_id": {"$eq": doc_id}},
            {"document_version": {"$lt": int(version)}},
        ]
    }


def _doc_filter(doc_id: str) -> dict:
    return {"document_id": {"$eq": doc_id}}


def _deleted_count(result) -> int:
    """chromadb 1.5 delete() trả {'deleted': n}; bản cũ trả {'ids': [...]} hoặc int."""
    if isinstance(result, dict):
        if "deleted" in result:
            return int(result["deleted"])
        return len(result.get("ids", []))
    return int(result or 0)


def upsert_document(
    client,
    doc_type: str,
    doc_id: str,
    sections: dict[str, str],
    embed: EmbedFn,
    embed_model: str,
    version: int = DEFAULT_VERSION,
    language: Optional[str] = None,
    status: str = _STATUS_READY,
    owner_id: Optional[str] = None,
    visibility: Optional[str] = None,
    batch_size: int = 128,
) -> dict:
    """Upsert toàn bộ vector (section + `_full`) của một tài liệu."""
    records = []
    for section, text in sections.items():
        vectors = embed([text]).astype(np.float32)
        records.append((section, text, vectors[0]))
    return upsert_vectors(
        client,
        doc_type,
        doc_id,
        records,
        embed_model,
        version=version,
        language=language,
        status=status,
        owner_id=owner_id,
        visibility=visibility,
        batch_size=batch_size,
    )


def upsert_vectors(
    client,
    doc_type: str,
    doc_id: str,
    records: "Iterable[tuple[str, str, np.ndarray]]",
    embed_model: str,
    version: int = DEFAULT_VERSION,
    language: Optional[str] = None,
    status: str = _STATUS_READY,
    owner_id: Optional[str] = None,
    visibility: Optional[str] = None,
    batch_size: int = 128,
) -> dict:
    """Upsert vector cho sẵn (records: section, text, embedding).

    - Xoá vector thuộc version cũ của tài liệu.
    - Upsert vector version hiện tại.
    - Dọn các vector version hiện tại của section đã biến mất.

    Trả stats: {upserted, deleted_stale, deleted_leftover}.
    """
    collection = client.get_collection(collection_name(doc_type))
    stats = {"upserted": 0, "deleted_stale": 0, "deleted_leftover": 0}

    stale = collection.delete(where=_stale_filter(doc_id, version))
    stats["deleted_stale"] = _deleted_count(stale)

    collected = list(records)
    ids: list[str] = []
    texts: list[str] = []
    embeddings: list[list] = []
    metadatas: list[dict] = []
    for section, text, vector in collected:
        ids.append(vector_id(doc_type, doc_id, version, section))
        texts.append(text)
        embeddings.append(np.asarray(vector, dtype=np.float32).tolist())
        metadatas.append(
            _meta(
                doc_type,
                doc_id,
                version,
                section,
                embed_model,
                section_hash(text),
                language,
                status,
                owner_id,
                visibility,
            )
        )

    for start in range(0, len(ids), batch_size):
        chunk = slice(start, start + batch_size)
        collection.upsert(
            ids=ids[chunk],
            embeddings=embeddings[chunk],
            documents=texts[chunk],
            metadatas=metadatas[chunk],
        )
        stats["upserted"] += len(ids[chunk])

    existing = get_document(client, doc_type, doc_id, version=version)
    known = {vector_id(doc_type, doc_id, version, s) for s, _, _ in collected}
    leftovers = [rec["id"] for rec in existing if rec["id"] not in known]
    if leftovers:
        collection.delete(ids=leftovers)
        stats["deleted_leftover"] = len(leftovers)
    return stats


def delete_document(
    client, doc_type: str, doc_id: str, version: Optional[int] = None
) -> int:
    """Xoá toàn bộ vector của một tài liệu (option: chỉ 1 version)."""
    collection = client.get_collection(collection_name(doc_type))
    where = _doc_filter(doc_id)
    if version is not None:
        where = {
            "$and": [
                {"document_id": {"$eq": doc_id}},
                {"document_version": {"$eq": int(version)}},
            ]
        }
    deleted = collection.delete(where=where)
    return _deleted_count(deleted)


def get_document(
    client,
    doc_type: str,
    doc_id: str,
    version: Optional[int] = None,
    include: Iterable[str] = ("metadatas", "embeddings", "documents"),
) -> list[dict]:
    """Trả danh sách vector của tài liệu (id, metadatas, embeddings, documents)."""
    collection = client.get_collection(collection_name(doc_type))
    result = collection.get(where=_vfilter(doc_id, version), include=list(include))
    rows = result["ids"]
    out: list[dict] = []
    for i, rid in enumerate(rows):
        out.append(
            {
                "id": rid,
                "metadata": result["metadatas"][i] if "metadatas" in include else None,
                "embedding": result["embeddings"][i] if "embeddings" in include else None,
                "document": result["documents"][i] if "documents" in include else None,
            }
        )
    return out


def _vfilter(doc_id: str, version: Optional[int]) -> dict:
    if version is None:
        return {"document_id": {"$eq": doc_id}}
    return {
        "$and": [
            {"document_id": {"$eq": doc_id}},
            {"document_version": {"$eq": int(version)}},
        ]
    }


def query(
    client,
    doc_type: str,
    query_embeddings: np.ndarray,
    n_results: int = 100,
    where: Optional[dict] = None,
    status: Optional[str] = None,
    include: Iterable[str] = ("metadatas", "documents", "distances"),
) -> list[dict]:
    """Tìm vector gần nhất. Trả list các vector kèm distance (nhỏ hơn = gần hơn)."""
    collection = client.get_collection(collection_name(doc_type))
    if status is not None:
        status_filter: dict = {"status": {"$eq": status}}
        where = (
            {"$and": [where, status_filter]}
            if where
            else status_filter
        )
    result = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where=where,
        include=list(include),
    )
    ids: list[str] = result["ids"][0]
    out: list[dict] = []
    for i, rid in enumerate(ids):
        out.append(
            {
                "id": rid,
                "metadata": result["metadatas"][0][i] if "metadatas" in include else None,
                "document": result["documents"][0][i] if "documents" in include else None,
                "distance": result["distances"][0][i] if "distances" in include else None,
            }
        )
    return out


def count_vectors(client, doc_type: str) -> int:
    collection = client.get_collection(collection_name(doc_type))
    return collection.count()