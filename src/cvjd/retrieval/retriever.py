"""Chroma-backed retrieval + section-aware semantic scoring (plan §5).

Luồng (plan §5.2, §6.5):
  1. Query bằng vector full-text của tài liệu truy vấn -> candidate pool top-K.
  2. Lấy vector theo section (gồm `_full`) của từng candidate từ Chroma.
  3. Tính điểm semantic theo section (ghép cặp fixed/cross, tổng hợp mean/max).

Hai chiều: JD -> CV (`doc_type="cv"`) và CV -> JD (`doc_type="jd"`).
`query_doc` cho `retrieve`/`section_scoring` là `Document`; `candidate_pool`
nhận bất kỳ đối tượng nào có thuộc tính `raw_text` (hoặc dùng `pool_by_text`).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..models.matching import aggregate, section_components
from ..storage import chroma

SectionVectors = dict[str, np.ndarray]


class _RawQuery:
    """Tối thiểu hoá đầu vào cho full-text query (chỉ cần `.raw_text`)."""

    __slots__ = ("raw_text",)

    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text


def _as_query(query) -> object:
    return query if hasattr(query, "raw_text") else _RawQuery(str(query))


class RetrievalResult:
    """Một candidate sau khi chấm điểm semantic theo section."""

    __slots__ = ("components", "distance", "doc_id", "semantic", "version")

    def __init__(
        self,
        doc_id: str,
        semantic: float,
        components: dict[str, float] | None = None,
        distance: float | None = None,
        version: int = 1,
    ) -> None:
        self.doc_id = doc_id
        self.semantic = semantic
        self.components = components or {}
        self.distance = distance
        self.version = version

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"RetrievalResult(doc_id={self.doc_id!r}, semantic={self.semantic:.4f}, "
            f"distance={self.distance})"
        )


class ChromaRetriever:
    def __init__(
        self,
        client,
        embedder,
        embed_model: str = "mock",
        status: str = "READY",
        pairing: str = "cross",
        aggregation: str = "mean",
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.embed_model = embed_model
        self.status = status
        self.pairing = pairing
        self.aggregation = aggregation

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def index_corpus(
        self, docs, version: int = 1, status: str | None = None,
        meta: dict | None = None,
    ) -> dict:
        """Batch-encode mọi section + full-text rồi upsert vào Chroma.

        `meta` (tuỳ chọn): {doc_id: {"owner_id": str, "visibility": str}} để gắn
        metadata phục vụ `access_where` khi retrieval.
        """
        from ..storage import chroma as chroma_mod

        extra = meta or {}
        entries: list[tuple[str, str, str, str, str]] = []  # type,id,lang,section,text
        for doc in docs:
            for section, text in chroma_mod.section_vectors(
                doc.sections, doc.raw_text
            ).items():
                entries.append((doc.doc_type, doc.doc_id, doc.lang, section, text))
        stats = {"upserted": 0, "deleted_stale": 0, "deleted_leftover": 0}
        if not entries:
            return stats

        vectors = self.embedder.encode([e[4] for e in entries])
        by_doc: dict = defaultdict(list)
        for (doc_type, doc_id, lang, section, text), vec in zip(entries, vectors):
            by_doc[(doc_type, doc_id, lang)].append((section, text, vec))

        for (doc_type, doc_id, lang), records in by_doc.items():
            doc_meta = extra.get(doc_id, {})
            deltas = chroma_mod.upsert_vectors(
                self.client,
                doc_type,
                doc_id,
                records,
                self.embed_model,
                version=version,
                language=lang,
                status=status or self.status,
                owner_id=doc_meta.get("owner_id"),
                visibility=doc_meta.get("visibility"),
            )
            for key in stats:
                stats[key] += deltas[key]
        return stats

    def index_document(
        self,
        doc,
        version: int = 1,
        status: str | None = None,
        owner_id: str | None = None,
        visibility: str | None = None,
    ) -> dict:
        sections = chroma.section_vectors(doc.sections, doc.raw_text)
        return chroma.upsert_document(
            self.client,
            doc.doc_type,
            doc.doc_id,
            dict(sections),
            self.embedder.encode,
            self.embed_model,
            version=version,
            language=doc.lang,
            status=status or self.status,
            owner_id=owner_id,
            visibility=visibility,
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def pool_by_text(
        self,
        text: str,
        doc_type: str,
        pool_size: int = 100,
        status: str | None = None,
        oversample: int = 3,
        where: dict | None = None,
    ) -> list[dict]:
        """Full-text candidate pool; trả [{doc_id, version, distance}] đã dedup."""
        total = chroma.count_vectors(self.client, doc_type)
        if total == 0:
            return []
        n_results = min(max(pool_size, pool_size * max(oversample, 1)), total)
        query_vec = self.embedder.encode([text])
        hits = chroma.query(
            self.client, doc_type, query_vec, n_results=n_results,
            status=status if status is not None else self.status,
            where=where,
        )
        best: dict[str, dict] = {}
        for hit in hits:
            try:
                _, doc_id, version, _section = chroma.parse_vector_id(hit["id"])
            except ValueError:
                continue
            distance = hit["distance"]
            current = best.get(doc_id)
            if current is None or (
                distance is not None and distance < current["distance"]
            ):
                best[doc_id] = {"doc_id": doc_id, "version": version, "distance": distance}
        return sorted(
            best.values(),
            key=lambda item: (item["distance"] is None, item["distance"], item["doc_id"]),
        )

    def candidate_pool(
        self, query_doc, doc_type: str, pool_size: int = 100, status: str | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        return self.pool_by_text(
            _raw_text(query_doc), doc_type, pool_size=pool_size, status=status, where=where
        )

    def section_vectors(self, doc_type: str, doc_id: str) -> dict[str, np.ndarray]:
        """Vectors theo section (gồm `_full`) của version mới nhất."""
        rows = chroma.get_document(
            self.client, doc_type, doc_id, version=None,
            include=["embeddings", "metadatas"],
        )
        if not rows:
            return {}
        latest = max((r["metadata"] or {}).get("document_version", 0) for r in rows)
        out: dict[str, np.ndarray] = {}
        for row in rows:
            meta = row["metadata"] or {}
            section = meta.get("section")
            if not section or meta.get("document_version") != latest:
                continue
            if row.get("embedding") is not None:
                out[section] = _as_vector(row["embedding"])
        return out

    def section_scoring(
        self,
        query_doc,
        doc_type: str,
        candidate_ids,
        pairing: str | None = None,
        aggregation: str | None = None,
    ) -> dict[str, RetrievalResult]:
        """Điểm semantic theo section cho danh sách candidate (đã có sẵn pool)."""
        query_sections = _section_vectors(query_doc, self.embedder)
        results: dict[str, RetrievalResult] = {}
        for doc_id in candidate_ids:
            vectors = self.section_vectors(doc_type, doc_id)
            if not vectors:
                results[doc_id] = RetrievalResult(doc_id, 0.0, {})
                continue
            components = section_components(
                query_sections, vectors, pairing=pairing or self.pairing
            )
            semantic = aggregate(components, aggregation or self.aggregation)
            results[doc_id] = RetrievalResult(doc_id, semantic, components)
        return results

    def retrieve(
        self,
        query_doc,
        doc_type: str,
        pool_size: int = 100,
        top_k: int = 10,
        pairing: str | None = None,
        aggregation: str | None = None,
        candidate_ids=None,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Full-text pool -> section-aware semantic score -> top-k."""
        pool = self.candidate_pool(query_doc, doc_type, pool_size, where=where)
        scored = self.section_scoring(
            query_doc, doc_type, [c["doc_id"] for c in pool], pairing, aggregation
        )
        distance_by_id = {c["doc_id"]: c["distance"] for c in pool}
        ranked = [
            RetrievalResult(
                rid, res.semantic, res.components, distance_by_id.get(rid)
            )
            for rid, res in scored.items()
        ]
        if candidate_ids is not None:
            keep = set(candidate_ids)
            ranked = [r for r in ranked if r.doc_id in keep]
        ranked.sort(key=lambda item: (-item.semantic, item.doc_id))
        return ranked[:top_k] if top_k else ranked


def _raw_text(query_doc) -> str:
    from ..schema import Document

    if isinstance(query_doc, Document):
        return query_doc.raw_text
    return str(query_doc)


def _section_vectors(doc, embedder) -> dict[str, np.ndarray]:
    from ..models.matching import embed_corpus

    return embed_corpus([doc], embedder)[doc.doc_id]


def _as_vector(embedding) -> np.ndarray:
    return np.asarray(embedding, dtype=np.float32)
