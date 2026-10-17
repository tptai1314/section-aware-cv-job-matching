"""End-to-end matching service (plan §13.4-13.5): retrieval -> hybrid -> gate.

Gom `ChromaRetriever`, `HybridRanker` và (tuỳ chọn) PostgreSQL hard-constraint
thành một API thống nhất cho hai chiều:

    JD -> CV   (`recommend_cvs_for_job`)
    CV -> JD   (`recommend_jobs_for_cv`)

Kết quả là `Recommendation` có thể serialize (`to_dict`) để trả về NestJS/UI.
Service KHÔNG tự kết nối DB: caller truyền `conn`/`ctx` khi muốn kiểm tra
quyền truy cập, còn lại chạy thuần vector + công thức.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .ranking.engine import HybridRanker, RankedCandidate
from .ranking.hybrid import CandidateProfile, job_from_document, profile_from_document
from .ranking.structured import (
    candidate_profile,
    job_requirement,
)
from .retrieval.retriever import ChromaRetriever
from .schema import Document
from .storage import postgres
from .utils.config import get_retrieval_settings, load_config


@dataclass(frozen=True)
class Recommendation:
    doc_id: str
    score: float
    semantic: float
    skills: float
    experience: float
    budget: float
    preferences: float
    section_components: dict[str, float] = field(default_factory=dict)
    keyword: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "components": {
                "semantic": round(self.semantic, 6),
                "skills": round(self.skills, 6),
                "experience": round(self.experience, 6),
                "budget": round(self.budget, 6),
                "preferences": round(self.preferences, 6),
                "keyword": None if self.keyword is None else round(self.keyword, 6),
            },
            "section_components": {
                name: round(value, 6) for name, value in self.section_components.items()
            },
        }


class MatchingService:
    """Orchestrate retrieval + hybrid ranking cho hai chiều match."""

    def __init__(
        self,
        client,
        embedder,
        embed_model: str | None = None,
        config: dict | None = None,
        ranker: HybridRanker | None = None,
        status: str | None = None,
        pairing: str | None = None,
        aggregation: str | None = None,
    ) -> None:
        config = config if config is not None else load_config()
        settings = get_retrieval_settings(config)
        self.settings = settings
        self.retriever = ChromaRetriever(
            client,
            embedder,
            embed_model=embed_model or getattr(embedder, "model_id", "mock"),
            status=status or settings["status"],
            pairing=pairing or settings["pairing"],
            aggregation=aggregation or settings["aggregation"],
        )
        self.ranker = ranker if ranker is not None else HybridRanker(config=config)

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def index(
        self, documents: Iterable[Document], version: int = 1, meta: dict | None = None
    ) -> dict:
        """Index tài liệu; `meta` {doc_id: {owner_id, visibility}} gắn metadata truy cập."""
        return self.retriever.index_corpus(documents, version=version, meta=meta)

    def delete(self, doc_id: str, doc_type: str, version: int | None = None) -> int:
        """Xoá vector của một tài liệu khỏi Chroma (privacy/H26).

        Trả về số vector đã xoá; dùng chung với việc xoá payload ở NestJS/payload
        store để đồng bộ PostgreSQL ↔ Chroma.
        """
        from .storage import chroma

        return chroma.delete_document(self.retriever.client, doc_type, doc_id, version)

    # ------------------------------------------------------------------ #
    # Recommendation
    # ------------------------------------------------------------------ #

    def recommend(
        self,
        query_doc: Document,
        target_type: str,
        documents: Mapping[str, Document] | None = None,
        profiles: Mapping[str, CandidateProfile] | None = None,
        pool_size: int | None = None,
        top_k: int | None = None,
        keyword_scores: Mapping[str, float] | None = None,
        keyword_alpha: float | None = None,
        conn=None,
        ctx=None,
        require_ready: bool = True,
        allowed_roles: Sequence[str] = (),
        where: Mapping[str, Any] | None = None,
    ) -> list[Recommendation]:
        """Retrieve candidate pool -> hybrid score -> lọc hard-constraint -> top-k.

        - `documents`/`profiles`: cần để tính skills/experience (nếu bỏ trống chỉ
          trả điểm semantic).
        - `keyword_scores`: điểm lexical (vd BM25) để fuse với semantic.
        - `conn`/`ctx`: nếu có, chạy thêm PostgreSQL hard-constraint gate.
        - `where`: metadata filter Chroma (vd `chroma.access_where(owner_id, vis)`)
          áp ngay ở bước retrieval.
        """
        pool_size = self.settings["pool_size"] if pool_size is None else pool_size
        top_k = self.settings["top_k"] if top_k is None else top_k
        alpha = self.settings["keyword_alpha"] if keyword_alpha is None else keyword_alpha

        results = self.retriever.retrieve(
            query_doc, target_type, pool_size=pool_size, top_k=0,
            where=None if where is None else dict(where),
        )

        if documents is None and profiles is None:
            recommendations = self._score_semantic(results, keyword_scores, alpha)
        else:
            recommendations = self._score_hybrid(
                query_doc, target_type, results, documents, profiles, keyword_scores, alpha
            )

        recommendations = self._gate(
            recommendations, target_type, conn, ctx, require_ready, allowed_roles
        )
        recommendations.sort(key=lambda item: (-item.score, item.doc_id))
        return recommendations[:top_k] if top_k else recommendations

    def recommend_cvs_for_job(self, job_doc: Document, **kwargs) -> list[Recommendation]:
        return self.recommend(job_doc, "cv", **kwargs)

    def recommend_jobs_for_cv(self, cv_doc: Document, **kwargs) -> list[Recommendation]:
        return self.recommend(cv_doc, "jd", **kwargs)

    def rank_structured(
        self,
        query: Mapping[str, Any],
        target_type: str,
        candidates: Sequence[Mapping[str, Any]],
        semantic_scores: Mapping[str, float] | None = None,
        section_components: Mapping[str, Mapping[str, float]] | None = None,
        top_k: int | None = None,
        keyword_scores: Mapping[str, float] | None = None,
        keyword_alpha: float | None = None,
        conn=None,
        ctx=None,
        require_ready: bool = True,
        allowed_roles: Sequence[str] = (),
    ) -> list[Recommendation]:
        """Ranking thuần công thức trên payload đã trích xuất (không cần Chroma).

        `query` và mỗi phần tử `candidates` theo schema `extract-cv`/`extract-job`
        (khoá `id`/`doc_id`, `skills`/`required_skills`, ...). `semantic_scores`
        (nếu có) là điểm semantic theo section tính sẵn cho từng candidate;
        `section_components` mang breakdown section tương ứng để trả kèm.
        """
        top_k = self.settings["top_k"] if top_k is None else top_k
        alpha = self.settings["keyword_alpha"] if keyword_alpha is None else keyword_alpha
        scores = semantic_scores or {}

        requirement = job_requirement(query) if target_type == "cv" else None
        query_profile = candidate_profile(query) if target_type == "jd" else None

        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            doc_id = str(candidate.get("id") or candidate.get("doc_id") or "").strip()
            if not doc_id:
                continue
            semantic = float(scores.get(doc_id, candidate.get("semantic", 0.0)) or 0.0)
            if target_type == "cv":
                job, profile = requirement, candidate_profile(candidate)
            else:
                job, profile = job_requirement(candidate), query_profile
            if job is None or profile is None:
                continue
            components = dict((section_components or {}).get(doc_id, {}))
            ranked.append(self.ranker.score(job, profile, semantic, doc_id, components))

        recommendations = self._finalize(ranked, keyword_scores, alpha)
        recommendations = self._gate(
            recommendations, target_type, conn, ctx, require_ready, allowed_roles
        )
        recommendations.sort(key=lambda item: (-item.score, item.doc_id))
        return recommendations[:top_k] if top_k else recommendations

    def _gate(
        self,
        recommendations: list[Recommendation],
        target_type: str,
        conn,
        ctx,
        require_ready: bool,
        allowed_roles: Sequence[str],
    ) -> list[Recommendation]:
        if conn is None or ctx is None:
            return recommendations
        allowed = set(
            postgres.validate_candidates(
                conn,
                target_type,
                [recommendation.doc_id for recommendation in recommendations],
                ctx,
                require_ready=require_ready,
                allowed_roles=allowed_roles,
            )
        )
        return [r for r in recommendations if r.doc_id in allowed]

    # ------------------------------------------------------------------ #
    # Scoring helpers
    # ------------------------------------------------------------------ #

    def _score_semantic(
        self,
        results,
        keyword_scores: Mapping[str, float] | None,
        alpha: float,
    ) -> list[Recommendation]:
        semantic = {result.doc_id: result for result in results}
        fused = self.ranker.fuse(
            {doc_id: result.semantic for doc_id, result in semantic.items()},
            keyword_scores,
            alpha,
        )
        return [
            self._to_recommendation(
                doc_id,
                fused[doc_id],
                semantic.get(doc_id),
                None if keyword_scores is None else keyword_scores.get(doc_id),
            )
            for doc_id in fused
        ]

    def _score_hybrid(
        self,
        query_doc: Document,
        target_type: str,
        results,
        documents: Mapping[str, Document] | None,
        profiles: Mapping[str, CandidateProfile] | None,
        keyword_scores: Mapping[str, float] | None,
        alpha: float,
    ) -> list[Recommendation]:
        """Chấm hybrid theo đúng chiều: JD là requirement, CV là profile.

        - target_type "cv" (JD -> CV): requirement từ query, profile của candidate.
        - target_type "jd" (CV -> JD): candidate là JD (requirement), query là profile.
        """
        query_requirement = job_from_document(query_doc) if target_type == "cv" else None
        query_profile = profile_from_document(query_doc) if target_type == "jd" else None

        candidates: list[RankedCandidate] = []
        for result in results:
            if target_type == "cv":
                requirement = query_requirement
                profile = self._lookup_profile(result.doc_id, documents, profiles)
            else:
                candidate_doc = None if documents is None else documents.get(result.doc_id)
                if candidate_doc is None:
                    continue
                requirement = job_from_document(candidate_doc)
                profile = query_profile
            if requirement is None or profile is None:
                continue
            candidates.append(
                self.ranker.score(
                    requirement, profile, result.semantic, result.doc_id, result.components
                )
            )

        return self._finalize(candidates, keyword_scores, alpha)

    def _finalize(
        self,
        candidates: Sequence[RankedCandidate],
        keyword_scores: Mapping[str, float] | None,
        alpha: float,
    ) -> list[Recommendation]:
        """Áp fusion semantic↔keyword (nếu có) và chuyển sang `Recommendation`."""
        if not keyword_scores:
            return [self._from_ranked(c, c.score, None) for c in candidates]

        fused = self.ranker.fuse(
            {candidate.doc_id: candidate.score for candidate in candidates},
            keyword_scores,
            alpha,
        )
        return [
            self._from_ranked(
                candidate, fused[candidate.doc_id], keyword_scores.get(candidate.doc_id)
            )
            for candidate in candidates
            if candidate.doc_id in fused
        ]

    def _lookup_profile(
        self,
        doc_id: str,
        documents: Mapping[str, Document] | None,
        profiles: Mapping[str, CandidateProfile] | None,
    ) -> CandidateProfile | None:
        if profiles is not None and doc_id in profiles:
            return profiles[doc_id]
        if documents is not None and doc_id in documents:
            return profile_from_document(documents[doc_id])
        return None

    @staticmethod
    def _to_recommendation(
        doc_id: str,
        score: float,
        result,
        keyword: float | None,
    ) -> Recommendation:
        return Recommendation(
            doc_id=doc_id,
            score=score,
            semantic=0.0 if result is None else result.semantic,
            skills=0.0,
            experience=0.0,
            budget=0.0,
            preferences=0.0,
            section_components={} if result is None else dict(result.components),
            keyword=keyword,
        )

    @staticmethod
    def _from_ranked(
        candidate: RankedCandidate, score: float, keyword: float | None
    ) -> Recommendation:
        return Recommendation(
            doc_id=candidate.doc_id,
            score=score,
            semantic=candidate.semantic,
            skills=candidate.skills,
            experience=candidate.experience,
            budget=candidate.budget,
            preferences=candidate.preferences,
            section_components=dict(candidate.section_components),
            keyword=keyword,
        )


__all__ = ["MatchingService", "Recommendation"]
