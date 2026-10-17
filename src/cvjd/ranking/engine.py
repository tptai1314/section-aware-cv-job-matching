"""Hybrid ranking engine (plan §7): fusion của semantic + keyword + cấu trúc.

Kết hợp `RetrievalResult` (semantic theo section) với `hybrid_score` (skills,
experience, budget, preferences) để ra thứ hạng cuối cùng. Tách biệt khỏi
`hybrid.py` (thuần công thức) — engine này gom candidate, gọi fusion và xuất
`RankedCandidate` kèm giải thích từng thành phần.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..utils.config import get_ranking_weights
from .hybrid import (
    DEFAULT_WEIGHTS,
    CandidateProfile,
    JobRequirement,
    hybrid_score,
    job_from_document,
    profile_from_document,
)

COMPONENTS = ("semantic", "skills", "experience", "budget", "preferences")


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def min_max(values):
    """Chuẩn hoá min-max về [0, 1]; dãy hằng -> tất cả 1.0."""
    values = list(values)
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 0:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


@dataclass(frozen=True)
class RankedCandidate:
    doc_id: str
    score: float
    semantic: float
    skills: float
    experience: float
    budget: float
    preferences: float
    section_components: dict[str, float] = field(default_factory=dict)

    def as_breakdown(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "skills": self.skills,
            "experience": self.experience,
            "budget": self.budget,
            "preferences": self.preferences,
        }


class HybridRanker:
    """Fuse semantic (Section-aware) với coverage/experience/budget/preferences."""

    def __init__(self, weights: dict[str, float] | None = None, config: dict | None = None):
        if weights is not None:
            self.weights = dict(weights)
        elif config is not None:
            self.weights = get_ranking_weights(config)
        else:
            self.weights = get_ranking_weights()
        if not self.weights:
            self.weights = dict(DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------ #
    # Fusion semantic <-> keyword (B5)
    # ------------------------------------------------------------------ #

    def fuse(
        self,
        semantic_scores,
        keyword_scores=None,
        alpha: float = 0.5,
        semantic_ids=None,
        keyword_ids=None,
    ):
        """Fuse hai bảng điểm về cùng tập id: alpha*sem + (1-alpha)*keyword.

        `semantic_scores`/`keyword_scores` là mapping doc_id -> điểm (chưa chuẩn
        hoá); thiếu ở một bên thì coi là 0 sau khi chuẩn hoá trên hợp các id.
        """
        sem = dict(semantic_scores or {})
        kw = {} if keyword_scores is None else dict(keyword_scores)
        if keyword_scores is None:
            return {doc_id: clamp01(score) for doc_id, score in sem.items()}

        ids = list(dict.fromkeys([*sem, *kw]))
        sem_norm = dict(zip(ids, min_max([sem.get(i, 0.0) for i in ids])))
        kw_norm = dict(zip(ids, min_max([kw.get(i, 0.0) for i in ids])))
        alpha = clamp01(alpha)
        return {
            doc_id: clamp01(alpha * sem_norm[doc_id] + (1.0 - alpha) * kw_norm[doc_id])
            for doc_id in ids
        }

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score(
        self,
        requirement: JobRequirement,
        profile: CandidateProfile,
        semantic: float,
        doc_id: str = "",
        section_components: dict[str, float] | None = None,
    ) -> RankedCandidate:
        req = replace(requirement, semantic_base=clamp01(semantic))
        breakdown = hybrid_score(req, profile, self.weights)
        return RankedCandidate(
            doc_id=doc_id,
            score=breakdown["overall"],
            semantic=breakdown["semantic"],
            skills=breakdown["skills"],
            experience=breakdown["experience"],
            budget=breakdown["budget"],
            preferences=breakdown["preferences"],
            section_components=dict(section_components or {}),
        )

    def rank(self, requirement: JobRequirement, items, top_k: int | None = None):
        """`items`: iterable (doc_id, semantic, profile[, section_components])."""
        ranked: list[RankedCandidate] = []
        for item in items:
            doc_id, semantic, profile = item[0], item[1], item[2]
            components = item[3] if len(item) > 3 else None
            ranked.append(self.score(requirement, profile, semantic, doc_id, components))
        ranked.sort(key=lambda c: (-c.score, c.doc_id))
        return ranked[:top_k] if top_k else ranked

    def rank_documents(
        self,
        job_doc,
        retrieval_results,
        cv_documents,
        top_k: int | None = None,
    ):
        """Bridge `Document`: JD + list `RetrievalResult` + dict cv_id -> Document."""
        requirement = job_from_document(job_doc)
        items = []
        for result in retrieval_results:
            cv_doc = cv_documents.get(result.doc_id)
            if cv_doc is None:
                continue
            profile = profile_from_document(cv_doc)
            items.append(
                (result.doc_id, result.semantic, profile, result.components)
            )
        return self.rank(requirement, items, top_k)


__all__ = [
    "COMPONENTS",
    "HybridRanker",
    "RankedCandidate",
    "clamp01",
    "min_max",
]
