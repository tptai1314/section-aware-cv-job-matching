"""Cross-encoder reranking: heavy upper bound and two-stage pipeline."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..schema import Document
from .factory import Ranking

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderRanker:
    """Frozen cross-encoder scoring (JD, CV) text pairs on CPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderRanker"
            ) from e
        self.model = CrossEncoder(model_name, device=device)

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict(list(pairs))
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def rank_all_cross_encoder(
    jds: Sequence[Document],
    cvs: Sequence[Document],
    ranker: Optional[CrossEncoderRanker] = None,
) -> Dict[str, Ranking]:
    """Upper bound: cross-encode every (JD, CV) pair directly."""
    ranker = ranker or CrossEncoderRanker()
    cv_ids = [cv.doc_id for cv in cvs]
    cv_texts = {cv.doc_id: cv.raw_text for cv in cvs}
    out: Dict[str, Ranking] = {}
    for jd in jds:
        pairs = [(jd.raw_text, cv_texts[cv_id]) for cv_id in cv_ids]
        scores = ranker.score_pairs(pairs)
        out[jd.doc_id] = sorted(
            zip(cv_ids, (float(s) for s in scores)),
            key=lambda item: (-item[1], item[0]),
        )
    return out


def rerank_query(
    jd: Document,
    ranking: Ranking,
    cvs_by_id: Mapping[str, Document],
    ranker: CrossEncoderRanker,
    top_k: Optional[int] = None,
) -> Ranking:
    """Cross-encode the top-k candidates of one ranking; the tail keeps
    its first-stage order behind the reranked head."""
    head = ranking if top_k is None else ranking[:top_k]
    tail = ranking if top_k is None else ranking[top_k:]
    pairs = [(jd.raw_text, cvs_by_id[cv_id].raw_text) for cv_id, _ in head]
    scores = ranker.score_pairs(pairs)
    reranked = sorted(
        zip([cv_id for cv_id, _ in head], (float(s) for s in scores)),
        key=lambda item: (-item[1], item[0]),
    )
    if top_k is None:
        return reranked
    return reranked + tail


def rerank(
    first_stage: Dict[str, Ranking],
    jds: Sequence[Document],
    cvs: Sequence[Document],
    ranker: Optional[CrossEncoderRanker] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Ranking]:
    ranker = ranker or CrossEncoderRanker()
    jd_by_id = {jd.doc_id: jd for jd in jds}
    cvs_by_id = {cv.doc_id: cv for cv in cvs}
    return {
        jd_id: rerank_query(jd_by_id[jd_id], ranking, cvs_by_id, ranker, top_k)
        for jd_id, ranking in first_stage.items()
    }
