"""Build a rank-one callable for any supported first-stage method."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from ..schema import Document
from .bm25 import BM25Ranker
from .chunking import embed_chunks, max_similarity
from .embedding import Embedder
from .matching import embed_corpus, rank_query

Ranking = List[Tuple[str, float]]
RankOne = Callable[[str], Ranking]

METHODS = (
    "bm25",
    "single_vector",
    "section_aware",
    "chunk_max_sim",
    "single_pca",
    "single_int8",
    "two_stage",
)


def sorted_scores(scored: List[Tuple[str, float]]) -> Ranking:
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def prepare_ranker(
    method: str,
    jds: Sequence[Document],
    cvs: Sequence[Document],
    embedder: Embedder | None = None,
    pairing: str = "cross",
    aggregation: str = "mean",
    chunk_max_tokens: int = 64,
    chunk_overlap: int = 16,
    pca_dim: int = 64,
    top_k_retrieve: int = 10,
    stage1_compression: str = "none",
) -> RankOne:
    """Build the index/embeddings once, then rank any single JD by id."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}; choose from {METHODS}")

    if method == "bm25":
        ranker = BM25Ranker(cvs)
        jd_by_id = {jd.doc_id: jd for jd in jds}
        return lambda jd_id: ranker.rank(jd_by_id[jd_id])

    if embedder is None:
        raise ValueError(f"embedder is required for method {method!r}")

    if method == "chunk_max_sim":
        jd_chunks = embed_chunks(jds, embedder, chunk_max_tokens, chunk_overlap)
        cv_chunks = embed_chunks(cvs, embedder, chunk_max_tokens, chunk_overlap)
        return lambda jd_id: sorted_scores([
            (cv_id, max_similarity(jd_chunks[jd_id], cv_chunks[cv_id]))
            for cv_id in cv_chunks
        ])

    if method in ("single_pca", "single_int8"):
        from .acceleration import prepare_compressed_single

        compression = "pca" if method == "single_pca" else "int8"
        return prepare_compressed_single(
            jds, cvs, embedder, compression=compression, pca_dim=pca_dim
        )

    if method == "two_stage":
        from .acceleration import prepare_two_stage

        return prepare_two_stage(
            jds, cvs, embedder,
            top_k=top_k_retrieve,
            pairing=pairing,
            aggregation=aggregation,
            stage1_compression=stage1_compression,
            pca_dim=pca_dim,
        )

    jd_emb = embed_corpus(jds, embedder)
    cv_emb = embed_corpus(cvs, embedder)
    return lambda jd_id: rank_query(
        jd_emb[jd_id], cv_emb, method=method, pairing=pairing, aggregation=aggregation
    )


def build_rankings(
    method: str,
    jds: Sequence[Document],
    cvs: Sequence[Document],
    embedder: Embedder | None = None,
    **kwargs,
) -> Dict[str, Ranking]:
    rank_one = prepare_ranker(method, jds, cvs, embedder, **kwargs)
    return {jd.doc_id: rank_one(jd.doc_id) for jd in jds}
