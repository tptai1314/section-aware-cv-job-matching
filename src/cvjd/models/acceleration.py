"""Embedding compression and two-stage retrieval (acceleration, RQ3)."""

from __future__ import annotations

from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np

from ..schema import Document
from .embedding import Embedder
from .factory import RankOne, Ranking, sorted_scores
from .matching import Aggregation, Pairing, embed_corpus, score_pair

Compression = Literal["none", "pca", "int8"]


def quantize_int8(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8 quantization -> (codes, scales)."""
    if vectors.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {vectors.shape}")
    scales = np.max(np.abs(vectors), axis=1) / 127.0
    scales = np.where(scales == 0.0, 1.0, scales)
    codes = np.clip(np.round(vectors / scales[:, None]), -127, 127).astype(np.int8)
    return codes, scales


def dequantize_int8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * scales[:, None]


def fit_pca(vectors: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA on `vectors`, return (components, transformed)."""
    from sklearn.decomposition import PCA

    n = min(n_components, vectors.shape[0], vectors.shape[1])
    if n < 1:
        raise ValueError("no components available for PCA")
    pca = PCA(n_components=n, svd_solver="full")
    return pca, pca.fit_transform(vectors)


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    m = matrix.astype(np.float64)
    qn = np.linalg.norm(q)
    mn = np.linalg.norm(m, axis=1)
    denom = qn * mn
    safe = np.where(denom == 0.0, 1.0, denom)
    scores = (m @ q) / safe
    return np.where(denom == 0.0, 0.0, scores)


def _apply_compression(
    jd_vecs: np.ndarray,
    cv_vecs: np.ndarray,
    compression: Compression,
    pca_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if compression == "none":
        return jd_vecs, cv_vecs
    if compression == "int8":
        jd_codes, jd_scales = quantize_int8(jd_vecs)
        cv_codes, cv_scales = quantize_int8(cv_vecs)
        return dequantize_int8(jd_codes, jd_scales), dequantize_int8(cv_codes, cv_scales)
    if compression == "pca":
        combined = np.vstack([jd_vecs, cv_vecs])
        pca, transformed = fit_pca(combined, pca_dim)
        return transformed[: len(jd_vecs)], transformed[len(jd_vecs):]
    raise ValueError(f"unknown compression: {compression!r}")


def prepare_compressed_single(
    jds: Sequence[Document],
    cvs: Sequence[Document],
    embedder: Embedder,
    compression: Compression = "none",
    pca_dim: int = 64,
) -> RankOne:
    """Single-vector ranking over compressed full-text embeddings."""
    jd_emb = embed_corpus(jds, embedder)
    cv_emb = embed_corpus(cvs, embedder)
    from .matching import FULL_KEY

    jd_ids = [jd.doc_id for jd in jds]
    jd_matrix = np.stack([jd_emb[jd_id][FULL_KEY] for jd_id in jd_ids])
    cv_ids = [cv.doc_id for cv in cvs]
    cv_matrix = np.stack([cv_emb[cv_id][FULL_KEY] for cv_id in cv_ids])
    jd_matrix, cv_matrix = _apply_compression(jd_matrix, cv_matrix, compression, pca_dim)

    def rank_one(jd_id: str) -> Ranking:
        idx = jd_ids.index(jd_id)
        scores = _cosine_scores(jd_matrix[idx], cv_matrix)
        return sorted_scores(list(zip(cv_ids, (float(s) for s in scores))))

    return rank_one


def prepare_two_stage(
    jds: Sequence[Document],
    cvs: Sequence[Document],
    embedder: Embedder,
    top_k: int = 10,
    pairing: Pairing = "cross",
    aggregation: Aggregation = "mean",
    stage1_compression: Compression = "none",
    pca_dim: int = 64,
) -> RankOne:
    """Cheap single-vector top-k retrieval, section-aware rerank of the head."""
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    stage1 = prepare_compressed_single(
        jds, cvs, embedder, compression=stage1_compression, pca_dim=pca_dim
    )
    jd_emb = embed_corpus(jds, embedder)
    cv_emb = embed_corpus(cvs, embedder)

    def rank_one(jd_id: str) -> Ranking:
        first: Ranking = stage1(jd_id)
        head = first[:top_k]
        tail = first[top_k:]
        reranked = [
            (cv_id, score_pair(
                jd_emb[jd_id], cv_emb[cv_id],
                method="section_aware", pairing=pairing, aggregation=aggregation,
            ))
            for cv_id, _ in head
        ]
        return sorted_scores(reranked) + tail

    return rank_one
