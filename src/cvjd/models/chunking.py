"""Fixed-size chunking with max-similarity scoring (chunk baseline)."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..schema import Document
from .embedding import Embedder

_WORD_RE = re.compile(r"\S+")


def chunk_text(text: str, max_tokens: int = 64, overlap: int = 16) -> List[str]:
    """Sliding-window word chunks; the last window always includes the tail."""
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError(f"overlap must be in [0, {max_tokens}), got {overlap}")
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return []
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]
    step = max_tokens - overlap
    chunks: List[str] = []
    start = 0
    while True:
        chunks.append(" ".join(tokens[start:start + max_tokens]))
        if start + max_tokens >= len(tokens):
            break
        start += step
    return chunks


def embed_chunks(
    docs: Sequence[Document],
    embedder: Embedder,
    max_tokens: int = 64,
    overlap: int = 16,
) -> Dict[str, List[np.ndarray]]:
    """Embed every chunk of every document in one batched encode call."""
    texts: List[str] = []
    counts: List[int] = []
    for doc in docs:
        chunks = chunk_text(doc.raw_text, max_tokens, overlap) or [""]
        counts.append(len(chunks))
        texts.extend(chunks)
    vectors = embedder.encode(texts)
    out: Dict[str, List[np.ndarray]] = {}
    offset = 0
    for doc, n in zip(docs, counts):
        out[doc.doc_id] = [vectors[offset + i] for i in range(n)]
        offset += n
    return out


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def max_similarity(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> float:
    """Max cosine over all chunk pairs, 0.0 on zero vectors."""
    if not a or not b:
        return 0.0
    left = _l2_normalize(np.stack(a).astype(np.float64))
    right = _l2_normalize(np.stack(b).astype(np.float64))
    return float(np.max(left @ right.T))


def rank_all_chunk(
    jds: Sequence[Document],
    cvs: Sequence[Document],
    embedder: Embedder,
    max_tokens: int = 64,
    overlap: int = 16,
) -> Dict[str, List[Tuple[str, float]]]:
    jd_chunks = embed_chunks(jds, embedder, max_tokens, overlap)
    cv_chunks = embed_chunks(cvs, embedder, max_tokens, overlap)
    rankings: Dict[str, List[Tuple[str, float]]] = {}
    for jd in jds:
        scored = [
            (cv_id, max_similarity(jd_chunks[jd.doc_id], cv_chunks[cv_id]))
            for cv_id in cv_chunks
        ]
        rankings[jd.doc_id] = sorted(scored, key=lambda item: (-item[1], item[0]))
    return rankings
