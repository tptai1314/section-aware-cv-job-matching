"""Rankers: section-aware multi-vector and single-vector baseline."""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np

from ..schema import Document
from .chunking import rank_all_chunk
from .embedding import Embedder

FULL_KEY = "_full"

Pairing = Literal["fixed", "cross"]
Aggregation = Literal["max", "mean"]
Method = Literal["section_aware", "single_vector", "chunk_max_sim"]

SectionVectors = Dict[str, np.ndarray]
CorpusEmbeddings = Dict[str, SectionVectors]


def embed_corpus(
    docs: Iterable[Document], embedder: Embedder
) -> CorpusEmbeddings:
    """Embed every section plus the full text of each document."""
    docs = list(docs)
    keys_per_doc: List[List[str]] = []
    texts: List[str] = []
    for doc in docs:
        keys = sorted(
            doc.sections,
            key=lambda name: doc.sections[name].order,
        )
        keys_per_doc.append(keys + [FULL_KEY])
        texts.extend(doc.sections[name].text for name in keys)
        texts.append(doc.raw_text)

    vectors = embedder.encode(texts)

    out: CorpusEmbeddings = {}
    offset = 0
    for doc, keys in zip(docs, keys_per_doc):
        out[doc.doc_id] = {
            key: vectors[offset + i] for i, key in enumerate(keys)
        }
        offset += len(keys)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def section_components(
    jd_emb: SectionVectors,
    cv_emb: SectionVectors,
    pairing: Pairing = "cross",
) -> Dict[str, float]:
    """Per-section similarity of one (JD, CV) pair.

    fixed: each JD section is compared only with the same-name CV section.
    cross: each JD section takes its best match among all CV sections.
    """
    jd_names = [n for n in jd_emb if n != FULL_KEY]
    cv_names = [n for n in cv_emb if n != FULL_KEY]

    components: Dict[str, float] = {}
    for name in jd_names:
        if pairing == "fixed":
            other = cv_emb.get(name)
            components[name] = cosine(jd_emb[name], other) if other is not None else 0.0
        else:
            components[name] = max(
                (cosine(jd_emb[name], cv_emb[other]) for other in cv_names),
                default=0.0,
            )
    return components


def aggregate(components: Dict[str, float], aggregation: Aggregation = "mean") -> float:
    if not components:
        return 0.0
    values = list(components.values())
    if aggregation == "max":
        return float(max(values))
    return float(sum(values) / len(values))


def score_pair(
    jd_emb: SectionVectors,
    cv_emb: SectionVectors,
    method: Method = "section_aware",
    pairing: Pairing = "cross",
    aggregation: Aggregation = "mean",
) -> float:
    if method == "single_vector":
        return cosine(jd_emb[FULL_KEY], cv_emb[FULL_KEY])
    if method == "section_aware":
        return aggregate(section_components(jd_emb, cv_emb, pairing), aggregation)
    if method == "chunk_max_sim":
        raise ValueError("chunk_max_sim is only supported through rank_all")
    raise ValueError(f"unknown method: {method!r}")


def rank_query(
    jd_emb: SectionVectors,
    cv_embs: CorpusEmbeddings,
    method: Method = "section_aware",
    pairing: Pairing = "cross",
    aggregation: Aggregation = "mean",
) -> List[Tuple[str, float]]:
    """Score one JD against all CVs, best first (ties broken by cv_id)."""
    scored = [
        (cv_id, score_pair(jd_emb, emb, method, pairing, aggregation))
        for cv_id, emb in cv_embs.items()
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def rank_all(
    jds: Sequence[Document],
    cvs: Sequence[Document],
    embedder: Embedder,
    method: Method = "section_aware",
    pairing: Pairing = "cross",
    aggregation: Aggregation = "mean",
    chunk_max_tokens: int = 64,
    chunk_overlap: int = 16,
) -> Dict[str, List[Tuple[str, float]]]:
    if method == "chunk_max_sim":
        return rank_all_chunk(jds, cvs, embedder, chunk_max_tokens, chunk_overlap)
    jd_embeddings = embed_corpus(jds, embedder)
    cv_embeddings = embed_corpus(cvs, embedder)
    return {
        jd_id: rank_query(jd_embeddings[jd_id], cv_embeddings, method, pairing, aggregation)
        for jd_id in jd_embeddings
    }
