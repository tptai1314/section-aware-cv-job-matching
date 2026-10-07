"""Embedding backends: MockEmbedder (deterministic) and SentenceTransformer."""

from __future__ import annotations

import re
import zlib
from typing import Iterable, Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that maps a list of texts to an (n, dim) float32 matrix."""

    dim: int

    def encode(self, texts: Iterable[str]) -> np.ndarray: ...


class MockEmbedder:
    """Deterministic hashing embedder (no model download).

    Each token is hashed into a dimension; texts sharing tokens end up with
    higher cosine similarity, so the mock data keeps a hidden semantic signal.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in _TOKEN_RE.findall(text.lower()):
                idx = zlib.crc32(token.encode("utf-8")) % self.dim
                counts[idx] = counts.get(idx, 0.0) + 1.0
            for idx, count in counts.items():
                out[i, idx] = 1.0 + np.log(count)
            norm = float(np.linalg.norm(out[i]))
            if norm > 0.0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    """Frozen SentenceTransformer on CPU."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder"
            ) from e
        self.model = SentenceTransformer(model_name, device=device)
        getter = getattr(self.model, "get_embedding_dimension", None)
        if getter is None:
            getter = self.model.get_sentence_embedding_dimension
        self.dim = int(getter())

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def create_embedder(name: str = "mock", **kwargs) -> Embedder:
    if name == "mock":
        return MockEmbedder(**kwargs)
    if name in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder: {name!r}")
