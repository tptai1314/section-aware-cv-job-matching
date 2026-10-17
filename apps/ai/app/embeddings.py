"""Embedding văn bản bằng BGE-M3 (lazy singleton)."""

from __future__ import annotations

import threading

from . import errors

_lock = threading.Lock()
_embedder = None


class Embedder:
    def __init__(self, model_id: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
        self.model_id = model_id
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [vec.tolist() for vec in vectors]


def get_embedder(model_id: str | None = None, device: str = "cpu") -> Embedder:
    global _embedder
    with _lock:
        if _embedder is None or (model_id and _embedder.model_id != model_id):
            _embedder = Embedder(model_id or "BAAI/bge-m3", device)
        return _embedder


def embed(texts: list[str], model_id: str | None = None, device: str = "cpu") -> list[list[float]]:
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise errors.cv_file_invalid()
    return get_embedder(model_id, device).encode(texts)