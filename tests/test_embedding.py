"""Unit tests for the embedder factory (no model downloads)."""

from __future__ import annotations

import cvjd.models.embedding as emb
from cvjd.models.embedding import BGE_M3_MODEL_ID, create_embedder


def test_create_embedder_bge_m3_maps_to_bge(monkeypatch):
    calls: dict = {}

    class Fake(emb.SentenceTransformerEmbedder):
        def __init__(self, model_name="", device="cpu", trust_remote_code=False):
            calls["model_name"] = model_name
            calls["trust_remote_code"] = trust_remote_code
            self.dim = 1024

    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", Fake)

    embedder = create_embedder("bge-m3")
    assert calls["model_name"] == BGE_M3_MODEL_ID
    assert calls["trust_remote_code"] is True
    assert embedder.dim == 1024


def test_create_embedder_bge_alias_same_behavior(monkeypatch):
    calls: dict = {}

    class Fake(emb.SentenceTransformerEmbedder):
        def __init__(self, model_name="", device="cpu", trust_remote_code=False):
            calls["model_name"] = model_name
            calls["trust_remote_code"] = trust_remote_code
            self.dim = 1024

    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", Fake)

    create_embedder("bge")
    assert calls["model_name"] == BGE_M3_MODEL_ID


def test_create_embedder_unknown_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown embedder"):
        create_embedder("nope")