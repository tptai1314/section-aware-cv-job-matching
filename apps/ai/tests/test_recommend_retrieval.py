"""Retrieval mode: `/internal/index` -> Chroma -> `/internal/recommend` (no candidates)."""

import pytest
from app import recommend as rec
from app.config import Settings
from app.main import create_app
from starlette.testclient import TestClient

from cvjd.models.embedding import MockEmbedder
from cvjd.service import MatchingService
from cvjd.storage import chroma


def make_settings() -> Settings:
    return Settings(
        ollama_url="http://ollama:11434",
        ollama_model="qwen3:4b",
        ollama_timeout_s=5,
        max_upload_mb=1,
        max_pages=20,
        max_llm_chars=24000,
        ocr_enabled=False,
        ocr_langs="vie+eng",
        strict_evidence=False,
        allow_legacy_doc=False,
        embed_model="mock",
        embed_device="cpu",
        log_level="INFO",
        recommend_pool_size=50,
        recommend_top_k=10,
    )


@pytest.fixture()
def client():
    return TestClient(create_app(make_settings()))


@pytest.fixture()
def fake_backend(monkeypatch):
    chroma_client = chroma.build_client(None)
    chroma.ensure_collections(chroma_client)
    service = MatchingService(
        chroma_client, MockEmbedder(dim=32), embed_model="mock", config={}
    )
    store = rec.DocumentStore(None)
    monkeypatch.setattr(rec, "get_service", lambda settings: service)
    monkeypatch.setattr(rec, "get_store", lambda settings: store)
    return service, store


def test_index_then_retrieval_recommend(client, fake_backend):
    indexed = client.post(
        "/internal/index",
        json={
            "documents": [
                {
                    "id": "cv_a",
                    "type": "cv",
                    "skills": [{"name": "python"}, {"name": "postgresql"}],
                    "experience": [
                        {"title": "Backend", "start_date": "2019-01", "end_date": "2024-01"}
                    ],
                },
                {
                    "id": "cv_b",
                    "type": "cv",
                    "skills": [{"name": "react"}, {"name": "css"}],
                },
            ]
        },
    )
    assert indexed.status_code == 200
    assert indexed.json()["indexed"] == 2

    response = client.post(
        "/internal/recommend",
        json={
            "direction": "jd2cv",
            "query": {
                "required_skills": [{"name": "python"}, {"name": "postgresql"}],
                "min_years_experience": 3,
            },
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results and results[0]["id"] == "cv_a"
    assert "skills" in results[0]["section_components"]


def test_index_document_requires_type(client):
    response = client.post("/internal/index", json={"documents": [{"id": "x"}]})
    assert response.status_code == 422


def test_retrieval_empty_index_returns_empty(client, fake_backend):
    response = client.post(
        "/internal/recommend",
        json={"direction": "jd2cv", "query": {"required_skills": [{"name": "go"}]}},
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_retrieval_recommend_with_explain(client, fake_backend, monkeypatch):
    indexed = client.post(
        "/internal/index",
        json={
            "documents": [
                {
                    "id": "cv_a",
                    "type": "cv",
                    "skills": [{"name": "python"}, {"name": "postgresql"}],
                }
            ]
        },
    )
    assert indexed.status_code == 200

    def fake_chat(system, user):
        return {
            "summary": "Khớp kỹ năng",
            "claims": [
                {"statement": "Có python", "verdict": "hit", "evidence": "python", "source": "cv"}
            ],
        }

    monkeypatch.setattr(client.app.state.llm, "chat", fake_chat)
    response = client.post(
        "/internal/recommend",
        json={
            "direction": "jd2cv",
            "query": {"required_skills": [{"name": "python"}]},
            "top_k": 5,
            "explain": True,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results and results[0]["id"] == "cv_a"
    assert results[0]["explanation"]["ready"] is True
    assert results[0]["explanation"]["summary"] == "Khớp kỹ năng"


def test_retrieval_respects_visibility(client, fake_backend):
    response = client.post(
        "/internal/index",
        json={
            "documents": [
                {
                    "id": "cv_pub",
                    "type": "cv",
                    "visibility": "public",
                    "skills": [{"name": "python"}, {"name": "postgres"}],
                },
                {
                    "id": "cv_priv",
                    "type": "cv",
                    "owner_id": "u2",
                    "visibility": "private",
                    "skills": [{"name": "python"}, {"name": "fastapi"}],
                },
            ]
        },
    )
    assert response.status_code == 200

    body = {"direction": "jd2cv", "query": {"required_skills": [{"name": "python"}]}}
    guest = client.post("/internal/recommend", json=body).json()["results"]
    ids = {item["id"] for item in guest}
    assert "cv_pub" in ids
    assert "cv_priv" not in ids

    owner = client.post(
        "/internal/recommend", json={**body, "owner_id": "u2"}
    ).json()["results"]
    assert "cv_priv" in {item["id"] for item in owner}

