import pytest
from app.config import Settings
from app.main import create_app
from starlette.testclient import TestClient


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
        embed_model="BAAI/bge-m3",
        embed_device="cpu",
        log_level="INFO",
    )


@pytest.fixture()
def client():
    app = create_app(make_settings())
    return TestClient(app)


def test_recommend_jd2cv_ranks_matching_candidate(client):
    body = {
        "direction": "jd2cv",
        "query": {
            "required_skills": [{"name": "React"}, {"name": "Node"}],
            "min_years_experience": 3,
        },
        "candidates": [
            {
                "id": "c1",
                "skills": [{"name": "reactjs"}, {"name": "node"}],
                "years_experience": 5,
                "semantic": 0.9,
            },
            {"id": "c2", "skills": [{"name": "PHP"}], "years_experience": 1, "semantic": 0.2},
        ],
        "top_k": 5,
    }
    r = client.post("/internal/recommend", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["direction"] == "jd2cv"
    assert [item["id"] for item in payload["results"]] == ["c1", "c2"]
    assert payload["results"][0]["components"]["skills"] == 1.0


def test_recommend_cv2jd_with_keyword(client):
    body = {
        "direction": "cv2jd",
        "query": {"skills": [{"name": "React"}], "years_experience": 4},
        "candidates": [
            {"id": "j1", "required_skills": [{"name": "React"}], "min_years_experience": 2},
            {"id": "j2", "required_skills": [{"name": "Docker"}], "min_years_experience": 5},
        ],
        "keyword_scores": {"j1": 0.1, "j2": 0.9},
        "keyword_alpha": 0.5,
        "top_k": 5,
    }
    r = client.post("/internal/recommend", json=body)
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["id"] == "j1"
    assert results[0]["components"]["keyword"] == 0.1


def test_recommend_rejects_empty_candidates(client):
    r = client.post(
        "/internal/recommend",
        json={"direction": "jd2cv", "query": {}, "candidates": []},
    )
    assert r.status_code == 422


def test_recommend_rejects_bad_direction(client):
    r = client.post(
        "/internal/recommend",
        json={"direction": "sideways", "query": {}, "candidates": [{"id": "c1"}]},
    )
    assert r.status_code == 422


def test_recommend_with_explain_attaches_only_top_n(client, monkeypatch):
    def fake_chat(system, user):
        return {
            "summary": "Phù hợp",
            "claims": [
                {"statement": "Có React", "verdict": "hit", "evidence": "React", "source": "cv", "confidence": 0.9}
            ],
        }

    monkeypatch.setattr(client.app.state.llm, "chat", fake_chat)
    body = {
        "direction": "jd2cv",
        "query": {"required_skills": [{"name": "React"}], "min_years_experience": 1},
        "candidates": [
            {"id": "c1", "skills": [{"name": "React"}], "years_experience": 3, "semantic": 0.9},
            {"id": "c2", "skills": [{"name": "PHP"}], "years_experience": 1, "semantic": 0.2},
        ],
        "top_k": 5,
        "explain": True,
        "explain_top_n": 1,
    }
    r = client.post("/internal/recommend", json=body)
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["id"] == "c1"
    assert results[0]["explanation"]["ready"] is True
    assert results[0]["explanation"]["summary"] == "Phù hợp"
    assert results[0]["explanation"]["claims"][0]["verdict"] == "hit"
    assert results[1]["explanation"] is None


def test_recommend_without_explain_has_no_explanation(client):
    body = {
        "direction": "jd2cv",
        "query": {"required_skills": [{"name": "React"}]},
        "candidates": [{"id": "c1", "skills": [{"name": "React"}], "semantic": 0.9}],
        "top_k": 5,
    }
    r = client.post("/internal/recommend", json=body)
    assert r.status_code == 200
    assert r.json()["results"][0]["explanation"] is None


def test_recommend_explain_llm_error_falls_back(client, monkeypatch):
    from app import errors as errors_mod

    def boom(system, user):
        raise errors_mod.llm_unavailable()

    monkeypatch.setattr(client.app.state.llm, "chat", boom)
    body = {
        "direction": "jd2cv",
        "query": {"required_skills": [{"name": "React"}]},
        "candidates": [{"id": "c1", "skills": [{"name": "React"}], "semantic": 0.9}],
        "top_k": 5,
        "explain": True,
    }
    r = client.post("/internal/recommend", json=body)
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["explanation"]["ready"] is False
    assert results[0]["explanation"]["fallback_reason"] == "LLM_UNAVAILABLE"
