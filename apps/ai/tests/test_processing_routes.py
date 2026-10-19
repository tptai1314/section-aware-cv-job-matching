"""Test endpoint hàng đợi: POST/GET/DELETE /internal/jobs + retry (design §4)."""

import dataclasses
import time

import pytest
from app import recommend as recommend_mod
from app import routes as routes_mod
from app.config import Settings
from app.main import create_app
from app.processing import pipeline as pipeline_mod
from app.processing.store import CANCELLED, FAILED, PENDING, READY
from starlette.testclient import TestClient

LONG_TEXT = "Senior React developer with 6 years building web applications at Acme Corp."

BASE = Settings(
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
    job_max_attempts=3,
    job_retry_base_delay_s=0.0,
    job_poll_interval_s=0.05,
)


def make_settings(tmp_path, **overrides) -> Settings:
    defaults = {
        "processing_db_path": str(tmp_path / "jobs.db"),
        "processing_files_dir": str(tmp_path / "files"),
    }
    return dataclasses.replace(BASE, **(defaults | overrides))


@pytest.fixture()
def store(tmp_path):
    app = create_app(make_settings(tmp_path))
    return app.state.processing_store


@pytest.fixture()
def client(tmp_path):
    app = create_app(make_settings(tmp_path))
    return TestClient(app)


def test_readyz_reports_queue_depth_and_worker(client):
    """/readyz thêm check store + queue depth (design §8)."""
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "unavailable")
    assert body["queue_depth"] == 0
    assert body["worker_alive"] is False  # fixture dựng app, không chạy lifespan

    client.post("/internal/jobs", data={"document_type": "jd", "document_id": "jd_r",
                                        "text": LONG_TEXT})
    assert client.get("/readyz").json()["queue_depth"] == 1


def test_create_jd_job_returns_202(client):
    r = client.post(
        "/internal/jobs",
        data={"document_type": "jd", "document_id": "jd_1", "text": LONG_TEXT,
              "title": "Tuyen React Dev"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == PENDING
    assert body["deduplicated"] is False
    assert body["document_type"] == "jd"

    status = client.get(f"/internal/jobs/{body['job_id']}").json()
    assert status["status"] == PENDING
    assert status["step"] == "RECEIVED"
    assert status["progress"] == 5
    assert status["max_attempts"] == 3


def test_create_job_is_idempotent(client):
    payload = {"document_type": "jd", "document_id": "jd_dup", "text": LONG_TEXT}
    first = client.post("/internal/jobs", data=payload).json()
    second = client.post("/internal/jobs", data=payload).json()
    assert second["job_id"] == first["job_id"]
    assert second["deduplicated"] is True


def test_cv_requires_source(client):
    r = client.post("/internal/jobs", data={"document_type": "cv", "document_id": "cv_1"})
    assert r.status_code == 422
    assert r.json()["detail"] == "CV_SOURCE_REQUIRED"


def test_cv_accepts_preextracted_text(client):
    r = client.post(
        "/internal/jobs",
        data={"document_type": "cv", "document_id": "cv_txt", "text": LONG_TEXT},
    )
    assert r.status_code == 202


def test_jd_requires_text(client):
    r = client.post(
        "/internal/jobs", data={"document_type": "jd", "document_id": "jd_x"}
    )
    assert r.status_code == 422
    assert r.json()["detail"] == "JD_TEXT_REQUIRED"


def test_invalid_document_type(client):
    r = client.post(
        "/internal/jobs", data={"document_type": "xx", "document_id": "a", "text": LONG_TEXT}
    )
    assert r.status_code == 422
    assert r.json()["detail"] == "INVALID_DOCUMENT_TYPE"


def test_invalid_visibility_rejected(client):
    r = client.post(
        "/internal/jobs",
        data={"document_type": "jd", "document_id": "jd_v", "text": LONG_TEXT,
              "visibility": "hidden"},
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "VISIBILITY_INVALID"


def test_get_unknown_job_404(client):
    assert client.get("/internal/jobs/khong-biet").status_code == 404


def test_file_upload_saves_source_and_cancel_removes_it(client, store):
    pdf_bytes = b"%PDF-1.7\n%" + b"x" * 64
    r = client.post(
        "/internal/jobs",
        data={"document_type": "cv", "document_id": "cv_file"},
        files={"file": ("cv.pdf", pdf_bytes, "application/pdf")},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    record = store.get(job_id)
    assert record.source_kind == "file"
    from pathlib import Path

    source = Path(record.source_ref)
    assert source.exists() and source.read_bytes() == pdf_bytes

    cancelled = client.delete(f"/internal/jobs/{job_id}")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == CANCELLED
    assert not source.exists()  # file tam bi xoa khi huy


def test_cancel_rules(client):
    created = client.post(
        "/internal/jobs",
        data={"document_type": "jd", "document_id": "jd_c", "text": LONG_TEXT},
    ).json()
    job_id = created["job_id"]
    assert client.delete(f"/internal/jobs/{job_id}").json()["status"] == CANCELLED
    assert client.delete(f"/internal/jobs/{job_id}").status_code == 409  # da ket thuc


def test_retry_rules(client, store):
    created = client.post(
        "/internal/jobs",
        data={"document_type": "jd", "document_id": "jd_r", "text": LONG_TEXT},
    ).json()
    job_id = created["job_id"]
    assert client.post(f"/internal/jobs/{job_id}/retry").status_code == 409  # van PENDING

    store.claim_next()
    store.mark_failed(job_id, "PARSING: LLM_UNAVAILABLE: ollama chet")
    retried = client.post(f"/internal/jobs/{job_id}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] == PENDING
    assert retried.json()["attempt_count"] == 0


def test_internal_key_required_when_configured(tmp_path):
    app = create_app(
        make_settings(tmp_path, internal_api_key="bi-mat")
    )
    with TestClient(app) as client:
        r = client.post(
            "/internal/jobs",
            data={"document_type": "jd", "document_id": "jd_auth", "text": LONG_TEXT},
        )
        assert r.status_code == 401
        ok = client.post(
            "/internal/jobs",
            data={"document_type": "jd", "document_id": "jd_auth", "text": LONG_TEXT},
            headers={"X-Internal-Token": "bi-mat"},
        )
        assert ok.status_code == 202


def test_end_to_end_job_becomes_ready(tmp_path, monkeypatch):
    """Worker thread that: POST -> PENDING -> ... -> READY (progress 100)."""

    def fake_parse(job, cleaned, settings, llm, warnings):
        assert job.document_type == "cv"
        return {"summary": "React developer", "skills": [{"name": "React"}]}

    monkeypatch.setattr(pipeline_mod, "parse_payload", fake_parse)
    monkeypatch.setattr(recommend_mod, "index", lambda docs, s, version=1: {"indexed": 1})

    settings = make_settings(tmp_path, processing_worker_enabled=True)
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/internal/jobs",
            data={"document_type": "cv", "document_id": "cv_e2e", "text": LONG_TEXT,
                  "owner_id": "freelancer_001", "visibility": "public"},
        ).json()
        deadline = time.time() + 10
        status = client.get(f"/internal/jobs/{created['job_id']}").json()
        while time.time() < deadline and status["status"] not in (READY, FAILED):
            time.sleep(0.02)
            status = client.get(f"/internal/jobs/{created['job_id']}").json()
        assert status["status"] == READY
        assert status["progress"] == 100
        assert status["duration_ms"] is not None
        assert status["attempt_count"] == 1


def test_analyze_cv_route_unchanged(client):
    """Tinh tuong thich: analyze-cv van dong bo nhu truoc."""
    r = client.post(
        "/api/v1/ai/analyze-cv",
        files={"file": ("cv.txt", b"hello world", "application/pdf")},
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "CV_FILE_INVALID"


def test_routes_module_has_no_processing_cycle():
    # routes import processing.store (khong qua pipeline) -> khong co vong lap
    assert hasattr(routes_mod, "create_processing_job")
