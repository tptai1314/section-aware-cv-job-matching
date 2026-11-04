"""Test worker + pipeline: happy path, retry tạm thời, lỗi vĩnh viễn, stale (§5–§6)."""

import dataclasses
import hashlib
import time

import pytest
from app import errors
from app.config import Settings
from app.processing import pipeline as pipeline_mod
from app.processing import store as store_mod
from app.processing.store import FAILED, PENDING, READY, SqliteProcessingStore
from app.processing.worker import ProcessingWorker

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
    processing_db_path=":memory:",
    job_max_attempts=3,
    job_retry_base_delay_s=0.0,  # test: retry ngay lap tuc
    job_poll_interval_s=0.01,
)


def make_settings(**overrides) -> Settings:
    return dataclasses.replace(BASE, **overrides)


def make_store(tmp_path) -> SqliteProcessingStore:
    return SqliteProcessingStore(tmp_path / "jobs.db")


def fake_payload(job, cleaned, settings, llm, warnings):
    return {"summary": "React developer", "skills": [{"name": "React"}]}


def test_run_once_happy_path(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(pipeline_mod, "parse_payload", fake_payload)

    def fake_index(documents, settings, version=1):
        seen["documents"] = documents
        seen["version"] = version
        return {"indexed": len(documents)}

    worker = ProcessingWorker(store, make_settings(), llm=None, index_fn=fake_index)
    job, _ = store.create(
        document_type="cv", document_id="cv_1", owner_id="freelancer_001",
        visibility="public", source_kind="text", text=LONG_TEXT,
    )

    assert worker.run_once() is True
    record = store.get(job.id)
    assert record.status == READY
    assert record.step == store_mod.STEP_FINALIZING
    assert record.attempt_count == 1
    assert record.content_hash and len(record.content_hash) == 64
    assert record.duration_ms is not None
    assert worker.run_once() is False  # het job

    document = seen["documents"][0]
    assert document["id"] == "cv_1" and document["type"] == "cv"
    assert document["owner_id"] == "freelancer_001"
    assert document["visibility"] == "public"
    assert seen["version"] == 1


def prime_content_hash(store, job, text: str = LONG_TEXT) -> str:
    """Gia lap 'lan chay truoc da hash text nhung chua xong': PENDING + content_hash."""
    cleaned = pipeline_mod.text_mod.merge_broken_lines(
        pipeline_mod.text_mod.clean_text(text)
    )
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    store.claim_next()  # -> PROCESSING (set_content_hash doi trang thai nay)
    store.set_content_hash(job.id, digest)
    store.requeue(job.id)  # -> PENDING, giu nguyen content_hash
    return digest


def test_skip_reembed_when_index_already_ready(tmp_path, monkeypatch):
    """OQ6: text khong doi + index con nguyen -> khong goi LLM, khong nhung lai."""
    store = make_store(tmp_path)
    job, _ = store.create(
        document_type="cv", document_id="cv_reuse", source_kind="text", text=LONG_TEXT
    )
    digest = prime_content_hash(store, job)

    def must_not_parse(*args, **kwargs):
        raise AssertionError("phai bo qua parse khi index da san")

    monkeypatch.setattr(pipeline_mod, "parse_payload", must_not_parse)
    worker = ProcessingWorker(
        store,
        make_settings(),
        llm=None,
        index_fn=lambda *a, **k: pytest.fail("khong duoc index lai"),
        index_lookup=lambda settings, doc_type, doc_id, version: True,
    )

    assert worker.run_once() is True
    record = store.get(job.id)
    assert record.status == READY
    assert record.step == store_mod.STEP_FINALIZING
    assert record.content_hash == digest
    assert record.warnings == ["REUSED_INDEX"]
    assert record.duration_ms is not None
    assert record.attempt_count == 2  # lan claim thu 2 (lan dau da bi requeue)


def test_no_skip_when_index_missing(tmp_path, monkeypatch):
    """Index khong con (da bi xoa / chua tung upsert) -> phai parse + index lai."""
    store = make_store(tmp_path)
    job, _ = store.create(
        document_type="cv", document_id="cv_noidx", source_kind="text", text=LONG_TEXT
    )
    prime_content_hash(store, job)
    seen = {"parsed": 0, "indexed": 0}

    def counting_parse(job, cleaned, settings, llm, warnings):
        seen["parsed"] += 1
        return fake_payload(job, cleaned, settings, llm, warnings)

    def counting_index(documents, settings, version=1):
        seen["indexed"] += 1
        return {"indexed": len(documents)}

    monkeypatch.setattr(pipeline_mod, "parse_payload", counting_parse)
    worker = ProcessingWorker(
        store, make_settings(), llm=None, index_fn=counting_index,
        index_lookup=lambda settings, doc_type, doc_id, version: False,
    )
    assert worker.run_once() is True
    assert seen == {"parsed": 1, "indexed": 1}
    assert store.get(job.id).warnings == []


def test_no_skip_when_text_changed(tmp_path, monkeypatch):
    """File/text moi lam hash doi -> luon parse + index lai du index con."""
    store = make_store(tmp_path)
    job, _ = store.create(
        document_type="cv", document_id="cv_changed", source_kind="text", text=LONG_TEXT
    )
    prime_content_hash(store, job, text="Senior Java developer, 10 nam kinh nghiem.")

    parsed = {"count": 0}

    def counting_parse(job, cleaned, settings, llm, warnings):
        parsed["count"] += 1
        return fake_payload(job, cleaned, settings, llm, warnings)

    monkeypatch.setattr(pipeline_mod, "parse_payload", counting_parse)
    worker = ProcessingWorker(
        store,
        make_settings(),
        llm=None,
        index_fn=lambda docs, s, version=1: {"indexed": len(docs)},
        index_lookup=lambda settings, doc_type, doc_id, version: True,
    )
    assert worker.run_once() is True
    assert store.get(job.id).status == READY
    assert parsed["count"] == 1
    assert store.get(job.id).warnings == []


def test_no_skip_when_disabled(tmp_path, monkeypatch):
    """PROCESSING_SKIP_REEMBED=0 -> luon parse lai du index con nguyen."""
    store = make_store(tmp_path)
    job, _ = store.create(
        document_type="cv", document_id="cv_off", source_kind="text", text=LONG_TEXT
    )
    prime_content_hash(store, job)
    monkeypatch.setattr(pipeline_mod, "parse_payload", fake_payload)
    worker = ProcessingWorker(
        store,
        make_settings(skip_reembed=False),
        llm=None,
        index_fn=lambda docs, s, version=1: {"indexed": len(docs)},
        index_lookup=lambda settings, doc_type, doc_id, version: True,
    )
    assert worker.run_once() is True
    record = store.get(job.id)
    assert record.status == READY
    assert record.warnings == []


def test_lookup_error_falls_back_to_reindex(tmp_path, monkeypatch):
    """Lookup loi (Chroma lock...) -> coi nhu chua index, khong lam job FAILED."""
    store = make_store(tmp_path)
    job, _ = store.create(
        document_type="cv", document_id="cv_err", source_kind="text", text=LONG_TEXT
    )
    prime_content_hash(store, job)
    monkeypatch.setattr(pipeline_mod, "parse_payload", fake_payload)

    def boom(settings, doc_type, doc_id, version):
        raise RuntimeError("chroma bi lock")

    worker = ProcessingWorker(
        store,
        make_settings(),
        llm=None,
        index_fn=lambda docs, s, version=1: {"indexed": len(docs)},
        index_lookup=boom,
    )
    assert worker.run_once() is True
    assert store.get(job.id).status == READY


def test_job_with_file_source_cleans_up_after_ready(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    monkeypatch.setattr(pipeline_mod, "parse_payload", fake_payload)
    # patch trich xuat text: noi dung file chi can dung magic bytes PDF
    monkeypatch.setattr(
        pipeline_mod.text_mod, "extract_text",
        lambda path, kind, settings: (LONG_TEXT, 1, False, []),
    )
    source = tmp_path / "cv.upload"
    source.write_bytes(b"%PDF-1.7\n%" + b"x" * 64)
    worker = ProcessingWorker(
        store, make_settings(), llm=None, index_fn=lambda docs, s, version=1: {}
    )
    job, _ = store.create(
        document_type="cv", document_id="cv_file", source_kind="file",
        source_ref=str(source),
    )
    worker.run_once()
    assert store.get(job.id).status == READY
    assert not source.exists()  # file tam duoc xoa sau khi READY


def test_transient_failure_retries_until_max_attempts(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    calls = {"count": 0}

    def boom(job, cleaned, settings, llm, warnings):
        calls["count"] += 1
        raise errors.llm_unavailable()

    monkeypatch.setattr(pipeline_mod, "parse_payload", boom)
    worker = ProcessingWorker(store, make_settings(), llm=None, index_fn=None)
    job, _ = store.create(
        document_type="cv", document_id="cv_2", source_kind="text", text=LONG_TEXT
    )

    for attempt in (1, 2):
        assert worker.run_once() is True
        record = store.get(job.id)
        assert record.status == PENDING  # loi tam thoi -> quay ve cho retry
        assert record.attempt_count == attempt
        assert "LLM_UNAVAILABLE" in record.last_error
        assert record.step == store_mod.STEP_PARSING
        # last_error phai ghi DUNG buoc gay loi (thiet ke §2.3), khong phai buoc luc claim
        assert record.last_error.startswith(f"{store_mod.STEP_PARSING}: LLM_UNAVAILABLE:")

    assert worker.run_once() is True  # lan thu 3 -> het luot
    record = store.get(job.id)
    assert record.status == FAILED
    assert record.attempt_count == 3
    assert "LLM_UNAVAILABLE" in record.last_error
    assert calls["count"] == 3
    assert worker.run_once() is False


def test_permanent_failure_fails_immediately(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    monkeypatch.setattr(pipeline_mod, "parse_payload", fake_payload)
    worker = ProcessingWorker(store, make_settings(), llm=None, index_fn=None)
    job, _ = store.create(
        document_type="cv", document_id="cv_3", source_kind="text", text="ngan qua"
    )

    assert worker.run_once() is True
    record = store.get(job.id)
    assert record.status == FAILED  # khong retry
    assert record.attempt_count == 1
    assert "CV_TEXT_EMPTY" in record.last_error
    assert record.step == store_mod.STEP_EXTRACTING


def test_unknown_exception_is_transient(tmp_path, monkeypatch):
    store = make_store(tmp_path)

    def boom(job, cleaned, settings, llm, warnings):
        raise RuntimeError("chroma bi lock")

    monkeypatch.setattr(pipeline_mod, "parse_payload", boom)
    worker = ProcessingWorker(store, make_settings(), llm=None, index_fn=None)
    job, _ = store.create(
        document_type="jd", document_id="jd_9", source_kind="text", text=LONG_TEXT
    )
    worker.run_once()
    record = store.get(job.id)
    assert record.status == PENDING
    assert "INTERNAL" in record.last_error
    assert "chroma bi lock" in record.last_error


def test_jd_job_uses_job_prompt_and_builds_record(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    seen: dict = {}

    def fake_parse(job, cleaned, settings, llm, warnings):
        assert job.document_type == "jd"
        return {"title": "Tuyen React Dev", "required_skills": [{"name": "React"}]}

    monkeypatch.setattr(pipeline_mod, "parse_payload", fake_parse)

    def fake_index(documents, settings, version=1):
        seen["documents"] = documents
        return {}

    worker = ProcessingWorker(store, make_settings(), llm=None, index_fn=fake_index)
    job, _ = store.create(
        document_type="jd", document_id="jd_1", title="Tuyen React Dev",
        source_kind="text", text=LONG_TEXT, visibility="public",
    )
    worker.run_once()
    assert store.get(job.id).status == READY
    assert seen["documents"][0]["title"] == "Tuyen React Dev"


def test_read_source_text_label():
    from app.processing.store import JobRecord

    cv = JobRecord(id="a", document_type="cv", document_id="cv_1", text="hello")
    assert pipeline_mod.read_source(cv, BASE).label == "text"
    jd = JobRecord(id="b", document_type="jd", document_id="jd_1", text="hello")
    assert pipeline_mod.read_source(jd, BASE).label == "job_text"


def test_read_source_missing_file_is_permanent():
    from app.processing.store import JobRecord

    job = JobRecord(id="a", document_type="cv", document_id="cv_1",
                    source_kind="file", source_ref="duong-dan-khong-ton-tai.pdf")
    with pytest.raises(pipeline_mod.PermanentError) as excinfo:
        pipeline_mod.read_source(job, BASE)
    assert excinfo.value.code == "SOURCE_MISSING"


# ---- vòng đời worker ----


def test_start_stop_is_idempotent(tmp_path):
    store = make_store(tmp_path)
    worker = ProcessingWorker(store, make_settings(), llm=None, index_fn=None)
    assert not worker.is_alive()
    worker.start()
    worker.start()  # goi lai khong tao thread thu hai
    assert worker.is_alive()
    worker.stop()
    assert not worker.is_alive()
    worker.stop()  # stop lai van on


def test_start_requeues_stale_jobs(tmp_path, monkeypatch):
    # parse_payload mac dinh se goi Ollama that -> patch loi tam thoi cho determinism
    monkeypatch.setattr(
        pipeline_mod, "parse_payload",
        lambda *a, **k: (_ for _ in ()).throw(errors.llm_unavailable()),
    )
    store = make_store(tmp_path)
    job, _ = store.create(
        document_type="cv", document_id="cv_stale", source_kind="text", text=LONG_TEXT
    )
    assert store.claim_next() is not None  # bi claim roi worker chet
    store._conn.execute(
        "UPDATE processing_jobs SET updated_at=? WHERE id=?",
        (time.time() - 3600, job.id),
    )
    store._conn.commit()

    worker = ProcessingWorker(store, make_settings(job_stale_seconds=60), llm=None)
    worker.start()
    try:
        deadline = time.time() + 5
        record = store.get(job.id)
        while time.time() < deadline and record.status != FAILED:
            time.sleep(0.02)
            record = store.get(job.id)
        assert record.status == FAILED  # requeue -> retry den khi het luot
        assert record.attempt_count >= 2  # van giu attempt cua lan claim bi ket
    finally:
        worker.stop()
