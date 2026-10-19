"""Test state machine + SQLite store (docs/DESIGN_processing_queue.md §2–§3)."""

import time

import pytest
from app.processing import store as store_mod
from app.processing.store import (
    CANCELLED,
    FAILED,
    PENDING,
    PROCESSING,
    READY,
    SqliteProcessingStore,
    TransitionError,
    format_error,
    progress_for,
)

LONG_TEXT = "Senior React developer with 6 years building web applications at Acme Corp."


def make_store(tmp_path) -> SqliteProcessingStore:
    return SqliteProcessingStore(tmp_path / "jobs.db")


def create_cv(store, doc_id="cv_1", version=1, **kw):
    kw.setdefault("source_kind", "text")
    kw.setdefault("text", LONG_TEXT)
    return store.create(document_type="cv", document_id=doc_id, version=version, **kw)


def backdate(store, job_id, seconds):
    """Mo phong worker chet giua chung: day updated_at + next_attempt_at lui lai."""
    then = time.time() - seconds
    store._conn.execute(
        "UPDATE processing_jobs SET updated_at=?, next_attempt_at=? WHERE id=?",
        (then, then, job_id),
    )
    store._conn.commit()


# ---- create / idempotency ----


def test_create_and_get(tmp_path):
    store = make_store(tmp_path)
    record, dedup = create_cv(store)
    assert dedup is False
    assert record.status == PENDING
    assert record.step == store_mod.STEP_RECEIVED
    assert store.get(record.id).id == record.id
    assert store.get("khong-ton-tai") is None
    assert store.get_by_doc("cv", "cv_1", 1).id == record.id


def test_create_dedups_same_document(tmp_path):
    store = make_store(tmp_path)
    first, _ = create_cv(store)
    second, dedup = create_cv(store, text="nguon moi")
    assert dedup is True
    assert second.id == first.id  # khong tao job moi cho cung ban ghi


def test_create_dedups_when_still_pending_or_processing(tmp_path):
    store = make_store(tmp_path)
    first, _ = create_cv(store)
    store.claim_next()  # -> PROCESSING
    again, dedup = create_cv(store)
    assert dedup is True
    assert again.id == first.id
    assert again.status == PROCESSING


def test_create_requeues_failed_job_with_new_source(tmp_path):
    store = make_store(tmp_path)
    first, _ = create_cv(store)
    store.claim_next()
    store.mark_failed(first.id, "PARSING: LLM_UNAVAILABLE: ollama chet")
    reused, dedup = create_cv(store, text="noi dung moi")
    assert dedup is True
    assert reused.id == first.id
    assert reused.status == PENDING
    assert reused.attempt_count == 0
    assert reused.last_error is None
    assert reused.text == "noi dung moi"


def test_create_requeues_cancelled_job(tmp_path):
    store = make_store(tmp_path)
    first, _ = create_cv(store)
    store.cancel(first.id)
    reused, dedup = create_cv(store)
    assert dedup is True
    assert reused.status == PENDING


def test_create_rejects_invalid_input(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ValueError):
        store.create(document_type="foo", document_id="x")
    with pytest.raises(ValueError):
        store.create(document_type="cv", document_id="x", visibility="hidden")
    with pytest.raises(ValueError):
        store.create(document_type="cv", document_id="x", version=0)


# ---- claim ----


def test_claim_next_moves_to_processing(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    claimed = store.claim_next()
    assert claimed is not None and claimed.id == job.id
    assert claimed.status == PROCESSING
    assert claimed.attempt_count == 1
    assert claimed.started_at is not None
    assert store.claim_next() is None  # khong con job PENDING


def test_claim_next_respects_backoff(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    store.claim_next()
    store.requeue(job.id, delay_s=60)
    assert store.claim_next() is None  # chua den han thu lai
    backdate(store, job.id, 120)  # xuyen next_attempt_at qua qua khu
    claimed = store.claim_next()
    assert claimed is not None and claimed.attempt_count == 2


def test_claim_next_orders_by_created_at(tmp_path):
    store = make_store(tmp_path)
    first, _ = create_cv(store, doc_id="cv_a")
    second, _ = create_cv(store, doc_id="cv_b")
    assert store.claim_next().id == first.id
    assert store.claim_next().id == second.id


# ---- steps / terminal ----


def test_set_step_and_mark_ready(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    store.claim_next()
    for step in (store_mod.STEP_EXTRACTING, store_mod.STEP_PARSING,
                 store_mod.STEP_EMBEDDING, store_mod.STEP_FINALIZING):
        assert store.set_step(job.id, step).step == step
    store.set_content_hash(job.id, "abc123")
    ready = store.mark_ready(job.id, warnings=["EMPTY_SECTION:education"])
    assert ready.status == READY
    assert ready.content_hash == "abc123"
    assert ready.warnings == ["EMPTY_SECTION:education"]
    assert ready.finished_at is not None
    assert ready.duration_ms is not None
    assert progress_for(ready.status, ready.step) == 100


def test_set_step_rejects_unknown_step(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    store.claim_next()
    with pytest.raises(ValueError):
        store.set_step(job.id, "Nonsense")


def test_terminal_states_are_locked(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    store.claim_next()
    store.mark_ready(job.id)
    with pytest.raises(TransitionError):
        store._apply(job.id, status=PROCESSING)  # READY la trang thai ket thuc
    with pytest.raises(KeyError):
        store.set_step("id-sai", store_mod.STEP_PARSING)


# ---- cancel / retry ----


def test_cancel_only_pending(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    cancelled = store.cancel(job.id)
    assert cancelled.status == CANCELLED
    with pytest.raises(TransitionError):
        store.cancel(job.id)  # khong huy duoc job da ket thuc

    other, _ = create_cv(store, doc_id="cv_2")
    store.claim_next()
    with pytest.raises(TransitionError):
        store.cancel(other.id)  # khong dung giua chung khi dang xu ly


def test_retry_failed_resets_attempts(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    store.claim_next()
    store.mark_failed(job.id, "PARSING: LLM_TIMEOUT: het gio cho Ollama")
    retried = store.retry(job.id)
    assert retried.status == PENDING
    assert retried.attempt_count == 0
    assert retried.last_error is None
    assert retried.finished_at is None
    with pytest.raises(TransitionError):
        store.retry(job.id)  # chi retry duoc tu FAILED


def test_mark_failed_keeps_error_and_duration(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    store.claim_next()
    failed = store.mark_failed(job.id, format_error("PARSING", "LLM_UNAVAILABLE", "loi"))
    assert failed.status == FAILED
    assert failed.last_error == "PARSING: LLM_UNAVAILABLE: loi"
    assert failed.duration_ms is not None


# ---- stale recovery (§5.2) ----


def test_requeue_stale_requeues_then_fails_when_out_of_attempts(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    store.claim_next()
    backdate(store, job.id, 600)
    assert store.requeue_stale(max_age_s=300, max_attempts=3) == 1
    record = store.get(job.id)
    assert record.status == PENDING
    assert record.attempt_count == 1  # van giu attempt da dung

    # gia lap worker chet lien tuc: moi vong mot attempt cho den khi het luot
    for _ in range(2):
        store.claim_next()
        backdate(store, job.id, 600)
        store.requeue_stale(max_age_s=300, max_attempts=3)
    record = store.get(job.id)
    assert record.status == FAILED
    assert "STALE_TIMEOUT" in record.last_error
    assert record.attempt_count == 3


def test_requeue_stale_ignores_fresh_jobs(tmp_path):
    store = make_store(tmp_path)
    job, _ = create_cv(store)
    store.claim_next()
    assert store.requeue_stale(max_age_s=300, max_attempts=3) == 0
    assert store.get(job.id).status == PROCESSING


# ---- helpers ----


def test_progress_mapping():
    assert progress_for(PENDING, store_mod.STEP_RECEIVED) == 5
    assert progress_for(PROCESSING, store_mod.STEP_EXTRACTING) == 25
    assert progress_for(PROCESSING, store_mod.STEP_PARSING) == 55
    assert progress_for(PROCESSING, store_mod.STEP_EMBEDDING) == 85
    assert progress_for(READY, store_mod.STEP_FINALIZING) == 100
    assert progress_for(FAILED, store_mod.STEP_PARSING) == 55  # dung o buoc loi


def test_list_active(tmp_path):
    store = make_store(tmp_path)
    first, _ = create_cv(store, doc_id="cv_a")
    second, _ = create_cv(store, doc_id="cv_b")
    store.claim_next()
    active_ids = {record.id for record in store.list_active()}
    assert active_ids == {first.id, second.id}  # PENDING + PROCESSING
    store.claim_next()
    assert {r.id for r in store.list_active()} == {first.id, second.id}
