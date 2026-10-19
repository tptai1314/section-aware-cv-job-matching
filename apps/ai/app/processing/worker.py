"""Worker nền: claim job -> chạy pipeline -> retry/backoff hoặc FAILED (thiết kế §5).

Chạy dưới dạng 1 thread daemon trong process FastAPI (main.lifespan). SQLite là
nguồn sự thật duy nhất — thread chỉ *wake up* sớm khi có `submit()`, còn lại poll
theo `job_poll_interval_s`; nhờ vậy job gửi trước khi worker khởi động vẫn được xử
lý và restart service không mất job (PENDING còn trong DB).
"""

from __future__ import annotations

import logging
import queue
import threading

from ..config import Settings
from . import pipeline as pipeline_mod
from . import store as store_mod
from .store import JobRecord, SqliteProcessingStore, format_error

logger = logging.getLogger("app.processing")

_SENTINEL = object()


class ProcessingWorker:
    """1 thread xử lý tuần tự (concurrency=1 — BGE-M3/OCR chạy CPU, thiết kế §5.4)."""

    def __init__(
        self,
        store: SqliteProcessingStore,
        settings: Settings,
        llm,
        *,
        index_fn=None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.llm = llm
        self.index_fn = index_fn
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # Vòng đời
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Rà job kẹt (worker cũ chết) rồi chạy thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        requeued = self.store.requeue_stale(
            self.settings.job_stale_seconds, self.settings.job_max_attempts
        )
        if requeued:
            logger.warning("requeued %d job kẹt quá %ss", requeued,
                           self.settings.job_stale_seconds)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="processing-worker", daemon=True
        )
        self._thread.start()
        logger.info("processing worker started (poll=%ss, max_attempts=%d)",
                    self.settings.job_poll_interval_s, self.settings.job_max_attempts)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._queue.put(_SENTINEL)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def submit(self, job_id: str) -> None:
        """Đánh thức worker (không chặn) — DB vẫn là nguồn sự thật."""
        self._queue.put(job_id)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    # Xử lý
    # ------------------------------------------------------------------ #

    def run_once(self) -> bool:
        """Claim + xử lý tối đa 1 job. Trả False khi không có job tới hạn."""
        job = self.store.claim_next()
        if job is None:
            return False
        self.execute(job)
        return True

    def execute(self, job: JobRecord) -> None:
        try:
            record = pipeline_mod.run(
                job, self.store, self.settings, self.llm, index_fn=self.index_fn
            )
            store_mod.remove_source_file(record)  # READY -> file tạm hết vai trò
        except pipeline_mod.PermanentError as exc:
            self._fail(job, exc.code, str(exc), transient=False)
        except pipeline_mod.TransientError as exc:
            self._fail(job, exc.code, str(exc), transient=True)
        except Exception as exc:  # phòng thủ: run() đã bọc, nhưng không được chết thread
            logger.exception("job %s lỗi không lường trước", job.id)
            self._fail(job, "INTERNAL", f"{type(exc).__name__}: {exc}", transient=True)

    def _fail(self, job: JobRecord, code: str, message: str, *, transient: bool) -> None:
        error = format_error(job.step, code, message)
        max_attempts = self.settings.job_max_attempts
        if transient and job.attempt_count < max_attempts:
            delay = self.settings.job_retry_base_delay_s * (2 ** (job.attempt_count - 1))
            record = self.store.requeue(job.id, delay_s=delay, last_error=error)
            logger.warning(
                "job=%s doc=%s:%s lỗi tạm thời (%s), thử lại lần %d sau %.1fs: %s",
                job.id, job.document_type, job.document_id, code,
                record.attempt_count, delay, message,
            )
            return
        record = self.store.mark_failed(job.id, error)
        logger.error(
            "job=%s doc=%s:%s FAILED sau %d lần: %s",
            job.id, job.document_type, job.document_id, record.attempt_count, error,
        )

    # ------------------------------------------------------------------ #
    # Vòng lặp
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        poll = max(0.01, self.settings.job_poll_interval_s)
        while not self._stop.is_set():
            try:
                ran = self.run_once()
            except Exception:  # store lock/disk lỗi v.v. — không để thread chết
                logger.exception("processing worker lỗi lặp; thử lại sau %.1fs", poll)
                ran = False
            if ran:
                continue  # còn job dồn thì xử lý liền, không chờ poll
            try:
                self._queue.get(timeout=poll)
            except queue.Empty:
                pass


__all__ = ["ProcessingWorker"]
