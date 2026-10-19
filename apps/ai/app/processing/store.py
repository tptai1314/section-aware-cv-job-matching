"""Nguồn trạng thái xử lý file: state machine + backend SQLite (plan §9.3, §11.2).

`DocumentProcessingJob` theo plan §11.2 được hiện thực hoá ở đây; repo này chỉ ĐỌC
PostgreSQL nên backend là SQLite (`data/processing.db` theo mặc định). Khi AI service
có quyền ghi PG, chỉ cần thêm `PostgresProcessingStore` cùng interface (thiết kế §3.2)
— pipeline/worker không đổi.

State machine (thiết kế §2):

    PENDING ──▶ PROCESSING ──▶ READY
       │            │  ▲
       │            │  └──retry (lỗi tạm thời, backoff)
       ▼            ▼
   CANCELLED     FAILED ──(retry thủ công)──▶ PENDING

Trạng thái record này KHÁC với metadata `status` trong Chroma (cờ `READY` cho
retrieval): record phản ánh tiến trình của *job*, metadata Chroma phản ánh điều
kiện *truy xuất* và chỉ được ghi `READY` khi job hoàn tất (pipeline §INDEXING).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---- Trạng thái (plan §9.3) ----

PENDING = "PENDING"
PROCESSING = "PROCESSING"
READY = "READY"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

STATUSES = (PENDING, PROCESSING, READY, FAILED, CANCELLED)
TERMINAL_STATUSES = (READY, FAILED, CANCELLED)

# Từ -> tập trạng thái được phép chuyển tới (chỉ liệt kê khi ĐỔI trạng thái;
# update cùng trạng thái — ví dụ ghi `step` — luôn hợp lệ).
TRANSITIONS: dict[str, set[str]] = {
    PENDING: {PROCESSING, CANCELLED},
    PROCESSING: {READY, FAILED, PENDING},
    READY: set(),
    FAILED: {PENDING},  # retry thủ công
    CANCELLED: set(),
}

# ---- Bước xử lý (thiết kế §2.3) ----

STEP_RECEIVED = "RECEIVED"
STEP_EXTRACTING = "EXTRACTING"
STEP_PARSING = "PARSING"
STEP_EMBEDDING = "EMBEDDING"
STEP_FINALIZING = "FINALIZING"

STEPS = (STEP_RECEIVED, STEP_EXTRACTING, STEP_PARSING, STEP_EMBEDDING, STEP_FINALIZING)

# % tiến độ cho UI (thiết kế §4.2)
PROGRESS_BY_STEP = {
    STEP_RECEIVED: 5,
    STEP_EXTRACTING: 25,
    STEP_PARSING: 55,
    STEP_EMBEDDING: 85,
    STEP_FINALIZING: 95,
    READY: 100,
}

VISIBILITIES = ("public", "private", "unlisted")
DOC_TYPES = ("cv", "jd")

ERROR_MAX_CHARS = 2000  # không để last_error phình to


class TransitionError(ValueError):
    """Chuyển trạng thái không hợp lệ (gọi API sai thứ tự — không phải bug nội bộ)."""


@dataclass
class JobRecord:
    """Một bản ghi `DocumentProcessingJob`."""

    id: str
    document_type: str  # cv | jd
    document_id: str
    version: int = 1
    status: str = PENDING
    step: str = STEP_RECEIVED
    attempt_count: int = 0
    next_attempt_at: float = 0.0
    last_error: str | None = None
    content_hash: str | None = None
    source_kind: str = "text"  # file | text
    source_ref: str | None = None  # đường dẫn file tạm (khi source_kind=file)
    text: str | None = None  # nội dung JD / CV đã trích (khi source_kind=text)
    title: str | None = None
    owner_id: str | None = None
    visibility: str | None = None
    warnings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


_COLUMNS = (
    "id, document_type, document_id, version, status, step, attempt_count, "
    "next_attempt_at, last_error, content_hash, source_kind, source_ref, text, title, "
    "owner_id, visibility, warnings, created_at, updated_at, started_at, finished_at, "
    "duration_ms"
)

_DDL = """
CREATE TABLE IF NOT EXISTS processing_jobs (
    id            TEXT PRIMARY KEY,
    document_type TEXT NOT NULL,
    document_id   TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    status        TEXT NOT NULL DEFAULT 'PENDING',
    step          TEXT NOT NULL DEFAULT 'RECEIVED',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    last_error    TEXT,
    content_hash  TEXT,
    source_kind   TEXT NOT NULL DEFAULT 'text',
    source_ref    TEXT,
    text          TEXT,
    title         TEXT,
    owner_id      TEXT,
    visibility    TEXT,
    warnings      TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    duration_ms   INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_processing_jobs_doc
    ON processing_jobs (document_type, document_id, version);
CREATE INDEX IF NOT EXISTS ix_processing_jobs_claim
    ON processing_jobs (status, next_attempt_at, created_at);
"""


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    data = dict(row)
    raw_warnings = data.pop("warnings", None)
    try:
        warnings = json.loads(raw_warnings) if raw_warnings else []
    except (TypeError, ValueError):
        warnings = []
    data["warnings"] = [str(w) for w in warnings] if isinstance(warnings, list) else []
    return JobRecord(**data)


def progress_for(status: str, step: str) -> int:
    """% tiến độ để UI hiển thị (thiết kế §4.2)."""
    if status == READY:
        return PROGRESS_BY_STEP[READY]
    return PROGRESS_BY_STEP.get(step, 0)


def remove_source_file(record: JobRecord) -> None:
    """Xoá file tạm của job (sau READY/CANCELLED) — không để rác trên đĩa.

    Job FAILED GIỮ file để retry thủ công; file được dọn khi retry thành công
    hoặc khi job bị thay nguồn (create requeue).
    """
    if record.source_kind == "file" and record.source_ref:
        try:
            Path(record.source_ref).unlink(missing_ok=True)
        except OSError:
            pass


def format_error(step: str, code: str, message: str) -> str:
    """`last_error` theo thiết kế §6: `{step}: {code}: {message}` (cắt ngắn)."""
    return f"{step}: {code}: {message}"[:ERROR_MAX_CHARS]


class SqliteProcessingStore:
    """`ProcessingStore` trên SQLite — an toàn cho 1 process nhiều thread.

    Toàn bộ thao tác được serialize bằng RLock (worker thread + request threads);
    claim dùng `BEGIN IMMEDIATE` để hai thread không claim cùng một job (§5.3).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        # Kết nối LAZY: create_app() không tự tạo thư mục/file DB khi chưa có job nào
        # (tránh rác `data/processing.db` trong test/dev chỉ dựng app để healthcheck).
        self._conn: sqlite3.Connection | None = None

    def _ensure_conn(self) -> sqlite3.Connection:
        """Mở kết nối + tạo bảng lần đầu (gọi trong `self._lock`)."""
        if self._conn is None:
            if str(self.path) != ":memory:":
                self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript(_DDL)
            conn.commit()
            self._conn = conn
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------ #
    # CRUD cơ bản
    # ------------------------------------------------------------------ #

    def create(
        self,
        *,
        document_type: str,
        document_id: str,
        version: int = 1,
        owner_id: str | None = None,
        visibility: str | None = None,
        source_kind: str = "text",
        source_ref: str | None = None,
        text: str | None = None,
        title: str | None = None,
    ) -> tuple[JobRecord, bool]:
        """Tạo job; `(record, deduplicated)`.

        Idempotency (thiết kế §5.2): job đã có cho cùng `(document_type, document_id,
        version)`:
        - `READY`/`PENDING`/`PROCESSING` -> trả job cũ, KHÔNG tạo bản mới.
        - `FAILED`/`CANCELLED` -> *requeue* job cũ với nguồn mới (attempt reset).
        """
        if document_type not in DOC_TYPES:
            raise ValueError(f"invalid document_type: {document_type!r}")
        if visibility is not None and visibility not in VISIBILITIES:
            raise ValueError(f"invalid visibility: {visibility!r}")
        if not (0 < version <= 1000):
            raise ValueError(f"invalid version: {version!r}")
        now = time.time()
        with self._lock:
            existing = self.get_by_doc(document_type, document_id, version)
            if existing is not None:
                if existing.status in (FAILED, CANCELLED):
                    self._conn.execute(
                        "UPDATE processing_jobs SET status=?, step=?, attempt_count=0, "
                        "next_attempt_at=?, last_error=NULL, content_hash=NULL, "
                        "source_kind=?, source_ref=?, text=?, title=?, owner_id=?, "
                        "visibility=?, warnings=NULL, updated_at=?, started_at=NULL, "
                        "finished_at=NULL, duration_ms=NULL "
                        "WHERE id=? AND status IN (?, ?)",
                        (
                            PENDING, STEP_RECEIVED, now, source_kind, source_ref, text,
                            title, owner_id, visibility, now, existing.id, FAILED, CANCELLED,
                        ),
                    )
                    self._conn.commit()
                    return self.get(existing.id), True
                return existing, True

            record = JobRecord(
                id=uuid.uuid4().hex,
                document_type=document_type,
                document_id=document_id,
                version=version,
                owner_id=owner_id,
                visibility=visibility,
                source_kind=source_kind,
                source_ref=source_ref,
                text=text,
                title=title,
                next_attempt_at=now,
                created_at=now,
                updated_at=now,
            )
            try:
                self._insert(record)
                self._conn.commit()
            except sqlite3.IntegrityError:
                # hai request đua tạo cùng bản ghi: trả job vừa được kia (idempotency)
                self._conn.rollback()
                raced = self.get_by_doc(document_type, document_id, version)
                if raced is None:
                    raise
                return raced, True
            return record, False

    def _insert(self, record: JobRecord) -> None:
        data = record.to_dict()
        data["warnings"] = json.dumps(record.warnings, ensure_ascii=False)
        placeholders = ", ".join(f":{name}" for name in data)
        self._conn.execute(
            f"INSERT INTO processing_jobs ({', '.join(data)}) VALUES ({placeholders})",
            data,
        )

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._ensure_conn().execute(
                "SELECT * FROM processing_jobs WHERE id=?", (job_id,)
            ).fetchone()
            return _row_to_record(row) if row else None

    def get_by_doc(self, document_type: str, document_id: str, version: int) -> JobRecord | None:
        with self._lock:
            row = self._ensure_conn().execute(
                "SELECT * FROM processing_jobs "
                "WHERE document_type=? AND document_id=? AND version=?",
                (document_type, document_id, version),
            ).fetchone()
            return _row_to_record(row) if row else None

    def list_active(self) -> list[JobRecord]:
        """Job chưa kết thúc — dùng cho monitoring/reconciliation."""
        with self._lock:
            rows = self._ensure_conn().execute(
                "SELECT * FROM processing_jobs WHERE status IN (?, ?) ORDER BY created_at",
                (PENDING, PROCESSING),
            ).fetchall()
            return [_row_to_record(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Chuyển trạng thái (plan §9.3)
    # ------------------------------------------------------------------ #

    def claim_next(self, now: float | None = None) -> JobRecord | None:
        """Atomic claim 1 job `PENDING` tới hạn (thiết kế §5.3): -> PROCESSING, attempt+1."""
        now = time.time() if now is None else now
        with self._lock:
            conn = self._ensure_conn()
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id FROM processing_jobs WHERE status=? AND next_attempt_at<=? "
                "ORDER BY created_at LIMIT 1",
                (PENDING, now),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            cursor = conn.execute(
                "UPDATE processing_jobs SET status=?, attempt_count=attempt_count+1, "
                "started_at=COALESCE(started_at, ?), updated_at=? "
                "WHERE id=? AND status=?",
                (PROCESSING, now, now, row["id"], PENDING),
            )
            if cursor.rowcount != 1:  # phòng ngừa: có ai giành giữa chừng
                conn.commit()
                return None
            conn.commit()
            return self.get(row["id"])

    def set_step(self, job_id: str, step: str) -> JobRecord:
        """Checkpoint tiến độ — job phải đang PROCESSING (thiết kế §5.2)."""
        if step not in STEPS:
            raise ValueError(f"invalid step: {step!r}")
        return self._apply(job_id, status=PROCESSING, step=step, _require=PROCESSING)

    def set_content_hash(self, job_id: str, content_hash: str) -> JobRecord:
        return self._apply(job_id, content_hash=content_hash, _require=PROCESSING)

    def mark_ready(self, job_id: str, warnings: list[str] | None = None) -> JobRecord:
        return self._apply(
            job_id,
            status=READY,
            warnings=list(warnings or []),
            finished_at=time.time(),
            _duration=True,
            _require=PROCESSING,
        )

    def mark_failed(self, job_id: str, last_error: str) -> JobRecord:
        return self._apply(
            job_id,
            status=FAILED,
            last_error=last_error[:ERROR_MAX_CHARS],
            finished_at=time.time(),
            _duration=True,
            _require=PROCESSING,
        )

    def requeue(
        self, job_id: str, delay_s: float = 0.0, last_error: str | None = None
    ) -> JobRecord:
        """Lỗi tạm thời: PROCESSING -> PENDING, lùi lượt thử sau `delay_s` (backoff §5.2).

        `last_error` được ghi lại để UI thấy lỗi gần nhất ngay cả khi đang chờ retry.
        """
        now = time.time()
        return self._apply(
            job_id,
            status=PENDING,
            last_error=None if last_error is None else last_error[:ERROR_MAX_CHARS],
            next_attempt_at=now + max(0.0, delay_s),
            updated_at=now,
            _require=PROCESSING,
        )

    def retry(self, job_id: str) -> JobRecord:
        """Retry thủ công: FAILED -> PENDING, reset attempt (thiết kế §2.2)."""
        now = time.time()
        return self._apply(
            job_id,
            status=PENDING,
            attempt_count=0,
            last_error=None,
            next_attempt_at=now,
            updated_at=now,
            started_at=None,
            finished_at=None,
            duration_ms=None,
            _require=FAILED,
        )

    def cancel(self, job_id: str) -> JobRecord:
        """Chỉ hủy được job còn PENDING (đã claim thì không dừng giữa chừng)."""
        now = time.time()
        return self._apply(
            job_id,
            status=CANCELLED,
            finished_at=now,
            updated_at=now,
            _duration=True,
            _require=PENDING,
        )

    def requeue_stale(self, max_age_s: float, max_attempts: int) -> int:
        """Worker chết giữa chừng: PROCESSING quá hạn -> PENDING (hoặc FAILED nếu hết lượt).

        Treated như lỗi tạm thời: mỗi lần kẹt tính 1 attempt (claim đã đếm 1) — job
        crash-loop không lặp vô hạn (thiết kế §5.2, §9.4).
        """
        now = time.time()
        cutoff = now - max_age_s
        requeued = 0
        with self._lock:
            rows = self._ensure_conn().execute(
                "SELECT * FROM processing_jobs WHERE status=? AND updated_at<?",
                (PROCESSING, cutoff),
            ).fetchall()
            for row in rows:
                record = _row_to_record(row)
                error = format_error(record.step, "STALE_TIMEOUT",
                                     f"job kẹt quá {max_age_s:.0f}s, worker có thể đã chết")
                if record.attempt_count >= max_attempts:
                    self._apply(record.id, status=FAILED, last_error=error,
                                finished_at=now, updated_at=now, _duration=True)
                else:
                    self._apply(record.id, status=PENDING, next_attempt_at=now,
                                updated_at=now)
                requeued += 1
        return requeued

    # ------------------------------------------------------------------ #
    # Nội bộ
    # ------------------------------------------------------------------ #

    def _apply(self, job_id: str, **fields: Any) -> JobRecord:
        """Cập nhật trường + kiểm tra transition; raise TransitionError nếu sai thứ tự.

        `_require`: trạng thái BẮT BUỘC đang có để thao tác này hợp lệ (vd. retry
        chỉ từ FAILED) — chặn cả thao tác "cùng trạng thái" vô nghĩa.
        """
        required = fields.pop("_require", None)
        want_duration = fields.pop("_duration", False)
        status = fields.get("status")
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT * FROM processing_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"job không tồn tại: {job_id!r}")
            record = _row_to_record(row)
            now = time.time()
            if required is not None and record.status != required:
                raise TransitionError(
                    f"thao tác này chỉ hợp lệ khi job ở trạng thái {required}, "
                    f"hiện đang {record.status}"
                )
            if (
                status is not None
                and status != record.status
                and status not in TRANSITIONS.get(record.status, set())
            ):
                raise TransitionError(f"không hợp lệ: {record.status} -> {status}")
            updates = dict(fields)
            if want_duration:
                base = record.started_at or record.created_at
                updates["duration_ms"] = max(0, int((now - base) * 1000))
            updates["updated_at"] = now
            if "warnings" in updates:
                updates["warnings"] = json.dumps(updates["warnings"], ensure_ascii=False)
            assignments = ", ".join(f"{name}=?" for name in updates)
            values = list(updates.values()) + [job_id]
            conn.execute(
                f"UPDATE processing_jobs SET {assignments} WHERE id=?", values
            )
            conn.commit()
            return self.get(job_id)


def make_store(settings) -> SqliteProcessingStore:
    """Dựng store từ settings; mặc định `data/processing.db` (CWD của service)."""
    path = (getattr(settings, "processing_db_path", None) or "data/processing.db").strip()
    return SqliteProcessingStore(path)


__all__ = [
    "CANCELLED",
    "DOC_TYPES",
    "FAILED",
    "PENDING",
    "PROCESSING",
    "PROGRESS_BY_STEP",
    "READY",
    "STATUSES",
    "STEPS",
    "STEP_EMBEDDING",
    "STEP_EXTRACTING",
    "STEP_FINALIZING",
    "STEP_PARSING",
    "STEP_RECEIVED",
    "TERMINAL_STATUSES",
    "TRANSITIONS",
    "VISIBILITIES",
    "JobRecord",
    "SqliteProcessingStore",
    "TransitionError",
    "format_error",
    "make_store",
    "progress_for",
]
