"""Pipeline xử lý một job: EXTRACTING → PARSING → EMBEDDING → FINALIZING.

Tái dùng đúng các khối đã có của AI service (thiết kế §5.2):
- `text.py`      — đọc file, magic bytes, trích xuất PDF/DOCX/OCR, làm sạch văn bản
- `routes.py`    — prompt LLM + `normalize_cv`/`normalize_job` (import TRỄN trong
                   `parse_payload` để tránh vòng phụ thuộc routes ↔ processing)
- `recommend.py` — `index()` nhúng + upsert Chroma (metadata `status=READY`) +
                   payload store cho hybrid ranking

Phân loại lỗi (thiết kế §6): lỗi *vĩnh viễn* (file hỏng, text rỗng...) -> FAILED
ngay; lỗi *tạm thời* (Ollama timeout, Chroma lock...) -> worker retry có backoff.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .. import errors
from .. import recommend as recommend_mod
from .. import text as text_mod
from ..config import Settings
from . import store as store_mod
from .store import JobRecord

logger = logging.getLogger("app.processing")

# Mã lỗi coi là vĩnh viễn — retry không cứu được (thiết kế §6)
PERMANENT_CODES = {"CV_FILE_REQUIRED", "CV_FILE_INVALID", "CV_FILE_TOO_LARGE",
                   "CV_TEXT_EMPTY", "CV_ENCRYPTED"}

MIN_TEXT_CHARS = 50  # đồng bộ route analyze-cv


class JobError(Exception):
    """Lỗi đã phân loại; code khớp `errors.ApiError.detail` hoặc mã nội bộ."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class PermanentError(JobError):
    """Không retry — chuyển thẳng FAILED."""


class TransientError(JobError):
    """Có thể retry (mạng, model, lock...) — worker backoff rồi thử lại."""


@dataclass(frozen=True)
class Source:
    """Kết quả bước EXTRACTING."""

    text: str
    kind: str  # pdf | docx | doc | text
    label: str  # pdf_text | pdf_ocr | docx | doc | job_text | text
    pages: int = 0
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)


def read_source(job: JobRecord, settings: Settings) -> Source:
    """Đọc nội dung job: file đã lưu (CV upload) hoặc text gửi thẳng (JD).

    File thiếu/không đọc được là lỗi vĩnh viễn (không thuộc loại sự cố tạm thời).
    """
    if job.source_kind == "text":
        label = "job_text" if job.document_type == "jd" else "text"
        return Source(text=job.text or "", kind="text", label=label)

    if not job.source_ref:
        raise PermanentError("SOURCE_MISSING", "job file nhưng thiếu source_ref")
    try:
        raw = Path(job.source_ref).read_bytes()
    except OSError as exc:
        raise PermanentError("SOURCE_MISSING", f"không đọc được file nguồn: {exc}") from exc

    kind = text_mod.classify(raw[:16], settings.allow_legacy_doc)
    if kind == "docx":  # zip hợp lệ phải có word/document.xml (đồng bộ read_upload)
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                if "word/document.xml" not in zf.namelist():
                    raise errors.cv_file_invalid()
        except errors.ApiError:
            raise
        except Exception as exc:  # zip hỏng / thiếu entry -> file không hợp lệ
            raise errors.cv_file_invalid() from exc

    with text_mod.temp_file(raw) as path:
        text, pages, ocr_used, warnings = text_mod.extract_text(path, kind, settings)
    label = "pdf_ocr" if (kind == "pdf" and ocr_used) else (
        "pdf_text" if kind == "pdf" else kind
    )
    return Source(text=text, kind=kind, label=label, pages=pages,
                  ocr_used=ocr_used, warnings=list(warnings))


def prepare_text(source: Source, settings: Settings, warnings: list[str]) -> str:
    """Làm sạch + gộp dòng ngắt quèn + giới hạn độ dài (đồng bộ route analyze-cv)."""
    cleaned = text_mod.merge_broken_lines(text_mod.clean_text(source.text))
    if len(cleaned) > settings.max_llm_chars:
        cleaned = cleaned[: settings.max_llm_chars]
        warnings.append("TRUNCATED")
    return cleaned


def parse_payload(
    job: JobRecord, cleaned: str, settings: Settings, llm, warnings: list[str]
) -> dict:
    """Gọi LLM trích xuất rồi chuẩn hoá payload (prompt + normalize dùng chung route)."""
    from .. import routes as routes_mod  # import trễ: routes phụ thuộc processing.store

    if job.document_type == "cv":
        data = llm.chat(routes_mod.llm_mod.SYSTEM_PROMPT,
                        routes_mod._user_cv_prompt(settings, cleaned))
        return routes_mod.normalize_cv(data, cleaned, settings, warnings)
    data = llm.chat(routes_mod.llm_mod.SYSTEM_PROMPT,
                    routes_mod._user_job_prompt(settings, job.title, cleaned))
    return routes_mod.normalize_job(data, settings, warnings)


def build_record(job: JobRecord, payload: dict) -> dict:
    """Tài liệu cho `/internal/index`: id + type + payload + metadata truy cập."""
    record: dict = {"id": job.document_id, "type": job.document_type, **payload}
    if job.owner_id:
        record["owner_id"] = job.owner_id
    if job.visibility:
        record["visibility"] = job.visibility
    return record


def _from_api_error(exc: errors.ApiError) -> JobError:
    if exc.detail in PERMANENT_CODES:
        return PermanentError(exc.detail, exc.message)
    return TransientError(exc.detail, exc.message)


def run(
    job: JobRecord,
    store: store_mod.SqliteProcessingStore,
    settings: Settings,
    llm,
    *,
    index_fn=None,
) -> JobRecord:
    """Chạy hết một job đã claim (status PROCESSING); cập nhật store qua checkpoint.

    `index_fn(documents, settings, version=...)` mặc định là `recommend.index`
    (nhúng BGE-M3 + upsert Chroma + payload store) — inject được cho test.

    Nâng cấp TransientError/PermanentError cho worker xử lý retry (không tự đổi
    trạng thái tại đây để mọi transition đi qua store — kiểm soát tập trung theo §2.2).
    """
    index_fn = index_fn or recommend_mod.index
    warnings: list[str] = []

    try:
        store.set_step(job.id, store_mod.STEP_EXTRACTING)
        source = read_source(job, settings)
        warnings.extend(source.warnings)
        cleaned = prepare_text(source, settings, warnings)
        if len(cleaned) < MIN_TEXT_CHARS:
            raise errors.cv_text_empty()

        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        store.set_content_hash(job.id, content_hash)

        store.set_step(job.id, store_mod.STEP_PARSING)
        payload = parse_payload(job, cleaned, settings, llm, warnings)

        store.set_step(job.id, store_mod.STEP_EMBEDDING)
        record = build_record(job, payload)
        index_result = index_fn([record], settings, version=job.version)
        logger.debug("indexed job=%s result=%s", job.id, index_result)

        store.set_step(job.id, store_mod.STEP_FINALIZING)
        return store.mark_ready(job.id, warnings=warnings)
    except JobError:
        raise
    except errors.ApiError as exc:
        raise _from_api_error(exc) from exc
    except Exception as exc:  # sự cố không lường trước -> tạm thời (worker retry)
        raise TransientError("INTERNAL", f"{type(exc).__name__}: {exc}") from exc


__all__ = [
    "JobError", "PermanentError", "Source", "TransientError", "build_record",
    "parse_payload", "prepare_text", "read_source", "run",
]
