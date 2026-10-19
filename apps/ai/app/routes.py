"""Các router: analyze-cv / extract-cv, extract-job, embeddings, jobs (hàng đợi xử lý)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from . import embeddings as emb_mod
from . import errors, normalize, schemas
from . import explain as explain_mod
from . import llm as llm_mod
from . import recommend as recommend_mod
from . import text as text_mod
from .config import Settings
from .processing import store as store_mod
from .security import require_internal_token

router = APIRouter(dependencies=[Depends(require_internal_token)])

JSON_ONLY_NOTE = "Trả về DUY NHẤT JSON, không kèm giải thích."

CV_SCHEMA_DOC = """{
  "skills": [{"name": "str", "proficiency_level": 1..10|null, "confidence": 0..1|null, "evidence": "đoạn text ngắn trong CV"}],
  "experience": [{"title", "company", "start_date": "YYYY-MM|YYYY|null", "end_date": "YYYY-MM|YYYY|null",
                  "is_current": bool, "description", "confidence"}],
  "education": [{"degree", "school", "start_year": null|int, "end_year": null|int, "confidence"}],
  "projects": [{"name", "technologies": ["str"], "description", "confidence"}],
  "summary": "str|null"
}"""

JOB_SCHEMA_DOC = """{
  "title": "str|null",
  "summary": "str|null (mô tả công việc đã làm sạch, không thêm nhận xét)",
  "required_skills": [{"name", "confidence", "evidence": "đoạn text trong văn bản"}],
  "optional_skills": [{"name", "confidence", "evidence": "đoạn text trong văn bản"}],
  "responsibilities": ["str"],
  "min_years_experience": int|null,
  "budget": {"min": number|null, "max": number|null, "currency": "str|null"} | null
}"""


def _user_cv_prompt(settings: Settings, text: str) -> str:
    return (
        f"Schema json cần trả về:\n{CV_SCHEMA_DOC}\n\n"
        "Quy tắc:\n"
        "- Mỗi skill PHẢI có evidence là đoạn ngắn xuất hiện y hệt trong văn bản CV; không tìm thấy thì loại bỏ.\n"
        "- Không tự chia nhỏ/ghép kỹ năng (ReactJS != React Native). Ngày tháng chỉ lấy khi CV ghi rõ.\n"
        f"{JSON_ONLY_NOTE}\n\n"
        f"{llm_mod.UNTRUSTED_NOTE}\n\n"
        f"NỘI DUNG CV (tối đa {settings.max_llm_chars} ký t�? đã chuẩn hoá):\n---\n{text}\n---"
    )


def _user_job_prompt(settings: Settings, title: str | None, text: str) -> str:
    return (
        f"Schema json cần trả về:\n{JOB_SCHEMA_DOC}\n\n"
        "Quy tắc:\n"
        "- Mỗi skill chỉ trích xuất khi ĐƯỢC NHẮC ĐẾN trong văn bản, kèm evidence.\n"
        "- min_years_experience là số nguyên; không có thì null. Budget chỉ lấy khi có số liệu cụ thể.\n"
        f"{JSON_ONLY_NOTE}\n\n"
        f"{llm_mod.UNTRUSTED_NOTE}\n\n"
        f"Tiêu đ�?job (nếu có): {title or '(không rõ)'}\n"
        f"NỘI DUNG JOB (tối đa {settings.max_llm_chars} ký t�?:\n---\n{text}\n---"
    )


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _clamp_confidence(value) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return min(max(float(value), 0.0), 1.0)


def _norm_skill(item, settings: Settings) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    if not name or len(name) > 100:
        return None
    prof = item.get("proficiency_level")
    if isinstance(prof, str):
        prof = int(prof) if prof.strip().isdigit() else None
    prof = prof if isinstance(prof, int) and 1 <= prof <= 10 else None
    return {
        "name": name,
        "proficiency_level": prof,
        "confidence": _clamp_confidence(item.get("confidence")),
        "evidence": (str(item.get("evidence") or "").strip()[:200]) or None,
    }


def _norm_experience(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    title = (str(item.get("title") or "")).strip()[:150] or None
    company = (str(item.get("company") or "")).strip()[:150] or None
    end_raw = item.get("end_date")
    is_current = bool(item.get("is_current")) or normalize.is_current_marker(
        str(end_raw) if end_raw else None
    )
    end = None if is_current else normalize.normalize_date(str(end_raw) if end_raw else None)
    return {
        "title": title,
        "company": company,
        "start_date": normalize.normalize_date(item.get("start_date")),
        "end_date": end,
        "is_current": is_current,
        "description": (str(item.get("description") or "").strip()[:1000]) or None,
        "confidence": _clamp_confidence(item.get("confidence")),
    }


def _to_year(value) -> int | None:
    try:
        y = int(str(value))
    except (TypeError, ValueError):
        return None
    return y if 1950 <= y <= 2100 else None


def _norm_education(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    return {
        "degree": (str(item.get("degree") or "").strip()[:150]) or None,
        "school": (str(item.get("school") or "").strip()[:150]) or None,
        "start_year": _to_year(item.get("start_year")),
        "end_year": _to_year(item.get("end_year")),
        "description": (str(item.get("description") or "").strip()[:1000]) or None,
        "confidence": _clamp_confidence(item.get("confidence")),
    }


def _norm_project(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    techs = _as_list(item.get("technologies"))
    techs = normalize.dedupe_technologies([str(t) for t in techs])[:30]
    return {
        "name": (str(item.get("name") or "").strip()[:150]) or None,
        "technologies": techs,
        "description": (str(item.get("description") or "").strip()[:1000]) or None,
        "confidence": _clamp_confidence(item.get("confidence")),
    }


def normalize_cv(data: dict, cv_full_text: str, settings: Settings, warnings: list[str]) -> dict:
    """Biến raw LLM output thành payload đúng schema, thực thi mục 7."""
    raw_skills = _as_list(data.get("skills"))
    skills = []
    for s in raw_skills:
        item = _norm_skill(s, settings)
        if item:
            skills.append(item)
    if not skills:
        warnings.append("EMPTY_SECTION:skills")
    dropped = normalize.check_evidence(skills, cv_full_text, settings.strict_evidence)
    if dropped and settings.strict_evidence:
        warnings.append(f"EVIDENCE_MISSING:{len(dropped)}")
    skills = normalize.dedupe_skills(skills)
    skills = normalize.clamp(skills, schemas.LIMIT_SKILLS, warnings, "skills")

    experiences = [_norm_experience(e) for e in _as_list(data.get("experience"))]
    experiences = [e for e in experiences if e]
    if not experiences:
        warnings.append("EMPTY_SECTION:experience")
    experiences = normalize.clamp(experiences, schemas.LIMIT_EXPERIENCE, warnings, "experience")

    educations = [_norm_education(e) for e in _as_list(data.get("education"))]
    educations = [e for e in educations if e]
    if not educations:
        warnings.append("EMPTY_SECTION:education")
    educations = normalize.clamp(educations, schemas.LIMIT_EDUCATION, warnings, "education")

    projects = [_norm_project(p) for p in _as_list(data.get("projects"))]
    projects = [p for p in projects if p]
    if not projects:
        warnings.append("EMPTY_SECTION:projects")
    projects = normalize.clamp(projects, schemas.LIMIT_PROJECTS, warnings, "projects")

    summary = (str(data.get("summary") or "").strip())[:1500] or None

    return {
        "skills": skills,
        "experience": experiences,
        "education": educations,
        "projects": projects,
        "summary": summary,
    }


def _norm_job_skill(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    if not name or len(name) > 100:
        return None
    return {
        "name": name,
        "confidence": _clamp_confidence(item.get("confidence")),
        "evidence": (str(item.get("evidence") or "").strip()[:200]) or None,
    }


def normalize_job(data: dict, settings: Settings, warnings: list[str]) -> dict:
    req = [_norm_job_skill(s) for s in _as_list(data.get("required_skills"))]
    req = normalize.dedupe_skills([r for r in req if r])
    req = normalize.clamp(req, schemas.LIMIT_SKILLS, warnings, "required_skills")

    opt = [_norm_job_skill(s) for s in _as_list(data.get("optional_skills"))]
    opt = normalize.dedupe_skills([o for o in opt if o])
    opt = normalize.clamp(opt, schemas.LIMIT_SKILLS, warnings, "optional_skills")

    responsibilities = [str(r).strip()[:200] for r in _as_list(data.get("responsibilities"))]
    responsibilities = normalize.dedupe([r for r in responsibilities if r], lambda it: it)
    responsibilities = normalize.clamp(responsibilities, 20, warnings, "responsibilities")

    title = (str(data.get("title") or "").strip())[:150] or None
    summary = (str(data.get("summary") or "").strip())[:3000] or None

    years = data.get("min_years_experience")
    try:
        years = int(years) if years is not None else None
    except (TypeError, ValueError):
        years = None
    if years is not None and not (0 <= years <= 20):
        years = None

    budget = None
    raw_budget = data.get("budget")
    if isinstance(raw_budget, dict):
        budget = {
            "min": float(raw_budget["min"]) if isinstance(raw_budget.get("min"), (int, float)) and raw_budget["min"] >= 0 else None,
            "max": float(raw_budget["max"]) if isinstance(raw_budget.get("max"), (int, float)) and raw_budget["max"] >= 0 else None,
            "currency": (str(raw_budget.get("currency") or "").strip()[:10]) or None,
        }

    return {
        "title": title,
        "summary": summary,
        "required_skills": req,
        "optional_skills": opt,
        "responsibilities": responsibilities,
        "min_years_experience": years,
        "budget": budget,
    }


def _source(kind: str, ocr_used: bool) -> str:
    if kind == "pdf":
        return "pdf_ocr" if ocr_used else "pdf_text"
    return "docx" if kind == "docx" else "doc"


@router.post("/api/v1/ai/analyze-cv", response_model=schemas.CvResponse)
@router.post("/internal/extract-cv", response_model=schemas.CvResponse, include_in_schema=False)
async def analyze_cv(file: UploadFile | None = File(default=None), request: Request = None):
    settings: Settings = request.app.state.settings
    client: llm_mod.OllamaClient = request.app.state.llm
    started = time.monotonic()

    raw, kind = await text_mod.read_upload(file, settings)
    with text_mod.temp_file(raw) as path:
        text, pages, ocr_used, warnings = await run_in_threadpool(
            text_mod.extract_text, path, kind, settings
        )
    cleaned = text_mod.merge_broken_lines(text_mod.clean_text(text))
    chars = len(cleaned)
    if chars < 50:
        raise errors.cv_text_empty()
    if chars > settings.max_llm_chars:
        cleaned = cleaned[: settings.max_llm_chars]
        warnings.append("TRUNCATED")

    data = await run_in_threadpool(
        client.chat, llm_mod.SYSTEM_PROMPT, _user_cv_prompt(settings, cleaned)
    )
    body = normalize_cv(data, cleaned, settings, warnings)

    meta = schemas.Meta(
        source=_source(kind, ocr_used),
        pages=pages,
        chars=chars,
        truncated="TRUNCATED" in warnings,
        ocr_used=ocr_used,
        model=settings.ollama_model,
        duration_ms=int((time.monotonic() - started) * 1000),
        warnings=list(dict.fromkeys(warnings))[:20],
    )
    return schemas.CvResponse(**body, meta=meta)


@router.post("/internal/extract-job", response_model=schemas.JobResponse)
async def extract_job(payload: dict, request: Request):
    settings: Settings = request.app.state.settings
    client: llm_mod.OllamaClient = request.app.state.llm
    started = time.monotonic()

    text = (str(payload.get("text") or "")).strip()
    if len(text) < 10:
        raise errors.cv_text_empty()
    title = (str(payload.get("title") or "").strip()) or None
    cleaned = text_mod.merge_broken_lines(text_mod.clean_text(text))
    if len(cleaned) > settings.max_llm_chars:
        cleaned = cleaned[: settings.max_llm_chars]

    data = await run_in_threadpool(
        client.chat, llm_mod.SYSTEM_PROMPT, _user_job_prompt(settings, title, cleaned)
    )
    warnings: list[str] = []
    body = normalize_job(data, settings, warnings)
    meta = schemas.Meta(
        source="job_text",
        chars=len(cleaned),
        truncated=len(text) > settings.max_llm_chars,
        model=settings.ollama_model,
        duration_ms=int((time.monotonic() - started) * 1000),
        warnings=list(dict.fromkeys(warnings))[:20],
    )
    return schemas.JobResponse(**body, meta=meta)


@router.post("/internal/embeddings", response_model=schemas.EmbeddingsResponse)
async def embeddings(body: schemas.EmbeddingsRequest, request: Request):
    settings: Settings = request.app.state.settings
    vectors = await run_in_threadpool(
        emb_mod.embed, body.texts, settings.embed_model, settings.embed_device
    )
    return schemas.EmbeddingsResponse(model=settings.embed_model, dim=len(vectors[0]), vectors=vectors)


@router.post("/internal/explain", response_model=schemas.ExplainResponse, include_in_schema=False)
async def explain(body: schemas.ExplainRequest, request: Request):
    """Lời giải thích evidence-grounded (E17/E19). LLM lỗi -> ready=false, không 5xx."""
    settings: Settings = request.app.state.settings
    client: llm_mod.OllamaClient = request.app.state.llm
    started = time.monotonic()
    result = await run_in_threadpool(
        explain_mod.explain_with_llm, client, settings, body.job, body.cv, body.match
    )
    result["duration_ms"] = int((time.monotonic() - started) * 1000)
    return result


@router.post("/internal/rerank", response_model=schemas.RerankResponse, include_in_schema=False)
async def rerank(body: schemas.RerankRequest, request: Request):
    """Pointwise rubric reranking (E18). Mỗi candidate ready=False nếu LLM lỗi."""
    settings: Settings = request.app.state.settings
    client: llm_mod.OllamaClient = request.app.state.llm
    started = time.monotonic()
    results = await run_in_threadpool(
        explain_mod.rerank_with_llm, client, settings, body.job, body.candidates
    )
    return schemas.RerankResponse(
        model=settings.ollama_model,
        duration_ms=int((time.monotonic() - started) * 1000),
        results=results,
    )


@router.post(
    "/internal/index",
    response_model=schemas.IndexResponse,
    include_in_schema=False,
)
async def index(body: schemas.IndexRequest, request: Request):
    """Upsert tài liệu đã trích xuất vào Chroma + payload store (BGE-M3)."""
    settings: Settings = request.app.state.settings
    result = await run_in_threadpool(
        recommend_mod.index,
        [document.model_dump() for document in body.documents],
        settings,
    )
    return schemas.IndexResponse(**result)


@router.post(
    "/internal/index/delete",
    response_model=schemas.IndexDeleteResponse,
    include_in_schema=False,
)
async def delete_index(body: schemas.IndexDeleteRequest, request: Request):
    """Xoá vector (Chroma) + payload của tài liệu (H26: đồng bộ xoá Postgres↔Chroma)."""
    settings: Settings = request.app.state.settings
    result = await run_in_threadpool(
        recommend_mod.delete,
        [document.model_dump() for document in body.documents],
        settings,
    )
    return schemas.IndexDeleteResponse(**result)


@router.post(
    "/internal/recommend",
    response_model=schemas.RecommendResponse,
    include_in_schema=False,
)
async def recommend(body: schemas.RecommendRequest, request: Request):
    """Hybrid Ranking (§7): stateless nếu có `candidates`, ngược lại retrieval Chroma."""
    settings: Settings = request.app.state.settings
    candidates = None if body.candidates is None else [c.model_dump() for c in body.candidates]
    results = await run_in_threadpool(
        recommend_mod.recommend,
        body.query,
        candidates,
        body.direction,
        body.top_k,
        body.keyword_scores,
        body.keyword_alpha,
        settings,
        pool_size=body.pool_size,
        owner_id=body.owner_id,
        visibilities=body.visibility,
        llm_client=request.app.state.llm,
        explain=body.explain,
        explain_top_n=body.explain_top_n,
    )
    return schemas.RecommendResponse(direction=body.direction, results=results)


# --------------------------------------------------------------------------- #
# Hàng đợi xử lý file (docs/DESIGN_processing_queue.md §4; plan §9.3, §10.1)
# --------------------------------------------------------------------------- #


def _status_response(record: store_mod.JobRecord, settings: Settings) -> schemas.JobStatusResponse:
    warnings: list[str] = []
    if record.warnings:
        try:
            warnings = json.loads(record.warnings) if isinstance(record.warnings, str) else list(record.warnings)
        except (TypeError, ValueError):
            warnings = []
    return schemas.JobStatusResponse(
        job_id=record.id,
        status=record.status,
        step=record.step,
        progress=store_mod.progress_for(record.status, record.step),
        attempt_count=record.attempt_count,
        max_attempts=settings.job_max_attempts,
        last_error=record.last_error,
        warnings=[str(w) for w in warnings],
        document_type=record.document_type,
        document_id=record.document_id,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        finished_at=record.finished_at,
        duration_ms=record.duration_ms,
    )


@router.post("/internal/jobs", response_model=schemas.JobCreateResponse, status_code=202)
async def create_processing_job(
    request: Request,
    document_type: str = Form(...),
    document_id: str = Form(...),
    version: int = Form(1),
    owner_id: str | None = Form(default=None),
    visibility: str | None = Form(default=None),
    title: str | None = Form(default=None),
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    """Nộp job xử lý nền: CV (file upload hoặc text đã trích) / JD (text).

    Idempotency (design §5.2): job đã có cho cùng (document_type, document_id,
    version) được dùng lại — không tạo bản trùng.
    """
    settings: Settings = request.app.state.settings
    store: store_mod.SqliteProcessingStore = request.app.state.processing_store
    worker = request.app.state.processing_worker

    doc_type = (document_type or "").strip().lower()
    if doc_type not in store_mod.DOC_TYPES:
        raise errors.invalid_document_type()
    doc_id = (document_id or "").strip()
    if not doc_id or len(doc_id) > 100:
        raise errors.bad_request("DOCUMENT_ID_INVALID", "`document_id` phải 1–100 ký tự")
    if not (0 < version <= 1000):
        raise errors.bad_request("VERSION_INVALID", "`version` phải trong khoảng 1–1000")
    if visibility is not None and visibility.strip().lower() not in store_mod.VISIBILITIES:
        raise errors.bad_request(
            "VISIBILITY_INVALID", "`visibility` phải là public | private | unlisted"
        )
    visibility = visibility.strip().lower() if visibility else None
    title = (title or "").strip()[:150] or None

    source_kind, source_ref, raw_text = "text", None, None
    if doc_type == "jd":
        if file is not None:
            raise errors.bad_request(
                "JD_FILE_NOT_ALLOWED", "JD gửi dạng text (form field `text`), không nhận file"
            )
        if not text or len(text.strip()) < 10:
            raise errors.job_source_required("JD_TEXT_REQUIRED", "JD cần `text` >= 10 ký tự")
        raw_text = text[:100_000]
    elif file is not None:
        raw, kind = await text_mod.read_upload(file, settings)  # magic bytes + giới hạn size
        files_dir = Path(settings.processing_files_dir or "data/processing_files")
        files_dir.mkdir(parents=True, exist_ok=True)
        path = files_dir / f"{uuid.uuid4().hex}.{kind or 'bin'}"
        path.write_bytes(raw)
        source_kind, source_ref = "file", str(path)
    elif text and len(text.strip()) >= 10:
        raw_text = text[:100_000]  # CV đã trích xuất sẵn (bỏ bước EXTRACTING)
    else:
        raise errors.job_source_required(
            "CV_SOURCE_REQUIRED", "CV cần `file` (upload) hoặc `text` (đã trích xuất)"
        )

    existing = store.get_by_doc(doc_type, doc_id, version)
    try:
        record, deduplicated = store.create(
            document_type=doc_type,
            document_id=doc_id,
            version=version,
            owner_id=(owner_id or "").strip()[:100] or None,
            visibility=visibility,
            source_kind=source_kind,
            source_ref=source_ref,
            text=raw_text,
            title=title,
        )
    except ValueError as exc:
        if source_ref:
            Path(source_ref).unlink(missing_ok=True)  # không để file mồ côi khi lỗi
        raise errors.bad_request("JOB_INVALID", str(exc)) from exc

    # file tạm không thuộc job dùng lại (dedup) hoặc của job bị thay nguồn (requeue)
    if source_ref and record.source_ref != source_ref:
        Path(source_ref).unlink(missing_ok=True)
    if existing and existing.source_ref and existing.source_ref != record.source_ref:
        Path(existing.source_ref).unlink(missing_ok=True)
    if record.status == store_mod.PENDING:
        worker.submit(record.id)  # đánh thức worker (vẫn PENDING trong DB nếu chưa chạy)

    return schemas.JobCreateResponse(
        job_id=record.id,
        status=record.status,
        document_type=record.document_type,
        document_id=record.document_id,
        version=record.version,
        deduplicated=deduplicated,
        created_at=record.created_at,
    )


@router.get("/internal/jobs/{job_id}", response_model=schemas.JobStatusResponse)
def get_processing_job(job_id: str, request: Request):
    """Trạng thái xử lý để NestJS poll `/api/cvs/:id/processing-status`."""
    settings: Settings = request.app.state.settings
    store: store_mod.SqliteProcessingStore = request.app.state.processing_store
    record = store.get(job_id)
    if record is None:
        raise errors.job_not_found()
    return _status_response(record, settings)


@router.delete("/internal/jobs/{job_id}", response_model=schemas.JobStatusResponse)
def cancel_processing_job(job_id: str, request: Request):
    """Hủy job còn chờ (PENDING). Job đang xử lý không dừng giữa chừng."""
    settings: Settings = request.app.state.settings
    store: store_mod.SqliteProcessingStore = request.app.state.processing_store
    record = store.get(job_id)
    if record is None:
        raise errors.job_not_found()
    try:
        record = store.cancel(job_id)
    except store_mod.TransitionError as exc:
        raise errors.job_conflict("JOB_NOT_CANCELLABLE", str(exc)) from exc
    store_mod.remove_source_file(record)
    return _status_response(record, settings)


@router.post("/internal/jobs/{job_id}/retry", response_model=schemas.JobStatusResponse)
def retry_processing_job(job_id: str, request: Request):
    """Retry thủ công job FAILED (attempt reset, design §2.2)."""
    settings: Settings = request.app.state.settings
    store: store_mod.SqliteProcessingStore = request.app.state.processing_store
    worker = request.app.state.processing_worker
    record = store.get(job_id)
    if record is None:
        raise errors.job_not_found()
    try:
        record = store.retry(job_id)
    except store_mod.TransitionError as exc:
        raise errors.job_conflict("JOB_NOT_RETRYABLE", str(exc)) from exc
    worker.submit(record.id)
    return _status_response(record, settings)