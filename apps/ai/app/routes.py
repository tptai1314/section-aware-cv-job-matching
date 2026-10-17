"""Các router: analyze-cv / extract-cv, extract-job, embeddings."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from . import embeddings as emb_mod
from . import errors, explain as explain_mod, llm as llm_mod, normalize, schemas, text as text_mod
from . import recommend as recommend_mod
from .config import Settings
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