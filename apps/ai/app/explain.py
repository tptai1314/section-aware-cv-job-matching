"""Qwen explanation + rubric reranking (plan §8, §6 — E17/E18/E19).

- explain: tạo lời giải thích evidence-grounded (từng claim có bằng chứng
  trích từ job/cv, verdict hit|missing|unknown). Không LLM -> ready=false
  (fallback, Không để lỗi LLM làm hỏng recommendation).
- rerank: pointwise chấm 0-100 theo rubric (§8.3) cho từng candidate,
  trả breakdown từng thành phần + bằng chứng + thiếu hụt.
"""

from __future__ import annotations

import json

from . import errors
from . import llm as llm_mod
from .config import Settings

JSON_ONLY_NOTE = "Trả về DUY NHẤT JSON, không kèm giải thích."

EXPLAIN_SCHEMA_DOC = """{
  "summary": "str (tóm tắt 2-3 câu vì sao CV này phù hợp/không phù hợp job)",
  "claims": [
    {
      "statement": "str (khẳng định ngắn)",
      "verdict": "hit|missing|unknown",
      "evidence": "trích y hệt đoạn trong JOB hoặc CV, hoặc null khi không có bằng chứng",
      "source": "job|cv|null",
      "confidence": 0..1|null
    }
  ]
}"""

EXPLAIN_RULES = """Quy tắc:
- Mỗi claim PHẢI có evidence trích y hệt đoạn văn bản đã cung cấp (job hoặc cv).
  Không được bịa bằng chứng -> nếu không tìm thấy, evidence=null, source=null.
- verdict: "hit" = CV đáp ứng yêu cầu job; "missing" = job yêu cầu nhưng CV thiếu;
  "unknown" = chưa xác định được (không đủ thông tin).
- Chỉ liệt kê claims có ý nghĩa (yêu cầu trúng/thiếu quan trọng), tối đa 12 claim."""

RERANK_SCHEMA_DOC = """{
  "results": [
    {
      "id": "str (chuỗi id của candidate tương ứng)",
      "score": 0..100 (số nguyên),
      "breakdown": {
        "skills_required": 0..40,
        "skills_priority": 0..20,
        "experience": 0..20,
        "responsibilities": 0..10,
        "condition_extra": 0..10
      },
      "reason": "str ngắn (1 câu)",
      "evidence": ["đoạn trích trong cv của candidate"],
      "missing": ["yêu cầu job mà cv thiếu"]
    }
  ]
}"""

RERANK_RULES = """Rubric (tổng tối đa 100):
- skills_required: 0..40 (kỹ năng bắt buộc)
- skills_priority: 0..20 (kỹ năng ưu tiên)
- experience: 0..20 (kinh nghiệm liên quan)
- responsibilities: 0..10 (mức hợp nhiệm vụ công việc)
- condition_extra: 0..10 (điều kiện bổ sung như ngôn ngữ, budget, thời gian)
Chỉ trích evidence có trong văn bản cv được cung cấp. Phải trả về ĐÚNG mọi id
candidate có trong input. Không thêm id không có. Score phải là số nguyên."""


def _preview(value, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…[CẮT]"


def _haystack(value) -> str:
    """Gộp nội dung job/cv thành chuỗi chuẩn hoá để kiểm chứng evidence."""
    if value is None:
        return ""
    return " ".join(json.dumps(value, ensure_ascii=False).split()).lower()


def _grounded(haystack: str, evidence: str) -> bool:
    needle = " ".join(evidence.split()).lower()
    return bool(needle) and needle in haystack


def explain_prompt(settings: Settings, job: dict, cv: dict, match: dict | None) -> str:
    job_txt = _preview(job, settings.max_llm_chars // 2)
    cv_txt = _preview(cv, settings.max_llm_chars // 2)
    match_txt = _preview(match, 2000) if match else "Không có số điểm match sẵn có."
    return (
        f"Schema json cần trả về:\n{EXPLAIN_SCHEMA_DOC}\n\n"
        f"{EXPLAIN_RULES}\n"
        f"{JSON_ONLY_NOTE}\n\n"
        f"{llm_mod.UNTRUSTED_NOTE}\n\n"
        "THÔNG TIN JOB (JSON):\n---\n"
        f"{job_txt}\n---\n\n"
        "THÔNG TIN CV (JSON):\n---\n"
        f"{cv_txt}\n---\n\n"
        "ĐIỂM MATCH (nếu có):\n---\n"
        f"{match_txt}\n---"
    )


def rerank_job_preview(settings: Settings, job: dict) -> str:
    return _preview(job, settings.max_llm_chars // 3)


def rerank_candidate_prompt(settings: Settings, job: dict, candidate: dict, index: int) -> str:
    cid = str(candidate.get("id") or f"c{index}")
    return (
        f"Schema json cần trả về:\n{RERANK_SCHEMA_DOC}\n\n"
        f"{RERANK_RULES}\n"
        f"{JSON_ONLY_NOTE}\n\n"
        f"{llm_mod.UNTRUSTED_NOTE}\n\n"
        "JOB (JSON):\n---\n"
        f"{rerank_job_preview(settings, job)}\n---\n\n"
        f"CANDIDATE CẦN CHẤM (JSON cv id={cid}):\n---\n"
        f"{_preview(candidate, settings.max_llm_chars // 3)}\n---\n\n"
        f"LƯU Ý: chỉ chấm candidate này, id={cid} phải giữ nguyên trong kết quả."
    )


def parse_explain(
    data: dict, job: dict | None = None, cv: dict | None = None
) -> tuple[list[dict], str | None, str | None]:
    """(claims, summary, error) — trả summary=None khi thiếu.

    Khi có `job`/`cv`, mọi claim `hit`/`missing` PHẢI có evidence trích y hệt
    nguồn; claim không đủ căn cứ bị hạ về `unknown` (§15 — kiểm tra mọi claim).
    Bỏ trống `job`/`cv` thì không kiểm chứng (giữ hành vi cũ).
    """
    if not isinstance(data, dict):
        return [], None, "NOT_OBJECT"
    verify = job is not None or cv is not None
    job_text = _haystack(job)
    cv_text = _haystack(cv)
    claims = []
    for item in data.get("claims") or []:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or "").strip()
        if not statement:
            continue
        verdict = item.get("verdict")
        if verdict not in ("hit", "missing", "unknown"):
            verdict = "unknown"
        confidence = item.get("confidence")
        if isinstance(confidence, str):
            confidence = float(confidence) if confidence.strip().lstrip("-").replace(".", "", 1).isdigit() else None
        if not isinstance(confidence, (int, float)):
            confidence = None
        evidence = (str(item.get("evidence") or "").strip()[:500]) or None
        source = item.get("source") if item.get("source") in ("job", "cv") else None
        if verify:
            verdict, evidence, source = _verify_claim(
                verdict, evidence, source, job_text, cv_text
            )
        if verdict == "unknown":
            # unknown = chưa xác định -> không có bằng chứng khẳng định nào.
            evidence = None
            source = None
        claims.append({
            "statement": statement[:400],
            "verdict": verdict,
            "evidence": evidence,
            "source": source,
            "confidence": min(max(float(confidence), 0.0), 1.0) if confidence is not None else None,
        })
    summary = (str(data.get("summary") or "").strip()[:2000]) or None
    return claims, summary, None


def _verify_claim(
    verdict: str,
    evidence: str | None,
    source: str | None,
    job_text: str,
    cv_text: str,
) -> tuple[str, str | None, str | None]:
    """Đối chiếu evidence với job/cv; hạ claim không có căn cứ về `unknown`."""
    if verdict == "unknown":
        return verdict, evidence, source
    if evidence is None:
        return "unknown", None, None
    if source == "job":
        haystack = job_text
    elif source == "cv":
        haystack = cv_text
    else:
        haystack = f"{job_text} {cv_text}".strip()
        if _grounded(job_text, evidence):
            source = "job"
        elif _grounded(cv_text, evidence):
            source = "cv"
    if _grounded(haystack, evidence):
        return verdict, evidence, source
    return "unknown", None, None


def parse_rerank(expect_ids: set[str], data: dict) -> dict:
    """Chuẩn hoá raw LLM thành {id: result} cho từng candidate kỳ vọng."""
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict] = {}
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip()
        if not cid or cid not in expect_ids:
            continue  # không thêm id lạ
        score = item.get("score")
        if isinstance(score, str):
            score = int(score) if score.strip().isdigit() else None
        score = int(score) if isinstance(score, (int, float)) else None
        if score is not None:
            score = max(0, min(100, score))
        breakdown = item.get("breakdown") if isinstance(item.get("breakdown"), dict) else {}
        evidence = [str(e)[:500] for e in (item.get("evidence") or []) if isinstance(e, str) and e.strip()][:10]
        missing = [str(m)[:300] for m in (item.get("missing") or []) if isinstance(m, str) and m.strip()][:10]
        out[cid] = {
            "id": cid,
            "score": score,
            "breakdown": breakdown,
            "reason": (str(item.get("reason") or "").strip()[:500]) or None,
            "evidence": evidence,
            "missing": missing,
            "ready": score is not None,
        }
    return out


def rerank_with_llm(client: llm_mod.OllamaClient, settings: Settings,
                    job: dict, candidates: list[dict]) -> list[dict]:
    """Chấm từng candidate (pointwise) rồi sắp xếp điểm giảm dần.

    Mỗi candidate một lượt gọi LLM để giữ prompt gọn (tối đa LLM chars).
    LLM lỗi/dữ liệu hỏng -> candidate đó ready=false (fallback từng phần).
    """
    for i, c in enumerate(candidates):
        if not c.get("id"):
            c["id"] = f"c{i}"

    results: dict[str, dict] = {}
    for c in candidates:
        cid = str(c["id"])
        results[cid] = {
            "id": cid, "score": None, "breakdown": {},
            "reason": None, "evidence": [], "missing": [], "ready": False,
        }
        try:
            prompt = rerank_candidate_prompt(settings, job, c, candidates.index(c))
            raw = client.chat(llm_mod.SYSTEM_PROMPT, prompt)
        except errors.ApiError as exc:
            results[cid]["reason"] = exc.detail
            continue
        parsed = parse_rerank({cid}, raw)
        if cid not in parsed:
            results[cid]["reason"] = "INVALID_OUTPUT"
            continue
        results[cid].update(parsed[cid])

    ordered = sorted(results.values(), key=lambda r: (r["score"] is not None, r["score"] or -1), reverse=True)
    return ordered


def explain_with_llm(client: llm_mod.OllamaClient, settings: Settings,
                     job: dict, cv: dict, match: dict | None) -> dict:
    """Tạo explanation. LLM lỗi -> ready=false (fallback §8.5), không raise."""
    base = {
        "ready": False,
        "summary": None,
        "claims": [],
        "fallback_reason": None,
        "model": settings.ollama_model,
    }
    try:
        prompt = explain_prompt(settings, job, cv, match)
        raw = client.chat(llm_mod.SYSTEM_PROMPT, prompt)
    except errors.ApiError as exc:
        base["fallback_reason"] = exc.detail
        return base
    claims, summary, err = parse_explain(raw, job, cv)
    if not claims:
        base["fallback_reason"] = err or "NO_CLAIMS"
        return base
    base.update({
        "ready": True,
        "summary": summary,
        "claims": claims[:12],
    })
    return base


__all__ = [
    "EXPLAIN_SCHEMA_DOC",
    "RERANK_SCHEMA_DOC",
    "explain_prompt",
    "explain_with_llm",
    "parse_explain",
    "parse_rerank",
    "rerank_candidate_prompt",
    "rerank_with_llm",
]