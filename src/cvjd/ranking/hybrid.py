"""Hybrid ranking: semantic + skills + experience + budget + preferences (plan §7).

S_hybrid = 0.40*S_semantic + 0.30*S_skills + 0.15*S_experience + 0.10*S_budget
         + 0.05*S_preferences

Mỗi thành phần được chuẩn hoá về [0, 1]. Đầu vào là cấu trúc đã trích xuất
(giống payload `/internal/extract-cv`, `/internal/extract-job` hoặc `Document`
nghiên cứu qua `profile_from_document`), nên module này không embed lại gì cả.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..schema import Document

# Trọng số §7
DEFAULT_WEIGHTS = {
    "semantic": 0.40,
    "skills": 0.30,
    "experience": 0.15,
    "budget": 0.10,
    "preferences": 0.05,
}

# Trọng số ràng buộc bắt buộc vs ưu tiên trong S_skills
REQUIRED_BONUS = 0.8
OPTIONAL_BONUS = 0.2

_ALIASES = {
    "node": "node.js",
    "javascript": "js",
    "typescript": "ts",
    "reactjs": "react",
    "react.js": "react",
    "nextjs": "next.js",
    "nestjs": "nest",
    "postgresql": "postgres",
}


@dataclass(frozen=True)
class BudgetRange:
    min: float | None = None
    max: float | None = None
    currency: str | None = None


@dataclass(frozen=True)
class SkillItem:
    name: str
    weight: float = 1.0
    level: int | None = None  # proficiency 1..10


@dataclass(frozen=True)
class JobRequirement:
    required_skills: Sequence[SkillItem] = ()
    optional_skills: Sequence[SkillItem] = ()
    min_years_experience: float | None = None
    budget: BudgetRange | None = None
    preferences: dict[str, str] = field(default_factory=dict)
    semantic_base: float = 1.0  # S_semantic đã có (cosine, clamp về [0,1])

    def with_counts(self) -> JobRequirement:
        return self


@dataclass(frozen=True)
class CandidateProfile:
    skills: Sequence[SkillItem] = ()
    years_experience: float | None = None
    budget: BudgetRange | None = None
    preferences: dict[str, str] = field(default_factory=dict)


def normalize_skill(name: str) -> str:
    """Chuẩn hoá tên kỹ năng để so khớp (alias + lowercase + collapse)."""
    key = " ".join(str(name).lower().split())
    return _ALIASES.get(key, key)


def _match(a: str, b: str) -> bool:
    x, y = normalize_skill(a), normalize_skill(b)
    if x == y:
        return True
    # con trỏ theo token duy nhất (tránh "java" ~ "javascript")
    ax = {t for t in x.replace("-", " ").split() if t}
    bx = {t for t in y.replace("-", " ").split() if t}
    return bool(ax and bx and ax <= bx or bx <= ax)


def _coverage(skills: Sequence[SkillItem], targets: Sequence[SkillItem]) -> float:
    """Tỷ lệ khối lượng target được phủ, trọng số theo `weight`."""
    total = sum(max(t.weight, 0.0) for t in targets)
    if total <= 0.0:
        return 0.0
    covered = 0.0
    matched_names = set()
    for target in targets:
        for skill in skills:
            if _match(skill.name, target.name):
                key = normalize_skill(target.name)
                if key in matched_names:
                    continue
                matched_names.add(key)
                covered += max(target.weight, 0.0)
                break
    return min(covered / total, 1.0)


def skills_score(cv: CandidateProfile, job: JobRequirement) -> float:
    """S_skills: 80% bắt buộc + 20% ưu tiên (nếu không có bắt buộc, 100% ưu tiên)."""
    required = list(job.required_skills)
    optional = list(job.optional_skills)
    if not required and not optional:
        return 1.0
    if not required:
        return _coverage(cv.skills, optional)

    req = _coverage(cv.skills, required)
    opt = _coverage(cv.skills, optional) if optional else req
    return REQUIRED_BONUS * req + OPTIONAL_BONUS * opt


def experience_score(cv: CandidateProfile, job: JobRequirement) -> float:
    """S_experience: tỷ lệ năm kinh nghiệm so với yêu cầu tối thiểu (cắt ở 1)."""
    required = job.min_years_experience
    if required is None or required <= 0:
        return 1.0
    years = cv.years_experience or 0.0
    return float(min(years / required, 1.0))


def _overlap(a: BudgetRange | None, b: BudgetRange | None) -> float | None:
    """Độ phủ khoảng [hỏi, cho]: diện tích chung / diện tích job, clamp [0,1]."""
    if a is None or b is None or (a.min is None and a.max is None):
        return None
    job_lo = a.min or 0.0
    job_hi = a.max if a.max is not None else job_lo
    cand_lo = b.min if b is not None and b.min is not None else job_lo
    cand_hi = b.max if b is not None and b.max is not None else job_lo

    if a.min is None and b.min is not None:
        job_lo, cand_lo = cand_lo, job_lo
    if a.max is None and b.max is not None:
        job_hi, cand_hi = cand_hi, job_hi

    span = job_hi - job_lo
    if span <= 0:
        # ngân sách điểm
        return 1.0 if cand_hi >= job_lo >= cand_lo else 0.0
    overlap = max(0.0, min(job_hi, cand_hi) - max(job_lo, cand_lo))
    return min(overlap / span, 1.0)


def budget_score(cv: CandidateProfile, job: JobRequirement) -> float:
    val = _overlap(job.budget, cv.budget)
    return 1.0 if val is None else val


def preferences_score(cv: CandidateProfile, job: JobRequirement) -> float:
    wanted = job.preferences
    if not wanted:
        return 1.0
    if not cv.preferences:
        return 0.0
    hits = 0
    for key, value in wanted.items():
        cv_val = cv.preferences.get(key)
        if cv_val and (
            cv_val.lower() == value.lower() or value == "*" or cv_val == "*"
        ):
            hits += 1
    return hits / len(wanted)


def hybrid_score(
    job: JobRequirement,
    cv: CandidateProfile,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Trả {overall} + từng component, mỗi component clamp về [0,1]."""
    w = dict(weights or DEFAULT_WEIGHTS)
    components = {
        "semantic": min(max(float(job.semantic_base), 0.0), 1.0),
        "skills": skills_score(cv, job),
        "experience": experience_score(cv, job),
        "budget": budget_score(cv, job),
        "preferences": preferences_score(cv, job),
    }
    overall = sum(w.get(name, 0.0) * score for name, score in components.items())
    weights_sum = sum(w.values())
    if weights_sum > 0.0:
        overall /= weights_sum
    return {"overall": float(min(max(overall, 0.0), 1.0)), **components}


def _extract_skills(text: str) -> list[SkillItem]:
    out = []
    seen = set()
    for raw in text.replace(";", ",").split(","):
        token = normalize_skill(raw)
        if token and len(token) <= 60 and token not in seen:
            seen.add(token)
            out.append(SkillItem(name=token))
    return out


def years_from_document(doc: Document) -> float | None:
    """Ước lượng tổng năm kinh nghiệm từ section `experience` của Document."""
    section = doc.sections.get("experience")
    if section is None or not section.text.strip():
        return None
    import re as _re

    matches = [float(x) for x in _re.findall(r"(\d{1,2})\s*(?:năm|nam|yrs?|years?)", section.text.lower())]
    if matches:
        return float(max(matches))
    return None


def profile_from_document(doc: Document) -> CandidateProfile:
    """Bridge từ `Document` nghiên cứu → CandidateProfile (best-effort)."""
    skills: list[SkillItem] = []
    section = doc.sections.get("skills")
    if section is not None:
        skills = _extract_skills(section.text)
    return CandidateProfile(
        skills=skills,
        years_experience=years_from_document(doc),
    )


def job_from_document(doc: Document) -> JobRequirement:
    """Bridge: JD nghiên cứu → JobRequirement (best-effort)."""
    section = doc.sections.get("skills")
    skills = _extract_skills(section.text) if section is not None else []
    return JobRequirement(
        required_skills=skills,
        min_years_experience=years_from_document(doc),
        semantic_base=1.0,
    )


__all__ = [
    "DEFAULT_WEIGHTS",
    "BudgetRange",
    "CandidateProfile",
    "JobRequirement",
    "SkillItem",
    "budget_score",
    "experience_score",
    "hybrid_score",
    "job_from_document",
    "normalize_skill",
    "preferences_score",
    "profile_from_document",
    "skills_score",
]