"""Hybrid ranking: semantic + skills + experience + budget + preferences (plan §7).

S_hybrid = 0.40*S_semantic + 0.30*S_skills + 0.15*S_experience + 0.10*S_budget
         + 0.05*S_preferences

Mỗi thành phần được chuẩn hoá về [0, 1]. Đầu vào là cấu trúc đã trích xuất
(giống payload `/internal/extract-cv`, `/internal/extract-job` hoặc `Document`
nghiên cứu qua `profile_from_document`), nên module này không embed lại gì cả.
"""

from __future__ import annotations

import unicodedata
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
    "nodejs": "node.js",
    "javascript": "js",
    "typescript": "ts",
    "reactjs": "react",
    "react.js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "nextjs": "next.js",
    "nestjs": "nest",
    "postgresql": "postgres",
    "golang": "go",
    "angularjs": "angular",
    "k8s": "kubernetes",
    "py": "python",
    "docker container": "docker",
    "mongo": "mongodb",
    "amazon web services": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "ms azure": "azure",
    "microsoft azure": "azure",
    "tailwind": "tailwind css",
    "tailwindcss": "tailwind css",
    "tailwind.css": "tailwind css",
    "restful api": "rest api",
    "restful": "rest api",
    "restful apis": "rest api",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
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
    norm = unicodedata.normalize("NFC", str(name).strip().lower())
    key = " ".join(norm.split())
    return _ALIASES.get(key, key)


def _match(a: str, b: str) -> bool:
    x, y = normalize_skill(a), normalize_skill(b)
    if x == y:
        return True
    # Word boundary token guard (prevent "java" ~ "javascript", "c" ~ "c++", etc.)
    ax = {t for t in x.replace("-", " ").split() if t}
    bx = {t for t in y.replace("-", " ").split() if t}
    if "java" in ax and "javascript" in bx:
        return False
    if "javascript" in ax and "java" in bx:
        return False
    if "c" in ax and ("c++" in bx or "c#" in bx):
        return False
    if ("c++" in ax or "c#" in ax) and "c" in bx:
        return False
    if ax and bx and (ax <= bx or bx <= ax):
        return True
    return False


def _coverage(skills: Sequence[SkillItem], targets: Sequence[SkillItem], allow_related: bool = True) -> float:
    """Tỷ lệ khối lượng target được phủ, trọng số theo `weight`, hỗ trợ điểm một phần cho kỹ năng liên quan."""
    total = sum(max(t.weight, 0.0) for t in targets)
    if total <= 0.0:
        return 0.0
    covered = 0.0
    matched_names = set()

    # Nhập taxonomy an toàn
    tax = None
    if allow_related:
        try:
            from ..data.taxonomy import get_taxonomy
            tax = get_taxonomy()
        except Exception:
            tax = None

    for target in targets:
        target_norm = normalize_skill(target.name)
        matched = False

        # 1. Khớp chính xác hoặc bí danh alias (100% điểm)
        for skill in skills:
            if _match(skill.name, target.name):
                if target_norm not in matched_names:
                    matched_names.add(target_norm)
                    covered += max(target.weight, 0.0)
                matched = True
                break

        # 2. Khớp kỹ năng theo quan hệ phân cấp trong taxonomy (Lớp 2: is_superset_of, framework_of, companion)
        if not matched and tax is not None:
            best_rel_weight = 0.0
            for skill in skills:
                rel_w = tax.get_relation_weight(skill.name, target.name)
                if rel_w > best_rel_weight:
                    best_rel_weight = rel_w

            if best_rel_weight > 0.0:
                if target_norm not in matched_names:
                    matched_names.add(target_norm)
                    covered += best_rel_weight * max(target.weight, 0.0)
                matched = True

    return min(covered / total, 1.0)


def skills_score(
    cv: CandidateProfile,
    job: JobRequirement,
    must_have_penalty: bool = False,
) -> float:
    """S_skills: 80% bắt buộc + 20% ưu tiên (nếu không có bắt buộc, 100% ưu tiên).
    
    Khi `must_have_penalty=True`, nếu ứng viên thiếu trên 50% kỹ năng bắt buộc (req < 0.5),
    áp dụng hệ số phạt nhân tử gamma = req để tránh việc bù đắp bởi tiêu chí phụ (paper §3.4).
    """
    required = list(job.required_skills)
    optional = list(job.optional_skills)
    if not required and not optional:
        return 0.0
    if not required:
        return _coverage(cv.skills, optional)

    req = _coverage(cv.skills, required)
    opt = _coverage(cv.skills, optional) if optional else req
    base = REQUIRED_BONUS * req + OPTIONAL_BONUS * opt
    if must_have_penalty and req < 0.5:
        base *= req
    return float(min(max(base, 0.0), 1.0))


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
    renormalize_missing: bool = False,
    must_have_penalty: bool = False,
) -> dict[str, float]:
    """Trả {overall} + từng component, mỗi component clamp về [0,1].

    Khi `renormalize_missing=True`, các tiêu chí không áp dụng (ví dụ cả 2 bên
    đều không quy định ngân sách hoặc sở thích, hoặc job không yêu cầu kinh nghiệm)
    sẽ được loại bỏ và tái chuẩn hoá trọng số trên các tiêu chí còn lại thay vì
    gán điểm mặc định 1.0 làm sai lệch phân phối.
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    components = {
        "semantic": min(max(float(job.semantic_base), 0.0), 1.0),
        "skills": skills_score(cv, job, must_have_penalty=must_have_penalty),
        "experience": experience_score(cv, job),
        "budget": budget_score(cv, job),
        "preferences": preferences_score(cv, job),
    }

    if renormalize_missing:
        # Xác định các thành phần thực sự có dữ liệu
        active_keys = ["semantic", "skills"]
        if job.min_years_experience is not None and job.min_years_experience > 0:
            active_keys.append("experience")
        if not (
            (job.budget is None or (job.budget.min is None and job.budget.max is None))
            and (cv.budget is None or (cv.budget.min is None and cv.budget.max is None))
        ):
            active_keys.append("budget")
        if bool(job.preferences):
            active_keys.append("preferences")

        active_w = {k: w.get(k, 0.0) for k in active_keys}
        total_w = sum(active_w.values())
        if total_w > 0.0:
            overall = sum(active_w[k] * components[k] for k in active_keys) / total_w
        else:
            overall = components["semantic"]
    else:
        overall = sum(w.get(name, 0.0) * score for name, score in components.items())
        weights_sum = sum(w.values())
        if weights_sum > 0.0:
            overall /= weights_sum

    return {"overall": float(min(max(overall, 0.0), 1.0)), **components}


def _extract_skills(text: str) -> list[SkillItem]:
    out: list[SkillItem] = []
    seen = set()

    # 1. Trích xuất chính xác theo taxonomy có ranh giới từ
    try:
        from ..data.taxonomy import get_taxonomy
        tax = get_taxonomy()
        extracted_canonicals = tax.extract_skills_from_text(text)
        for c_name in extracted_canonicals:
            token = normalize_skill(c_name)
            if token and token not in seen:
                seen.add(token)
                out.append(SkillItem(name=token))
    except Exception:
        pass

    # 2. Hỗ trợ tách bổ sung theo dấu phẩy / chấm phẩy / gạch đầu dòng cho cụm từ tự do
    for raw in text.replace(";", ",").replace("\n", ",").replace("•", ",").replace("-", ",").split(","):
        cleaned = raw.strip()
        if not cleaned:
            continue
        token = normalize_skill(cleaned)
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
    """Bridge từ `Document` nghiên cứu → CandidateProfile (hỗ trợ trích xuất toàn diện)."""
    skills: list[SkillItem] = []
    seen = set()

    # 1. Trích xuất ưu tiên từ section skills
    section = doc.sections.get("skills")
    if section is not None and section.text.strip():
        for s in _extract_skills(section.text):
            if s.name not in seen:
                seen.add(s.name)
                skills.append(s)

    # 2. Quét bổ sung từ experience và toàn văn bản để không bỏ sót kỹ năng nêu trong quá trình làm việc
    try:
        from ..data.taxonomy import get_taxonomy
        tax = get_taxonomy()
        exp_text = doc.sections.get("experience").text if doc.sections.get("experience") else ""
        text_to_scan = f"{exp_text}\n{doc.raw_text}"
        for c_name in tax.extract_skills_from_text(text_to_scan):
            token = normalize_skill(c_name)
            if token and token not in seen:
                seen.add(token)
                skills.append(SkillItem(name=token))
    except Exception:
        pass

    return CandidateProfile(
        skills=skills,
        years_experience=years_from_document(doc),
    )


def job_from_document(doc: Document) -> JobRequirement:
    """Bridge: JD nghiên cứu → JobRequirement (phân định must-have vs nice-to-have)."""
    must_skills: list[SkillItem] = []
    nice_skills: list[SkillItem] = []
    seen_must = set()
    seen_nice = set()

    # 1. Phân loại Must-have và Nice-to-have từ toàn văn JD
    try:
        from ..data.taxonomy import get_taxonomy
        tax = get_taxonomy()
        must_names, nice_names = tax.classify_jd_skills(doc.raw_text)
        for name in must_names:
            norm = normalize_skill(name)
            if norm not in seen_must:
                seen_must.add(norm)
                must_skills.append(SkillItem(name=norm))
        for name in nice_names:
            norm = normalize_skill(name)
            if norm not in seen_must and norm not in seen_nice:
                seen_nice.add(norm)
                nice_skills.append(SkillItem(name=norm))
    except Exception:
        pass

    # 2. Bổ sung từ section skills tường minh nếu có
    section = doc.sections.get("skills")
    if section is not None and section.text.strip():
        for s in _extract_skills(section.text):
            if s.name not in seen_must and s.name not in seen_nice:
                seen_must.add(s.name)
                must_skills.append(s)

    return JobRequirement(
        required_skills=must_skills,
        optional_skills=nice_skills,
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