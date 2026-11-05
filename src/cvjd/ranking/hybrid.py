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
    # Ánh xạ Việt - Anh thuật ngữ IT
    "cơ sở dữ liệu": "database",
    "co so du lieu": "database",
    "csdl": "database",
    "hệ cơ sở dữ liệu": "database management system",
    "he co so du lieu": "database management system",
    "hệ quản trị cơ sở dữ liệu": "database management system",
    "he quan tri co so du lieu": "database management system",
    "cơ sở dữ liệu quan hệ": "relational database",
    "co so du lieu quan he": "relational database",
    "cơ sở dữ liệu phi quan hệ": "nosql database",
    "co so du lieu phi quan he": "nosql database",
    "lập trình web": "web development",
    "lap trinh web": "web development",
    "phát triển web": "web development",
    "phat trien web": "web development",
    "lập trình backend": "backend development",
    "lap trinh backend": "backend development",
    "phát triển backend": "backend development",
    "phat trien backend": "backend development",
    "lập trình frontend": "frontend development",
    "lap trinh frontend": "frontend development",
    "phát triển frontend": "frontend development",
    "phat trien frontend": "frontend development",
    "lập trình fullstack": "fullstack development",
    "lap trinh fullstack": "fullstack development",
    "phát triển fullstack": "fullstack development",
    "phat trien fullstack": "fullstack development",
    "kỹ sư phần mềm": "software engineer",
    "ky su phan mem": "software engineer",
    "phát triển phần mềm": "software development",
    "phat trien phan mem": "software development",
    "phần mềm": "software",
    "phan mem": "software",
    "hệ điều hành": "operating system",
    "he dieu hanh": "operating system",
    "mạng máy tính": "computer network",
    "mang may tinh": "computer network",
    "lập trình di động": "mobile development",
    "lap trinh di dong": "mobile development",
    "lập trình mobile": "mobile development",
    "khoa học dữ liệu": "data science",
    "khoa hoc du lieu": "data science",
    "học máy": "machine learning",
    "hoc may": "machine learning",
    "học sâu": "deep learning",
    "hoc sau": "deep learning",
    "xử lý ngôn ngữ tự nhiên": "natural language processing",
    "xu ly ngon ngu tu nhien": "natural language processing",
    "nlp": "natural language processing",
    "thị giác máy tính": "computer vision",
    "thi giac may tinh": "computer vision",
    "điện toán đám mây": "cloud computing",
    "dien toan dam may": "cloud computing",
    "quản lý dự án": "project management",
    "quan ly du an": "project management",
    "bảo mật thông tin": "information security",
    "bao mat thong tin": "information security",
    "an toàn thông tin": "information security",
    "an toan thong tin": "information security",
    "an ninh mạng": "cybersecurity",
    "an ninh mang": "cybersecurity",
    "trí tuệ nhân tạo": "artificial intelligence",
    "tri tue nhan tao": "artificial intelligence",
    "quản trị cơ sở dữ liệu": "database administration",
    "quan tri co so du lieu": "database administration",
    "kiểm thử phần mềm": "software testing",
    "kiem thu phan mem": "software testing",
    "kiểm thử tự động": "automation testing",
    "kiem thu tu dong": "automation testing",
    "phân tích dữ liệu": "data analytics",
    "phan tich du lieu": "data analytics",
    "thiết kế giao diện": "ui/ux design",
    "thiet ke giao dien": "ui/ux design",
    "thiết kế đồ họa": "graphic design",
    "thiet ke do hoa": "graphic design",
    "tiếng anh giao tiếp": "english communication",
    "tieng anh giao tiep": "english communication",
}


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để so khớp tương đồng không dấu."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")


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
    if x == y or strip_accents(x) == strip_accents(y):
        return True
    # con trỏ theo token duy nhất (tránh "java" ~ "javascript")
    ax = {t for t in x.replace("-", " ").split() if t}
    bx = {t for t in y.replace("-", " ").split() if t}
    if ax and bx and (ax <= bx or bx <= ax):
        return True
    # Thử đối sánh không dấu
    ax_no_acc = {strip_accents(t) for t in ax}
    bx_no_acc = {strip_accents(t) for t in bx}
    return bool(ax_no_acc and bx_no_acc and (ax_no_acc <= bx_no_acc or bx_no_acc <= ax_no_acc))


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
        return 1.0
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