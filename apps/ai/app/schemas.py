from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CURRENT_YEAR_MAX = 2100

# Giới hạn số lượng item (spec 3.4)
LIMIT_SKILLS = 50
LIMIT_EXPERIENCE = 20
LIMIT_EDUCATION = 10
LIMIT_PROJECTS = 20


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    proficiency_level: int | None = Field(default=None, ge=1, le=10)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = Field(default=None, max_length=200)


class Experience(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=150)
    company: str | None = Field(default=None, max_length=150)
    start_date: str | None = Field(default=None)
    end_date: str | None = Field(default=None)
    is_current: bool = False
    description: str | None = Field(default=None, max_length=1000)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Education(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degree: str | None = Field(default=None, max_length=150)
    school: str | None = Field(default=None, max_length=150)
    start_year: int | None = Field(default=None, ge=1950, le=CURRENT_YEAR_MAX)
    end_year: int | None = Field(default=None, ge=1950, le=CURRENT_YEAR_MAX)
    description: str | None = Field(default=None, max_length=1000)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=150)
    technologies: list[str] = Field(default_factory=list, max_length=30)
    description: str | None = Field(default=None, max_length=1000)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(pattern="^(pdf_text|pdf_ocr|docx|doc|job_text)$")
    pages: int = 0
    chars: int = 0
    truncated: bool = False
    ocr_used: bool = False
    model: str = ""
    duration_ms: int = 0
    warnings: list[str] = Field(default_factory=list)


class CvResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: list[Skill] = Field(default_factory=list, max_length=LIMIT_SKILLS)
    experience: list[Experience] = Field(default_factory=list, max_length=LIMIT_EXPERIENCE)
    education: list[Education] = Field(default_factory=list, max_length=LIMIT_EDUCATION)
    projects: list[Project] = Field(default_factory=list, max_length=LIMIT_PROJECTS)
    summary: str | None = Field(default=None, max_length=1500)
    meta: Meta


class JobSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = Field(default=None, max_length=200)


class JobBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | None = Field(default=None, ge=0)
    max: float | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, max_length=10)


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=150)
    summary: str | None = Field(default=None, max_length=3000)
    required_skills: list[JobSkill] = Field(default_factory=list, max_length=LIMIT_SKILLS)
    optional_skills: list[JobSkill] = Field(default_factory=list, max_length=LIMIT_SKILLS)
    responsibilities: list[str] = Field(default_factory=list, max_length=20)  
    min_years_experience: int | None = Field(default=None, ge=0, le=20)
    budget: JobBudget | None = None
    meta: Meta


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(min_length=1, max_length=100)


class EmbeddingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    dim: int
    vectors: list[list[float]]


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
    message: str


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job: dict
    cv: dict
    match: dict | None = None


class ExplainClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(max_length=400)
    verdict: str = Field(pattern="^(hit|missing|unknown)$")
    evidence: str | None = Field(default=None, max_length=500)
    source: str | None = Field(default=None, pattern="^(job|cv)$")
    confidence: float | None = Field(default=None, ge=0, le=1)


class ExplainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    summary: str | None = Field(default=None, max_length=2000)
    claims: list[ExplainClaim] = Field(default_factory=list, max_length=12)
    fallback_reason: str | None = Field(default=None, max_length=50)
    model: str
    duration_ms: int = 0


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job: dict
    candidates: list[dict] = Field(min_length=1, max_length=50)


class RerankItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    score: int | None = Field(default=None, ge=0, le=100)
    breakdown: dict = Field(default_factory=dict)
    reason: str | None = Field(default=None, max_length=500)
    evidence: list[str] = Field(default_factory=list, max_length=10)
    missing: list[str] = Field(default_factory=list, max_length=10)
    ready: bool = False


class RerankResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    duration_ms: int = 0
    results: list[RerankItem]


class RecommendCandidate(BaseModel):
    """Candidate đã trích xuất (skills/experience/budget...); giữ thêm field tuỳ ý."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1, max_length=100)
    semantic: float | None = Field(default=None, ge=0, le=1)


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal["jd2cv", "cv2jd"] = "jd2cv"
    query: dict
    candidates: list[RecommendCandidate] | None = Field(
        default=None, min_length=1, max_length=200
    )
    top_k: int = Field(default=10, ge=1, le=100)
    pool_size: int | None = Field(default=None, ge=1, le=1000)
    keyword_alpha: float | None = Field(default=None, ge=0, le=1)
    keyword_scores: dict[str, float] | None = None
    owner_id: str | None = Field(default=None, max_length=100)
    visibility: list[Literal["public", "private", "unlisted"]] | None = None
    explain: bool = False
    explain_top_n: int = Field(default=3, ge=0, le=50)


class RecommendComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic: float = 0.0
    skills: float = 0.0
    experience: float = 0.0
    budget: float = 0.0
    preferences: float = 0.0
    keyword: float | None = None


class RecommendItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
    components: RecommendComponents
    section_components: dict[str, float] = Field(default_factory=dict)
    explanation: ExplainResponse | None = None


class RecommendResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: str
    results: list[RecommendItem]


class IndexDocument(BaseModel):
    """Tài liệu đã trích xuất để index (giữ thêm field structured tuỳ ý)."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1, max_length=100)
    type: Literal["cv", "jd"]


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[IndexDocument] = Field(min_length=1, max_length=500)


class IndexResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indexed: int
    stats: dict = Field(default_factory=dict)


class IndexDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[IndexDocument] = Field(min_length=1, max_length=500)


class IndexDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deleted: int
    stats: dict = Field(default_factory=dict)


# ---- Hàng đợi xử lý file (docs/DESIGN_processing_queue.md §4) ----


class JobCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str = Field(pattern="^(PENDING|PROCESSING|READY|FAILED|CANCELLED)$")
    document_type: Literal["cv", "jd"]
    document_id: str
    version: int = Field(ge=1, le=1000)
    deduplicated: bool = False  # True: dùng lại job đã có cho cùng bản ghi
    created_at: float


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str = Field(pattern="^(PENDING|PROCESSING|READY|FAILED|CANCELLED)$")
    step: str
    progress: int = Field(ge=0, le=100)  # % cho UI (design §4.2)
    attempt_count: int = 0
    max_attempts: int = 3
    last_error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    document_type: Literal["cv", "jd"]
    document_id: str
    version: int
    created_at: float
    updated_at: float
    finished_at: float | None = None
    duration_ms: int | None = None