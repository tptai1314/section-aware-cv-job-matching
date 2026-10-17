"""Hard constraints appendix (plan §7.4): lọc CV/JD trước khi ranking.

Theo plan, hard constraints được enforce tại NestJS + PostgreSQL (job còn mở,
CV được chia sẻ, quyền truy cập, tài liệu chưa xoá) — KHÔNG lưu trong metadata
Chroma. Module này định nghĩa policy thuần (Python) để tái sử dụng trong
NestJS service (D/H) và validate cho pipeline nghiên cứu.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AccessContext:
    user_id: str | None = None
    roles: frozenset = field(default_factory=frozenset)
    is_admin: bool = False


@dataclass(frozen=True)
class CandidateRecord:
    doc_id: str
    doc_type: str
    status: str = "READY"
    visibility: str = "public"  # public | private | unlisted
    owner_id: str | None = None
    deleted_at: str | None = None  # None = chưa xoá
    version: int = 1


def is_allowed(
    record: CandidateRecord,
    ctx: AccessContext,
    require_ready: bool = True,
    allowed_roles: Sequence[str] = (),
) -> bool:
    """One-record policy check."""
    if record.deleted_at is not None:
        return False
    if require_ready and record.status != "READY":
        return False
    if record.visibility == "private":
        if ctx.is_admin:
            return True
        if record.owner_id and record.owner_id == ctx.user_id:
            return True
        if record.owner_id is None and record.doc_type == "cv":
            return False
        return False
    if record.visibility == "unlisted":
        if ctx.is_admin or (record.owner_id and record.owner_id == ctx.user_id):
            return True
        return any(role in allowed_roles for role in ctx.roles)
    return True


def apply_hard_constraints(
    records: Sequence[CandidateRecord],
    ctx: AccessContext,
    require_ready: bool = True,
    allowed_roles: Sequence[str] = (),
) -> list[CandidateRecord]:
    """Lọc danh sách, giữ nguyên thứ tự; bỏ qua record không hợp lệ doc_type."""
    return [
        record
        for record in records
        if record.doc_type in ("cv", "jd") and is_allowed(record, ctx, require_ready, allowed_roles)
    ]


def public_candidates(records: Sequence[CandidateRecord], ctx: AccessContext) -> list[CandidateRecord]:
    """Kết quả công khai: chỉ public + READY + chưa xoá."""
    return [
        r for r in records
        if r.status == "READY" and r.deleted_at is None and r.visibility == "public"
    ]


__all__ = [
    "AccessContext",
    "CandidateRecord",
    "apply_hard_constraints",
    "is_allowed",
    "public_candidates",
]