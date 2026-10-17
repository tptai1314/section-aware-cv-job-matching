"""Structured payload -> ranking objects (bridge cho API `extract-*`).

Nhận dict đúng schema `/internal/extract-cv` và `/internal/extract-job`
(`apps/ai/app/routes.py`) rồi dựng `JobRequirement`/`CandidateProfile` mà không
cần tới `Document`/text. Dùng cho endpoint `/internal/recommend` và NestJS khi
đã có dữ liệu đã trích xuất sẵn.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .hybrid import BudgetRange, CandidateProfile, JobRequirement, SkillItem

_YEAR_MONTH = re.compile(r"^(\d{4})(?:[-/](\d{1,2}))?")


def _get(data: Mapping[str, Any] | None, *keys: str) -> Any:
    if not isinstance(data, Mapping):
        return None
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def skill_items(items: Sequence[Any] | None) -> list[SkillItem]:
    """Chuẩn hoá list kỹ năng (dict `{name, weight?, proficiency_level?}` hoặc str)."""
    out: list[SkillItem] = []
    for item in items or ():
        if isinstance(item, Mapping):
            name = str(item.get("name") or "").strip()
            weight = _get(item, "weight", "priority")
            level = item.get("proficiency_level", item.get("level"))
        else:
            name, weight, level = str(item).strip(), None, None
        if not name:
            continue
        level = level if isinstance(level, int) and 1 <= level <= 10 else None
        weight = weight if isinstance(weight, (int, float)) and weight > 0 else 1.0
        out.append(SkillItem(name=name, weight=float(weight), level=level))
    return out


def budget_range(data: Mapping[str, Any] | None) -> BudgetRange | None:
    if not isinstance(data, Mapping):
        return None
    if data.get("min") is None and data.get("max") is None:
        return None
    return BudgetRange(
        min=_as_float(data.get("min")),
        max=_as_float(data.get("max")),
        currency=(str(data.get("currency")).strip() or None)
        if data.get("currency")
        else None,
    )


def preferences(data: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(data, Mapping):
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def _year_month(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    match = _YEAR_MONTH.match(str(value).strip())
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2) or 1)
    if not 1 <= month <= 12:
        return None
    return year, month


def years_from_experience(
    entries: Sequence[Mapping[str, Any]] | None,
    now: datetime | None = None,
) -> float | None:
    """Cộng dồn thời lượng kinh nghiệm từ start/end date (YYYY-MM); None nếu thiếu."""
    total_months = 0
    found = False
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        start = _year_month(entry.get("start_date"))
        if start is None:
            continue
        end = _year_month(entry.get("end_date"))
        if end is None and entry.get("is_current"):
            moment = now or datetime.now(timezone.utc)
            end = (moment.year, moment.month)
        if end is None:
            continue
        months = (end[0] - start[0]) * 12 + (end[1] - start[1])
        if months > 0:
            total_months += months
            found = True
    return round(total_months / 12, 2) if found else None


def job_requirement(data: Mapping[str, Any]) -> JobRequirement:
    """JD đã trích xuất -> JobRequirement (required/optional/min-years/budget/prefs)."""
    return JobRequirement(
        required_skills=skill_items(data.get("required_skills") or data.get("skills")),
        optional_skills=skill_items(data.get("optional_skills")),
        min_years_experience=_as_float(data.get("min_years_experience")),
        budget=budget_range(data.get("budget")),
        preferences=preferences(data.get("preferences")),
    )


def candidate_profile(data: Mapping[str, Any]) -> CandidateProfile:
    """CV đã trích xuất -> CandidateProfile (skills/years/budget/prefs)."""
    years = _as_float(data.get("years_experience"))
    if years is None:
        years = years_from_experience(data.get("experience"))
    budget = budget_range(data.get("budget") or data.get("expected_budget"))
    return CandidateProfile(
        skills=skill_items(data.get("skills")),
        years_experience=years,
        budget=budget,
        preferences=preferences(data.get("preferences")),
    )


__all__ = [
    "budget_range",
    "candidate_profile",
    "job_requirement",
    "preferences",
    "skill_items",
    "years_from_experience",
]
