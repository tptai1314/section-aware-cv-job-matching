"""Chuẩn hoá kết quả LLM (spec mục 7): ngày, dedupe, giới hạn, kiểm tra evidence."""

from __future__ import annotations

import re

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "thang1": 1, "thang2": 2, "thang3": 3, "thang4": 4, "thang5": 5, "thang6": 6,
    "thang7": 7, "thang8": 8, "thang9": 9, "thang10": 10, "thang11": 11, "thang12": 12,
}

_DATE_FULL = re.compile(r"(?P<y>\d{4})\s*-\s*(?P<m>\d{1,2})\b")
_DATE_IT = re.compile(r"\b(?P<y>\d{4})\s*[/-]\s*(?P<m>\d{1,2})\b")
_DATE_VI_WORD = re.compile(r"\b(?P<m>\d{1,2})/\s*(?P<y>\d{4})\b")
_DATE_EN_WORD = re.compile(r"\b(?P<m>[A-Za-z]{3,9})[\.,]?\s+(?P<y>\d{4})\b")
_DATE_YEAR = re.compile(r"(?<![0-9/])(?:19|20)\d{2}\b")


def normalize_date(value: str | None) -> str | None:
    """Chuẩn hoá ngày về YYYY-MM hoặc YYYY; không suy diễn => None."""
    if not value:
        return None
    s = value.strip().lower().replace("tháng", "thang").replace(".", "/")
    m = _DATE_FULL.search(s)
    if m and 1 <= int(m.group("m")) <= 12:
        return f"{m.group('y')}-{int(m.group('m')):02d}"
    m = _DATE_IT.search(s)
    if m and 1 <= int(m.group("m")) <= 12:
        return f"{m.group('y')}-{int(m.group('m')):02d}"
    m = _DATE_VI_WORD.search(s)
    if m and 1 <= int(m.group("m")) <= 12:
        return f"{m.group('y')}-{int(m.group('m')):02d}"
    m = _DATE_EN_WORD.search(s)
    if m:
        month = _MONTHS.get(m.group("m")[:3])
        if month:
            return f"{m.group('y')}-{month:02d}"
    m = _DATE_YEAR.search(s)
    if m:
        return m.group(0)
    return None


def is_current_marker(value: str | None) -> bool:
    if not value:
        return False
    s = value.strip().lower().replace("tháng", "thang")
    return s in ("present", "hiện tại", "now", "current", "đang làm", "đến nay")


def dedupe(items: list[dict], key_fn) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for item in items:
        key = key_fn(item)
        if key is None:
            result.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def dedupe_skills(skills: list[dict]) -> list[dict]:
    return dedupe(skills, lambda it: (it.get("name") or "").strip().lower() or None)


def dedupe_technologies(techs: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for t in techs:
        key = t.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(t.strip())
    return result


def _norm_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def check_evidence(skills: list[dict], cv_text: str, strict: bool) -> list[str]:
    """Skill có `evidence` không xuất hiện trong CV (đã chuẩn hoá, không phân biệt hoa/thường/
    khoảng trắng) => hạ confidence <= 0.3, hoặc loại bỏ nếu strict."""
    if not cv_text:
        return []
    body = _norm_for_match(cv_text)
    dropped = []
    kept: list[dict] = []
    for sk in skills:
        ev = (sk.get("evidence") or "").strip()
        if ev and _norm_for_match(ev) not in body:
            if strict:
                dropped.append(sk.get("name") or "?")
                continue
            sk["confidence"] = min(sk.get("confidence") if sk.get("confidence") is not None else 2, 0.3)
            sk["confidence"] = float(sk["confidence"])
        kept.append(sk)
    if dropped:
        skills[:] = kept
    return dropped


def clamp(items: list[dict], limit: int, warnings: list[str], label: str) -> list[dict]:
    if len(items) > limit:
        warnings.append(f"LIMIT_{label.upper()}_TRIMMED")
        return items[:limit]
    return items