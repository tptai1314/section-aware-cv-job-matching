"""Rule-based section splitting for raw CV/JD text."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..schema import Document, Section

_HEADING_RULES = [
    ("summary", re.compile(
        r"^(?:tóm tắt(?: bản thân)?|summary|profile|professional summary)$", re.I)),
    ("skills", re.compile(
        r"^(?:kỹ năng(?: chuyên môn)?|skills?|technical skills|chuyên môn)$", re.I)),
    ("experience", re.compile(
        r"^(?:kinh nghiệm(?: làm việc)?|experiences?|work experiences?"
        r"|employment(?: history)?)$", re.I)),
    ("education", re.compile(
        r"^(?:học vấn|trình độ học vấn|education|academic background)$", re.I)),
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_HEADING_LEN = 60
_MAX_HEADING_WORDS = 8


def normalize_heading(heading: str) -> Optional[str]:
    """Map a heading to a normalized section name, or None when unknown."""
    cleaned = heading.strip().rstrip(":：").strip()
    if not cleaned:
        return None
    for name, pattern in _HEADING_RULES:
        if pattern.match(cleaned.lower()):
            return name
    return None


def _slug(heading: str) -> str:
    cleaned = heading.strip().rstrip(":：").strip()
    return _SLUG_RE.sub("_", cleaned.lower()).strip("_") or "other"


def _is_heading(line: str) -> bool:
    cleaned = line.strip().rstrip(":：").strip()
    if not cleaned or len(cleaned) > _MAX_HEADING_LEN:
        return False
    if normalize_heading(cleaned):
        return True
    if cleaned.endswith((".", ",", ";", ")", "…", "]")):
        return False
    words = cleaned.split()
    if len(words) > _MAX_HEADING_WORDS:
        return False
    if cleaned.isupper() and any(c.isalpha() for c in cleaned):
        return True
    if any(c.isdigit() for c in cleaned):
        return False
    return all(w[0].isupper() for w in words)


def split_sections(text: str) -> List[Section]:
    """Split raw text into sections on heading lines.

    Text before the first heading becomes section "preamble". Unknown
    headings keep a slug of the raw heading as the section name.
    """
    sections: List[Section] = []
    heading_raw = ""
    buffer: List[str] = []
    order = 0

    def flush() -> None:
        nonlocal heading_raw, buffer, order
        body = "\n".join(buffer).strip()
        if not heading_raw and not body:
            buffer = []
            return
        if heading_raw:
            name = normalize_heading(heading_raw) or _slug(heading_raw)
        else:
            name = "preamble"
        sections.append(Section(name=name, text=body, heading_raw=heading_raw, order=order))
        order += 1
        heading_raw = ""
        buffer = []

    for line in text.splitlines():
        if _is_heading(line):
            flush()
            heading_raw = line.strip().rstrip(":：").strip()
        else:
            buffer.append(line)
    flush()
    return sections


def sections_to_dict(sections: List[Section]) -> Dict[str, Section]:
    """Key sections by name; repeated names get a numeric suffix."""
    out: Dict[str, Section] = {}
    counts: Dict[str, int] = {}
    for section in sections:
        counts[section.name] = counts.get(section.name, 0) + 1
        key = section.name if counts[section.name] == 1 else f"{section.name}_{counts[section.name]}"
        out[key] = section
    return out


def resplit_document(doc: Document) -> Document:
    """Rebuild a document's sections from its raw text."""
    return Document(
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        lang=doc.lang,
        raw_text=doc.raw_text,
        sections=sections_to_dict(split_sections(doc.raw_text)),
        schema_version=doc.schema_version,
    )
