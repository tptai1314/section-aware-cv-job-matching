"""Rule-based section splitting for raw CV/JD text.

Supports both:
1. Standard standalone headings on separate lines (e.g., 'SUMMARY', 'WORK EXPERIENCE').
2. Glued line-start headings from HTML-scraped resumes (e.g., 'SummaryHighly...', 'ExperienceAccountant...').
3. Domain stop-headings for Job Descriptions (Responsibilities, Requirements, Benefits, About Us, EEO).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from ..schema import Document, Section

# Standard standalone headings
_STANDALONE_RULES = [
    ("summary", re.compile(
        r"^(?:summary|profile|professional\s+summary|career\s+profile|personal\s+profile|about\s+me|objective|career\s+objective|overview"
        r"|tóm\s+tắt(?: bản thân)?|tom\s+tat(?: ban than)?|giới\s+thiệu(?: bản thân)?|gioi\s+thieu(?: ban than)?)$", re.I)),
    ("skills", re.compile(
        r"^(?:skills?|technical\s+skills|technical\s+skills\s+and\s+qualifications|core\s+skills|core\s+competencies|expertise|tech\s+skills|tech\s+stack|key\s+skills|technologies|highlights"
        r"|kỹ\s+năng(?: chuyên môn)?|ky\s+nang(?: chuyen mon)?|chuyên\s+môn|chuyen\s+mon)$", re.I)),
    ("experience", re.compile(
        r"^(?:experiences?|work\s+experiences?|professional\s+experience|employment(?: history)?|work\s+history|exp|work\s+exp"
        r"|kinh\s+nghiệm(?: làm việc)?|kinh\s+nghiem(?: lam viec)?|lịch\s+sử\s+(?:công tác|làm việc))$", re.I)),
    ("education", re.compile(
        r"^(?:certifications?|education|education\s+and\s+training|education\s+and\s+qualifications|academic\s+background|qualifications|edu|certs?|degrees?"
        r"|học\s+vấn|hoc\s+van|trình\s+độ\s+học\s+vấn|trinh\s+do\s+hoc\s+van|bằng\s+cấp|bang\s+cap)$", re.I)),
]

# Glued line-start headings (strictly requires uppercase letter, digit, or dash immediately after)
# Notice: (?i:...) is applied to keyword only, while the lookahead is strictly case-sensitive to avoid matching lowercase prose
_GLUED_CV_RULES = [
    ("summary", re.compile(r"^(?i:professional\s+summary|career\s+profile|personal\s+profile|summary|profile|overview)(?=[A-Z0-9\-])")),
    ("skills", re.compile(r"^(?i:technical\s+skills\s+and\s+qualifications|technical\s+skills|core\s+competencies|key\s+skills|highlights|skills?)(?=[A-Z0-9\-])")),
    ("experience", re.compile(r"^(?i:professional\s+experience|work\s+experience|work\s+history|employment\s+history|experience)(?=[A-Z0-9\-])")),
    ("education", re.compile(r"^(?i:education\s+and\s+training|education\s+and\s+qualifications|academic\s+background|education|certifications?)(?=[A-Z0-9\-])")),
    ("projects", re.compile(r"^(?i:key\s+projects|personal\s+projects|projects?)(?=[A-Z0-9\-])")),
]

# Job description specific headings and stop-headings
_JD_HEADING_RULES = [
    ("responsibilities", re.compile(r"^(?:key\s+responsibilities|responsibilities|duties|essential\s+functions|what\s+you(?:\'ll|\s+will)\s+do|what\s+you\s+will\s+be\s+doing)(?:[:\-]|\s+[A-Z0-9]|$)", re.I)),
    ("requirements", re.compile(r"^(?:basic\s+qualifications|minimum\s+qualifications|required\s+qualifications|qualifications|requirements|what\s+you\s+need|what\s+were\s+looking\s+for|what\s+we\s+look\s+for)(?:[:\-]|\s+[A-Z0-9]|$)", re.I)),
    ("skills", re.compile(r"^(?:technical\s+skills|core\s+skills|key\s+skills|required\s+skills|skills\s+and\s+qualifications|skills\s+required|skills)(?:[:\-]|\s+[A-Z0-9]|$)", re.I)),
    ("experience", re.compile(r"^(?:experience\s+required|work\s+experience|professional\s+experience|role\s+specific\s+experience)(?:[:\-]|\s+[A-Z0-9]|$)", re.I)),
    ("education", re.compile(r"^(?:education\s+required|education\s+and\s+training|education\s+and\s+experience|education\s+desired|academic\s+requirements)(?:[:\-]|\s+[A-Z0-9]|$)", re.I)),
    ("benefits", re.compile(r"^(?:what\s+we\s+offer|benefits|perks|compensation|pay\s+range|salary)(?:[:\-]|\s+[A-Z0-9]|$)", re.I)),
    ("about_company", re.compile(r"^(?:about\s+us|about\s+the\s+company|who\s+we\s+are|company\s+overview|life\s+at\s+[a-z0-9]+)(?:[:\-]|\s+[A-Z0-9]|$)", re.I)),
    ("eeo", re.compile(r"^(?:equal\s+opportunity\s+employer|eeo\s+statement|equal\s+employment\s+opportunity)(?:[:\-]|\s+[A-Z0-9]|$)", re.I)),
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_HEADING_LEN = 60
_MAX_HEADING_WORDS = 8


def normalize_heading(heading: str) -> Optional[str]:
    """Map a heading string to a normalized section name, or None when unknown."""
    cleaned = unicodedata.normalize("NFC", heading.strip().rstrip(":：").strip())
    if not cleaned:
        return None
    for name, pattern in _STANDALONE_RULES:
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


def split_sections(text: str, is_jd: bool = False) -> List[Section]:
    """Split raw CV or JD text into structured functional sections.

    Supports:
    - Traditional standalone heading lines.
    - Glued heading lines where text starts immediately without newline.
    - JD-specific stop-headings (requirements, responsibilities, benefits, about_company).
    """
    sections: List[Section] = []
    lines = text.splitlines()
    curr_name = "preamble"
    curr_heading_raw = ""
    buf: List[str] = []
    order = 0

    def flush() -> None:
        nonlocal curr_name, curr_heading_raw, buf, order
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(
                name=curr_name,
                text=body,
                heading_raw=curr_heading_raw,
                order=order,
            ))
            order += 1
        buf = []
        curr_name = "other"
        curr_heading_raw = ""

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        matched_name: Optional[str] = None
        matched_raw: Optional[str] = None
        remaining_body: Optional[str] = None

        # 1. Standalone heading line
        cleaned_h = line_clean.rstrip(":：").strip()
        if len(cleaned_h.split()) <= _MAX_HEADING_WORDS:
            # Check standard rules
            norm = normalize_heading(cleaned_h)
            if norm:
                matched_name = norm
                matched_raw = cleaned_h
                remaining_body = ""
            elif is_jd:
                for s_name, pat in _JD_HEADING_RULES:
                    if pat.match(cleaned_h):
                        matched_name = s_name
                        matched_raw = cleaned_h
                        remaining_body = ""
                        break
            elif _is_heading(line_clean):
                matched_name = _slug(cleaned_h)
                matched_raw = cleaned_h
                remaining_body = ""

        # 2. Glued heading line (if not standalone)
        if not matched_name and not is_jd:
            for s_name, pat in _GLUED_CV_RULES:
                m = pat.match(line_clean)
                if m:
                    matched_name = s_name
                    matched_raw = m.group(0)
                    remaining_body = line_clean[m.end():].strip()
                    break

        if not matched_name and is_jd:
            for s_name, pat in _JD_HEADING_RULES:
                m = pat.match(line_clean)
                if m:
                    matched_name = s_name
                    matched_raw = m.group(0)
                    remaining_body = line_clean[m.end():].strip()
                    break

        if matched_name:
            flush()
            curr_name = matched_name
            curr_heading_raw = matched_raw or ""
            if remaining_body:
                buf.append(remaining_body)
        else:
            buf.append(line_clean)

    flush()
    return sections


def sections_to_dict(sections: List[Section]) -> Dict[str, Section]:
    """Map section list into a dictionary, handling duplicates with numerical suffixes."""
    out: Dict[str, Section] = {}
    counts: Dict[str, int] = {}
    for sec in sections:
        name = sec.name
        counts[name] = counts.get(name, 0) + 1
        key = name if counts[name] == 1 else f"{name}_{counts[name]}"
        out[key] = sec
    return out


def resplit_document(doc: Document) -> Document:
    """Reparse document text into sections."""
    is_jd = (doc.doc_type == "jd")
    sections_list = split_sections(doc.raw_text, is_jd=is_jd)
    return Document(
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        lang=doc.lang,
        raw_text=doc.raw_text,
        sections=sections_to_dict(sections_list),
        metadata=dict(doc.metadata),
        schema_version=doc.schema_version,
    )
