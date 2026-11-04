"""Build a research `Document` from a structured payload (`extract-cv`/`extract-job`).

Dùng cho nhánh retrieval của AI service: payload đã trích xuất vẫn cần một
`Document` (sections + raw_text) để tạo vector section + full-text trong Chroma.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..schema import DocType, Document, Lang, Section


def _names(items: Sequence[Any] | None) -> str:
    names = []
    for item in items or ():
        if isinstance(item, Mapping):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    return ", ".join(names)


def _experience_text(items: Sequence[Any] | None) -> str:
    parts = []
    for item in items or ():
        if not isinstance(item, Mapping):
            continue
        head = " at ".join(
            str(value).strip()
            for value in (item.get("title"), item.get("company"))
            if str(value or "").strip()
        )
        description = str(item.get("description") or "").strip()
        line = ", ".join(part for part in (head, description) if part)
        if line:
            parts.append(line)
    return "\n".join(parts)


def _projects_text(items: Sequence[Any] | None) -> str:
    parts = []
    for item in items or ():
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        techs = _names(item.get("technologies"))
        description = str(item.get("description") or "").strip()
        head = " — ".join(part for part in (name, techs) if part)
        line = ": ".join(part for part in (head, description) if part)
        if line:
            parts.append(line)
    return "\n".join(parts)


def _education_text(items: Sequence[Any] | None) -> str:
    parts = []
    for item in items or ():
        if not isinstance(item, Mapping):
            continue
        line = ", ".join(
            str(value).strip()
            for value in (item.get("degree"), item.get("school"))
            if str(value or "").strip()
        )
        if line:
            parts.append(line)
    return "\n".join(parts)


def _cv_sections(payload: Mapping[str, Any]) -> dict[str, Section]:
    sections: dict[str, Section] = {}
    skills = _names(payload.get("skills"))
    if skills:
        sections["skills"] = Section("skills", skills)
    experience = _experience_text(payload.get("experience"))
    if experience:
        sections["experience"] = Section("experience", experience)
    projects = _projects_text(payload.get("projects"))
    if projects:
        sections["projects"] = Section("projects", projects)
    education = _education_text(payload.get("education"))
    if education:
        sections["education"] = Section("education", education)
    summary = str(payload.get("summary") or "").strip()
    if summary:
        sections["summary"] = Section("summary", summary)
    return sections


def _jd_sections(payload: Mapping[str, Any]) -> dict[str, Section]:
    sections: dict[str, Section] = {}
    required = payload.get("required_skills") or payload.get("skills")
    required_text = _names(required)
    if required_text:
        sections["skills"] = Section("skills", required_text)
    optional = _names(payload.get("optional_skills"))
    if optional:
        sections["optional_skills"] = Section("optional_skills", optional)
    responsibilities = payload.get("responsibilities")
    if isinstance(responsibilities, Sequence) and not isinstance(responsibilities, str):
        text = "\n".join(str(r).strip() for r in responsibilities if str(r).strip())
        if text:
            sections["responsibilities"] = Section("responsibilities", text)
    summary = str(payload.get("summary") or payload.get("title") or "").strip()
    if summary:
        sections["summary"] = Section("summary", summary)
    return sections


def document_from_payload(
    doc_id: str,
    doc_type: DocType,
    payload: Mapping[str, Any],
    lang: Lang = "vi",
) -> Document:
    """Dựng `Document` (sections + raw_text) từ payload đã trích xuất."""
    if doc_type == "cv":
        sections = _cv_sections(payload)
    elif doc_type == "jd":
        sections = _jd_sections(payload)
    else:
        raise ValueError(f"doc_type must be 'cv' or 'jd', got {doc_type!r}")

    raw_text = "\n\n".join(section.text for section in sections.values()).strip()
    if not raw_text:
        raw_text = str(payload.get("summary") or payload.get("title") or doc_id).strip()
    return Document(
        doc_id=doc_id,
        doc_type=doc_type,
        lang=lang,
        raw_text=raw_text,
        sections=sections,
    )


__all__ = ["document_from_payload"]
