"""Load real CV/JD documents from directories or JSONL into schema objects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..io import load_documents
from .preprocessing import sections_to_dict, split_sections
from ..schema import Document, DocType

_TEXT_SUFFIXES = {".txt", ".md"}
_VIETNAMESE_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
    r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
    r"ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)


def detect_lang(text: str) -> str:
    """Heuristic: Vietnamese diacritic density decides vi / en / mixed."""
    diacritics = len(_VIETNAMESE_RE.findall(text))
    letters = sum(c.isalpha() for c in text) or 1
    ratio = diacritics / letters
    if ratio >= 0.02:
        return "vi"
    if diacritics > 0:
        return "mixed"
    return "en"


def document_from_text(
    doc_id: str,
    doc_type: DocType,
    text: str,
    lang: Optional[str] = None,
) -> Document:
    """Build a Document by splitting raw text with the section rules."""
    return Document(
        doc_id=doc_id,
        doc_type=doc_type,
        lang=lang or detect_lang(text),
        raw_text=text,
        sections=sections_to_dict(split_sections(text)),
    )


def load_text_dir(directory: Path, doc_type: DocType) -> List[Document]:
    """Load every .txt/.md file in a directory; doc_id = file stem."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    files = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in _TEXT_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"no .txt/.md files in {directory}")
    docs: List[Document] = []
    seen: dict[str, Path] = {}
    for path in files:
        if path.stem in seen:
            raise ValueError(
                f"duplicate doc_id {path.stem!r}: {seen[path.stem]} and {path}"
            )
        seen[path.stem] = path
        docs.append(document_from_text(
            path.stem, doc_type, path.read_text(encoding="utf-8")
        ))
    return docs


def load_raw_dir(raw_dir: Path, layout: str = "auto") -> Tuple[List[Document], List[Document]]:
    """Load (cvs, jds) from raw_dir.

    Layouts:
      jsonl — raw_dir/cvs.jsonl and raw_dir/jds.jsonl (schema documents)
      dir   — raw_dir/cvs/*.txt|md and raw_dir/jds/*.txt|md (split on load)
      auto  — jsonl first, then dir
    """
    raw_dir = Path(raw_dir)
    if layout not in ("auto", "jsonl", "dir"):
        raise ValueError(f"unknown layout: {layout!r}")

    batches: List[List[Document]] = []
    for name, doc_type in (("cvs", "cv"), ("jds", "jd")):
        jsonl = raw_dir / f"{name}.jsonl"
        directory = raw_dir / name
        docs: Optional[List[Document]] = None
        if layout in ("auto", "jsonl") and jsonl.exists():
            docs = list(load_documents(jsonl).values())
        elif layout in ("auto", "dir") and directory.is_dir():
            docs = load_text_dir(directory, doc_type)
        if docs is None:
            raise FileNotFoundError(
                f"nothing found for {name}: expected {jsonl} or {directory}/"
            )
        batches.append(docs)
    return batches[0], batches[1]
