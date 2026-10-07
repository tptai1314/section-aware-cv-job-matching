"""JSONL I/O: load/save Document, Pair, Query per cvjd schema."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Union

from .schema import Document, Pair, Query, Section

PathLike = Union[str, Path]

_VALID_DOC_TYPES = ("cv", "jd")
_MIN_LABEL = -1
_MAX_LABEL = 3


def iter_jsonl(path: PathLike) -> Iterator[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON line") from e


def save_jsonl(path: PathLike, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def document_to_dict(doc: Document) -> dict:
    return asdict(doc)


def document_from_dict(data: dict) -> Document:
    data = dict(data)
    sections_raw = data.pop("sections", None) or {}
    sections = {name: Section(**sec) for name, sec in sections_raw.items()}
    doc_type = data.get("doc_type")
    if doc_type not in _VALID_DOC_TYPES:
        raise ValueError(f"invalid doc_type: {doc_type!r}")
    return Document(sections=sections, **data)


def pair_from_dict(data: dict) -> Pair:
    pair = Pair(**data)
    if not _MIN_LABEL <= pair.label <= _MAX_LABEL:
        raise ValueError(
            f"label out of range [{_MIN_LABEL}, {_MAX_LABEL}]: {pair.label}"
        )
    return pair


def query_from_dict(data: dict) -> Query:
    query = Query(**data)
    if len(query.cv_ids) != len(query.relevance):
        raise ValueError(
            f"cv_ids/relevance length mismatch for jd {query.jd_id}: "
            f"{len(query.cv_ids)} != {len(query.relevance)}"
        )
    return query


def save_documents(path: PathLike, docs: Iterable[Document]) -> None:
    save_jsonl(path, (document_to_dict(d) for d in docs))


def load_documents(path: PathLike) -> Dict[str, Document]:
    docs: Dict[str, Document] = {}
    for data in iter_jsonl(path):
        doc = document_from_dict(data)
        if doc.doc_id in docs:
            raise ValueError(f"duplicate doc_id: {doc.doc_id}")
        docs[doc.doc_id] = doc
    return docs


def save_pairs(path: PathLike, pairs: Iterable[Pair]) -> None:
    save_jsonl(path, (asdict(p) for p in pairs))


def load_pairs(path: PathLike) -> List[Pair]:
    return [pair_from_dict(data) for data in iter_jsonl(path)]


def save_queries(path: PathLike, queries: Iterable[Query]) -> None:
    save_jsonl(path, (asdict(q) for q in queries))


def load_queries(path: PathLike) -> List[Query]:
    return [query_from_dict(data) for data in iter_jsonl(path)]


def build_queries(
    pairs: Iterable[Pair],
    jd_ids: Sequence[str],
    cv_ids: Sequence[str],
) -> List[Query]:
    """Build full ranking queries: every JD ranks all CVs, unlabeled = -1."""
    labels = {(p.jd_id, p.cv_id): p.label for p in pairs}
    cv_ids = list(cv_ids)
    return [
        Query(
            jd_id=jd_id,
            cv_ids=cv_ids,
            relevance=[labels.get((jd_id, cv_id), -1) for cv_id in cv_ids],
        )
        for jd_id in jd_ids
    ]
