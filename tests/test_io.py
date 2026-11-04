"""Round-trip tests: schema objects -> JSONL -> schema objects."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvjd.io import (  # noqa: E402
    build_queries,
    load_documents,
    load_pairs,
    load_queries,
    pair_from_dict,
    query_from_dict,
    save_documents,
    save_pairs,
    save_queries,
)
from cvjd.schema import Document, Pair, Query, Section  # noqa: E402


def _fixture_docs():
    cv = Document(
        doc_id="cv_0001",
        doc_type="cv",
        lang="en",
        raw_text="SUMMARY\nData Scientist with 3 years of experience.\n\nSKILLS\npython, sql",
        sections={
            "summary": Section("summary", "Data Scientist with 3 years of experience.", "SUMMARY", 0),
            "skills": Section("skills", "python, sql", "SKILLS", 1),
        },
    )
    jd = Document(
        doc_id="jd_0001",
        doc_type="jd",
        lang="en",
        raw_text="SUMMARY\nHiring Data Scientist.\n\nSKILLS\npython",
        sections={
            "summary": Section("summary", "Hiring Data Scientist.", "SUMMARY", 0),
            "skills": Section("skills", "python", "SKILLS", 1),
        },
    )
    return [cv, jd]


def test_document_round_trip(tmp_path):
    docs = _fixture_docs()
    path = tmp_path / "docs.jsonl"
    save_documents(path, docs)
    loaded = load_documents(path)
    assert loaded == {d.doc_id: d for d in docs}


def test_pair_round_trip(tmp_path):
    pairs = [Pair("cv_0001", "jd_0001", 3), Pair("cv_0002", "jd_0001", -1)]
    path = tmp_path / "pairs.jsonl"
    save_pairs(path, pairs)
    assert load_pairs(path) == pairs


def test_query_round_trip(tmp_path):
    queries = [Query("jd_0001", ["cv_0001", "cv_0002"], [3, 0])]
    path = tmp_path / "queries.jsonl"
    save_queries(path, queries)
    assert load_queries(path) == queries


def test_build_queries_unlabeled_fallback():
    pairs = [Pair("cv_0001", "jd_0001", 2)]
    queries = build_queries(pairs, ["jd_0001"], ["cv_0001", "cv_0002"])
    assert len(queries) == 1
    assert queries[0].relevance == [2, -1]


def test_invalid_label_rejected():
    with pytest.raises(ValueError, match="label out of range"):
        pair_from_dict({"cv_id": "cv_1", "jd_id": "jd_1", "label": 4})


def test_invalid_doc_type_rejected():
    from cvjd.io import document_from_dict

    with pytest.raises(ValueError, match="invalid doc_type"):
        document_from_dict({
            "doc_id": "x", "doc_type": "email", "lang": "en",
            "raw_text": "", "sections": {},
        })


def test_query_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        query_from_dict({"jd_id": "jd_1", "cv_ids": ["a", "b"], "relevance": [1]})


def test_duplicate_doc_id_rejected(tmp_path):
    docs = _fixture_docs()
    path = tmp_path / "docs.jsonl"
    save_documents(path, docs)
    path.write_text(
        path.read_text(encoding="utf-8") + path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_documents(path)
