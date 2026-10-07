"""Round-trip tests: schema objects -> JSONL -> schema objects."""

import importlib.util
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


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_mock_generator_round_trip(tmp_path):
    mock = _load_script("make_mock_data")
    cvs, jds, pairs, queries = mock.build_mock(
        n_cvs=5, n_jds=3, seed=7, noise=0.2, coverage=0.8
    )

    save_documents(tmp_path / "cvs.jsonl", cvs)
    save_documents(tmp_path / "jds.jsonl", jds)
    save_pairs(tmp_path / "pairs.jsonl", pairs)
    save_queries(tmp_path / "queries.jsonl", queries)

    assert load_documents(tmp_path / "cvs.jsonl") == {d.doc_id: d for d in cvs}
    assert load_documents(tmp_path / "jds.jsonl") == {d.doc_id: d for d in jds}
    assert load_pairs(tmp_path / "pairs.jsonl") == pairs
    assert load_queries(tmp_path / "queries.jsonl") == queries

    cv_ids = {d.doc_id for d in cvs}
    jd_ids = {d.doc_id for d in jds}
    for pair in pairs:
        assert pair.cv_id in cv_ids and pair.jd_id in jd_ids
        assert -1 <= pair.label <= 3
    for query in queries:
        assert query.jd_id in jd_ids
        assert len(query.cv_ids) == len(query.relevance)
        assert set(query.cv_ids) == cv_ids
