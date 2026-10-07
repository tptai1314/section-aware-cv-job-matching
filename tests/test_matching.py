"""Tests for MockEmbedder and section-aware / single-vector matching."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvjd.models.embedding import MockEmbedder  # noqa: E402
from cvjd.models.matching import (  # noqa: E402
    FULL_KEY,
    aggregate,
    embed_corpus,
    rank_all,
    rank_query,
    score_pair,
    section_components,
)
from cvjd.schema import Document, Section  # noqa: E402


def _doc(doc_id: str, doc_type: str, sections: dict[str, str]) -> Document:
    objs = {
        name: Section(name, text, name.upper(), i)
        for i, (name, text) in enumerate(sections.items())
    }
    raw = "\n\n".join(f"{s.heading_raw}\n{s.text}" for s in objs.values())
    return Document(
        doc_id=doc_id, doc_type=doc_type, lang="en",
        raw_text=raw, sections=objs,
    )


def _cv(doc_id: str, skills: str, summary: str = "Experienced engineer.") -> Document:
    return _doc(doc_id, "cv", {"summary": summary, "skills": skills})


def _jd(doc_id: str, skills: str, summary: str = "Hiring engineer.") -> Document:
    return _doc(doc_id, "jd", {"summary": summary, "skills": skills})


def test_mock_embedder_is_deterministic():
    emb = MockEmbedder(dim=64)
    a = emb.encode(["python sql machine learning", "java"])
    b = emb.encode(["python sql machine learning", "java"])
    assert a.shape == (2, 64)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)


def test_mock_embedder_similar_texts_are_closer():
    emb = MockEmbedder(dim=384)
    query, related, unrelated = emb.encode([
        "python sql machine learning",
        "python sql data analysis",
        "cooking recipes kitchen",
    ])
    sim_related = float(np.dot(query, related))
    sim_unrelated = float(np.dot(query, unrelated))
    assert sim_related > sim_unrelated


def test_mock_embedder_unit_norm_and_empty_text():
    emb = MockEmbedder(dim=32)
    vectors = emb.encode(["hello world", ""])
    assert float(np.linalg.norm(vectors[0])) == pytest.approx(1.0)
    assert float(np.linalg.norm(vectors[1])) == 0.0


def test_embed_corpus_has_sections_and_full_text():
    docs = [_cv("cv_1", "python, sql"), _jd("jd_1", "python")]
    emb = embed_corpus(docs, MockEmbedder(dim=32))
    assert set(emb) == {"cv_1", "jd_1"}
    assert set(emb["cv_1"]) == {"summary", "skills", FULL_KEY}
    assert emb["cv_1"]["skills"].shape == (32,)


def test_single_vector_scores_matching_text_higher():
    docs = [
        _jd("jd_1", "python sql"),
        _cv("cv_match", "python sql machine learning"),
        _cv("cv_miss", "cooking recipes"),
    ]
    emb = embed_corpus(docs, MockEmbedder(dim=128))
    match = score_pair(emb["jd_1"], emb["cv_match"], method="single_vector")
    miss = score_pair(emb["jd_1"], emb["cv_miss"], method="single_vector")
    assert match > miss


def test_fixed_pairing_only_compares_same_name_sections():
    docs = [
        _jd("jd_1", "python", summary="hiring engineer"),
        _cv("cv_1", "python sql", summary="cooking recipes kitchen"),
    ]
    emb = embed_corpus(docs, MockEmbedder(dim=128))
    components = section_components(emb["jd_1"], emb["cv_1"], pairing="fixed")
    assert set(components) == {"summary", "skills"}
    assert components["skills"] > 0.0
    assert components["summary"] == 0.0


def test_cross_pairing_takes_best_cv_section():
    docs = [
        _jd("jd_1", "python"),
        _cv("cv_1", "cooking recipes", summary="python sql python sql"),
    ]
    emb = embed_corpus(docs, MockEmbedder(dim=128))
    fixed = section_components(emb["jd_1"], emb["cv_1"], pairing="fixed")
    cross = section_components(emb["jd_1"], emb["cv_1"], pairing="cross")
    assert cross["skills"] > fixed["skills"]


def test_aggregate_max_and_mean():
    components = {"a": 0.2, "b": 0.8}
    assert aggregate(components, "max") == 0.8
    assert aggregate(components, "mean") == 0.5
    assert aggregate({}, "mean") == 0.0


def test_rank_query_sorted_descending():
    docs = [
        _jd("jd_1", "python sql"),
        _cv("cv_high", "python sql"),
        _cv("cv_low", "cooking recipes"),
    ]
    emb = embed_corpus(docs, MockEmbedder(dim=128))
    ranking = rank_query(emb["jd_1"], {k: emb[k] for k in ("cv_high", "cv_low")})
    assert ranking[0][0] == "cv_high"
    assert ranking[0][1] >= ranking[1][1]


def test_rank_all_covers_every_jd():
    jds = [_jd("jd_1", "python"), _jd("jd_2", "sql")]
    cvs = [_cv("cv_1", "python sql"), _cv("cv_2", "java")]
    rankings = rank_all(jds, cvs, MockEmbedder(dim=64))
    assert set(rankings) == {"jd_1", "jd_2"}
    for ranking in rankings.values():
        assert {cv_id for cv_id, _ in ranking} == {"cv_1", "cv_2"}
    assert rankings["jd_1"][0][0] == "cv_1"
    assert rankings["jd_2"][0][0] == "cv_1"
