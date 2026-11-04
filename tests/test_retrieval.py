"""Tests Chroma-backed retrieval + section-aware semantic scoring."""

from pathlib import Path

import pytest

from cvjd.data.loader import document_from_text
from cvjd.models.embedding import MockEmbedder
from cvjd.retrieval import ChromaRetriever, RetrievalResult
from cvjd.storage import chroma

REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture(scope="module")
def client():
    c = chroma.build_client(None)
    chroma.ensure_collections(c)
    return c


@pytest.fixture(scope="module")
def embedder():
    return MockEmbedder(dim=32)


def make_retriever(client, embedder, **kwargs):
    return ChromaRetriever(client, embedder, "mock", **kwargs)


def test_index_corpus_upserts_sections_and_full(client, embedder):
    retriever = make_retriever(client, embedder)
    doc = document_from_text(
        "rt_cv_1", "cv", "SKILLS\n- python, fastapi\n\nEXPERIENCE\n- 3 years backend"
    )
    stats = retriever.index_corpus([doc])
    assert stats["upserted"] >= 2
    vectors = retriever.section_vectors("cv", "rt_cv_1")
    assert chroma.FULL_SECTION in vectors
    assert "skills" in vectors


def test_candidate_pool_dedups_documents(client, embedder):
    retriever = make_retriever(client, embedder)
    cvs = [
        document_from_text("rt_cv_p1", "cv", "SKILLS\n- python, postgres, docker"),
        document_from_text("rt_cv_p2", "cv", "SKILLS\n- react, typescript, css"),
        document_from_text("rt_cv_p3", "cv", "SKILLS\n- python, pandas, numpy"),
    ]
    retriever.index_corpus(cvs)
    jd = document_from_text("rt_jd_p", "jd", "Backend python developer with postgres")
    pool = retriever.pool_by_text(jd.raw_text, "cv", pool_size=10)
    ids = [item["doc_id"] for item in pool]
    assert len(ids) == len(set(ids))
    assert all(isinstance(item["distance"], float) for item in pool)


def test_pool_respects_status_filter(client, embedder):
    retriever = make_retriever(client, embedder, status="READY")
    retriever.index_document(
        document_from_text("rt_cv_r", "cv", "SKILLS\n- golang"), status="READY"
    )
    retriever.index_document(
        document_from_text("rt_cv_x", "cv", "SKILLS\n- golang"), status="PROCESSING"
    )
    pool = retriever.pool_by_text("golang", "cv", pool_size=20, status="READY")
    ids = {item["doc_id"] for item in pool}
    assert "rt_cv_r" in ids
    assert "rt_cv_x" not in ids


def test_retrieve_ranks_related_cv_first(client, embedder):
    retriever = make_retriever(client, embedder, pairing="cross", aggregation="mean")
    cvs = [
        document_from_text(
            "rt_cv_rel",
            "cv",
            "SKILLS\n- python, fastapi, postgresql\n\nEXPERIENCE\n- 5 years backend",
        ),
        document_from_text(
            "rt_cv_irr",
            "cv",
            "SKILLS\n- react, css, html\n\nEXPERIENCE\n- 2 years frontend",
        ),
    ]
    retriever.index_corpus(cvs)
    jd = document_from_text("rt_jd_rel", "jd", "Backend python developer postgres fastapi")

    pool = retriever.pool_by_text(jd.raw_text, "cv", pool_size=50)
    pool_order = {item["doc_id"]: i for i, item in enumerate(pool)}
    assert pool_order["rt_cv_rel"] < pool_order["rt_cv_irr"]

    results = retriever.retrieve(jd, "cv", pool_size=50, top_k=50)
    by_id = {r.doc_id: r for r in results}
    assert "rt_cv_rel" in by_id and "rt_cv_irr" in by_id
    assert all(0.0 <= r.semantic <= 1.0 for r in results)


def test_section_scoring_skips_full_key_and_unknown(client, embedder):
    retriever = make_retriever(client, embedder)
    doc = document_from_text("rt_cv_sc", "cv", "SKILLS\n- python\n\nEXPERIENCE\n- 3 years")
    retriever.index_document(doc)
    jd = document_from_text("rt_jd_sc", "jd", "SKILLS\n- python backend")
    scored = retriever.section_scoring(jd, "cv", ["rt_cv_sc", "rt_cv_missing"])
    assert chroma.FULL_SECTION not in scored["rt_cv_sc"].components
    assert 0.0 <= scored["rt_cv_sc"].semantic <= 1.0
    assert scored["rt_cv_missing"].semantic == 0.0


def test_retrieve_top_k_limits_results(client, embedder):
    retriever = make_retriever(client, embedder)
    cvs = [
        document_from_text(f"rt_cv_k{i}", "cv", "SKILLS\n- python, data science")
        for i in range(5)
    ]
    retriever.index_corpus(cvs)
    jd = document_from_text("rt_jd_k", "jd", "python data science")
    results = retriever.retrieve(jd, "cv", pool_size=10, top_k=3)
    assert len(results) <= 3
    assert all(isinstance(r, RetrievalResult) for r in results)
