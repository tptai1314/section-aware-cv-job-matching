"""Tests MatchingService: retrieval -> hybrid -> (optional) hard-constraint gate."""

from pathlib import Path

import pytest

from cvjd.data.loader import document_from_text
from cvjd.models.embedding import MockEmbedder
from cvjd.ranking.constraints import AccessContext
from cvjd.service import MatchingService, Recommendation
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


def _docs():
    cvs = {
        "svc_cv_1": document_from_text(
            "svc_cv_1",
            "cv",
            "SKILLS\n- python, postgresql, docker\n\nEXPERIENCE\n- 5 years backend",
        ),
        "svc_cv_2": document_from_text(
            "svc_cv_2",
            "cv",
            "SKILLS\n- react, css\n\nEXPERIENCE\n- 2 years frontend",
        ),
    }
    job = document_from_text(
        "svc_jd_1", "jd", "Backend\n\nSKILLS\n- python, postgresql\n\nNeed 3 years experience"
    )
    return cvs, job


def _service(client, embedder):
    service = MatchingService(client, embedder, embed_model="mock", config={})
    cvs, job = _docs()
    service.index(list(cvs.values()) + [job])
    return service, cvs, job


def test_recommend_semantic_only(client, embedder):
    service, _cvs, job = _service(client, embedder)
    recs = service.recommend_cvs_for_job(job, pool_size=50, top_k=50)
    assert recs and all(isinstance(r, Recommendation) for r in recs)
    ids = {r.doc_id for r in recs}
    assert {"svc_cv_1", "svc_cv_2"} <= ids
    payload = next(r for r in recs if r.doc_id == "svc_cv_1").to_dict()
    assert {"doc_id", "score", "components", "section_components"} <= set(payload)


def test_recommend_hybrid_ranks_matching_cv_first(client, embedder):
    service, cvs, job = _service(client, embedder)
    recs = service.recommend_cvs_for_job(
        job, documents=cvs, pool_size=50, top_k=50, keyword_scores=None
    )
    order = {r.doc_id: i for i, r in enumerate(recs)}
    assert order["svc_cv_1"] < order["svc_cv_2"]
    assert next(r for r in recs if r.doc_id == "svc_cv_1").skills > 0.0


def test_recommend_fuses_keyword_scores(client, embedder):
    service, cvs, job = _service(client, embedder)
    recs = service.recommend_cvs_for_job(
        job,
        documents=cvs,
        keyword_scores={"svc_cv_1": 1.0, "svc_cv_2": 0.0},
        keyword_alpha=0.5,
        pool_size=50,
        top_k=50,
    )
    by_id = {r.doc_id: r for r in recs}
    assert by_id["svc_cv_1"].keyword == 1.0
    assert by_id["svc_cv_1"].score >= by_id["svc_cv_2"].score


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.rows = self.conn.resolve(sql, params)

    def fetchall(self):
        return self.rows


class _FakeConn:
    def __init__(self):
        self._catalog = [("FreelancerProfile", "id"), ("FreelancerProfile", "userId")]

    def cursor(self, row_factory=None):
        return _FakeCursor(self)

    def resolve(self, sql, params=None):
        if "information_schema" in sql:
            return self._catalog
        if '"FreelancerProfile"' in sql:
            wanted = set(params[0]) if params else set()
            if 1 in wanted:
                return [{"id": 1, "userId": "u1"}]
            return []
        return []


def test_recommend_applies_postgres_gate(client, embedder):
    cvs = {
        "cv_1": document_from_text("cv_1", "cv", "SKILLS\n- python, postgresql"),
        "cv_2": document_from_text("cv_2", "cv", "SKILLS\n- python, postgresql"),
    }
    job = document_from_text("jd_1", "jd", "SKILLS\n- python, postgresql")
    service = MatchingService(client, embedder, embed_model="mock", config={})
    service.index(list(cvs.values()) + [job])

    recs = service.recommend_cvs_for_job(
        job,
        documents=cvs,
        pool_size=10,
        top_k=5,
        conn=_FakeConn(),
        ctx=AccessContext(),
    )
    assert [r.doc_id for r in recs] == ["cv_1"]


def test_rank_structured_without_chroma():
    service = MatchingService(None, None, config={})
    query = {
        "required_skills": [{"name": "React"}, {"name": "Node"}],
        "min_years_experience": 3,
    }
    candidates = [
        {"id": "c1", "skills": [{"name": "reactjs"}, {"name": "node"}], "years_experience": 5},
        {"id": "c2", "skills": [{"name": "PHP"}], "years_experience": 1},
    ]
    recs = service.rank_structured(
        query, "cv", candidates, semantic_scores={"c1": 0.9, "c2": 0.2}
    )
    assert [r.doc_id for r in recs] == ["c1", "c2"]
    assert recs[0].skills == 1.0
    assert recs[0].semantic == 0.9


def test_rank_structured_cv2jd_and_keyword():
    service = MatchingService(None, None, config={})
    query = {"skills": [{"name": "React"}], "years_experience": 4}
    candidates = [
        {"id": "j1", "required_skills": [{"name": "React"}], "min_years_experience": 2},
        {"id": "j2", "required_skills": [{"name": "Java"}], "min_years_experience": 8},
    ]
    recs = service.rank_structured(
        query,
        "jd",
        candidates,
        semantic_scores={"j1": 0.8, "j2": 0.7},
        keyword_scores={"j1": 0.0, "j2": 1.0},
        keyword_alpha=0.5,
    )
    by_id = {r.doc_id: r for r in recs}
    assert by_id["j1"].skills == 1.0
    assert by_id["j1"].keyword == 0.0
    assert by_id["j1"].score >= by_id["j2"].score


def test_recommend_jobs_for_cv_direction(client, embedder):
    service, cvs, job = _service(client, embedder)
    recs = service.recommend_jobs_for_cv(
        cvs["svc_cv_1"], documents={"svc_jd_1": job}, pool_size=10, top_k=5
    )
    assert recs[0].doc_id == "svc_jd_1"


def test_service_delete_removes_vectors(client, embedder):
    service = MatchingService(client, embedder, embed_model="mock", config={})
    doc = document_from_text("svc_del_1", "cv", "SKILLS\n- go, rust")
    service.index([doc])
    assert chroma.get_document(client, "cv", "svc_del_1")

    deleted = service.delete("svc_del_1", "cv")
    assert deleted > 0
    assert chroma.get_document(client, "cv", "svc_del_1") == []
