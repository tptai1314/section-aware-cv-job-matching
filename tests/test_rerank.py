"""Tests for cross-encoder reranking logic (uses a dummy ranker)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvjd.models.rerank import rank_all_cross_encoder, rerank, rerank_query  # noqa: E402
from cvjd.schema import Document  # noqa: E402


class DummyRanker:
    """Scores a pair by token overlap, no model needed."""

    def score_pairs(self, pairs):
        import numpy as np

        scores = []
        for a, b in pairs:
            ta = set(a.lower().split())
            tb = set(b.lower().split())
            scores.append(float(len(ta & tb)))
        return np.asarray(scores, dtype=np.float32)


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, doc_type="cv", lang="en", raw_text=text)


def _jd(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, doc_type="jd", lang="en", raw_text=text)


def test_cross_encoder_full_ranking_orders_by_score():
    jds = [_jd("jd_1", "python developer")]
    cvs = [
        _doc("cv_1", "unrelated cooking text"),
        _doc("cv_2", "python python developer"),
        _doc("cv_3", "python"),
    ]
    rankings = rank_all_cross_encoder(jds, cvs, ranker=DummyRanker())
    order = [cv_id for cv_id, _ in rankings["jd_1"]]
    assert order[0] == "cv_2"
    assert len(order) == 3


def test_rerank_query_top_k_keeps_tail_in_first_stage_order():
    jd = _jd("jd_1", "python developer")
    cvs_by_id = {
        "cv_1": _doc("cv_1", "python developer python"),
        "cv_2": _doc("cv_2", "python"),
        "cv_3": _doc("cv_3", "nothing here"),
        "cv_4": _doc("cv_4", "also nothing"),
    }
    first_stage = [("cv_3", 0.9), ("cv_4", 0.8), ("cv_1", 0.1), ("cv_2", 0.05)]
    result = rerank_query(jd, first_stage, cvs_by_id, DummyRanker(), top_k=2)
    assert [cv_id for cv_id, _ in result] == ["cv_3", "cv_4", "cv_1", "cv_2"]


def test_rerank_query_full_mode_reranks_everything():
    jd = _jd("jd_1", "python developer")
    cvs_by_id = {
        "cv_1": _doc("cv_1", "python developer python"),
        "cv_2": _doc("cv_2", "nothing"),
    }
    first_stage = [("cv_2", 0.9), ("cv_1", 0.1)]
    result = rerank_query(jd, first_stage, cvs_by_id, DummyRanker(), top_k=None)
    assert [cv_id for cv_id, _ in result] == ["cv_1", "cv_2"]


def test_rerank_query_empty_ranking():
    jd = _jd("jd_1", "python")
    result = rerank_query(jd, [], {}, DummyRanker(), top_k=5)
    assert result == []


def test_rerank_batch_over_queries():
    jds = [_jd("jd_1", "python developer"), _jd("jd_2", "cooking chef")]
    cvs = [
        _doc("cv_1", "python developer python"),
        _doc("cv_2", "cooking chef recipes"),
    ]
    first_stage = {
        "jd_1": [("cv_2", 0.5), ("cv_1", 0.4)],
        "jd_2": [("cv_1", 0.5), ("cv_2", 0.4)],
    }
    result = rerank(first_stage, jds, cvs, ranker=DummyRanker(), top_k=2)
    assert [cv_id for cv_id, _ in result["jd_1"]][0] == "cv_1"
    assert [cv_id for cv_id, _ in result["jd_2"]][0] == "cv_2"
    assert len(result["jd_1"]) == 2


def test_rerank_batch_top_k_keeps_unreranked_tail():
    jds = [_jd("jd_1", "python developer")]
    cvs = [_doc("cv_1", "python"), _doc("cv_2", "nothing")]
    first_stage = {"jd_1": [("cv_2", 0.9), ("cv_1", 0.1)]}
    result = rerank(first_stage, jds, cvs, ranker=DummyRanker(), top_k=1)
    assert [cv_id for cv_id, _ in result["jd_1"]] == ["cv_2", "cv_1"]


def test_rank_all_cross_encoder_empty_pairs_score():
    ranker = DummyRanker()
    import numpy as np

    assert np.array_equal(ranker.score_pairs([]), np.zeros(0))


def test_rerank_requires_valid_top_k():
    jd = _jd("jd_1", "python")
    cvs_by_id = {"cv_1": _doc("cv_1", "python")}
    first_stage = [("cv_1", 0.5)]
    result = rerank_query(jd, first_stage, cvs_by_id, DummyRanker(), top_k=100)
    assert [cv_id for cv_id, _ in result] == ["cv_1"]
