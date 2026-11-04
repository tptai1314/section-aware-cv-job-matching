"""Tests for IR and ranking evaluation metrics (P@K, Recall@K, nDCG@K, MRR, MAP)."""

import pytest

from cvjd.ranking.metrics import (
    aggregate_metrics,
    average_precision,
    dcg_at_k,
    evaluate_query,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    qrels = {"doc1": 3, "doc2": 2, "doc3": 1, "doc4": 0}
    # doc1 (rel>=2: yes), doc3 (no), doc2 (yes), doc4 (no)
    ranked = ["doc1", "doc3", "doc2", "doc4"]
    assert precision_at_k(ranked, qrels, k=2, threshold=2) == 0.5  # 1 out of 2
    assert precision_at_k(ranked, qrels, k=4, threshold=2) == 0.5  # 2 out of 4
    assert precision_at_k(ranked, qrels, k=1, threshold=2) == 1.0  # 1 out of 1


def test_recall_at_k():
    qrels = {"doc1": 3, "doc2": 2, "doc3": 0}
    # total relevant = 2 (doc1, doc2)
    assert recall_at_k(["doc1", "doc3"], qrels, k=2, threshold=2) == 0.5
    assert recall_at_k(["doc1", "doc2"], qrels, k=2, threshold=2) == 1.0
    assert recall_at_k([], qrels, k=5, threshold=2) == 0.0


def test_reciprocal_rank():
    qrels = {"doc1": 0, "doc2": 3, "doc3": 2}
    assert reciprocal_rank(["doc1", "doc2", "doc3"], qrels, threshold=2) == 0.5
    assert reciprocal_rank(["doc2", "doc1"], qrels, threshold=2) == 1.0
    assert reciprocal_rank(["doc1"], qrels, threshold=2) == 0.0


def test_ndcg_at_k_perfect_ranking():
    qrels = {"d1": 3, "d2": 2, "d3": 1, "d4": 0}
    assert ndcg_at_k(["d1", "d2", "d3", "d4"], qrels, k=4) == pytest.approx(1.0)


def test_ndcg_at_k_inverted_ranking():
    qrels = {"d1": 3, "d2": 0}
    perfect = ndcg_at_k(["d1", "d2"], qrels, k=2)
    inverted = ndcg_at_k(["d2", "d1"], qrels, k=2)
    assert perfect == pytest.approx(1.0)
    assert 0.0 < inverted < 1.0


def test_average_precision():
    qrels = {"d1": 2, "d2": 0, "d3": 2}
    # d1 hit at 1 (p=1/1), d2 miss at 2, d3 hit at 3 (p=2/3) -> map = (1 + 2/3) / 2 = 5/6 = 0.8333
    ap = average_precision(["d1", "d2", "d3"], qrels, threshold=2)
    assert pytest.approx(ap, abs=1e-4) == 5.0 / 6.0


def test_evaluate_query_and_aggregate():
    qrels = {"d1": 3, "d2": 2, "d3": 0}
    res = evaluate_query(["d1", "d3", "d2"], qrels, threshold=2)
    assert "ndcg@10" in res
    assert "mrr" in res
    agg = aggregate_metrics([res, res])
    assert agg["mrr"] == res["mrr"]

