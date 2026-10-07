"""Known-value tests for NDCG@k, MRR, Spearman."""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvjd.evaluation.metrics import (  # noqa: E402
    compute_metrics,
    dcg_at_k,
    mrr,
    ndcg_at_k,
    spearman,
)


def test_dcg_known_values():
    assert dcg_at_k([0, 0], 10) == 0.0
    assert dcg_at_k([1], 10) == pytest.approx(1.0)
    assert dcg_at_k([3, 2, 1], 3) == pytest.approx(
        7.0 / 1 + 3.0 / math.log2(3) + 1.0 / math.log2(4)
    )
    assert dcg_at_k([1, 2, 3, 4], 2) == pytest.approx(dcg_at_k([1, 2], 2))


def test_dcg_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        dcg_at_k([1], 0)


def test_ndcg_perfect_ranking():
    assert ndcg_at_k([3, 2, 1, 0], 10) == pytest.approx(1.0)
    assert ndcg_at_k([1, 1], 10) == pytest.approx(1.0)


def test_ndcg_worst_ranking():
    value = ndcg_at_k([0, 0, 3], 10)
    assert 0.0 < value < 1.0
    assert value == pytest.approx(
        (2.0**3 - 1.0) / math.log2(4) / ((2.0**3 - 1.0) / math.log2(2))
    )


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k([0, 0, 0], 10) == 0.0


def test_ndcg_respects_k():
    assert ndcg_at_k([1, 1, 0], 3) == pytest.approx(1.0)
    assert ndcg_at_k([1, 0, 1], 1) == pytest.approx(1.0)
    assert ndcg_at_k([1, 0, 1], 3) < 1.0


def test_mrr():
    assert mrr([0, 1, 0]) == pytest.approx(0.5)
    assert mrr([0, 0, 1]) == pytest.approx(1.0 / 3.0)
    assert mrr([2, 1]) == pytest.approx(1.0)
    assert mrr([0, 0, 0]) == 0.0
    assert mrr([1, 0], threshold=2) == 0.0
    assert mrr([2], threshold=2) == pytest.approx(1.0)


def test_spearman_identical_and_reversed():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_with_ties():
    assert spearman([1, 1, 2], [2, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2], [5, 5, 9]) == pytest.approx(1.0)


def test_spearman_constant_input_is_zero():
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_spearman_length_mismatch():
    with pytest.raises(ValueError):
        spearman([1, 2], [1])


def test_spearman_fewer_than_two_items():
    assert spearman([], []) == 0.0
    assert spearman([1], [2]) == 0.0


def test_compute_metrics_keys():
    result = compute_metrics(
        ranked_relevances=[3, 1, 0],
        scores=[0.9, 0.5, 0.2],
        labels=[3, 1, 0],
        k=10,
    )
    assert set(result) == {"ndcg@10", "mrr", "spearman"}
    assert result["ndcg@10"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)


def test_compute_metrics_without_scores_omits_spearman():
    result = compute_metrics(ranked_relevances=[0, 1], k=10)
    assert set(result) == {"ndcg@10", "mrr"}
