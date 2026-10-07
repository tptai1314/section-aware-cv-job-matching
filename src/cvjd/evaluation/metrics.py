"""Ranking metrics: NDCG@k, MRR, Spearman."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

from ..schema import Query


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with exponential gain 2^rel - 1."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return sum(
        (2.0**rel - 1.0) / math.log2(i + 2)
        for i, rel in enumerate(list(relevances)[:k])
    )


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    """NDCG@k of a ranked relevance list; 0.0 when no relevant item exists."""
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def mrr(relevances: Sequence[float], threshold: float = 1.0) -> float:
    """Reciprocal rank of the first relevance >= threshold."""
    for i, rel in enumerate(relevances, start=1):
        if rel >= threshold:
            return 1.0 / i
    return 0.0


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [v - mean_x for v in x]
    dy = [v - mean_y for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties.

    Returns 0.0 when either input is constant or fewer than 2 items.
    """
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} != {len(y)}")
    if len(x) < 2:
        return 0.0
    return _pearson(_average_ranks(x), _average_ranks(y))


def compute_metrics(
    ranked_relevances: Sequence[float],
    scores: Sequence[float] | None = None,
    labels: Sequence[float] | None = None,
    k: int = 10,
    threshold: float = 1.0,
) -> Dict[str, float]:
    """Metrics for a single query.

    ranked_relevances: relevance labels in ranked order (for NDCG/MRR).
    scores/labels: parallel arrays for Spearman (optional).
    """
    result = {
        f"ndcg@{k}": ndcg_at_k(ranked_relevances, k),
        "mrr": mrr(ranked_relevances, threshold),
    }
    if scores is not None and labels is not None:
        result["spearman"] = spearman(scores, labels)
    return result


def evaluate_rankings(
    rankings: Mapping[str, Sequence[Tuple[str, float]]],
    queries: Sequence[Query],
    k: int = 10,
    threshold: float = 1.0,
) -> Dict[str, float]:
    """Mean NDCG@k, MRR and Spearman over queries with at least one label.

    Unlabeled candidates (label -1) are dropped before scoring.
    """
    totals: Dict[str, float] = {}
    count = 0
    for query in queries:
        if query.jd_id not in rankings:
            raise ValueError(f"missing ranking for query {query.jd_id}")
        label_by_cv = dict(zip(query.cv_ids, query.relevance))
        labeled = [
            (score, label_by_cv[cv_id])
            for cv_id, score in rankings[query.jd_id]
            if label_by_cv.get(cv_id, -1) >= 0
        ]
        if not labeled:
            continue
        scores = [score for score, _ in labeled]
        labels = [label for _, label in labeled]
        metrics = compute_metrics(
            ranked_relevances=labels,
            scores=scores if len(labeled) >= 2 else None,
            labels=labels if len(labeled) >= 2 else None,
            k=k,
            threshold=threshold,
        )
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
    if count == 0:
        raise ValueError("no labeled queries to evaluate")
    return {name: value / count for name, value in totals.items()}
