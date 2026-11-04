"""Evaluation metrics for information retrieval and ranking (plan §12, §14).

Standard IR metrics:
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- nDCG@K (Normalized Discounted Cumulative Gain at K)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Dict, List, Tuple


def dcg_at_k(relevances: Sequence[float | int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K using 2^rel - 1 gain."""
    k = min(k, len(relevances))
    score = 0.0
    for i in range(k):
        rel = float(relevances[i])
        score += (math.pow(2.0, rel) - 1.0) / math.log2(i + 2.0)
    return score


def ndcg_at_k(
    ranked_ids: Sequence[str],
    qrels: Mapping[str, float | int],
    k: int = 10,
) -> float:
    """Compute Normalized DCG at K given ranked document IDs and ground-truth relevance dict."""
    if not qrels or k <= 0:
        return 0.0

    actual_rels = [float(qrels.get(doc_id, 0.0)) for doc_id in ranked_ids[:k]]
    # Pad with 0.0 if ranked_ids has fewer than k items
    if len(actual_rels) < k:
        actual_rels.extend([0.0] * (k - len(actual_rels)))

    ideal_rels = sorted((float(v) for v in qrels.values() if float(v) > 0.0), reverse=True)[:k]
    if len(ideal_rels) < k:
        ideal_rels.extend([0.0] * (k - len(ideal_rels)))

    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0.0:
        return 0.0

    dcg = dcg_at_k(actual_rels, k)
    return min(1.0, max(0.0, dcg / idcg))


def precision_at_k(
    ranked_ids: Sequence[str],
    qrels: Mapping[str, float | int],
    k: int = 10,
    threshold: float | int = 2,
) -> float:
    """Proportion of top-K retrieved docs that are relevant (rel >= threshold)."""
    if k <= 0 or not ranked_ids:
        return 0.0
    top_k = ranked_ids[:k]
    rel_count = sum(1 for doc_id in top_k if qrels.get(doc_id, 0) >= threshold)
    return float(rel_count / k)


def recall_at_k(
    ranked_ids: Sequence[str],
    qrels: Mapping[str, float | int],
    k: int = 10,
    threshold: float | int = 2,
) -> float:
    """Proportion of all relevant documents (rel >= threshold) retrieved in top-K."""
    total_relevant = sum(1 for rel in qrels.values() if rel >= threshold)
    if total_relevant <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    rel_count = sum(1 for doc_id in top_k if qrels.get(doc_id, 0) >= threshold)
    return min(1.0, float(rel_count / total_relevant))


def reciprocal_rank(
    ranked_ids: Sequence[str],
    qrels: Mapping[str, float | int],
    threshold: float | int = 2,
) -> float:
    """Reciprocal rank of the first relevant document in ranked_ids."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if qrels.get(doc_id, 0) >= threshold:
            return 1.0 / rank
    return 0.0


def average_precision(
    ranked_ids: Sequence[str],
    qrels: Mapping[str, float | int],
    threshold: float | int = 2,
) -> float:
    """Compute Average Precision for a single query."""
    total_relevant = sum(1 for rel in qrels.values() if rel >= threshold)
    if total_relevant <= 0:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if qrels.get(doc_id, 0) >= threshold:
            hit_count += 1
            precision_sum += hit_count / rank

    return precision_sum / total_relevant


def evaluate_query(
    ranked_ids: Sequence[str],
    qrels: Mapping[str, float | int],
    threshold: float | int = 2,
) -> Dict[str, float]:
    """Calculate all standard metrics for a single ranked list."""
    return {
        "p@5": precision_at_k(ranked_ids, qrels, k=5, threshold=threshold),
        "p@10": precision_at_k(ranked_ids, qrels, k=10, threshold=threshold),
        "recall@5": recall_at_k(ranked_ids, qrels, k=5, threshold=threshold),
        "recall@10": recall_at_k(ranked_ids, qrels, k=10, threshold=threshold),
        "ndcg@5": ndcg_at_k(ranked_ids, qrels, k=5),
        "ndcg@10": ndcg_at_k(ranked_ids, qrels, k=10),
        "mrr": reciprocal_rank(ranked_ids, qrels, threshold=threshold),
        "map": average_precision(ranked_ids, qrels, threshold=threshold),
    }


def aggregate_metrics(metrics_list: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Average a list of per-query metric dictionaries."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        values = [m.get(k, 0.0) for m in metrics_list]
        out[k] = round(sum(values) / len(values), 4)
    return out

