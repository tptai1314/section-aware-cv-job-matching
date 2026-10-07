"""End-to-end evaluation: JSONL -> embed -> rank -> metrics.

Usage:
    python scripts/evaluate.py --method section_aware --pairing cross
    python scripts/evaluate.py --method single_vector --embedder mock
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.evaluation.metrics import compute_metrics
from cvjd.io import load_documents, load_queries
from cvjd.models.embedding import create_embedder
from cvjd.models.matching import rank_all


def evaluate(
    rankings: Dict[str, List[Tuple[str, float]]],
    queries,
    k: int = 10,
) -> Dict[str, float]:
    """Average NDCG@k, MRR, Spearman over labeled queries."""
    totals: Dict[str, float] = {}
    count = 0
    for query in queries:
        label_by_cv = dict(zip(query.cv_ids, query.relevance))
        ranked = rankings[query.jd_id]
        labeled = [
            (score, label_by_cv[cv_id])
            for cv_id, score in ranked
            if label_by_cv.get(cv_id, -1) >= 0
        ]
        if not labeled:
            continue
        scores = [s for s, _ in labeled]
        labels = [r for _, r in labeled]
        metrics = compute_metrics(
            ranked_relevances=labels,
            scores=scores if len(labeled) >= 2 else None,
            labels=labels if len(labeled) >= 2 else None,
            k=k,
        )
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
    if count == 0:
        raise ValueError("no labeled queries to evaluate")
    return {name: value / count for name, value in totals.items()}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--embedder", default="mock", choices=["mock", "st"])
    parser.add_argument(
        "--method", default="section_aware",
        choices=["section_aware", "single_vector"],
    )
    parser.add_argument("--pairing", default="cross", choices=["fixed", "cross"])
    parser.add_argument("--aggregation", default="mean", choices=["max", "mean"])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    cvs = list(load_documents(args.data_dir / "raw" / "cvs.jsonl").values())
    jds = list(load_documents(args.data_dir / "raw" / "jds.jsonl").values())
    queries = load_queries(args.data_dir / "labels" / "queries.jsonl")

    embedder = create_embedder(args.embedder)

    start = time.perf_counter()
    rankings = rank_all(
        jds, cvs, embedder,
        method=args.method,
        pairing=args.pairing,
        aggregation=args.aggregation,
    )
    elapsed = time.perf_counter() - start

    metrics = evaluate(rankings, queries, k=args.k)

    print(
        f"embedder={args.embedder} method={args.method} "
        f"pairing={args.pairing} agg={args.aggregation}"
    )
    print(f"{'query':>10}  {'NDCG@' + str(args.k):>8}  {'MRR':>6}  {'Spearman':>9}")
    print(
        f"{'mean':>10}  {metrics.get(f'ndcg@{args.k}', 0.0):>8.4f}  "
        f"{metrics.get('mrr', 0.0):>6.4f}  {metrics.get('spearman', 0.0):>9.4f}"
    )
    print(f"rank_time={elapsed:.3f}s over {len(jds)} JDs x {len(cvs)} CVs")


if __name__ == "__main__":
    main()
