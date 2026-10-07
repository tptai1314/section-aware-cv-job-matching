"""End-to-end evaluation: JSONL -> rank -> metrics.

Usage:
    python scripts/evaluate.py --method section_aware --pairing cross
    python scripts/evaluate.py --method bm25
    python scripts/evaluate.py --method chunk_max_sim
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.evaluation.metrics import evaluate_rankings  # noqa: E402
from cvjd.io import load_documents, load_queries  # noqa: E402
from cvjd.models.bm25 import rank_all_bm25  # noqa: E402
from cvjd.models.embedding import create_embedder  # noqa: E402
from cvjd.models.matching import rank_all  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--embedder", default="mock", choices=["mock", "st"])
    parser.add_argument(
        "--method", default="section_aware",
        choices=["section_aware", "single_vector", "chunk_max_sim", "bm25"],
    )
    parser.add_argument("--pairing", default="cross", choices=["fixed", "cross"])
    parser.add_argument("--aggregation", default="mean", choices=["max", "mean"])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    cvs = list(load_documents(args.data_dir / "raw" / "cvs.jsonl").values())
    jds = list(load_documents(args.data_dir / "raw" / "jds.jsonl").values())
    queries = load_queries(args.data_dir / "labels" / "queries.jsonl")

    embedder = None if args.method == "bm25" else create_embedder(args.embedder)

    start = time.perf_counter()
    if args.method == "bm25":
        rankings = rank_all_bm25(jds, cvs)
        pairing_info = ""
    else:
        rankings = rank_all(
            jds, cvs, embedder,
            method=args.method,
            pairing=args.pairing,
            aggregation=args.aggregation,
        )
        pairing_info = f" pairing={args.pairing} agg={args.aggregation}"
    elapsed = time.perf_counter() - start

    metrics = evaluate_rankings(rankings, queries, k=args.k)

    print(
        f"embedder={args.embedder if embedder else '-'} method={args.method}"
        f"{pairing_info}"
    )
    print(f"{'query':>10}  {'NDCG@' + str(args.k):>8}  {'MRR':>6}  {'Spearman':>9}")
    print(
        f"{'mean':>10}  {metrics.get(f'ndcg@{args.k}', 0.0):>8.4f}  "
        f"{metrics.get('mrr', 0.0):>6.4f}  {metrics.get('spearman', 0.0):>9.4f}"
    )
    print(f"rank_time={elapsed:.3f}s over {len(jds)} JDs x {len(cvs)} CVs")


if __name__ == "__main__":
    main()
