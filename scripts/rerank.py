"""Cross-encoder upper bound and two-stage reranking evaluation.

Usage:
    python scripts/rerank.py --mode full
    python scripts/rerank.py --mode two-stage --first-stage section_aware --top-k 10
Writes results/rerank.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.evaluation.metrics import evaluate_rankings  # noqa: E402
from cvjd.io import load_documents, load_queries  # noqa: E402
from cvjd.models.embedding import create_embedder  # noqa: E402
from cvjd.models.factory import METHODS, Ranking, prepare_ranker  # noqa: E402
from cvjd.models.rerank import (  # noqa: E402
    DEFAULT_CROSS_ENCODER,
    CrossEncoderRanker,
    rank_all_cross_encoder,
    rerank_query,
)


def _stats(latencies_ms: List[float]) -> Tuple[float, float]:
    mean = sum(latencies_ms) / len(latencies_ms)
    ordered = sorted(latencies_ms)
    idx = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return mean, ordered[idx]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--mode", default="two-stage", choices=["full", "two-stage"])
    parser.add_argument("--first-stage", default="section_aware", choices=list(METHODS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedder", default="mock", choices=["mock", "st"])
    parser.add_argument("--pairing", default="cross", choices=["fixed", "cross"])
    parser.add_argument("--aggregation", default="mean", choices=["max", "mean"])
    parser.add_argument("--model", default=DEFAULT_CROSS_ENCODER)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "rerank.csv")
    args = parser.parse_args(argv)

    cvs = list(load_documents(args.data_dir / "raw" / "cvs.jsonl").values())
    jds = list(load_documents(args.data_dir / "raw" / "jds.jsonl").values())
    queries = load_queries(args.data_dir / "labels" / "queries.jsonl")
    cvs_by_id = {cv.doc_id: cv for cv in cvs}

    t0 = time.perf_counter()
    ranker = CrossEncoderRanker(args.model)
    build_s = time.perf_counter() - t0

    final: Dict[str, Ranking] = {}
    latencies_ms: List[float] = []

    if args.mode == "full":
        for jd in jds:
            t0 = time.perf_counter()
            final[jd.doc_id] = rank_all_cross_encoder([jd], cvs, ranker)[jd.doc_id]
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        stage_info = "mode=full (all pairs)"
    else:
        embedder = None if args.first_stage == "bm25" else create_embedder(args.embedder)
        first_rank_one = prepare_ranker(
            args.first_stage, jds, cvs, embedder,
            pairing=args.pairing, aggregation=args.aggregation,
        )
        for jd in jds:
            first = first_rank_one(jd.doc_id)
            t0 = time.perf_counter()
            final[jd.doc_id] = rerank_query(
                jd, first, cvs_by_id, ranker, top_k=args.top_k
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        stage_info = f"mode=two-stage first={args.first_stage} top_k={args.top_k}"

    metrics = evaluate_rankings(final, queries, k=args.k)
    mean_ms, p95_ms = _stats(latencies_ms)

    print(f"model={args.model} {stage_info}")
    print(f"{'query':>10}  {'NDCG@' + str(args.k):>8}  {'MRR':>6}  {'Spearman':>9}")
    print(
        f"{'mean':>10}  {metrics.get(f'ndcg@{args.k}', 0.0):>8.4f}  "
        f"{metrics.get('mrr', 0.0):>6.4f}  {metrics.get('spearman', 0.0):>9.4f}"
    )
    print(f"build={build_s:.3f}s query_mean={mean_ms:.2f}ms query_p95={p95_ms:.2f}ms")

    row = {
        "mode": args.mode,
        "first_stage": "" if args.mode == "full" else args.first_stage,
        "top_k": "" if args.mode == "full" else args.top_k,
        "model": args.model,
        "build_s": round(build_s, 3),
        "query_ms_mean": round(mean_ms, 3),
        "query_ms_p95": round(p95_ms, 3),
        f"ndcg@{args.k}": round(metrics.get(f"ndcg@{args.k}", 0.0), 4),
        "mrr": round(metrics.get("mrr", 0.0), 4),
        "spearman": round(metrics.get("spearman", 0.0), 4),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with args.out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended -> {args.out}")


if __name__ == "__main__":
    main()
