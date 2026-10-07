"""Benchmark latency, memory and quality of every ranking method.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --methods bm25,section_aware --embedder mock
Writes results/benchmark.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.evaluation.metrics import evaluate_rankings  # noqa: E402
from cvjd.io import load_documents, load_queries  # noqa: E402
from cvjd.models.embedding import create_embedder  # noqa: E402
from cvjd.models.factory import METHODS, prepare_ranker  # noqa: E402

ALL_METHODS = list(METHODS)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def benchmark_method(
    method: str,
    jds,
    cvs,
    queries,
    embedder,
    pairing: str,
    aggregation: str,
    k: int,
    repeats: int,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    extra = extra or {}
    rss_before = _rss_mb()
    start = time.perf_counter()
    rank_one = prepare_ranker(
        method, jds, cvs, embedder, pairing=pairing, aggregation=aggregation, **extra
    )
    build_s = time.perf_counter() - start
    rss_after_build = _rss_mb()

    params = ""
    if method == "single_pca":
        params = f"pca_dim={extra.get('pca_dim')}"
    elif method == "two_stage":
        params = (
            f"top_k={extra.get('top_k_retrieve')}"
            f",stage1={extra.get('stage1_compression')}"
        )

    for jd in jds:
        rank_one(jd.doc_id)

    latencies_ms: List[float] = []
    rankings: Dict[str, List[Tuple[str, float]]] = {}
    rss_peak = rss_after_build
    for _ in range(repeats):
        for jd in jds:
            t0 = time.perf_counter()
            rankings[jd.doc_id] = rank_one(jd.doc_id)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        rss_peak = max(rss_peak, _rss_mb())

    metrics = evaluate_rankings(rankings, queries, k=k)
    return {
        "method": method,
        "params": params,
        "pairing": pairing if method == "section_aware" else "",
        "aggregation": aggregation if method == "section_aware" else "",
        "build_s": round(build_s, 4),
        "query_ms_mean": round(float(np.mean(latencies_ms)), 4),
        "query_ms_p50": round(float(np.percentile(latencies_ms, 50)), 4),
        "query_ms_p95": round(float(np.percentile(latencies_ms, 95)), 4),
        "rss_before_mb": round(rss_before, 1),
        "rss_build_mb": round(rss_after_build, 1),
        "rss_peak_mb": round(rss_peak, 1),
        f"ndcg@{k}": round(metrics.get(f"ndcg@{k}", 0.0), 4),
        "mrr": round(metrics.get("mrr", 0.0), 4),
        "spearman": round(metrics.get("spearman", 0.0), 4),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--embedder", default="mock", choices=["mock", "st"])
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--pairing", default="cross", choices=["fixed", "cross"])
    parser.add_argument("--aggregation", default="mean", choices=["max", "mean"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--top-k-retrieve", type=int, default=10)
    parser.add_argument(
        "--stage1-compression", default="none",
        choices=["none", "pca", "int8"],
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "benchmark.csv")
    args = parser.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}; choose from {ALL_METHODS}")

    cvs = list(load_documents(args.data_dir / "raw" / "cvs.jsonl").values())
    jds = list(load_documents(args.data_dir / "raw" / "jds.jsonl").values())
    queries = load_queries(args.data_dir / "labels" / "queries.jsonl")

    needs_embeddings = any(m != "bm25" for m in methods)
    embedder = create_embedder(args.embedder) if needs_embeddings else None

    rows = []
    extra = {
        "pca_dim": args.pca_dim,
        "top_k_retrieve": args.top_k_retrieve,
        "stage1_compression": args.stage1_compression,
    }
    for method in methods:
        print(f"benchmarking {method} ...")
        rows.append(benchmark_method(
            method, jds, cvs, queries, embedder,
            pairing=args.pairing, aggregation=args.aggregation,
            k=args.k, repeats=args.repeats, extra=extra,
        ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.rjust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(str(row[c]).rjust(widths[c]) for c in columns))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
