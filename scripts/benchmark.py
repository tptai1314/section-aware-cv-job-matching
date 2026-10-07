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
from cvjd.models.bm25 import BM25Ranker  # noqa: E402
from cvjd.models.chunking import embed_chunks, max_similarity  # noqa: E402
from cvjd.models.embedding import create_embedder  # noqa: E402
from cvjd.models.matching import embed_corpus, rank_query  # noqa: E402

ALL_METHODS = ["bm25", "single_vector", "section_aware", "chunk_max_sim"]

RankOne = Callable[[str], List[Tuple[str, float]]]


def _sorted_scores(scored: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def prepare_ranker(
    method: str,
    jds,
    cvs,
    embedder,
    pairing: str,
    aggregation: str,
    chunk_max_tokens: int = 64,
    chunk_overlap: int = 16,
) -> RankOne:
    if method == "bm25":
        ranker = BM25Ranker(cvs)
        jd_by_id = {jd.doc_id: jd for jd in jds}
        return lambda jd_id: ranker.rank(jd_by_id[jd_id])

    if method == "chunk_max_sim":
        jd_chunks = embed_chunks(jds, embedder, chunk_max_tokens, chunk_overlap)
        cv_chunks = embed_chunks(cvs, embedder, chunk_max_tokens, chunk_overlap)
        return lambda jd_id: _sorted_scores([
            (cv_id, max_similarity(jd_chunks[jd_id], cv_chunks[cv_id]))
            for cv_id in cv_chunks
        ])

    jd_emb = embed_corpus(jds, embedder)
    cv_emb = embed_corpus(cvs, embedder)
    return lambda jd_id: rank_query(
        jd_emb[jd_id], cv_emb, method=method, pairing=pairing, aggregation=aggregation
    )


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
) -> Dict[str, object]:
    rss_before = _rss_mb()
    start = time.perf_counter()
    rank_one = prepare_ranker(method, jds, cvs, embedder, pairing, aggregation)
    build_s = time.perf_counter() - start
    rss_after_build = _rss_mb()

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
    for method in methods:
        print(f"benchmarking {method} ...")
        rows.append(benchmark_method(
            method, jds, cvs, queries, embedder,
            pairing=args.pairing, aggregation=args.aggregation,
            k=args.k, repeats=args.repeats,
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
