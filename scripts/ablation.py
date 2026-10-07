"""Ablation (leave-one-section-out, pairing, aggregation) + error analysis.

Usage: python scripts/ablation.py [--embedder mock] [--k 10]
Writes results/ablation.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.evaluation.metrics import evaluate_rankings, ndcg_at_k  # noqa: E402
from cvjd.io import load_documents, load_queries  # noqa: E402
from cvjd.models.embedding import create_embedder  # noqa: E402
from cvjd.models.matching import aggregate, embed_corpus, section_components  # noqa: E402


def build_variant_rankings(
    jds,
    cvs,
    jd_emb,
    cv_emb,
    pairing: str,
    aggregation: str,
    drop: Sequence[str] = (),
) -> Dict[str, List[Tuple[str, float]]]:
    rankings: Dict[str, List[Tuple[str, float]]] = {}
    for jd in jds:
        scored = []
        for cv in cvs:
            components = section_components(
                jd_emb[jd.doc_id], cv_emb[cv.doc_id], pairing=pairing
            )
            for name in drop:
                components.pop(name, None)
            scored.append((cv.doc_id, aggregate(components, aggregation)))
        rankings[jd.doc_id] = sorted(scored, key=lambda item: (-item[1], item[0]))
    return rankings


def per_query_ndcg(rankings, queries, k: int) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for query in queries:
        label_by_cv = dict(zip(query.cv_ids, query.relevance))
        labels = [
            label_by_cv[cv_id]
            for cv_id, _ in rankings[query.jd_id]
            if label_by_cv.get(cv_id, -1) >= 0
        ]
        if labels:
            result[query.jd_id] = ndcg_at_k(labels, k)
    return result


def error_analysis(rankings, queries, k: int, worst: int = 3) -> None:
    scores = per_query_ndcg(rankings, queries, k)
    order = sorted(scores, key=lambda jd_id: scores[jd_id])
    print(f"\nworst queries by NDCG@{k}:")
    for jd_id in order[:worst]:
        query = next(q for q in queries if q.jd_id == jd_id)
        label_by_cv = dict(zip(query.cv_ids, query.relevance))
        rank_by_cv = {cv_id: i + 1 for i, (cv_id, _) in enumerate(rankings[jd_id])}
        missed = [
            (cv_id, label_by_cv[cv_id], rank_by_cv[cv_id])
            for cv_id in query.cv_ids
            if label_by_cv.get(cv_id, -1) >= 2
        ]
        missed = sorted(missed, key=lambda t: -t[2])[:3]
        print(f"  {jd_id}: ndcg@{k}={scores[jd_id]:.4f}")
        for cv_id, label, rank in missed:
            print(f"    label {label} CV {cv_id} ranked #{rank}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--embedder", default="mock", choices=["mock", "st"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "ablation.csv")
    args = parser.parse_args(argv)

    cvs = list(load_documents(args.data_dir / "raw" / "cvs.jsonl").values())
    jds = list(load_documents(args.data_dir / "raw" / "jds.jsonl").values())
    queries = load_queries(args.data_dir / "labels" / "queries.jsonl")

    embedder = create_embedder(args.embedder)
    jd_emb = embed_corpus(jds, embedder)
    cv_emb = embed_corpus(cvs, embedder)

    section_names = sorted({
        name
        for doc in jds + cvs
        for name in doc.sections
    })

    variants = [
        ("full", "cross", "mean", ()),
        ("pairing_fixed", "fixed", "mean", ()),
        ("agg_max", "cross", "max", ()),
    ]
    variants += [
        (f"drop_{name}", "cross", "mean", (name,))
        for name in section_names
    ]

    rows = []
    full_rankings = None
    for variant, pairing, aggregation, drop in variants:
        rankings = build_variant_rankings(
            jds, cvs, jd_emb, cv_emb, pairing, aggregation, drop
        )
        metrics = evaluate_rankings(rankings, queries, k=args.k)
        rows.append({
            "variant": variant,
            "pairing": pairing,
            "aggregation": aggregation,
            "dropped": "+".join(drop),
            f"ndcg@{args.k}": round(metrics.get(f"ndcg@{args.k}", 0.0), 4),
            "mrr": round(metrics.get("mrr", 0.0), 4),
            "spearman": round(metrics.get("spearman", 0.0), 4),
        })
        if variant == "full":
            full_rankings = rankings

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    widths = {
        c: max(len(c), *(len(str(r[c])) for r in rows)) for c in rows[0]
    }
    print("  ".join(c.rjust(widths[c]) for c in rows[0]))
    for row in rows:
        print("  ".join(str(row[c]).rjust(widths[c]) for c in rows[0]))
    print(f"wrote {args.out}")

    error_analysis(full_rankings, queries, args.k)


if __name__ == "__main__":
    main()
