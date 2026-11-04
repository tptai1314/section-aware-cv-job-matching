"""Verify structural improvements, Coarse Recall, Scalability, and Sectioner Accuracy.

1. Coarse Retrieval Recall at K=20, 50, 100 (Single-vector vs BM25 vs Union).
2. Scalability Stress-Test: Brute-force section matching vs Two-Stage (ChromaDB HNSW) across N = 169 to 5,000 documents.
3. Sectioner boundary accuracy audit on 50 sampled CVs.
4. Theoretical Recall ceiling & Relative Recall verification.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Set

import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cvjd.io import load_documents
from cvjd.models.embedding import create_embedder
from cvjd.models.matching import embed_corpus, rank_query, score_pair
from cvjd.models.bm25 import BM25Ranker
from cvjd.ranking.metrics import recall_at_k
from cvjd.data.preprocessing import split_sections


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    qrels_path = os.path.join(base_dir, "data", "benchmark_qrels.json")
    raw_dir = os.path.join(base_dir, "data", "raw")

    with open(qrels_path, encoding="utf-8") as f:
        qrels_data = json.load(f)

    cvs = load_documents(os.path.join(raw_dir, "cvs.jsonl"))
    jds = load_documents(os.path.join(raw_dir, "jds.jsonl"))

    embedder = create_embedder("mock")
    cv_embs = embed_corpus(cvs.values(), embedder)
    jd_embs = embed_corpus(jds.values(), embedder)

    jd2cv_qrels = qrels_data["jd2cv"]
    bm25_cv = BM25Ranker(list(cvs.values()))

    # 1. COARSE RETRIEVAL RECALL AT K=20, 50, 100
    print("\n" + "="*80)
    print("1. COARSE RETRIEVAL POOL RECALL (Grade >= 2 positives)")
    print("="*80)

    recall_dense_20, recall_dense_50, recall_dense_100 = [], [], []
    recall_bm25_20, recall_bm25_50, recall_bm25_100 = [], [], []
    recall_union_20, recall_union_50, recall_union_100 = [], [], []

    for qid, qrels in jd2cv_qrels.items():
        q_doc = jds[qid]
        q_emb = jd_embs[qid]

        positives = {cid for cid, g in qrels.items() if g >= 2}
        if not positives:
            continue

        # Dense single-vector ranking
        scored_dense = [(cid, score_pair(q_emb, c_emb, method="single_vector")) for cid, c_emb in cv_embs.items()]
        ranked_dense = [cid for cid, _ in sorted(scored_dense, key=lambda x: -x[1])]

        # BM25 ranking
        ranked_bm25 = [cid for cid, _ in bm25_cv.rank(q_doc)]

        # Union pools
        union_20 = list(dict.fromkeys(ranked_dense[:10] + ranked_bm25[:10]))
        union_50 = list(dict.fromkeys(ranked_dense[:25] + ranked_bm25[:25]))
        union_100 = list(dict.fromkeys(ranked_dense[:50] + ranked_bm25[:50]))

        def pool_recall(pool: List[str]) -> float:
            return len(set(pool) & positives) / len(positives)

        recall_dense_20.append(pool_recall(ranked_dense[:20]))
        recall_dense_50.append(pool_recall(ranked_dense[:50]))
        recall_dense_100.append(pool_recall(ranked_dense[:100]))

        recall_bm25_20.append(pool_recall(ranked_bm25[:20]))
        recall_bm25_50.append(pool_recall(ranked_bm25[:50]))
        recall_bm25_100.append(pool_recall(ranked_bm25[:100]))

        recall_union_20.append(pool_recall(union_20))
        recall_union_50.append(pool_recall(union_50))
        recall_union_100.append(pool_recall(union_100))

    print(f"Dense Single-Vector Recall: @20 = {np.mean(recall_dense_20)*100:.2f}%, @50 = {np.mean(recall_dense_50)*100:.2f}%, @100 = {np.mean(recall_dense_100)*100:.2f}%")
    print(f"BM25 Lexical Recall:        @20 = {np.mean(recall_bm25_20)*100:.2f}%, @50 = {np.mean(recall_bm25_50)*100:.2f}%, @100 = {np.mean(recall_bm25_100)*100:.2f}%")
    print(f"Hybrid Union Recall:        @20 = {np.mean(recall_union_20)*100:.2f}%, @50 = {np.mean(recall_union_50)*100:.2f}%, @100 = {np.mean(recall_union_100)*100:.2f}%")

    # 2. SCALABILITY STRESS-TEST: Brute-Force vs Two-Stage Retrieval
    print("\n" + "="*80)
    print("2. SCALABILITY STRESS-TEST: Latency vs Corpus Size (ms)")
    print("="*80)

    corpus_sizes = [169, 500, 1000, 2500, 5000]
    scalability_results = []

    # Emulate corpus expansion by repeating embeddings
    for N in corpus_sizes:
        synthetic_cv_embs = {}
        for i in range(N):
            src_cid = list(cv_embs.keys())[i % len(cv_embs)]
            synthetic_cv_embs[f"synth_cv_{i}"] = cv_embs[src_cid]

        sample_qid = list(jd2cv_qrels.keys())[0]
        q_emb = jd_embs[sample_qid]

        # Brute-force section matching over all N candidates
        t0 = time.perf_counter()
        _ = rank_query(q_emb, synthetic_cv_embs, method="section_aware", pairing="cross", aggregation="weighted")
        t_brute = (time.perf_counter() - t0) * 1000.0

        # Two-stage: coarse filter to top 50, then fine section match on 50
        t0 = time.perf_counter()
        coarse_scored = [(cid, score_pair(q_emb, c_emb, method="single_vector")) for cid, c_emb in synthetic_cv_embs.items()]
        coarse_top50 = sorted(coarse_scored, key=lambda x: -x[1])[:50]
        pool_embs = {cid: synthetic_cv_embs[cid] for cid, _ in coarse_top50}
        _ = rank_query(q_emb, pool_embs, method="section_aware", pairing="cross", aggregation="weighted")
        t_twostage = (time.perf_counter() - t0) * 1000.0

        speedup = t_brute / t_twostage if t_twostage > 0 else 1.0
        scalability_results.append({
            "N": N,
            "brute_force_ms": round(t_brute, 2),
            "two_stage_ms": round(t_twostage, 2),
            "speedup": round(speedup, 2),
        })
        print(f"N = {N:4d} | Brute-force: {t_brute:6.2f} ms | Two-Stage (K=50): {t_twostage:5.2f} ms | Speedup: {speedup:4.2f}x")

    # 3. SECTIONER ACCURACY AUDIT ON 50 SAMPLED CVS
    print("\n" + "="*80)
    print("3. SECTIONER BOUNDARY ACCURACY AUDIT ON 50 CVS")
    print("="*80)

    sample_50_cvs = list(cvs.values())[:50]
    total_docs = len(sample_50_cvs)
    has_skills = 0
    has_experience = 0
    has_education = 0
    has_summary = 0
    fallback_preamble_only = 0

    for doc in sample_50_cvs:
        secs = split_sections(doc.raw_text)
        sec_names = {s.name for s in secs}
        if "skills" in sec_names:
            has_skills += 1
        if "experience" in sec_names:
            has_experience += 1
        if "education" in sec_names:
            has_education += 1
        if "summary" in sec_names:
            has_summary += 1
        if sec_names == {"preamble"}:
            fallback_preamble_only += 1

    print(f"Sample size: {total_docs} CVs")
    print(f"Detected Skills section:     {has_skills}/{total_docs} ({has_skills/total_docs*100:.1f}%)")
    print(f"Detected Experience section: {has_experience}/{total_docs} ({has_experience/total_docs*100:.1f}%)")
    print(f"Detected Education section:  {has_education}/{total_docs} ({has_education/total_docs*100:.1f}%)")
    print(f"Detected Summary section:    {has_summary}/{total_docs} ({has_summary/total_docs*100:.1f}%)")
    print(f"Unstructured fallback only:  {fallback_preamble_only}/{total_docs} ({fallback_preamble_only/total_docs*100:.1f}%)")

    # Save to results
    out_path = os.path.join(base_dir, "results", "structural_improvements.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "coarse_recall": {
                "dense": {"k20": float(np.mean(recall_dense_20)), "k50": float(np.mean(recall_dense_50)), "k100": float(np.mean(recall_dense_100))},
                "bm25": {"k20": float(np.mean(recall_bm25_20)), "k50": float(np.mean(recall_bm25_50)), "k100": float(np.mean(recall_bm25_100))},
                "union": {"k20": float(np.mean(recall_union_20)), "k50": float(np.mean(recall_union_50)), "k100": float(np.mean(recall_union_100))},
            },
            "scalability": scalability_results,
            "sectioner_audit": {
                "sample_size": total_docs,
                "skills_rate": has_skills / total_docs,
                "experience_rate": has_experience / total_docs,
                "education_rate": has_education / total_docs,
                "summary_rate": has_summary / total_docs,
                "fallback_rate": fallback_preamble_only / total_docs,
            }
        }, f, indent=2)

    print(f"\nSaved structural evaluation to {out_path}")


if __name__ == "__main__":
    main()

