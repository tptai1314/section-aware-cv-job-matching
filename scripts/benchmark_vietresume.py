"""Comprehensive Benchmark Evaluation Runner on VietResume-Bench.

Dataset: data/public_benchmark/vietresume/
  - cvs.jsonl (1,585 CVs)
  - jds.jsonl (60 JDs across 10 IT categories)
  - benchmark_qrels.json (3-tier relevance: 0=No Fit, 1=Potential Fit, 2=Good Fit)

Evaluates:
  1. Lexical Baselines: BM25 (pyvi) vs BM25 (whitespace)
  2. Dense Baselines: Single-Vector vs Chunk Max-Sim
  3. Proposed Architecture: Section-Aware Multi-Vector (Fixed & Cross)
  4. Hybrid Pipeline: Section-Aware + BM25 + Rules + Score Calibration
  5. Sliced Language Analysis (RQ1, RQ2): VI-VI, VI-EN (cross-lingual), Code-Mixed
  6. Robustness to Diacritic Removal (RQ4): Accented vs Unaccented Vietnamese text

Outputs:
  - results/vietresume_benchmark_results.json
  - paper/tables/tab_vietresume_main.tex
  - paper/tables/tab_vietresume_bilingual.tex
  - paper/tables/tab_vietresume_robustness.tex
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.data.preprocessing import split_sections
from cvjd.io import load_documents
from cvjd.models.bm25 import BM25Ranker
from cvjd.models.chunking import rank_all_chunk
from cvjd.models.embedding import create_embedder
from cvjd.models.matching import embed_corpus, rank_query
from cvjd.ranking.engine import HybridRanker
from cvjd.ranking.hybrid import (
    hybrid_score,
    job_from_document,
    profile_from_document,
    strip_accents,
)
from cvjd.ranking.metrics import aggregate_metrics, evaluate_query, ndcg_at_k
from cvjd.schema import Document

DATA_DIR = REPO_ROOT / "data" / "public_benchmark" / "vietresume"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name or 'mock' / 'bge-m3' / 'st'",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--direction",
        default="both",
        choices=["both", "jd2cv", "cv2jd"],
        help="Matching direction to evaluate",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Limit queries (for quick testing)",
    )
    return parser.parse_args()


def format_table_row(name: str, m: Dict[str, float]) -> str:
    return (
        f"| {name:<35} | {m.get('ndcg@5', 0.0):.4f} | {m.get('ndcg@10', 0.0):.4f} | "
        f"{m.get('mrr', 0.0):.4f} | {m.get('map', 0.0):.4f} | {m.get('p@5', 0.0):.4f} |"
    )


def evaluate_ranking_run(
    queries: Sequence[Document],
    rank_fn,
    qrels: Dict[str, Dict[str, int]],
    threshold: int = 1,
) -> Tuple[Dict[str, float], Dict[str, List[str]], float]:
    """Evaluate a ranking function across all queries with latency tracking."""
    per_query_metrics = []
    ranked_map = {}
    t0 = time.perf_counter()

    for q in queries:
        qid = q.doc_id
        if qid not in qrels or not qrels[qid]:
            continue
        ranked = rank_fn(q)
        ranked_ids = [doc_id for doc_id, _ in ranked]
        ranked_map[qid] = ranked_ids
        m = evaluate_query(ranked_ids, qrels[qid], threshold=threshold)
        per_query_metrics.append(m)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    latency_per_query = elapsed_ms / max(len(per_query_metrics), 1)
    agg = aggregate_metrics(per_query_metrics)
    agg["latency_ms"] = latency_per_query
    return agg, ranked_map, latency_per_query


def make_hybrid_runner(
    q_embs: dict,
    c_embs: dict,
    candidate_docs: dict[str, Document],
    bm25_ranker: BM25Ranker,
    hybrid_ranker: HybridRanker,
    target_type: str,
):
    """Build a standalone hybrid ranking closure."""
    def run_hybrid(q: Document) -> list[tuple[str, float]]:
        sem_ranked = rank_query(
            q_embs[q.doc_id], c_embs, method="section_aware", pairing="cross", aggregation="weighted"
        )
        sem_scores = {cid: sc for cid, sc in sem_ranked}
        kw_scores = dict(bm25_ranker.rank(q))
        fused_sem = hybrid_ranker.fuse(sem_scores, kw_scores, alpha=0.5)

        scored = []
        for cid, sem_score in fused_sem.items():
            c_doc = candidate_docs.get(cid)
            if not c_doc:
                continue
            if target_type == "cv":
                j_req = replace(job_from_document(q), semantic_base=sem_score)
                c_prof = profile_from_document(c_doc)
                tot = hybrid_score(j_req, c_prof, weights=hybrid_ranker.weights)["overall"]
            else:
                j_req = replace(job_from_document(c_doc), semantic_base=sem_score)
                c_prof = profile_from_document(q)
                tot = hybrid_score(j_req, c_prof, weights=hybrid_ranker.weights)["overall"]
            scored.append((cid, tot))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    return run_hybrid


def main():
    args = parse_args()
    print("=" * 75)
    print(" VietResume-Bench: Comprehensive Evaluation Benchmark")
    print(f" Embedder:  {args.embedder} ({args.device})")
    print(f" Data Dir:  {DATA_DIR}")
    print("=" * 75)

    # 1. Load Corpus & Qrels
    t0 = time.time()
    cv_docs = load_documents(DATA_DIR / "cvs.jsonl")
    jd_docs = load_documents(DATA_DIR / "jds.jsonl")
    with open(DATA_DIR / "benchmark_qrels.json", "r", encoding="utf-8") as f:
        qrels_data = json.load(f)

    print(f"Loaded {len(cv_docs)} CVs and {len(jd_docs)} JDs in {time.time()-t0:.2f}s")
    print(f"Rubric: {qrels_data['metadata']['rubric']}")

    # 2. Setup Embedder & Embed Corpora
    print(f"\n[1/5] Initializing Embedder: {args.embedder} ...")
    embedder = create_embedder(args.embedder, device=args.device)
    print(f"Embedder initialized. Embedding dimension: {embedder.dim}")

    print("Embedding CV corpus (sections + full text) ...", flush=True)
    t_emb = time.time()
    cv_embs = embed_corpus(cv_docs.values(), embedder)
    print(f"CVs embedded in {time.time()-t_emb:.2f}s")

    print("Embedding JD corpus (sections + full text) ...", flush=True)
    t_emb = time.time()
    jd_embs = embed_corpus(jd_docs.values(), embedder)
    print(f"JDs embedded in {time.time()-t_emb:.2f}s")

    # 3. Setup BM25 Rankers
    print("\n[2/5] Building BM25 Index (pyvi & whitespace) ...", flush=True)
    bm25_cv_pyvi = BM25Ranker(list(cv_docs.values()), tokenize_mode="pyvi")
    bm25_cv_ws = BM25Ranker(list(cv_docs.values()), tokenize_mode="whitespace")
    bm25_jd_pyvi = BM25Ranker(list(jd_docs.values()), tokenize_mode="pyvi")

    hybrid_ranker = HybridRanker()

    directions_to_run = ["jd2cv", "cv2jd"] if args.direction == "both" else [args.direction]
    all_results = {}

    for direction in directions_to_run:
        print(f"\n{'='*75}")
        print(f" Evaluating Direction: {direction.upper()} ({'JD -> Candidates' if direction == 'jd2cv' else 'CV -> Jobs'})")
        print(f"{'='*75}")

        if direction == "jd2cv":
            queries = list(jd_docs.values())
            candidate_docs = cv_docs
            q_embs = jd_embs
            c_embs = cv_embs
            qrels = qrels_data["jd2cv"]
            bm25_pyvi = bm25_cv_pyvi
            bm25_ws = bm25_cv_ws
            target_type = "cv"
        else:
            queries = [cv_docs[cid] for cid in qrels_data["cv2jd"].keys() if cid in cv_docs]
            candidate_docs = jd_docs
            q_embs = cv_embs
            c_embs = jd_embs
            qrels = qrels_data["cv2jd"]
            bm25_pyvi = bm25_jd_pyvi
            bm25_ws = bm25_jd_pyvi
            target_type = "jd"

        if args.max_queries > 0:
            queries = queries[: args.max_queries]

        dir_results = {}

        # 1. BM25 (pyvi)
        print("Running Baseline 1: BM25 (pyvi) ...")
        m, ranked_bm25, lat = evaluate_ranking_run(
            queries, lambda q: bm25_pyvi.rank(q), qrels
        )
        dir_results["bm25_pyvi"] = m

        # 2. BM25 (whitespace)
        print("Running Baseline 2: BM25 (whitespace naive) ...")
        m, ranked_bm25_ws, lat = evaluate_ranking_run(
            queries, lambda q: bm25_ws.rank(q), qrels
        )
        dir_results["bm25_whitespace"] = m

        # 3. Single-Vector Dense
        print("Running Baseline 3: Single-Vector (Dense full text) ...")
        m, ranked_single, lat = evaluate_ranking_run(
            queries,
            lambda q: rank_query(q_embs[q.doc_id], c_embs, method="single_vector"),
            qrels,
        )
        dir_results["single_vector"] = m

        # 4. Chunk Max-Sim
        print("Running Baseline 4: Chunk Max-Similarity ...")
        chunk_results = rank_all_chunk(queries, list(candidate_docs.values()), embedder)
        m, ranked_chunk, lat = evaluate_ranking_run(
            queries,
            lambda q: chunk_results.get(q.doc_id, []),
            qrels,
        )
        dir_results["chunk_max_sim"] = m

        # 5. Section-Aware Multi-Vector (Fixed)
        print("Running Method: Section-Aware (Fixed Pairing) ...")
        m, ranked_fixed, lat = evaluate_ranking_run(
            queries,
            lambda q: rank_query(q_embs[q.doc_id], c_embs, method="section_aware", pairing="fixed", aggregation="mean"),
            qrels,
        )
        dir_results["section_aware_fixed"] = m

        # 6. Section-Aware Multi-Vector (Cross - Ours)
        print("Running Method: Section-Aware (Cross Pairing - Ours) ...")
        m, ranked_cross, lat = evaluate_ranking_run(
            queries,
            lambda q: rank_query(q_embs[q.doc_id], c_embs, method="section_aware", pairing="cross", aggregation="weighted"),
            qrels,
        )
        dir_results["section_aware_cross"] = m

        # 7. Calibrated Hybrid Fusion
        print("Running Full Pipeline: Calibrated Hybrid Fusion ...")
        hybrid_runner = make_hybrid_runner(q_embs, c_embs, candidate_docs, bm25_pyvi, hybrid_ranker, target_type)
        m, ranked_hybrid, lat = evaluate_ranking_run(queries, hybrid_runner, qrels)
        dir_results["hybrid_calibrated"] = m

        # Print Markdown Table
        print(f"\n### Results Summary Table: {direction.upper()}")
        print("| Method                              | nDCG@5 | nDCG@10 | MRR    | MAP    | P@5    | Latency |")
        print("|:------------------------------------|:------:|:-------:|:------:|:------:|:------:|:-------:|")
        for k, res in dir_results.items():
            print(f"{format_table_row(k, res)} {res.get('latency_ms', 0):.1f} ms |")

        all_results[direction] = dir_results

    # 5. Sliced Language Breakdown (RQ1, RQ2 on JD->CV)
    print(f"\n{'='*75}")
    print(" [3/5] RQ1 & RQ2: Sliced Language Breakdown (VI-VI, VI-EN Cross-lingual, Code-Mixed)")
    print(f"{'='*75}")

    jd2cv_qrels = qrels_data["jd2cv"]
    cv_lang_map = {c.doc_id: c.lang for c in cv_docs.values()}

    # Language slices evaluated on candidate languages:
    # 1. Monolingual VI-VI: candidates with lang == 'vi'
    # 2. Cross-lingual VI-EN: candidates with lang == 'en'
    # 3. Code-Mixed: candidates with lang == 'mixed'
    slices = [
        ("VI <-> VI (Monolingual)", "vi"),
        ("VI <-> EN (Cross-lingual)", "en"),
        ("Code-Mixed", "mixed"),
    ]

    hybrid_jd2cv_runner = make_hybrid_runner(jd_embs, cv_embs, cv_docs, bm25_cv_pyvi, hybrid_ranker, "cv")

    eval_queries = list(jd_docs.values())
    if args.max_queries > 0:
        eval_queries = eval_queries[: args.max_queries]

    sliced_results = {}
    for sl_label, target_lang in slices:
        sl_metrics = {}
        for m_name, rank_fn in [
            ("bm25_whitespace", lambda q: bm25_cv_ws.rank(q)),
            ("bm25_pyvi", lambda q: bm25_cv_pyvi.rank(q)),
            ("single_vector", lambda q: rank_query(jd_embs[q.doc_id], cv_embs, method="single_vector")),
            ("section_aware", lambda q: rank_query(jd_embs[q.doc_id], cv_embs, method="section_aware", pairing="cross")),
            ("hybrid", hybrid_jd2cv_runner),
        ]:
            ndcg5_list, ndcg10_list, map_list = [], [], []
            for q in eval_queries:
                full_ranked = rank_fn(q)
                # Filter ranked list to the target candidate language
                ranked_filtered = [cid for cid, _ in full_ranked if cv_lang_map.get(cid) == target_lang]
                # Filter qrels to the target candidate language
                q_qrels = {cid: sc for cid, sc in jd2cv_qrels.get(q.doc_id, {}).items() if cv_lang_map.get(cid) == target_lang}
                if q_qrels and any(sc > 0 for sc in q_qrels.values()):
                    m = evaluate_query(ranked_filtered, q_qrels, threshold=1)
                    ndcg5_list.append(m["ndcg@5"])
                    ndcg10_list.append(m["ndcg@10"])
                    map_list.append(m["map"])

            sl_metrics[m_name] = {
                "ndcg@5": float(np.mean(ndcg5_list)) if ndcg5_list else 0.0,
                "ndcg@10": float(np.mean(ndcg10_list)) if ndcg10_list else 0.0,
                "map": float(np.mean(map_list)) if map_list else 0.0,
                "evaluated_queries": len(ndcg10_list),
            }
        sliced_results[sl_label] = sl_metrics

    print("\n| Language Slice               | Method           | nDCG@5 | nDCG@10 | MAP    |")
    print("|:-----------------------------|:-----------------|:------:|:-------:|:------:|")
    for sl_label, sl_dict in sliced_results.items():
        for m_name, m in sl_dict.items():
            print(f"| {sl_label:<28} | {m_name:<16} | {m.get('ndcg@5',0):.4f} | {m.get('ndcg@10',0):.4f} | {m.get('map',0):.4f} |")

    # 6. Robustness to Diacritic Removal (RQ4)
    print(f"\n{'='*75}")
    print(" [4/5] RQ4: Robustness to Diacritic Removal (tieng Viet khong dau)")
    print(f"{'='*75}")

    unaccented_queries = []
    test_jds = list(jd_docs.values())
    if args.max_queries > 0:
        test_jds = test_jds[: args.max_queries]

    for jd in test_jds:
        unaccented_text = strip_accents(jd.raw_text)
        unaccented_queries.append(
            Document(
                doc_id=jd.doc_id,
                doc_type=jd.doc_type,
                lang="vi",
                raw_text=unaccented_text,
                sections={name: sec for name, sec in jd.sections.items()},
                metadata=jd.metadata,
            )
        )

    unacc_embs = embed_corpus(unaccented_queries, embedder)

    robust_results = {}
    for m_name, rank_fn in [
        ("bm25_pyvi", lambda q: bm25_cv_pyvi.rank(q)),
        ("single_vector", lambda q: rank_query(unacc_embs[q.doc_id], cv_embs, method="single_vector")),
        ("section_aware_cross", lambda q: rank_query(unacc_embs[q.doc_id], cv_embs, method="section_aware", pairing="cross")),
    ]:
        agg, _, _ = evaluate_ranking_run(unaccented_queries, rank_fn, jd2cv_qrels)
        orig_ndcg10 = all_results["jd2cv"][m_name]["ndcg@10"]
        unacc_ndcg10 = agg["ndcg@10"]
        drop_pct = ((unacc_ndcg10 - orig_ndcg10) / max(orig_ndcg10, 1e-6)) * 100.0
        robust_results[m_name] = {
            "original_ndcg10": orig_ndcg10,
            "unaccented_ndcg10": unacc_ndcg10,
            "drop_percent": drop_pct,
        }
        print(f"| {m_name:<18} | Có dấu: {orig_ndcg10:.4f} | Không dấu: {unacc_ndcg10:.4f} | Biến thiên: {drop_pct:+.2f}% |")

    # 7. Export Outputs
    print(f"\n{'='*75}")
    print(" [5/5] Exporting Results and LaTeX Tables ...")
    print(f"{'='*75}")

    final_payload = {
        "metadata": {
            "benchmark": "VietResume-Bench",
            "embedder": args.embedder,
            "device": args.device,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_cvs": len(cv_docs),
            "corpus_jds": len(jd_docs),
        },
        "main_results": all_results,
        "sliced_bilingual": sliced_results,
        "robustness_diacritics": robust_results,
    }

    json_path = RESULTS_DIR / "vietresume_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {json_path}")

    # Generate LaTeX Tables
    # 1. Main Table
    tex_main = f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Main ranking performance on \\texttt{{VietResume-Bench}} (1,585 CVs, 60 JDs, 3-tier graded relevance).}}
\\label{{tab:vietresume_main}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\\hline
\\textbf{{Method}} & \\textbf{{nDCG@5}} & \\textbf{{nDCG@10}} & \\textbf{{MRR}} & \\textbf{{MAP}} & \\textbf{{Latency (ms)}} \\\\
\\hline
BM25 (whitespace) & {all_results['jd2cv']['bm25_whitespace']['ndcg@5']:.4f} & {all_results['jd2cv']['bm25_whitespace']['ndcg@10']:.4f} & {all_results['jd2cv']['bm25_whitespace']['mrr']:.4f} & {all_results['jd2cv']['bm25_whitespace']['map']:.4f} & {all_results['jd2cv']['bm25_whitespace']['latency_ms']:.1f} \\\\
BM25 (pyvi) & {all_results['jd2cv']['bm25_pyvi']['ndcg@5']:.4f} & {all_results['jd2cv']['bm25_pyvi']['ndcg@10']:.4f} & {all_results['jd2cv']['bm25_pyvi']['mrr']:.4f} & {all_results['jd2cv']['bm25_pyvi']['map']:.4f} & {all_results['jd2cv']['bm25_pyvi']['latency_ms']:.1f} \\\\
Single-Vector & {all_results['jd2cv']['single_vector']['ndcg@5']:.4f} & {all_results['jd2cv']['single_vector']['ndcg@10']:.4f} & {all_results['jd2cv']['single_vector']['mrr']:.4f} & {all_results['jd2cv']['single_vector']['map']:.4f} & {all_results['jd2cv']['single_vector']['latency_ms']:.1f} \\\\
Chunk Max-Sim & {all_results['jd2cv']['chunk_max_sim']['ndcg@5']:.4f} & {all_results['jd2cv']['chunk_max_sim']['ndcg@10']:.4f} & {all_results['jd2cv']['chunk_max_sim']['mrr']:.4f} & {all_results['jd2cv']['chunk_max_sim']['map']:.4f} & {all_results['jd2cv']['chunk_max_sim']['latency_ms']:.1f} \\\\
Section-Aware (Fixed) & {all_results['jd2cv']['section_aware_fixed']['ndcg@5']:.4f} & {all_results['jd2cv']['section_aware_fixed']['ndcg@10']:.4f} & {all_results['jd2cv']['section_aware_fixed']['mrr']:.4f} & {all_results['jd2cv']['section_aware_fixed']['map']:.4f} & {all_results['jd2cv']['section_aware_fixed']['latency_ms']:.1f} \\\\
\\textbf{{Section-Aware (Cross)}} & \\textbf{{{all_results['jd2cv']['section_aware_cross']['ndcg@5']:.4f}}} & \\textbf{{{all_results['jd2cv']['section_aware_cross']['ndcg@10']:.4f}}} & \\textbf{{{all_results['jd2cv']['section_aware_cross']['mrr']:.4f}}} & \\textbf{{{all_results['jd2cv']['section_aware_cross']['map']:.4f}}} & {all_results['jd2cv']['section_aware_cross']['latency_ms']:.1f} \\\\
\\textbf{{Calibrated Hybrid (Ours)}} & \\textbf{{{all_results['jd2cv']['hybrid_calibrated']['ndcg@5']:.4f}}} & \\textbf{{{all_results['jd2cv']['hybrid_calibrated']['ndcg@10']:.4f}}} & \\textbf{{{all_results['jd2cv']['hybrid_calibrated']['mrr']:.4f}}} & \\textbf{{{all_results['jd2cv']['hybrid_calibrated']['map']:.4f}}} & {all_results['jd2cv']['hybrid_calibrated']['latency_ms']:.1f} \\\\
\\hline
\\end{{tabular}}%
}}
\\end{{table}}
"""
    with open(TABLES_DIR / "tab_vietresume_main.tex", "w", encoding="utf-8") as f:
        f.write(tex_main)
    print("Saved tab_vietresume_main.tex")

    # 2. Bilingual Sliced Table
    tex_bi = f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Sliced performance across language groups on \\texttt{{VietResume-Bench}} (RQ1 \\& RQ2).}}
\\label{{tab:vietresume_bilingual}}
\\begin{{tabular}}{{@{{}}llccc@{{}}}}
\\hline
\\textbf{{Language Slice}} & \\textbf{{Method}} & \\textbf{{nDCG@5}} & \\textbf{{nDCG@10}} & \\textbf{{MAP}} \\\\
\\hline
VI $\\leftrightarrow$ VI (Monolingual) & BM25 (pyvi) & {sliced_results['VI <-> VI (Monolingual)']['bm25_pyvi']['ndcg@5']:.4f} & {sliced_results['VI <-> VI (Monolingual)']['bm25_pyvi']['ndcg@10']:.4f} & {sliced_results['VI <-> VI (Monolingual)']['bm25_pyvi']['map']:.4f} \\\\
 & Single-Vector & {sliced_results['VI <-> VI (Monolingual)']['single_vector']['ndcg@5']:.4f} & {sliced_results['VI <-> VI (Monolingual)']['single_vector']['ndcg@10']:.4f} & {sliced_results['VI <-> VI (Monolingual)']['single_vector']['map']:.4f} \\\\
 & Section-Aware & {sliced_results['VI <-> VI (Monolingual)']['section_aware']['ndcg@5']:.4f} & {sliced_results['VI <-> VI (Monolingual)']['section_aware']['ndcg@10']:.4f} & {sliced_results['VI <-> VI (Monolingual)']['section_aware']['map']:.4f} \\\\
\\hline
VI $\\leftrightarrow$ EN (Cross-lingual) & BM25 (whitespace) & {sliced_results['VI <-> EN (Cross-lingual)']['bm25_whitespace']['ndcg@5']:.4f} & {sliced_results['VI <-> EN (Cross-lingual)']['bm25_whitespace']['ndcg@10']:.4f} & {sliced_results['VI <-> EN (Cross-lingual)']['bm25_whitespace']['map']:.4f} \\\\
 & BM25 (pyvi) & {sliced_results['VI <-> EN (Cross-lingual)']['bm25_pyvi']['ndcg@5']:.4f} & {sliced_results['VI <-> EN (Cross-lingual)']['bm25_pyvi']['ndcg@10']:.4f} & {sliced_results['VI <-> EN (Cross-lingual)']['bm25_pyvi']['map']:.4f} \\\\
 & Single-Vector & {sliced_results['VI <-> EN (Cross-lingual)']['single_vector']['ndcg@5']:.4f} & {sliced_results['VI <-> EN (Cross-lingual)']['single_vector']['ndcg@10']:.4f} & {sliced_results['VI <-> EN (Cross-lingual)']['single_vector']['map']:.4f} \\\\
 & \\textbf{{Section-Aware (Ours)}} & \\textbf{{{sliced_results['VI <-> EN (Cross-lingual)']['section_aware']['ndcg@5']:.4f}}} & \\textbf{{{sliced_results['VI <-> EN (Cross-lingual)']['section_aware']['ndcg@10']:.4f}}} & \\textbf{{{sliced_results['VI <-> EN (Cross-lingual)']['section_aware']['map']:.4f}}} \\\\
\\hline
Code-Mixed & Single-Vector & {sliced_results['Code-Mixed']['single_vector']['ndcg@5']:.4f} & {sliced_results['Code-Mixed']['single_vector']['ndcg@10']:.4f} & {sliced_results['Code-Mixed']['single_vector']['map']:.4f} \\\\
 & \\textbf{{Section-Aware (Ours)}} & \\textbf{{{sliced_results['Code-Mixed']['section_aware']['ndcg@5']:.4f}}} & \\textbf{{{sliced_results['Code-Mixed']['section_aware']['ndcg@10']:.4f}}} & \\textbf{{{sliced_results['Code-Mixed']['section_aware']['map']:.4f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(TABLES_DIR / "tab_vietresume_bilingual.tex", "w", encoding="utf-8") as f:
        f.write(tex_bi)
    print("Saved tab_vietresume_bilingual.tex")

    # 3. Robustness Table
    tex_rob = f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Robustness to diacritic removal on \\texttt{{VietResume-Bench}} (RQ4).}}
\\label{{tab:vietresume_robustness}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\hline
\\textbf{{Method}} & \\textbf{{nDCG@10 (Accented)}} & \\textbf{{nDCG@10 (Unaccented)}} & \\textbf{{Change $\\Delta$ (\\%)}} \\\\
\\hline
BM25 (pyvi) & {robust_results['bm25_pyvi']['original_ndcg10']:.4f} & {robust_results['bm25_pyvi']['unaccented_ndcg10']:.4f} & {robust_results['bm25_pyvi']['drop_percent']:+.2f}\\% \\\\
Single-Vector & {robust_results['single_vector']['original_ndcg10']:.4f} & {robust_results['single_vector']['unaccented_ndcg10']:.4f} & {robust_results['single_vector']['drop_percent']:+.2f}\\% \\\\
\\textbf{{Section-Aware (Ours)}} & \\textbf{{{robust_results['section_aware_cross']['original_ndcg10']:.4f}}} & \\textbf{{{robust_results['section_aware_cross']['unaccented_ndcg10']:.4f}}} & \\textbf{{{robust_results['section_aware_cross']['drop_percent']:+.2f}\\%}} \\\\
\\hline
\\end{{tabular}}%
}}
\\end{{table}}
"""
    with open(TABLES_DIR / "tab_vietresume_robustness.tex", "w", encoding="utf-8") as f:
        f.write(tex_rob)
    print("Saved tab_vietresume_robustness.tex")

    print("\nDone! All benchmarks on VietResume-Bench executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
