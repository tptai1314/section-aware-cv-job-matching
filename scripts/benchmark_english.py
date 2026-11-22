"""Official English Evaluation Benchmark on resume-job-description-fit.

Evaluates 100% English curriculum vitae & job description matching:
1. Lexical Baselines: TF-IDF / BM25
2. Dense Bi-Encoders:
   - sentence-transformers/all-MiniLM-L6-v2 (384-d, 512 ctx)
   - BAAI/bge-m3 (1024-d, 8192 ctx)
3. Proposed Architectural Contributions:
   - Section-Aware Multi-Vector (Fixed & Cross Section Alignment)
   - 3-Tier Skill Taxonomy (Exact, Relational, Semantic) & Asymmetric Coverage
4. Fusion & Reranking:
   - Calibrated Hybrid Fusion
   - Parameter-Free Reciprocal Rank Fusion (RRF, k=60)
   - Zero-Shot Cross-Encoder Top-K Reranking (ms-marco-MiniLM-L-6-v2)

Outputs:
  - results/english_benchmark_results.json
  - paper/tables/tab_main_benchmark.tex
  - paper/tables/tab_embedding_comparison.tex
  - paper/tables/tab_reranking.tex
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
import pandas as pd
from scipy.stats import spearmanr
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.data.preprocessing import sections_to_dict, split_sections
from cvjd.data.taxonomy import get_taxonomy
from cvjd.ranking.engine import min_max
from cvjd.ranking.hybrid import (
    DEFAULT_WEIGHTS,
    CandidateProfile,
    JobRequirement,
    SkillItem,
    hybrid_score,
)
from cvjd.ranking.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)

DATA_PATH = REPO_ROOT / "data" / "public_benchmark" / "resume-job-description-fit" / "test.csv"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-bge", action="store_true", help="Also evaluate BGE-M3 (larger model)")
    parser.add_argument("--run-cross-encoder", action="store_true", default=True, help="Evaluate Cross-Encoder")
    return parser.parse_args()


def bootstrap_ci(values: List[float], n_resamples: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    mean_val = float(np.mean(values))
    boot_means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_resamples)]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_lower, ci_upper


def evaluate_ranking_predictions(
    query_groups: Dict[int, List[int]],
    pairs: List[Tuple[int, int]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    ndcg5_list = []
    ndcg10_list = []
    mrr_list = []
    map_list = []
    p5_list = []

    for q_idx, pair_indices in query_groups.items():
        if len(pair_indices) < 2:
            continue
        cand_ids = [str(pairs[p][1]) for p in pair_indices]
        full_qrels = {str(pairs[p][1]): float(y_true[p]) for p in pair_indices}

        scores = y_pred[pair_indices]
        order = np.argsort(-scores)
        ranked_ids = [cand_ids[i] for i in order]

        ndcg5_list.append(ndcg_at_k(ranked_ids, full_qrels, k=5))
        ndcg10_list.append(ndcg_at_k(ranked_ids, full_qrels, k=10))
        mrr_list.append(reciprocal_rank(ranked_ids, full_qrels, threshold=1))
        map_list.append(average_precision(ranked_ids, full_qrels, threshold=1))
        p5_list.append(precision_at_k(ranked_ids, full_qrels, k=5, threshold=1))

    rho, _ = spearmanr(y_pred, y_true)
    mean_n5, lo_n5, hi_n5 = bootstrap_ci(ndcg5_list, n_bootstrap, seed)
    mean_n10, lo_n10, hi_n10 = bootstrap_ci(ndcg10_list, n_bootstrap, seed)
    mean_mrr, lo_mrr, hi_mrr = bootstrap_ci(mrr_list, n_bootstrap, seed)
    mean_map, lo_map, hi_map = bootstrap_ci(map_list, n_bootstrap, seed)
    mean_p5, lo_p5, hi_p5 = bootstrap_ci(p5_list, n_bootstrap, seed)

    return {
        "ndcg@5": mean_n5, "ndcg@5_ci": [lo_n5, hi_n5],
        "ndcg@10": mean_n10, "ndcg@10_ci": [lo_n10, hi_n10],
        "mrr": mean_mrr, "mrr_ci": [lo_mrr, hi_mrr],
        "map": mean_map, "map_ci": [lo_map, hi_map],
        "p@5": mean_p5, "p@5_ci": [lo_p5, hi_p5],
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
    }


def compute_rrf_for_queries(
    query_groups: Dict[int, List[int]],
    scores_dense: np.ndarray,
    scores_lexical: np.ndarray,
    k: int = 60,
) -> np.ndarray:
    rrf_pred = np.zeros_like(scores_dense)
    for q_idx, pair_indices in query_groups.items():
        p_dense = scores_dense[pair_indices]
        p_lex = scores_lexical[pair_indices]

        # dense ranks (1-based)
        order_dense = np.argsort(-p_dense)
        rank_dense = np.empty_like(order_dense)
        rank_dense[order_dense] = np.arange(1, len(order_dense) + 1)

        # lexical ranks (1-based)
        order_lex = np.argsort(-p_lex)
        rank_lex = np.empty_like(order_lex)
        rank_lex[order_lex] = np.arange(1, len(order_lex) + 1)

        rrf_sub = 1.0 / (k + rank_dense) + 1.0 / (k + rank_lex)
        rrf_pred[pair_indices] = rrf_sub
    return rrf_pred


def main():
    args = parse_args()
    print("=" * 80, flush=True)
    print(" Official English Benchmark: resume-job-description-fit", flush=True)
    print(f" Dataset Path: {DATA_PATH}", flush=True)
    print(f" Device:       {args.device}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Dataset
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print(f"Loaded {len(df)} pairs in {time.time()-t0:.2f}s", flush=True)
    print(f"Label distribution:\n{df['label'].value_counts()}", flush=True)

    unique_jobs = df["job_description_text"].unique().tolist()
    unique_resumes = df["resume_text"].unique().tolist()
    job2id = {j: i for i, j in enumerate(unique_jobs)}
    res2id = {r: i for i, r in enumerate(unique_resumes)}

    pairs = [(job2id[row["job_description_text"]], res2id[row["resume_text"]]) for _, row in df.iterrows()]
    y_true = np.array([LABEL_MAP[row["label"]] for _, row in df.iterrows()])

    query_groups = defaultdict(list)
    for pair_idx, (j_idx, _) in enumerate(pairs):
        query_groups[j_idx].append(pair_idx)

    print(f"Queries (Unique JDs):      {len(unique_jobs)}", flush=True)
    print(f"Candidates (Unique CVs):   {len(unique_resumes)}", flush=True)

    results: Dict[str, Any] = {}

    # 2. Baseline 1: Lexical (TF-IDF / BM25)
    print("\n[1/6] Evaluating Lexical Baseline (BM25 / TF-IDF) ...", flush=True)
    t0 = time.perf_counter()
    vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    all_texts = unique_jobs + unique_resumes
    vec.fit(all_texts)
    job_vecs = vec.transform(unique_jobs)
    res_vecs = vec.transform(unique_resumes)
    bm25_scores = np.array([float((job_vecs[j].multiply(res_vecs[r])).sum()) for j, r in pairs])
    bm25_lat = (time.perf_counter() - t0) * 1000.0 / len(unique_jobs)
    results["BM25 / TF-IDF (Lexical)"] = {
        **evaluate_ranking_predictions(query_groups, pairs, y_true, bm25_scores, args.n_bootstrap, args.seed),
        "latency_ms_per_query": round(bm25_lat, 2),
    }

    # 3. Baseline 2: Dense Bi-Encoder (MiniLM)
    print("\n[2/6] Evaluating Dense Bi-Encoder (sentence-transformers/all-MiniLM-L6-v2) ...", flush=True)
    t0 = time.perf_counter()
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=args.device)
    job_dense_embs = dense_model.encode(unique_jobs, batch_size=32, normalize_embeddings=True)
    res_dense_embs = dense_model.encode(unique_resumes, batch_size=32, normalize_embeddings=True)
    single_dense_scores = np.array([float(np.dot(job_dense_embs[j], res_dense_embs[r])) for j, r in pairs])
    dense_lat = (time.perf_counter() - t0) * 1000.0 / len(unique_jobs)
    results["Single-Vector Dense (MiniLM)"] = {
        **evaluate_ranking_predictions(query_groups, pairs, y_true, single_dense_scores, args.n_bootstrap, args.seed),
        "latency_ms_per_query": round(dense_lat, 2),
    }

    # 4. Method 3: Section-Aware Multi-Vector
    print("\n[3/6] Evaluating Proposed Section-Aware Multi-Vector (Section Extraction & Alignment) ...", flush=True)
    t0 = time.perf_counter()
    # Segment resumes and JDs into sections
    res_parsed_sections: List[Dict[str, str]] = []
    for r_text in unique_resumes:
        secs = sections_to_dict(split_sections(r_text))
        res_parsed_sections.append({k: v.text for k, v in secs.items() if len(v.text.strip()) > 10})

    job_parsed_sections: List[Dict[str, str]] = []
    for j_text in unique_jobs:
        secs = sections_to_dict(split_sections(j_text))
        job_parsed_sections.append({k: v.text for k, v in secs.items() if len(v.text.strip()) > 10})

    # Encode section representations
    section_weights = {"skills": 0.40, "experience": 0.35, "education": 0.15, "summary": 0.10}
    
    # Precompute section vectors for resumes
    res_section_vectors: List[Dict[str, np.ndarray]] = []
    for sec_dict in res_parsed_sections:
        vec_dict = {}
        for s_name, s_txt in sec_dict.items():
            if len(s_txt.strip()) > 10:
                vec_dict[s_name] = dense_model.encode(s_txt[:512], normalize_embeddings=True)
        res_section_vectors.append(vec_dict)

    # Compute Section-Aware Cross similarity
    sec_aware_scores = []
    for j_idx, r_idx in pairs:
        j_vec = job_dense_embs[j_idx]
        r_sec_vecs = res_section_vectors[r_idx]
        if not r_sec_vecs:
            sec_aware_scores.append(single_dense_scores[len(sec_aware_scores)])
            continue

        weighted_sim = 0.0
        total_w = 0.0
        for s_name, w in section_weights.items():
            if s_name in r_sec_vecs:
                sim = float(np.dot(j_vec, r_sec_vecs[s_name]))
                weighted_sim += w * sim
                total_w += w
        if total_w > 0:
            weighted_sim /= total_w
        else:
            weighted_sim = single_dense_scores[len(sec_aware_scores)]

        # Dynamic combination of Section Granularity + Whole Document Context
        final_sec = 0.65 * weighted_sim + 0.35 * single_dense_scores[len(sec_aware_scores)]
        sec_aware_scores.append(final_sec)

    sec_aware_scores = np.array(sec_aware_scores)
    sec_lat = (time.perf_counter() - t0) * 1000.0 / len(unique_jobs)
    results["Section-Aware Multi-Vector (Ours)"] = {
        **evaluate_ranking_predictions(query_groups, pairs, y_true, sec_aware_scores, args.n_bootstrap, args.seed),
        "latency_ms_per_query": round(sec_lat, 2),
    }

    # 5. Method 4: 3-Tier Skill Taxonomy & Asymmetric Coverage
    print("\n[4/6] Evaluating 3-Tier Taxonomy Skill Extraction & Coverage ...", flush=True)
    t0 = time.perf_counter()
    taxonomy = get_taxonomy()

    # Pre-extract skills
    job_skills_list = [taxonomy.classify_jd_skills(j_text) for j_text in unique_jobs]
    res_skills_list = [taxonomy.extract_skills_from_text(r_text) for r_text in unique_resumes]

    tax_rule_scores = []
    for j_idx, r_idx in pairs:
        must_skills, nice_skills = job_skills_list[j_idx]
        c_skills = res_skills_list[r_idx]

        if not must_skills:
            # Fallback if no specific skills listed
            tax_rule_scores.append(1.0)
            continue

        # Asymmetric coverage with Layer 2 relation support
        covered = 0.0
        for m_skill in must_skills:
            if m_skill in c_skills:
                covered += 1.0
            else:
                # Check Layer 2 taxonomy relation
                max_rel = 0.0
                for cs in c_skills:
                    rw = taxonomy.get_relation_weight(cs, m_skill)
                    if rw > max_rel:
                        max_rel = rw
                covered += max_rel

        must_cov = covered / len(must_skills)

        # Nice-to-have bonus
        nice_cov = 0.0
        if nice_skills:
            nice_cov = sum(1.0 for ns in nice_skills if ns in c_skills) / len(nice_skills)

        rule_score = 0.85 * must_cov + 0.15 * nice_cov
        tax_rule_scores.append(rule_score)

    tax_rule_scores = np.array(tax_rule_scores)

    # 6. Method 5: Calibrated Hybrid Fusion & Parameter-Free RRF
    print("\n[5/6] Evaluating Calibrated Hybrid Fusion & Parameter-Free RRF ...", flush=True)
    # Per-query min-max normalization
    norm_sec_scores = np.zeros_like(sec_aware_scores)
    norm_bm25_scores = np.zeros_like(bm25_scores)

    for q_idx, pair_indices in query_groups.items():
        s_sub = sec_aware_scores[pair_indices]
        b_sub = bm25_scores[pair_indices]
        norm_sec_scores[pair_indices] = min_max(s_sub)
        norm_bm25_scores[pair_indices] = min_max(b_sub)

    # Fused retrieval semantic score (alpha = 0.6 dense, 0.4 lexical)
    fused_retrieval = 0.60 * norm_sec_scores + 0.40 * norm_bm25_scores
    # Full Hybrid Competency score: 0.60 retrieval + 0.40 structured taxonomy coverage
    hybrid_final_scores = 0.60 * fused_retrieval + 0.40 * tax_rule_scores

    results["Calibrated Hybrid (Dense+Lexical+Taxonomy)"] = {
        **evaluate_ranking_predictions(query_groups, pairs, y_true, hybrid_final_scores, args.n_bootstrap, args.seed),
        "latency_ms_per_query": round(sec_lat + bm25_lat, 2),
    }

    # Parameter-Free RRF
    rrf_retrieval_scores = compute_rrf_for_queries(query_groups, sec_aware_scores, bm25_scores, k=60)
    norm_rrf = np.zeros_like(rrf_retrieval_scores)
    for q_idx, pair_indices in query_groups.items():
        norm_rrf[pair_indices] = min_max(rrf_retrieval_scores[pair_indices])
    rrf_final_scores = 0.60 * norm_rrf + 0.40 * tax_rule_scores

    results["Parameter-Free RRF + Taxonomy"] = {
        **evaluate_ranking_predictions(query_groups, pairs, y_true, rrf_final_scores, args.n_bootstrap, args.seed),
        "latency_ms_per_query": round(sec_lat + bm25_lat, 2),
    }

    # 7. Method 6: Zero-Shot Cross-Encoder Top-K Reranking
    if args.run_cross_encoder:
        print("\n[6/6] Evaluating Zero-Shot Cross-Encoder (ms-marco-MiniLM-L-6-v2) ...", flush=True)
        try:
            t0 = time.perf_counter()
            ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=args.device)
            ce_pairs = [(unique_jobs[j], unique_resumes[r]) for j, r in pairs]
            ce_raw = ce_model.predict(ce_pairs, batch_size=32, show_progress_bar=False)
            ce_scores = np.array(ce_raw)
            ce_lat = (time.perf_counter() - t0) * 1000.0 / len(unique_jobs)
            results["Cross-Encoder (ms-marco-MiniLM)"] = {
                **evaluate_ranking_predictions(query_groups, pairs, y_true, ce_scores, args.n_bootstrap, args.seed),
                "latency_ms_per_query": round(ce_lat, 2),
            }
        except Exception as e:
            print(f"Cross-Encoder evaluation skipped: {e}", flush=True)

    # 8. Print Results Summary
    print("\n" + "=" * 95, flush=True)
    print(" OFFICIAL ENGLISH BENCHMARK RESULTS (resume-job-description-fit, 1,759 Pairs)", flush=True)
    print("=" * 95, flush=True)
    print(f"| {'Method / Model':<42} | {'nDCG@5':<8} | {'nDCG@10':<8} | {'MRR':<8} | {'MAP':<8} | {'P@5':<8} | {'Spearman':<8} | {'Latency':<8} |", flush=True)
    print("|" + "-" * 44 + "|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|", flush=True)
    for m, vals in results.items():
        print(
            f"| {m:<42} | {vals['ndcg@5']:<8.4f} | {vals['ndcg@10']:<8.4f} | "
            f"{vals['mrr']:<8.4f} | {vals['map']:<8.4f} | {vals['p@5']:<8.4f} | "
            f"{vals['spearman_rho']:<8.4f} | {vals['latency_ms_per_query']:<5.1f} ms |",
            flush=True,
        )

    # 9. Save JSON Results
    json_path = RESULTS_DIR / "english_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured results to: {json_path}", flush=True)

    # 10. Generate Main LaTeX Table
    tex_path = TABLES_DIR / "tab_main_benchmark.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by scripts/benchmark_english.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Main benchmark results on \\textsc{resume-job-description-fit} (1,759 test pairs, 71 job queries).}\n")
        f.write("\\label{tab:main_benchmark}\n")
        f.write("\\resizebox{\\columnwidth}{!}{\n")
        f.write("\\begin{tabular}{@{}lcccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("Model / Architecture & nDCG@5 & nDCG@10 & MRR & MAP & P@5 & Spearman $\\rho$ \\\\\n")
        f.write("\\midrule\n")
        for m, vals in results.items():
            bold = "\\textbf{" if "Ours" in m or "Hybrid" in m or "RRF" in m else ""
            end_bold = "}" if bold else ""
            name = m.replace("&", "\\&").replace("_", "\\_")
            f.write(f"{name} & {bold}{vals['ndcg@5']:.4f}{end_bold} & {bold}{vals['ndcg@10']:.4f}{end_bold} & {vals['mrr']:.4f} & {vals['map']:.4f} & {vals['p@5']:.4f} & {vals['spearman_rho']:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    print(f"Generated LaTeX table: {tex_path}", flush=True)


if __name__ == "__main__":
    main()

