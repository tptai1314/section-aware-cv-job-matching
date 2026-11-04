"""Evaluate baselines and section-aware matching on the public Hugging Face benchmark.

Dataset: cnamuangtoun/resume-job-description-fit (test.csv, 1,759 pairs, 71 jobs, 477 resumes)
Labels: No Fit (0), Potential Fit (1), Good Fit (2)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cvjd.ranking.metrics import ndcg_at_k, reciprocal_rank, average_precision, precision_at_k
from cvjd.data.preprocessing import split_sections, sections_to_dict

LABEL_MAP = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2,
}


def build_bm25_scores(jobs: List[str], resumes: List[str], pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Compute BM25 / TF-IDF lexical scores for (job, resume) pairs."""
    vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    all_texts = jobs + resumes
    vec.fit(all_texts)
    job_vecs = vec.transform(jobs)
    res_vecs = vec.transform(resumes)
    scores = []
    for j_idx, r_idx in pairs:
        s = float((job_vecs[j_idx].multiply(res_vecs[r_idx])).sum())
        scores.append(s)
    return np.array(scores)


def evaluate_ranking(
    query_groups: Dict[int, List[int]],
    pairs: List[Tuple[int, int]],
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate ranking metrics across all query groups."""
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

    return {
        "nDCG@5": float(np.mean(ndcg5_list)),
        "nDCG@10": float(np.mean(ndcg10_list)),
        "MRR": float(np.mean(mrr_list)),
        "MAP": float(np.mean(map_list)),
        "Precision@5": float(np.mean(p5_list)),
        "Spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
    }


def main():
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "public_benchmark", "resume-job-description-fit", "test.csv"
    ))
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    print("Loading test dataset from", data_path, flush=True)
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} pairs.", flush=True)

    unique_jobs = df["job_description_text"].unique().tolist()
    unique_resumes = df["resume_text"].unique().tolist()

    job2id = {j: i for i, j in enumerate(unique_jobs)}
    res2id = {r: i for i, r in enumerate(unique_resumes)}

    pairs = [(job2id[row["job_description_text"]], res2id[row["resume_text"]]) for _, row in df.iterrows()]
    y_true = np.array([LABEL_MAP[row["label"]] for _, row in df.iterrows()])

    # Group pairs by job (query)
    query_groups = defaultdict(list)
    for pair_idx, (j_idx, _) in enumerate(pairs):
        query_groups[j_idx].append(pair_idx)

    print(f"Number of queries (jobs): {len(unique_jobs)}", flush=True)
    print(f"Number of candidates (resumes): {len(unique_resumes)}", flush=True)

    results = {}

    # 1. Lexical Baseline: BM25 / TF-IDF
    print("\n--- Evaluating Baseline 1: Lexical (TF-IDF / BM25) ---", flush=True)
    t0 = time.time()
    bm25_scores = build_bm25_scores(unique_jobs, unique_resumes, pairs)
    bm25_time = time.time() - t0
    results["TF-IDF / BM25"] = {
        **evaluate_ranking(query_groups, pairs, y_true, bm25_scores),
        "latency_sec": round(bm25_time, 2),
    }

    # 2. Dense Bi-Encoder (MiniLM)
    print("\n--- Evaluating Baseline 2: Dense Bi-Encoder (all-MiniLM-L6-v2) ---", flush=True)
    t0 = time.time()
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    print("Encoding jobs and resumes...", flush=True)
    job_embeds = dense_model.encode(unique_jobs, batch_size=32, normalize_embeddings=True)
    res_embeds = dense_model.encode(unique_resumes, batch_size=32, normalize_embeddings=True)

    dense_scores = np.array([
        float(np.dot(job_embeds[j_idx], res_embeds[r_idx]))
        for j_idx, r_idx in pairs
    ])
    dense_time = time.time() - t0
    results["Dense Bi-Encoder (MiniLM)"] = {
        **evaluate_ranking(query_groups, pairs, y_true, dense_scores),
        "latency_sec": round(dense_time, 2),
    }

    # 3. Cross-Encoder (ms-marco-MiniLM-L-6-v2)
    print("\n--- Evaluating Baseline 3: Cross-Encoder (ms-marco-MiniLM-L-6-v2) ---", flush=True)
    try:
        t0 = time.time()
        ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        ce_pairs = [(unique_jobs[j], unique_resumes[r]) for j, r in pairs]
        ce_scores = ce_model.predict(ce_pairs, batch_size=32, show_progress_bar=False)
        ce_time = time.time() - t0
        results["Cross-Encoder (ms-marco)"] = {
            **evaluate_ranking(query_groups, pairs, y_true, np.array(ce_scores)),
            "latency_sec": round(ce_time, 2),
        }
    except Exception as e:
        print(f"Skipping Cross-Encoder due to error: {e}", flush=True)

    # 4. Section-Aware Multi-Vector
    print("\n--- Evaluating Method 4: Section-Aware Multi-Vector (Regex + Weighted) ---", flush=True)
    t0 = time.time()
    # Extract sections for resumes
    res_sections = []
    for r_text in unique_resumes:
        secs = sections_to_dict(split_sections(r_text))
        sec_texts = {k: v.text for k, v in secs.items() if len(v.text.strip()) > 10}
        res_sections.append(sec_texts)

    # Encode section texts with weights
    weights = {"skills": 0.40, "experience": 0.35, "education": 0.15, "summary": 0.10}
    section_scores = []
    for j_idx, r_idx in pairs:
        r_sec = res_sections[r_idx]
        j_vec = job_embeds[j_idx]
        weighted_sim = 0.0
        total_w = 0.0
        for s_name, w in weights.items():
            if s_name in r_sec and len(r_sec[s_name].strip()) > 10:
                s_vec = dense_model.encode(r_sec[s_name][:500], normalize_embeddings=True)
                sim = float(np.dot(j_vec, s_vec))
                weighted_sim += w * sim
                total_w += w
        if total_w > 0:
            weighted_sim /= total_w
        else:
            weighted_sim = dense_scores[len(section_scores)]
        section_scores.append(0.6 * weighted_sim + 0.4 * dense_scores[len(section_scores)])

    section_scores = np.array(section_scores)
    sec_time = time.time() - t0
    results["Section-Aware Multi-Vector"] = {
        **evaluate_ranking(query_groups, pairs, y_true, section_scores),
        "latency_sec": round(sec_time, 2),
    }

    # 5. Hybrid (Section-Aware + BM25)
    print("\n--- Evaluating Method 5: Hybrid (Section-Aware + BM25) ---", flush=True)
    norm_bm25 = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    norm_sec = (section_scores - section_scores.min()) / (section_scores.max() - section_scores.min() + 1e-8)
    hybrid_scores = 0.6 * norm_sec + 0.4 * norm_bm25
    results["Hybrid (Section-Aware + BM25)"] = {
        **evaluate_ranking(query_groups, pairs, y_true, hybrid_scores),
        "latency_sec": round(sec_time + bm25_time, 2),
    }

    # Save outputs
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "public_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*80, flush=True)
    print("PUBLIC BENCHMARK RESULTS (cnamuangtoun/resume-job-description-fit)", flush=True)
    print("="*80, flush=True)
    print("| Method / Model | nDCG@5 | nDCG@10 | MRR | MAP | Precision@5 | Spearman ρ | Latency (s) |", flush=True)
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|", flush=True)
    for m, vals in results.items():
        print(f"| **{m}** | {vals['nDCG@5']:.4f} | {vals['nDCG@10']:.4f} | {vals['MRR']:.4f} | {vals['MAP']:.4f} | {vals['Precision@5']:.4f} | {vals['Spearman_rho']:.4f} | {vals['latency_sec']} |", flush=True)

    # Save LaTeX table
    tex_path = os.path.join(results_dir, "tab_public_benchmark.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by scripts/eval_public_benchmark.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Cross-dataset generalizability on public benchmark \\texttt{cnamuangtoun/resume-job-description-fit} (1,759 test pairs, 71 job queries).}\n")
        f.write("\\label{tab:public_benchmark}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("\\textbf{Model / Pipeline} & \\textbf{nDCG@5} & \\textbf{nDCG@10} & \\textbf{MRR} & \\textbf{MAP} & \\textbf{P@5} & \\textbf{Spearman $\\rho$} \\\\ \\hline\n")
        for m, vals in results.items():
            f.write(f"{m} & {vals['nDCG@5']:.4f} & {vals['nDCG@10']:.4f} & {vals['MRR']:.4f} & {vals['MAP']:.4f} & {vals['Precision@5']:.4f} & {vals['Spearman_rho']:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"\nSaved results to {json_path} and {tex_path}", flush=True)


if __name__ == "__main__":
    main()
