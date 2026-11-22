"""Parameter Sensitivity Analysis conducted strictly on the TRAINING set (train.csv).

Evaluates:
1. Section Weight configurations:
   - Default Heuristic: Skills=0.40, Experience=0.35, Education=0.15, Summary=0.10
   - Equal Weight: Skills=0.25, Experience=0.25, Education=0.25, Summary=0.25
   - Skills-Heavy: Skills=0.60, Experience=0.20, Education=0.10, Summary=0.10
   - Experience-Heavy: Skills=0.20, Experience=0.60, Education=0.10, Summary=0.10
   - Core Only: Skills=0.50, Experience=0.50, Education=0.00, Summary=0.00
2. Alpha sweep (Dense vs Lexical): alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
3. Beta sweep (Retrieval vs Taxonomy): beta in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

Saves results to: results/train_sensitivity_analysis.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "public_benchmark" / "resume-job-description-fit"
RESULTS_DIR = REPO_ROOT / "results"

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
from cvjd.data.preprocessing import sections_to_dict, split_sections
from cvjd.data.taxonomy import get_taxonomy


def compute_ndcg_at_10(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)[:10]
    sorted_labels = labels[order]
    gains = (2 ** sorted_labels - 1) / np.log2(np.arange(2, len(sorted_labels) + 2))
    dcg = np.sum(gains)

    ideal_order = np.argsort(-labels)[:10]
    ideal_labels = labels[ideal_order]
    ideal_gains = (2 ** ideal_labels - 1) / np.log2(np.arange(2, len(ideal_labels) + 2))
    idcg = np.sum(ideal_gains)

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_map(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    pos_mask = (sorted_labels >= 1)
    n_pos = np.sum(pos_mask)
    if n_pos == 0:
        return 0.0
    hits = 0
    prec_sum = 0.0
    for rank, is_rel in enumerate(pos_mask, start=1):
        if is_rel:
            hits += 1
            prec_sum += hits / rank
    return float(prec_sum / n_pos)


def min_max(vec: np.ndarray) -> np.ndarray:
    mi, ma = np.min(vec), np.max(vec)
    if ma - mi < 1e-9:
        return np.zeros_like(vec)
    return (vec - mi) / (ma - mi)


def main() -> None:
    print("=" * 70)
    print("Starting Parameter Sensitivity Analysis strictly on TRAIN set...")
    print("=" * 70)

    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding_errors="replace")
    label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
    train_df["target"] = train_df["label"].map(label_map)

    # Filter to queries with >= 10 candidates to ensure meaningful ranking
    query_counts = train_df["job_description_text"].value_counts()
    valid_queries = query_counts[query_counts >= 10].index.tolist()
    print(f"Total train queries with >= 10 candidates: {len(valid_queries)} / {len(query_counts)}")

    # Sample 50 queries for efficient sweep
    np.random.seed(42)
    selected_queries = list(np.random.choice(valid_queries, size=min(50, len(valid_queries)), replace=False))
    train_sub = train_df[train_df["job_description_text"].isin(selected_queries)].copy()

    unique_jobs = train_sub["job_description_text"].unique().tolist()
    unique_cvs = train_sub["resume_text"].unique().tolist()
    print(f"Profiling subset: {len(unique_jobs)} JDs, {len(unique_cvs)} CVs, {len(train_sub)} pairs.")

    # Encode with MiniLM
    print("Encoding texts with MiniLM...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embedder.max_seq_length = 256

    job_embs = {j: embedder.encode(j, show_progress_bar=False) for j in unique_jobs}
    cv_embs = {c: embedder.encode(c, show_progress_bar=False) for c in unique_cvs}

    print("Parsing sections and encoding section embeddings...")
    cv_sec_embs: Dict[str, Dict[str, np.ndarray]] = {}
    for c in unique_cvs:
        secs = sections_to_dict(split_sections(c, is_jd=False))
        vdict = {}
        for sn in ["skills", "experience", "education", "summary"]:
            if sn in secs and secs[sn].text.strip():
                vdict[sn] = embedder.encode(secs[sn].text, show_progress_bar=False)
        cv_sec_embs[c] = vdict

    # Pre-extract taxonomy skills
    print("Pre-extracting taxonomy skills...")
    tax = get_taxonomy()
    job_skills = {j: tax.classify_jd_skills(j) for j in unique_jobs}
    cv_skills = {c: tax.extract_skills_from_text(c) for c in unique_cvs}

    # Pre-compute TF-IDF
    print("Fitting TF-IDF on train subset...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf.fit(unique_cvs + unique_jobs)
    job_tfidf = {j: tfidf.transform([j]) for j in unique_jobs}
    cv_tfidf = {c: tfidf.transform([c]) for c in unique_cvs}

    # Precompute pairwise baseline components
    pairs = list(zip(train_sub["job_description_text"], train_sub["resume_text"], train_sub["target"]))
    query_groups: Dict[str, List[Tuple[str, int]]] = {}
    for j, c, y in pairs:
        query_groups.setdefault(j, []).append((c, y))

    # Precompute Single-Vector and BM25 scores
    pair_single_scores: Dict[Tuple[str, str], float] = {}
    pair_bm25_scores: Dict[Tuple[str, str], float] = {}
    pair_tax_scores: Dict[Tuple[str, str], float] = {}

    for j, c, _ in pairs:
        pair_single_scores[(j, c)] = float(np.dot(job_embs[j], cv_embs[c]))
        pair_bm25_scores[(j, c)] = float((job_tfidf[j].multiply(cv_tfidf[c])).sum())
        must, nice = job_skills[j]
        all_j = set(must + nice)
        c_s = set(cv_skills[c])
        cov = len(all_j.intersection(c_s)) / len(all_j) if all_j else 1.0
        pair_tax_scores[(j, c)] = float(cov)

    # =========================================================================
    # Sweep 1: Section Weight Configurations
    # =========================================================================
    print("\n--- Running Sweep 1: Section Weights ---")
    weight_configs = {
        "Default (0.40/0.35/0.15/0.10)": {"skills": 0.40, "experience": 0.35, "education": 0.15, "summary": 0.10},
        "Equal (0.25/0.25/0.25/0.25)": {"skills": 0.25, "experience": 0.25, "education": 0.25, "summary": 0.25},
        "Skills-Heavy (0.60/0.20/0.10/0.10)": {"skills": 0.60, "experience": 0.20, "education": 0.10, "summary": 0.10},
        "Experience-Heavy (0.20/0.60/0.10/0.10)": {"skills": 0.20, "experience": 0.60, "education": 0.10, "summary": 0.10},
        "Core Only (0.50/0.50/0.00/0.00)": {"skills": 0.50, "experience": 0.50, "education": 0.00, "summary": 0.00},
    }

    section_sweep_results = {}
    for cfg_name, wdict in weight_configs.items():
        q_ndcgs, q_maps = [], []
        for j, c_list in query_groups.items():
            jv = job_embs[j]
            scores, labels = [], []
            for c, y in c_list:
                c_secs = cv_sec_embs[c]
                w_sum, tot_w = 0.0, 0.0
                for sn, w in wdict.items():
                    if sn in c_secs:
                        w_sum += w * float(np.dot(jv, c_secs[sn]))
                        tot_w += w
                scores.append(w_sum / tot_w if tot_w > 0.0 else pair_single_scores[(j, c)])
                labels.append(y)
            sc_arr = np.array(scores)
            lb_arr = np.array(labels)
            q_ndcgs.append(compute_ndcg_at_10(sc_arr, lb_arr))
            q_maps.append(compute_map(sc_arr, lb_arr))
        mean_n10 = float(np.mean(q_ndcgs))
        mean_map = float(np.mean(q_maps))
        section_sweep_results[cfg_name] = {
            "ndcg@10": round(mean_n10, 4),
            "map": round(mean_map, 4),
        }
        print(f"  {cfg_name:38s} | nDCG@10: {mean_n10:.4f} | MAP: {mean_map:.4f}")

    # =========================================================================
    # Sweep 2: Alpha (Dense vs Lexical in Fused Retrieval)
    # =========================================================================
    print("\n--- Running Sweep 2: Alpha (Dense vs Lexical) ---")
    alpha_sweep_results = {}
    default_sec_w = weight_configs["Default (0.40/0.35/0.15/0.10)"]

    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        q_ndcgs, q_maps = [], []
        for j, c_list in query_groups.items():
            jv = job_embs[j]
            dense_scores, bm25_scores, labels = [], [], []
            for c, y in c_list:
                c_secs = cv_sec_embs[c]
                w_sum, tot_w = 0.0, 0.0
                for sn, w in default_sec_w.items():
                    if sn in c_secs:
                        w_sum += w * float(np.dot(jv, c_secs[sn]))
                        tot_w += w
                dense_scores.append(w_sum / tot_w if tot_w > 0.0 else pair_single_scores[(j, c)])
                bm25_scores.append(pair_bm25_scores[(j, c)])
                labels.append(y)

            norm_dense = min_max(np.array(dense_scores))
            norm_bm25 = min_max(np.array(bm25_scores))
            fused = alpha * norm_dense + (1.0 - alpha) * norm_bm25
            lb_arr = np.array(labels)
            q_ndcgs.append(compute_ndcg_at_10(fused, lb_arr))
            q_maps.append(compute_map(fused, lb_arr))

        mean_n10 = float(np.mean(q_ndcgs))
        mean_map = float(np.mean(q_maps))
        alpha_sweep_results[f"alpha_{alpha:.1f}"] = {
            "alpha": alpha,
            "ndcg@10": round(mean_n10, 4),
            "map": round(mean_map, 4),
        }
        print(f"  alpha = {alpha:.1f} (Dense: {alpha*100:3.0f}%, Lexical: {(1-alpha)*100:3.0f}%) | nDCG@10: {mean_n10:.4f} | MAP: {mean_map:.4f}")

    # =========================================================================
    # Sweep 3: Beta (Retrieval vs Taxonomy in Final Ranking)
    # =========================================================================
    print("\n--- Running Sweep 3: Beta (Retrieval vs Taxonomy) ---")
    beta_sweep_results = {}
    for beta in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        q_ndcgs, q_maps = [], []
        for j, c_list in query_groups.items():
            jv = job_embs[j]
            dense_scores, bm25_scores, tax_scores, labels = [], [], [], []
            for c, y in c_list:
                c_secs = cv_sec_embs[c]
                w_sum, tot_w = 0.0, 0.0
                for sn, w in default_sec_w.items():
                    if sn in c_secs:
                        w_sum += w * float(np.dot(jv, c_secs[sn]))
                        tot_w += w
                dense_scores.append(w_sum / tot_w if tot_w > 0.0 else pair_single_scores[(j, c)])
                bm25_scores.append(pair_bm25_scores[(j, c)])
                tax_scores.append(pair_tax_scores[(j, c)])
                labels.append(y)

            norm_dense = min_max(np.array(dense_scores))
            norm_bm25 = min_max(np.array(bm25_scores))
            fused_retrieval = 0.60 * norm_dense + 0.40 * norm_bm25
            norm_tax = min_max(np.array(tax_scores))
            final_scores = beta * fused_retrieval + (1.0 - beta) * norm_tax

            lb_arr = np.array(labels)
            q_ndcgs.append(compute_ndcg_at_10(final_scores, lb_arr))
            q_maps.append(compute_map(final_scores, lb_arr))

        mean_n10 = float(np.mean(q_ndcgs))
        mean_map = float(np.mean(q_maps))
        beta_sweep_results[f"beta_{beta:.1f}"] = {
            "beta": beta,
            "ndcg@10": round(mean_n10, 4),
            "map": round(mean_map, 4),
        }
        print(f"  beta = {beta:.1f} (Retrieval: {beta*100:3.0f}%, Taxonomy: {(1-beta)*100:3.0f}%) | nDCG@10: {mean_n10:.4f} | MAP: {mean_map:.4f}")

    # Output JSON payload
    out_payload = {
        "dataset": "cnamuangtoun/resume-job-description-fit (train split)",
        "sample_queries": len(unique_jobs),
        "sample_candidates": len(unique_cvs),
        "sample_pairs": len(train_sub),
        "section_weights_sweep": section_sweep_results,
        "alpha_sweep": alpha_sweep_results,
        "beta_sweep": beta_sweep_results,
    }

    with open(RESULTS_DIR / "train_sensitivity_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved train sensitivity results to {RESULTS_DIR / 'train_sensitivity_analysis.json'}")


if __name__ == "__main__":
    main()
