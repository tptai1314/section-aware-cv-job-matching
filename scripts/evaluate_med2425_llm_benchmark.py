"""Evaluate Section-Aware CV-Job Matching on med2425/resume-job-fit-merged-v1.

Auditor protocol:
1. Secondary validation with LLM labels (Qwen2.5-32B recruiter judge).
2. Filter to target IT domain: resume_domain in {software, data, ai} and jd_domain in {software, data, ai}.
3. 41 unique JDs, 277 unique CVs, 8,575 pairs.
4. Evaluate: Random, BM25, MiniLM Flat, MiniLM Section-Aware, Calibrated Hybrid, Oracle.
5. Paired difference 95% Bootstrap CI and paired t-test.

Outputs:
- results/med2425_secondary_audit.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.data.preprocessing import sections_to_dict, split_sections
from cvjd.data.taxonomy import get_taxonomy

DATA_FILE = REPO_ROOT / "data" / "public_benchmark" / "med2425" / "test.parquet"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}

def dcg_at_k(rels: Sequence[float | int], k: int) -> float:
    score = 0.0
    for i in range(min(k, len(rels))):
        score += (math.pow(2.0, float(rels[i])) - 1.0) / math.log2(i + 2.0)
    return score

def ndcg_at_k(ranked_rels: Sequence[float | int], ideal_rels: Sequence[float | int], k: int) -> float:
    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0.0:
        return 0.0
    return min(1.0, max(0.0, dcg_at_k(ranked_rels, k) / idcg))

def reciprocal_rank(ranked_rels: Sequence[float | int], threshold: int) -> float:
    for rank, r in enumerate(ranked_rels, start=1):
        if r >= threshold:
            return 1.0 / rank
    return 0.0

def average_precision(ranked_rels: Sequence[float | int], threshold: int) -> float:
    total_rel = sum(1 for r in ranked_rels if r >= threshold)
    if total_rel <= 0:
        return 0.0
    hits = 0
    p_sum = 0.0
    for rank, r in enumerate(ranked_rels, start=1):
        if r >= threshold:
            hits += 1
            p_sum += hits / rank
    return p_sum / total_rel

def precision_at_k(ranked_rels: Sequence[float | int], k: int, threshold: int) -> float:
    if k <= 0 or len(ranked_rels) == 0:
        return 0.0
    return sum(1 for r in ranked_rels[:k] if r >= threshold) / float(k)

def evaluate_query_with_tie_breaking(
    scores: np.ndarray,
    rels: np.ndarray,
    k_vals: Tuple[int, int] = (5, 10),
    thresholds: Tuple[int, int] = (1, 2),
    n_tie_shuffles: int = 50,
    rng: np.random.RandomState | None = None,
) -> Dict[str, float]:
    if rng is None:
        rng = np.random.RandomState(42)

    n_items = len(scores)
    if n_items == 0:
        return {
            "ndcg@5": 0.0, "ndcg@10": 0.0,
            "mrr_th1": 0.0, "mrr_th2": 0.0,
            "map_th1": 0.0, "map_th2": 0.0,
            "p@5_th1": 0.0, "p@5_th2": 0.0,
        }

    ideal_order = np.argsort(-rels)
    ideal_rels = rels[ideal_order]

    metric_acc: Dict[str, float] = defaultdict(float)

    for _ in range(n_tie_shuffles):
        jitter = rng.uniform(-1e-7, 1e-7, size=n_items)
        perturbed_scores = scores + jitter
        ranked_indices = np.argsort(-perturbed_scores)
        ranked_rels = rels[ranked_indices]

        metric_acc["ndcg@5"] += ndcg_at_k(ranked_rels, ideal_rels, 5)
        metric_acc["ndcg@10"] += ndcg_at_k(ranked_rels, ideal_rels, 10)
        metric_acc["mrr_th1"] += reciprocal_rank(ranked_rels, threshold=1)
        metric_acc["mrr_th2"] += reciprocal_rank(ranked_rels, threshold=2)
        metric_acc["map_th1"] += average_precision(ranked_rels, threshold=1)
        metric_acc["map_th2"] += average_precision(ranked_rels, threshold=2)
        metric_acc["p@5_th1"] += precision_at_k(ranked_rels, 5, threshold=1)
        metric_acc["p@5_th2"] += precision_at_k(ranked_rels, 5, threshold=2)

    return {k: v / n_tie_shuffles for k, v in metric_acc.items()}

def min_max_norm(arr: np.ndarray) -> np.ndarray:
    mi, ma = np.min(arr), np.max(arr)
    if ma - mi < 1e-9:
        return np.zeros_like(arr)
    return (arr - mi) / (ma - mi)

def main() -> None:
    print("=" * 80, flush=True)
    print("SECONDARY BENCHMARK: med2425/resume-job-fit-merged-v1 (Qwen2.5-32B Labels)", flush=True)
    print("=" * 80, flush=True)

    df = pd.read_parquet(DATA_FILE)
    df["rel"] = df["label"].map(LABEL_MAP)

    target_domains = {"software", "data", "ai"}
    domain_mask = df["resume_domain"].isin(target_domains) & df["jd_domain"].isin(target_domains)
    sub_df = df[domain_mask].copy()

    print(f"Total rows in test.parquet: {len(df)}", flush=True)
    print(f"Target IT domain pairs (software, data, ai): {len(sub_df)}", flush=True)
    print(f"Target domain label distribution:\n{sub_df['label'].value_counts()}", flush=True)

    unique_jobs = sub_df["jd"].unique().tolist()
    unique_cvs = sub_df["resume"].unique().tolist()
    job_to_idx = {j: i for i, j in enumerate(unique_jobs)}
    cv_to_idx = {r: i for i, r in enumerate(unique_cvs)}

    print(f"Unique JDs: {len(unique_jobs)}, Unique Resumes: {len(unique_cvs)}", flush=True)

    pairs: List[Tuple[int, int]] = []
    y_true: List[int] = []
    query_groups: Dict[int, List[int]] = defaultdict(list)

    for _, row in sub_df.iterrows():
        j_idx = job_to_idx[row["jd"]]
        c_idx = cv_to_idx[row["resume"]]
        p_idx = len(pairs)
        query_groups[j_idx].append(p_idx)
        pairs.append((j_idx, c_idx))
        y_true.append(row["rel"])

    y_true_arr = np.array(y_true, dtype=int)
    n_queries = len(unique_jobs)

    # 1. Lexical BM25
    print("\n--> [1/4] Computing Lexical BM25 ...", flush=True)
    vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    vec.fit(unique_jobs + unique_cvs)
    job_tfidf = vec.transform(unique_jobs)
    cv_tfidf = vec.transform(unique_cvs)
    bm25_scores = np.array([float((job_tfidf[j].multiply(cv_tfidf[c])).sum()) for j, c in pairs])

    # 2. MiniLM Embeddings
    print("--> [2/4] Computing MiniLM Single-Vector & Section-Aware Embeddings ...", flush=True)
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    job_flat_embs = minilm.encode(unique_jobs, batch_size=32, normalize_embeddings=True)
    cv_flat_embs = minilm.encode(unique_cvs, batch_size=32, normalize_embeddings=True)
    flat_scores = np.array([float(np.dot(job_flat_embs[j], cv_flat_embs[c])) for j, c in pairs])

    # Section-Aware
    cv_parsed = [sections_to_dict(split_sections(c, is_jd=False)) for c in unique_cvs]
    jd_parsed = [sections_to_dict(split_sections(j, is_jd=True)) for j in unique_jobs]

    sec_keys = ["skills", "experience", "education", "summary"]
    sec_weights = {"skills": 0.40, "experience": 0.35, "education": 0.15, "summary": 0.10}

    sec_cv_texts = []
    sec_job_texts = []
    for sk in sec_keys:
        sec_cv_texts.extend([sd[sk].text if sk in sd else "" for sd in cv_parsed])
        sec_job_texts.extend([sd[sk].text if sk in sd else "" for sd in jd_parsed])

    unique_sec_texts = list(set([t for t in sec_cv_texts + sec_job_texts if t.strip()]))
    sec_emb_map = {}
    if unique_sec_texts:
        non_empty_embs = minilm.encode(unique_sec_texts, batch_size=32, normalize_embeddings=True)
        sec_emb_map = dict(zip(unique_sec_texts, non_empty_embs))
    zero_vec = np.zeros(384, dtype=np.float32)

    cv_sec_vecs = {sk: [sec_emb_map.get(sd[sk].text if sk in sd else "", zero_vec) for sd in cv_parsed] for sk in sec_keys}
    job_sec_vecs = {sk: [sec_emb_map.get(sd[sk].text if sk in sd else "", zero_vec) for sd in jd_parsed] for sk in sec_keys}

    sec_scores = []
    for j, c in pairs:
        score = 0.0
        w_sum = 0.0
        for sk, w in sec_weights.items():
            jv = job_sec_vecs[sk][j]
            cv = cv_sec_vecs[sk][c]
            if np.linalg.norm(jv) > 1e-6 and np.linalg.norm(cv) > 1e-6:
                sim = float(np.dot(jv, cv))
                score += w * sim
                w_sum += w
            elif np.linalg.norm(cv) > 1e-6:
                sim = float(np.dot(job_flat_embs[j], cv))
                score += (w * 0.5) * sim
                w_sum += (w * 0.5)
        if w_sum > 0:
            sec_scores.append(score / w_sum)
        else:
            sec_scores.append(float(np.dot(job_flat_embs[j], cv_flat_embs[c])))
    sec_scores = np.array(sec_scores)

    # 3. Taxonomy & Calibrated Hybrid
    print("--> [3/4] Computing Taxonomy Coverage and Hybrid Fusion ...", flush=True)
    taxonomy = get_taxonomy()
    job_taxonomy_skills = [taxonomy.classify_jd_skills(j) for j in unique_jobs]
    cv_taxonomy_skills = [taxonomy.extract_skills_from_text(c) for c in unique_cvs]

    tax_scores = []
    for j, c in pairs:
        j_must, j_nice = job_taxonomy_skills[j]
        all_j = set(j_must).union(set(j_nice))
        c_sk = set(cv_taxonomy_skills[c])
        if not all_j:
            tax_scores.append(0.0)
            continue
        cov = 0.0
        for js in all_j:
            if js in c_sk:
                cov += 1.0
            else:
                max_rel = max([taxonomy.get_relation_weight(cs, js) for cs in c_sk], default=0.0)
                cov += max_rel
        tax_scores.append(cov / len(all_j))
    tax_scores = np.array(tax_scores)

    dense_norm = min_max_norm(sec_scores)
    bm25_norm = min_max_norm(bm25_scores)
    l0_scores = 0.60 * dense_norm + 0.40 * bm25_norm

    tax_norm = min_max_norm(tax_scores)
    hybrid_scores = 0.60 * l0_scores + 0.40 * tax_norm

    # Random Baseline
    rng_rand = np.random.RandomState(42)
    random_scores = rng_rand.uniform(0, 1, size=len(pairs))

    models = {
        "Floor: Random Ranking": random_scores,
        "BM25 / TF-IDF (Lexical)": bm25_scores,
        "MiniLM Single-Vector (Flat)": flat_scores,
        "MiniLM Section-Aware Dense (Ours)": sec_scores,
        "Dense + BM25 (L0: No Taxonomy)": l0_scores,
        "Calibrated Hybrid (Dense+BM25+Tax)": hybrid_scores,
    }

    # 4. Evaluation
    print("--> [4/4] Ranking and Computing Metrics ...", flush=True)
    rng_eval = np.random.RandomState(42)
    eval_results = {}

    for m_name, m_scores in models.items():
        per_q = []
        for q in range(n_queries):
            p_idx = query_groups[q]
            if not p_idx:
                continue
            q_res = evaluate_query_with_tie_breaking(
                m_scores[p_idx], y_true_arr[p_idx], n_tie_shuffles=50, rng=rng_eval
            )
            per_q.append(q_res)
        avg_m = {k: float(np.mean([r[k] for r in per_q])) for k in per_q[0].keys()}
        eval_results[m_name] = {
            "metrics": avg_m,
            "per_query_n10": [r["ndcg@10"] for r in per_q],
            "per_query_map": [r["map_th1"] for r in per_q],
            "per_query_mrr": [r["mrr_th1"] for r in per_q],
        }

    # Oracle
    or_per_q = []
    for q in range(n_queries):
        p_idx = query_groups[q]
        if not p_idx:
            continue
        or_sc = y_true_arr[p_idx].astype(float)
        q_res = evaluate_query_with_tie_breaking(or_sc, y_true_arr[p_idx], n_tie_shuffles=20, rng=rng_eval)
        or_per_q.append(q_res)
    avg_or = {k: float(np.mean([r[k] for r in or_per_q])) for k in or_per_q[0].keys()}
    eval_results["Ceiling: Oracle Ranking"] = {
        "metrics": avg_or,
        "per_query_n10": [r["ndcg@10"] for r in or_per_q],
        "per_query_map": [r["map_th1"] for r in or_per_q],
        "per_query_mrr": [r["mrr_th1"] for r in or_per_q],
    }

    # Paired Statistical Tests: Section-Aware vs Single-Vector
    diffs_n10 = np.array(eval_results["MiniLM Section-Aware Dense (Ours)"]["per_query_n10"]) - np.array(eval_results["MiniLM Single-Vector (Flat)"]["per_query_n10"])
    rng_bs = np.random.RandomState(42)
    boot_means = [np.mean(diffs_n10[rng_bs.choice(len(diffs_n10), size=len(diffs_n10), replace=True)]) for _ in range(1000)]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    t_stat, p_val = ttest_rel(eval_results["MiniLM Section-Aware Dense (Ours)"]["per_query_n10"], eval_results["MiniLM Single-Vector (Flat)"]["per_query_n10"])
    w_stat, p_wilc = wilcoxon(diffs_n10, zero_method="wilcox")

    stat_test = {
        "comparison": "Section-Aware vs Single-Vector (nDCG@10)",
        "mean_diff": round(float(np.mean(diffs_n10)), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "t_stat": round(float(t_stat), 3),
        "paired_t_p": round(float(p_val), 4),
        "wilcoxon_p": round(float(p_wilc), 4),
    }

    output_data = {
        "dataset": "med2425/resume-job-fit-merged-v1 (Target IT Domains: software, data, ai)",
        "annotation_source": "Qwen2.5-32B LLM Recruiter Judge (Zero-shot temperature=0)",
        "n_queries": n_queries,
        "n_candidates": len(unique_cvs),
        "n_pairs": len(sub_df),
        "results": {k: {"metrics": v["metrics"]} for k, v in eval_results.items()},
        "statistical_test": stat_test,
    }

    with open(RESULTS_DIR / "med2425_secondary_audit.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n=== SECONDARY BENCHMARK RESULTS (med2425, Qwen2.5-32B Labels) ===")
    print(f"{'Model':40s} | {'nDCG@10':7s} | {'MRR':7s} | {'MAP':7s} | {'P@5':7s}")
    print("-" * 75)
    for m, d in eval_results.items():
        met = d["metrics"]
        print(f"{m:40s} | {met['ndcg@10']:.4f}  | {met['mrr_th1']:.4f}  | {met['map_th1']:.4f}  | {met['p@5_th1']:.4f}")

    print("\n=== PAIRED TEST: Section-Aware vs Single-Vector ===")
    print(f"Mean Diff: {stat_test['mean_diff']:+.4f} | 95% Bootstrap CI: [{stat_test['ci_lower']:+.4f}, {stat_test['ci_upper']:+.4f}]")
    print(f"Paired t-test p: {stat_test['paired_t_p']:.4f} | Wilcoxon p: {stat_test['wilcoxon_p']:.4f}")

if __name__ == "__main__":
    main()
