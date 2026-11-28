"""Run scientific benchmark evaluation strictly on the 76 clean held-out train queries.

Adheres strictly to the pre-registered plan: docs/PRE_REGISTERED_TRAIN_EVAL_PLAN.md
- 76 clean JDs, 1829 pairs
- Setting A (all 76 queries)
- Setting B (rel >= 1: 71 queries; rel == 2: 31 queries)
- Models: Random, BM25, MiniLM Flat, Chunk-Mean, Chunk-MaxSim, Section-Aware, Dense+BM25, Hybrid, RRF, Oracle
- Paired statistical significance tests: 1000-resample bootstrap 95% CI, paired t-test, Wilcoxon, Holm-Bonferroni
- Functional character coverage analysis

Outputs:
- results/clean_train_benchmark_results.json
- paper/tables/tab_clean_train_benchmark.tex
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from cvjd.ranking.engine import min_max

DATA_DIR = REPO_ROOT / "data" / "public_benchmark" / "resume-job-description-fit"
ANNOTATION_DIR = REPO_ROOT / "data" / "audit_annotation"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}

# ==============================================================================
# 1. METRICS WITH FAIR TIE-BREAKING
# ==============================================================================

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

def chunk_text(text: str, window_words: int = 128, stride_words: int = 64) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    if len(words) <= window_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + window_words])
        chunks.append(chunk)
        if start + window_words >= len(words):
            break
        start += stride_words
    return chunks

# ==============================================================================
# 2. MAIN RUNNER
# ==============================================================================

def main() -> None:
    print("=" * 80)
    print("RUNNING BENCHMARK EVALUATION ON CLEAN HELD-OUT TRAIN QUERIES")
    print("=" * 80)

    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="latin1")
    train_df["rel"] = train_df["label"].map(LABEL_MAP)

    unique_train_jds = train_df["job_description_text"].unique().tolist()

    # Load touched sets
    dev_jds_set = set()
    if (ANNOTATION_DIR / "train_dev_pool.json").exists():
        with open(ANNOTATION_DIR / "train_dev_pool.json", "r", encoding="utf-8") as f:
            dev_jds_set = set(json.load(f).get("dev_jds", []))

    eval_jds_set = set()
    if (ANNOTATION_DIR / "raw_eval_samples.json").exists():
        with open(ANNOTATION_DIR / "raw_eval_samples.json", "r", encoding="utf-8") as f:
            d = json.load(f)
            eval_jds_set = set(item["text"] for item in d if item.get("doc_type") == "JD")

    query_counts = train_df["job_description_text"].value_counts()
    valid_queries = query_counts[query_counts >= 10].index.tolist()
    np.random.seed(42)
    sensitivity_jds_set = set(np.random.choice(valid_queries, size=min(50, len(valid_queries)), replace=False))

    all_touched_jds = dev_jds_set.union(eval_jds_set).union(sensitivity_jds_set)
    clean_jds = [j for j in unique_train_jds if j not in all_touched_jds]
    print(f"Total Unique Train JDs: {len(unique_train_jds)}")
    print(f"Total Touched JDs: {len(all_touched_jds)}")
    print(f"Clean Held-Out Train JDs: {len(clean_jds)}")

    clean_train_df = train_df[train_df["job_description_text"].isin(clean_jds)].copy()
    print(f"Clean Train Pairs: {len(clean_train_df)}")

    unique_jobs = clean_jds
    job_to_idx = {j: i for i, j in enumerate(unique_jobs)}
    unique_cvs = clean_train_df["resume_text"].unique().tolist()
    cv_to_idx = {r: i for i, r in enumerate(unique_cvs)}

    pairs: List[Tuple[int, int]] = []
    y_true: List[int] = []
    query_groups: Dict[int, List[int]] = defaultdict(list)

    for _, row in clean_train_df.iterrows():
        j_idx = job_to_idx[row["job_description_text"]]
        c_idx = cv_to_idx[row["resume_text"]]
        p_idx = len(pairs)
        query_groups[j_idx].append(p_idx)
        pairs.append((j_idx, c_idx))
        y_true.append(row["rel"])

    y_true_arr = np.array(y_true, dtype=int)
    n_queries = len(unique_jobs)

    pos_queries_th1 = [q for q, p_idx in query_groups.items() if any(y_true_arr[p] >= 1 for p in p_idx)]
    pos_queries_th2 = [q for q, p_idx in query_groups.items() if any(y_true_arr[p] == 2 for p in p_idx)]
    print(f"Setting A: All {n_queries} clean queries")
    print(f"Setting B (rel >= 1): {len(pos_queries_th1)} queries with >= 1 positive ({n_queries - len(pos_queries_th1)} zero-pos)")
    print(f"Setting B (rel == 2): {len(pos_queries_th2)} queries with >= 1 positive ({n_queries - len(pos_queries_th2)} zero-pos)")

    # 1. Character coverage analysis
    print("\n--> [Step 1/6] Measuring Functional Character Coverage ...")
    functional_keys = {'skills', 'experience', 'education', 'summary', 'responsibilities', 'requirements', 'qualifications', 'benefits', 'about_company', 'projects'}
    
    cv_func_ratios = []
    cv_preamble_ratios = []
    for c in unique_cvs:
        tot = len(c)
        if tot == 0:
            continue
        secs = split_sections(c, is_jd=False)
        f_len = sum(len(s.text) for s in secs if s.name in functional_keys)
        p_len = sum(len(s.text) for s in secs if s.name == 'preamble')
        cv_func_ratios.append(f_len / tot)
        cv_preamble_ratios.append(p_len / tot)

    jd_func_ratios = []
    jd_preamble_ratios = []
    for j in unique_jobs:
        tot = len(j)
        if tot == 0:
            continue
        secs = split_sections(j, is_jd=True)
        f_len = sum(len(s.text) for s in secs if s.name in functional_keys)
        p_len = sum(len(s.text) for s in secs if s.name == 'preamble')
        jd_func_ratios.append(f_len / tot)
        jd_preamble_ratios.append(p_len / tot)

    char_coverage_stats = {
        "cv_mean_functional_pct": round(float(np.mean(cv_func_ratios) * 100), 2),
        "cv_median_functional_pct": round(float(np.median(cv_func_ratios) * 100), 2),
        "cv_mean_preamble_pct": round(float(np.mean(cv_preamble_ratios) * 100), 2),
        "jd_mean_functional_pct": round(float(np.mean(jd_func_ratios) * 100), 2),
        "jd_median_functional_pct": round(float(np.median(jd_func_ratios) * 100), 2),
        "jd_mean_preamble_pct": round(float(np.mean(jd_preamble_ratios) * 100), 2),
    }
    print("Functional Character Coverage:", char_coverage_stats)

    # 2. Section Parsing with Frozen Parser
    print("\n--> [Step 2/6] Parsing sections with frozen parser ...")
    cv_parsed = [sections_to_dict(split_sections(c, is_jd=False)) for c in unique_cvs]
    jd_parsed = [sections_to_dict(split_sections(j, is_jd=True)) for j in unique_jobs]

    cv_fallback_count = sum(1 for sd in cv_parsed if not any(k in sd for k in ["skills", "experience", "education", "summary"]))
    print(f"CV Fallback Count: {cv_fallback_count} / {len(unique_cvs)} ({cv_fallback_count/len(unique_cvs)*100:.2f}%)")

    # 3. Model 1: Lexical BM25 / TF-IDF
    print("\n--> [Step 3/6] Computing Lexical BM25 / TF-IDF ...")
    vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    vec.fit(unique_jobs + unique_cvs)
    job_tfidf = vec.transform(unique_jobs)
    cv_tfidf = vec.transform(unique_cvs)
    bm25_scores = np.array([float((job_tfidf[j].multiply(cv_tfidf[c])).sum()) for j, c in pairs])

    # 4. Model 2: MiniLM Embeddings
    print("\n--> [Step 4/6] Encoding with all-MiniLM-L6-v2 ...", flush=True)
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    CACHE_DIR = REPO_ROOT / "data" / "cache_embeddings"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Flat Single-Vector
    job_flat_path = CACHE_DIR / "clean_train_job_flat.npy"
    cv_flat_path = CACHE_DIR / "clean_train_cv_flat.npy"
    if job_flat_path.exists() and cv_flat_path.exists():
        job_flat_embs = np.load(job_flat_path)
        cv_flat_embs = np.load(cv_flat_path)
    else:
        job_flat_embs = minilm.encode(unique_jobs, batch_size=32, normalize_embeddings=True)
        cv_flat_embs = minilm.encode(unique_cvs, batch_size=32, normalize_embeddings=True)
        np.save(job_flat_path, job_flat_embs)
        np.save(cv_flat_path, cv_flat_embs)
    flat_scores = np.array([float(np.dot(job_flat_embs[j], cv_flat_embs[c])) for j, c in pairs])

    # Chunk-Mean & Chunk-MaxSim
    print("Computing Chunk-Mean and Chunk-MaxSim ...", flush=True)
    cv_chunks = [chunk_text(c, 128, 64) for c in unique_cvs]
    job_chunks = [chunk_text(j, 128, 64) for j in unique_jobs]

    all_cv_chunks = [ch for doc in cv_chunks for ch in doc]
    all_job_chunks = [ch for doc in job_chunks for ch in doc]

    cv_chunk_path = CACHE_DIR / "clean_train_cv_chunk.npy"
    job_chunk_path = CACHE_DIR / "clean_train_job_chunk.npy"
    if cv_chunk_path.exists() and job_chunk_path.exists():
        cv_chunk_embs_flat = np.load(cv_chunk_path)
        job_chunk_embs_flat = np.load(job_chunk_path)
    else:
        cv_chunk_embs_flat = minilm.encode(all_cv_chunks, batch_size=32, normalize_embeddings=True)
        job_chunk_embs_flat = minilm.encode(all_job_chunks, batch_size=32, normalize_embeddings=True)
        np.save(cv_chunk_path, cv_chunk_embs_flat)
        np.save(job_chunk_path, job_chunk_embs_flat)

    cv_chunk_embs = []
    idx = 0
    for doc in cv_chunks:
        n_ch = len(doc)
        cv_chunk_embs.append(cv_chunk_embs_flat[idx : idx + n_ch])
        idx += n_ch

    job_chunk_embs = []
    idx = 0
    for doc in job_chunks:
        n_ch = len(doc)
        job_chunk_embs.append(job_chunk_embs_flat[idx : idx + n_ch])
        idx += n_ch

    chunk_mean_scores = []
    chunk_maxsim_scores = []
    for j, c in pairs:
        j_embs = job_chunk_embs[j]
        c_embs = cv_chunk_embs[c]
        sim_matrix = np.dot(j_embs, c_embs.T)
        chunk_mean_scores.append(float(np.mean(sim_matrix)))
        chunk_maxsim_scores.append(float(np.mean(np.max(sim_matrix, axis=1))))

    chunk_mean_scores = np.array(chunk_mean_scores)
    chunk_maxsim_scores = np.array(chunk_maxsim_scores)

    # Section-Aware MiniLM
    print("Computing Section-Aware Dense Embeddings ...", flush=True)
    sec_keys = ["skills", "experience", "education", "summary"]
    sec_weights = {"skills": 0.40, "experience": 0.35, "education": 0.15, "summary": 0.10}

    # Encode section texts
    sec_cv_texts = []
    sec_job_texts = []
    for sk in sec_keys:
        sec_cv_texts.extend([sd[sk].text if sk in sd else "" for sd in cv_parsed])
        sec_job_texts.extend([sd[sk].text if sk in sd else "" for sd in jd_parsed])

    all_sec_texts = sec_cv_texts + sec_job_texts
    unique_sec_texts = list(set(all_sec_texts))
    sec_emb_map = {}
    sec_texts_path = CACHE_DIR / "clean_train_sec_texts.json"
    sec_embs_path = CACHE_DIR / "clean_train_sec_embs.npy"
    if sec_texts_path.exists() and sec_embs_path.exists():
        with open(sec_texts_path, "r", encoding="utf-8") as f:
            non_empty = json.load(f)
        non_empty_embs = np.load(sec_embs_path)
        sec_emb_map = dict(zip(non_empty, non_empty_embs))
    else:
        if unique_sec_texts:
            non_empty = [t for t in unique_sec_texts if t.strip()]
            if non_empty:
                non_empty_embs = minilm.encode(non_empty, batch_size=32, normalize_embeddings=True)
                sec_emb_map = dict(zip(non_empty, non_empty_embs))
                with open(sec_texts_path, "w", encoding="utf-8") as f:
                    json.dump(non_empty, f)
                np.save(sec_embs_path, non_empty_embs)
    zero_vec = np.zeros(384, dtype=np.float32)

    cv_sec_vecs = {sk: [] for sk in sec_keys}
    for sd in cv_parsed:
        for sk in sec_keys:
            t = sd[sk].text if sk in sd else ""
            cv_sec_vecs[sk].append(sec_emb_map.get(t, zero_vec))

    job_sec_vecs = {sk: [] for sk in sec_keys}
    for sd in jd_parsed:
        for sk in sec_keys:
            t = sd[sk].text if sk in sd else ""
            job_sec_vecs[sk].append(sec_emb_map.get(t, zero_vec))

    # Cross-section dynamic weighted scoring
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
                # Fallback to compare CV section with job flat
                sim = float(np.dot(job_flat_embs[j], cv))
                score += (w * 0.5) * sim
                w_sum += (w * 0.5)
        if w_sum > 0:
            sec_scores.append(score / w_sum)
        else:
            sec_scores.append(float(np.dot(job_flat_embs[j], cv_flat_embs[c])))
    sec_scores = np.array(sec_scores)

    # 5. Hybrid & Taxonomy
    print("\n--> [Step 5/6] Computing Taxonomy Coverage and Hybrid Fusion ...")
    taxonomy = get_taxonomy()
    taxonomy.attach_skill_embeddings(lambda xs: minilm.encode(xs, batch_size=64, normalize_embeddings=True))
    job_taxonomy_skills = [taxonomy.classify_jd_skills(j) for j in unique_jobs]
    cv_taxonomy_skills = [taxonomy.extract_skills_from_text(c) for c in unique_cvs]

    tax_scores = np.array([
        taxonomy.score_skills_coverage(
            job_taxonomy_skills[j][0],
            job_taxonomy_skills[j][1],
            cv_taxonomy_skills[c],
            must_weight=0.85,
            nice_weight=0.15,
            neutral_score=1.0,
            use_layer2=True,
            use_layer3=True,
        )
        for j, c in pairs
    ])

    # Dense + BM25 (L0)
    dense_norm = min_max_norm(sec_scores)
    bm25_norm = min_max_norm(bm25_scores)
    l0_scores = 0.60 * dense_norm + 0.40 * bm25_norm

    # Calibrated Hybrid
    tax_norm = min_max_norm(tax_scores)
    hybrid_scores = 0.60 * l0_scores + 0.40 * tax_norm

    # Parameter-Free RRF
    rrf_k = 60
    rrf_scores = np.zeros(len(pairs))
    for q_idx in range(n_queries):
        p_indices = query_groups[q_idx]
        if not p_indices:
            continue
        order_dense = np.argsort(-sec_scores[p_indices])
        order_bm25 = np.argsort(-bm25_scores[p_indices])
        order_tax = np.argsort(-tax_scores[p_indices])

        rank_d = np.empty_like(order_dense)
        rank_d[order_dense] = np.arange(1, len(p_indices) + 1)

        rank_b = np.empty_like(order_bm25)
        rank_b[order_bm25] = np.arange(1, len(p_indices) + 1)

        rank_t = np.empty_like(order_tax)
        rank_t[order_tax] = np.arange(1, len(p_indices) + 1)

        rrf_q = (1.0 / (rrf_k + rank_d)) + (1.0 / (rrf_k + rank_b)) + (1.0 / (rrf_k + rank_t))
        rrf_scores[p_indices] = rrf_q

    # Random Baseline
    rng_rand = np.random.RandomState(42)
    random_scores = rng_rand.uniform(0, 1, size=len(pairs))

    # All model scores dictionary
    models_dict = {
        "Floor: Random Ranking": random_scores,
        "BM25 / TF-IDF (Lexical)": bm25_scores,
        "MiniLM Single-Vector (Flat)": flat_scores,
        "MiniLM Chunk-Mean (128w, stride 64)": chunk_mean_scores,
        "MiniLM Chunk-MaxSim (ColBERT-style)": chunk_maxsim_scores,
        "MiniLM Section-Aware Dense (Ours)": sec_scores,
        "Dense + BM25 (L0: No Taxonomy)": l0_scores,
        "Calibrated Hybrid (Dense+BM25+Tax)": hybrid_scores,
        "Parameter-Free RRF + Taxonomy": rrf_scores,
    }

    # 6. Evaluation across Setting A and Setting B
    print("\n--> [Step 6/6] Computing metrics and statistical significance ...")
    rng_eval = np.random.RandomState(42)

    def evaluate_query_set(q_list: List[int]) -> Dict[str, Any]:
        results = {}
        for m_name, m_scores in models_dict.items():
            per_q = []
            for q in q_list:
                p_idx = query_groups[q]
                if not p_idx:
                    continue
                q_res = evaluate_query_with_tie_breaking(
                    m_scores[p_idx], y_true_arr[p_idx], n_tie_shuffles=50, rng=rng_eval
                )
                per_q.append(q_res)
            avg_m = {k: float(np.mean([r[k] for r in per_q])) for k in per_q[0].keys()}
            results[m_name] = {
                "metrics": avg_m,
                "per_query_n10": [r["ndcg@10"] for r in per_q],
                "per_query_map1": [r["map_th1"] for r in per_q],
                "per_query_map2": [r["map_th2"] for r in per_q],
                "per_query_mrr1": [r["mrr_th1"] for r in per_q],
                "per_query_mrr2": [r["mrr_th2"] for r in per_q],
            }

        # Oracle
        or_per_q = []
        for q in q_list:
            p_idx = query_groups[q]
            if not p_idx:
                continue
            or_sc = y_true_arr[p_idx].astype(float)
            q_res = evaluate_query_with_tie_breaking(or_sc, y_true_arr[p_idx], n_tie_shuffles=20, rng=rng_eval)
            or_per_q.append(q_res)
        avg_or = {k: float(np.mean([r[k] for r in or_per_q])) for k in or_per_q[0].keys()}
        results["Ceiling: Oracle Ranking"] = {
            "metrics": avg_or,
            "per_query_n10": [r["ndcg@10"] for r in or_per_q],
            "per_query_map1": [r["map_th1"] for r in or_per_q],
            "per_query_map2": [r["map_th2"] for r in or_per_q],
            "per_query_mrr1": [r["mrr_th1"] for r in or_per_q],
            "per_query_mrr2": [r["mrr_th2"] for r in or_per_q],
        }
        return results

    # Evaluate Setting A and Setting B
    all_clean_q = list(range(n_queries))
    eval_setting_a = evaluate_query_set(all_clean_q)
    eval_setting_b_th1 = evaluate_query_set(pos_queries_th1)
    eval_setting_b_th2 = evaluate_query_set(pos_queries_th2)

    # 7. Statistical Significance Testing on Pre-Registered Family
    def compute_statistical_tests(eval_dict: Dict[str, Any], metric_key: str = "per_query_n10") -> List[Dict[str, Any]]:
        comparisons = [
            ("RQ1: Section-Aware vs Single-Vector", "MiniLM Section-Aware Dense (Ours)", "MiniLM Single-Vector (Flat)"),
            ("RQ2: BM25 vs Single-Vector", "BM25 / TF-IDF (Lexical)", "MiniLM Single-Vector (Flat)"),
            ("RQ3: Hybrid vs Dense+BM25 (nDCG@10)", "Calibrated Hybrid (Dense+BM25+Tax)", "Dense + BM25 (L0: No Taxonomy)"),
            ("Chunk-Mean vs Single-Vector", "MiniLM Chunk-Mean (128w, stride 64)", "MiniLM Single-Vector (Flat)"),
            ("Chunk-MaxSim vs Single-Vector", "MiniLM Chunk-MaxSim (ColBERT-style)", "MiniLM Single-Vector (Flat)"),
            ("RRF vs Calibrated Hybrid", "Parameter-Free RRF + Taxonomy", "Calibrated Hybrid (Dense+BM25+Tax)"),
        ]

        raw_tests = []
        rng_bs = np.random.RandomState(42)

        for comp_name, m_a, m_b in comparisons:
            a_vals = np.array(eval_dict[m_a][metric_key])
            b_vals = np.array(eval_dict[m_b][metric_key])
            diffs = a_vals - b_vals
            mean_diff = float(np.mean(diffs))

            # Bootstrap 95% CI (1000 resamples)
            boot_means = []
            for _ in range(1000):
                boot_idx = rng_bs.choice(len(diffs), size=len(diffs), replace=True)
                boot_means.append(np.mean(diffs[boot_idx]))
            ci_lower = float(np.percentile(boot_means, 2.5))
            ci_upper = float(np.percentile(boot_means, 97.5))

            # t-test & wilcoxon
            t_stat, p_ttest = ttest_rel(a_vals, b_vals)
            try:
                w_stat, p_wilc = wilcoxon(diffs, zero_method="wilcox")
            except Exception:
                p_wilc = 1.0

            raw_tests.append({
                "hypothesis": comp_name,
                "model_a": m_a,
                "model_b": m_b,
                "mean_diff": round(mean_diff, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "t_stat": round(float(t_stat), 3),
                "raw_p_val": float(p_ttest) if not np.isnan(p_ttest) else 1.0,
                "wilcoxon_p": round(float(p_wilc), 4),
            })

        # Holm-Bonferroni correction
        m_tests = len(raw_tests)
        sorted_indices = np.argsort([t["raw_p_val"] for t in raw_tests])
        adjusted_p = [1.0] * m_tests

        cum_max = 0.0
        for rank, idx in enumerate(sorted_indices, start=1):
            raw_p = raw_tests[idx]["raw_p_val"]
            multiplier = m_tests - rank + 1
            cur_adj = min(1.0, raw_p * multiplier)
            cum_max = max(cum_max, cur_adj)
            adjusted_p[idx] = cum_max

        for i, t in enumerate(raw_tests):
            t["holm_p_val"] = round(float(adjusted_p[i]), 4)
            t["significant_05"] = bool(adjusted_p[i] < 0.05)

        return raw_tests

    stats_setting_a = compute_statistical_tests(eval_setting_a)
    stats_setting_b_th1 = compute_statistical_tests(eval_setting_b_th1)
    stats_setting_b_th2 = compute_statistical_tests(eval_setting_b_th2)

    # 8. Save JSON
    final_output = {
        "dataset": "cnamuangtoun/resume-job-description-fit (76 Clean Held-Out Train Queries)",
        "protocol": "Pre-registered evaluation plan (docs/PRE_REGISTERED_TRAIN_EVAL_PLAN.md)",
        "char_coverage_stats": char_coverage_stats,
        "setting_A_all_76": {k: {"metrics": v["metrics"]} for k, v in eval_setting_a.items()},
        "setting_B_filtered_th1_71": {k: {"metrics": v["metrics"]} for k, v in eval_setting_b_th1.items()},
        "setting_B_filtered_th2_31": {k: {"metrics": v["metrics"]} for k, v in eval_setting_b_th2.items()},
        "statistical_tests_setting_A": stats_setting_a,
        "statistical_tests_setting_B_th1": stats_setting_b_th1,
        "statistical_tests_setting_B_th2": stats_setting_b_th2,
    }

    with open(RESULTS_DIR / "clean_train_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print("\nSuccessfully saved results to results/clean_train_benchmark_results.json")

    # 9. Print Summary Tables
    print("\n=== SETTING A (All 76 Clean Queries) ===")
    print(f"{'Model':42s} | {'nDCG@10':7s} | {'MRR':7s} | {'MAP':7s} | {'P@5':7s}")
    print("-" * 75)
    for m_name, d in eval_setting_a.items():
        m = d["metrics"]
        print(f"{m_name:42s} | {m['ndcg@10']:.4f}  | {m['mrr_th1']:.4f}  | {m['map_th1']:.4f}  | {m['p@5_th1']:.4f}")

    print("\n=== STATISTICAL TESTS (Setting A, Holm-Bonferroni Correction) ===")
    print(f"{'Hypothesis':36s} | {'Mean Diff':9s} | {'95% CI':18s} | {'Raw p':7s} | {'Holm p':7s} | {'Sig.'}")
    print("-" * 90)
    for st in stats_setting_a:
        print(f"{st['hypothesis']:36s} | {st['mean_diff']:+7.4f}   | [{st['ci_lower']:+7.4f}, {st['ci_upper']:+7.4f}] | {st['raw_p_val']:.4f}  | {st['holm_p_val']:.4f}  | {st['significant_05']}")

    print("\n=== SETTING B (Filtered rel >= 1, N=71 Queries) ===")
    print(f"{'Model':42s} | {'nDCG@10':7s} | {'MRR':7s} | {'MAP':7s} | {'P@5':7s}")
    print("-" * 75)
    for m_name, d in eval_setting_b_th1.items():
        m = d["metrics"]
        print(f"{m_name:42s} | {m['ndcg@10']:.4f}  | {m['mrr_th1']:.4f}  | {m['map_th1']:.4f}  | {m['p@5_th1']:.4f}")

    print("\n=== STATISTICAL TESTS (Setting B rel >= 1, Holm-Bonferroni Correction) ===")
    print(f"{'Hypothesis':36s} | {'Mean Diff':9s} | {'95% CI':18s} | {'Raw p':7s} | {'Holm p':7s} | {'Sig.'}")
    print("-" * 90)
    for st in stats_setting_b_th1:
        print(f"{st['hypothesis']:36s} | {st['mean_diff']:+7.4f}   | [{st['ci_lower']:+7.4f}, {st['ci_upper']:+7.4f}] | {st['raw_p_val']:.4f}  | {st['holm_p_val']:.4f}  | {st['significant_05']}")

    # 10. Generate LaTeX Table
    latex_table = r"""% Auto-generated by scripts/run_clean_train_benchmark.py
\begin{table*}[t]
\centering
\caption{Held-Out Clean Train Set Benchmark (76 JDs, 1,829 pairs): Generalization and Overfitting Audit across Setting A (All 76 queries) and Setting B (Filtered $N=71$ queries with $\text{rel}\ge 1$).}
\label{tab:clean_train_benchmark}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\hline
& \multicolumn{4}{c}{\textbf{Setting A (All 76 Clean Queries)}} & \multicolumn{4}{c}{\textbf{Setting B (Filtered $N=71$, $\text{rel}\ge 1$)}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
\textbf{Model / Architecture} & \textbf{nDCG@10} & \textbf{MRR} & \textbf{MAP} & \textbf{P@5} & \textbf{nDCG@10} & \textbf{MRR} & \textbf{MAP} & \textbf{P@5} \\
\hline
"""
    model_order = [
        "Floor: Random Ranking",
        "BM25 / TF-IDF (Lexical)",
        "MiniLM Single-Vector (Flat)",
        "MiniLM Chunk-Mean (128w, stride 64)",
        "MiniLM Chunk-MaxSim (ColBERT-style)",
        "MiniLM Section-Aware Dense (Ours)",
        "Dense + BM25 (L0: No Taxonomy)",
        "Calibrated Hybrid (Dense+BM25+Tax)",
        "Parameter-Free RRF + Taxonomy",
        "Ceiling: Oracle Ranking",
    ]

    for m in model_order:
        ma = eval_setting_a[m]["metrics"]
        mb = eval_setting_b_th1[m]["metrics"]
        latex_table += f"{m:35s} & {ma['ndcg@10']:.4f} & {ma['mrr_th1']:.4f} & {ma['map_th1']:.4f} & {ma['p@5_th1']:.4f} & {mb['ndcg@10']:.4f} & {mb['mrr_th1']:.4f} & {mb['map_th1']:.4f} & {mb['p@5_th1']:.4f} \\\\\n"

    latex_table += r"""\hline
\end{tabular}%
}
\end{table*}
"""
    with open(TABLES_DIR / "tab_clean_train_benchmark.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    print("Saved LaTeX table to paper/tables/tab_clean_train_benchmark.tex")

if __name__ == "__main__":
    main()
