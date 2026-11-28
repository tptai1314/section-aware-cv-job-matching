"""Direction #3: Leakage-Free Learning-to-Rank (LTR) Training & Multi-Dataset Scientific Benchmark.

Protocol:
1. Stacking-Leakage Prevention:
   - For Frozen MiniLM (`all-MiniLM-L6-v2`): Fits LTR on all 280 clean train JDs (5,432 pairs, 0% overlap with test.csv).
   - For Fine-Tuned MiniLM (`models/minilm-cvjd-ft`): Fits LTR strictly on the 50 held-out Validation JDs (866 pairs)
     that were UNSEEN during Bi-Encoder gradient descent, preventing in-sample Dense feature over-weighting!
2. Evaluates 3 Monotonic LTR Variants (`cvjd.ranking.ltr`):
   - `LTR-Linear (Monotonic Simplex)`: 16 interpretable convex weights learned via within-query CoSENT + L2 prior.
   - `LTR-LambdaMART (5-Fold Monotonic LightGBM)`: Non-linear tree ensemble with `monotone_constraints = +1`.
   - `LTR-Ensemble (Linear + LambdaMART)`: Smooth tie-free calibrated combination (0.65 Linear + 0.35 LambdaMART).
3. Comprehensive Evaluation:
   - In-Domain `test.csv`: Setting A (71 JDs), Setting B (rel>=1: 61 JDs; rel==2: 28 JDs), Clean CVs (409 CVs).
   - Zero-Shot External Datasets: `med2425` IT (7,418 pairs), `job_resume_fit` (2,385 pairs), `resume-score-details` (851 pairs).
   - Paired Bootstrap 95% CI, Paired t-test, Wilcoxon signed-rank test, and CPU latency (ms/query for N=25).

Outputs:
- results/ltr_benchmark_results.json
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.data.preprocessing import sections_to_dict, split_sections
from cvjd.data.taxonomy import get_taxonomy
from cvjd.ranking.engine import min_max
from cvjd.ranking.ltr import (
    FEATURE_NAMES,
    MonotonicLambdaMART,
    MonotonicLinearLTR,
    extract_query_feature_matrix,
)

PUBLIC_DIR = REPO_ROOT / "data" / "public_benchmark"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FT_MODEL_DIR = REPO_ROOT / "models" / "minilm-cvjd-ft"
LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
SEC_KEYS = ["skills", "experience", "education", "summary"]
SEC_WEIGHTS = {"skills": 0.40, "experience": 0.35, "education": 0.15, "summary": 0.10}


# ==============================================================================
# 1. METRICS & STATISTICAL TESTS
# ==============================================================================

def dcg_at_k(rels: Sequence[float], k: int) -> float:
    score = 0.0
    for i in range(min(k, len(rels))):
        score += (math.pow(2.0, float(rels[i])) - 1.0) / math.log2(i + 2.0)
    return score


def ndcg_at_k(ranked_rels: Sequence[float], ideal_rels: Sequence[float], k: int) -> float:
    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0.0:
        return 0.0
    return min(1.0, max(0.0, dcg_at_k(ranked_rels, k) / idcg))


def reciprocal_rank(ranked_bin: Sequence[int]) -> float:
    for rank, r in enumerate(ranked_bin, start=1):
        if r >= 1:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_bin: Sequence[int]) -> float:
    total_rel = sum(1 for r in ranked_bin if r >= 1)
    if total_rel <= 0:
        return 0.0
    hits = 0
    p_sum = 0.0
    for rank, r in enumerate(ranked_bin, start=1):
        if r >= 1:
            hits += 1
            p_sum += hits / rank
    return p_sum / total_rel


def precision_at_k(ranked_bin: Sequence[int], k: int) -> float:
    if k <= 0 or len(ranked_bin) == 0:
        return 0.0
    return sum(1 for r in ranked_bin[:k] if r >= 1) / float(min(k, len(ranked_bin)))


def evaluate_query_tie_aware(
    scores: np.ndarray,
    graded_rels: np.ndarray,
    bin_rels: np.ndarray,
    n_tie_shuffles: int = 50,
    rng: np.random.RandomState | None = None,
) -> Dict[str, float]:
    if rng is None:
        rng = np.random.RandomState(42)
    ideal_graded = sorted(graded_rels, reverse=True)
    acc: Dict[str, float] = defaultdict(float)
    n_cands = len(scores)
    for _ in range(n_tie_shuffles):
        jitter = rng.uniform(-1e-9, 1e-9, size=n_cands)
        order = np.argsort(-(scores + jitter))
        r_gr = graded_rels[order]
        r_bn = bin_rels[order]
        acc["ndcg@5"] += ndcg_at_k(r_gr, ideal_graded, 5)
        acc["ndcg@10"] += ndcg_at_k(r_gr, ideal_graded, 10)
        acc["mrr"] += reciprocal_rank(r_bn)
        acc["map"] += average_precision(r_bn)
        acc["p@5"] += precision_at_k(r_bn, 5)
    return {k: v / n_tie_shuffles for k, v in acc.items()}


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    arr_a = np.asarray(scores_a, dtype=float)
    arr_b = np.asarray(scores_b, dtype=float)
    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(diffs[idx])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return mean_diff, float(ci_low), float(ci_high)


# ==============================================================================
# 2. FEATURE EXTRACTION PIPELINE FOR ANY DATASET
# ==============================================================================

def compute_dense_and_components(
    model: SentenceTransformer,
    unique_jobs: List[str],
    unique_cvs: List[str],
    cv_parsed: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    job_embs = model.encode(unique_jobs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    cv_embs = model.encode(unique_cvs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    flat_scores = np.array([float(np.dot(job_embs[j], cv_embs[c])) for j, c in pairs], dtype=float)

    all_sec_texts = []
    sec_meta = []
    for c_idx, sd in enumerate(cv_parsed):
        for s_name in SEC_KEYS:
            if s_name in sd and len(sd[s_name].text.strip()) > 5:
                sec_meta.append((c_idx, s_name))
                all_sec_texts.append(sd[s_name].text[:1024])

    cv_sec_vecs: List[Dict[str, np.ndarray]] = [{} for _ in range(len(unique_cvs))]
    if all_sec_texts:
        sec_vecs_flat = model.encode(all_sec_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        for (c_idx, s_name), vec in zip(sec_meta, sec_vecs_flat):
            cv_sec_vecs[c_idx][s_name] = vec

    sec_scores = np.empty(len(pairs), dtype=float)
    sec_comp_list: List[Dict[str, float]] = []
    for idx, (j, c) in enumerate(pairs):
        jv = job_embs[j]
        c_secs = cv_sec_vecs[c]
        w_sim, tot_w = 0.0, 0.0
        comp: Dict[str, float] = {}
        for sn, w in SEC_WEIGHTS.items():
            if sn in c_secs:
                sim_val = float(np.dot(jv, c_secs[sn]))
                comp[sn] = sim_val
                w_sim += w * sim_val
                tot_w += w
        sec_scores[idx] = (w_sim / tot_w) if tot_w > 0.0 else flat_scores[idx]
        sec_comp_list.append(comp)

    return flat_scores, sec_scores, sec_comp_list


def compute_lexical_and_taxonomy(
    unique_jobs: List[str],
    unique_cvs: List[str],
    pairs: List[Tuple[int, int]],
    taxonomy: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    vec.fit(unique_jobs + unique_cvs)
    j_tfidf = vec.transform(unique_jobs)
    c_tfidf = vec.transform(unique_cvs)
    bm25_scores = np.array([float((j_tfidf[j].multiply(c_tfidf[c])).sum()) for j, c in pairs], dtype=float)

    j_skills = [taxonomy.classify_jd_skills(j) for j in unique_jobs]
    c_skills = [taxonomy.extract_skills_from_text(c) for c in unique_cvs]
    tax_must, tax_nice, tax_full = [], [], []
    for j, c in pairs:
        must, nice = j_skills[j]
        c_sk = c_skills[c]
        c_set = set(c_sk)
        if not must and not nice:
            tax_must.append(1.0)
            tax_nice.append(0.0)
            tax_full.append(1.0)
            continue
        cov_m = sum(
            1.0 if m in c_set else max([taxonomy.get_effective_relation_weight(cs, m) for cs in c_sk], default=0.0)
            for m in must
        )
        cov_n = sum(
            1.0 if n in c_set else max([taxonomy.get_effective_relation_weight(cs, n) for cs in c_sk], default=0.0)
            for n in nice
        )
        m_sc = cov_m / len(must) if must else 1.0
        n_sc = cov_n / len(nice) if nice else 0.0
        tax_must.append(m_sc)
        tax_nice.append(n_sc)
        tax_full.append(
            taxonomy.score_skills_coverage(
                must, nice, c_sk, must_weight=0.85, nice_weight=0.15, neutral_score=1.0
            )
        )

    return (
        bm25_scores,
        np.array(tax_must, dtype=float),
        np.array(tax_nice, dtype=float),
        np.array(tax_full, dtype=float),
    )


def build_heuristic_hybrid(
    sec_scores: np.ndarray,
    bm25_scores: np.ndarray,
    tax_full: np.ndarray,
    query_groups: Dict[int, List[int]],
) -> np.ndarray:
    n_sec = np.zeros_like(sec_scores)
    n_bm = np.zeros_like(bm25_scores)
    for _, p_idx in query_groups.items():
        n_sec[p_idx] = min_max(sec_scores[p_idx])
        n_bm[p_idx] = min_max(bm25_scores[p_idx])
    d_bm = 0.60 * n_sec + 0.40 * n_bm
    return 0.60 * d_bm + 0.40 * tax_full


def build_ltr_ensemble_scores(
    linear_ltr: MonotonicLinearLTR,
    mart_ltr: MonotonicLambdaMART,
    X: np.ndarray,
    query_groups: Dict[int, List[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_lin = linear_ltr.predict(X)
    s_mart = mart_ltr.predict(X)
    s_ens = np.zeros_like(s_lin)
    for _, p_idx in query_groups.items():
        if not p_idx:
            continue
        n_l = np.asarray(min_max(s_lin[p_idx]), dtype=np.float64)
        n_m = np.asarray(min_max(s_mart[p_idx]), dtype=np.float64)
        s_ens[p_idx] = 0.65 * n_l + 0.35 * n_m
    return s_lin, s_mart, s_ens


# ==============================================================================
# 3. MAIN RUNNER
# ==============================================================================

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 100, flush=True)
    print(f"DIRECTION #3: LEARNING-TO-RANK (LTR) TRAINING & BENCHMARK SUITE (Device: {device})", flush=True)
    print("=" * 100, flush=True)

    frozen_model = SentenceTransformer(BASE_MODEL_NAME, device=device)
    frozen_model.max_seq_length = 256
    ft_model = SentenceTransformer(str(FT_MODEL_DIR), device=device)
    ft_model.max_seq_length = 256
    taxonomy = get_taxonomy()
    taxonomy.attach_skill_embeddings(lambda xs: frozen_model.encode(xs, batch_size=64, normalize_embeddings=True))

    # 1. Prepare Clean Train & Held-out Val Splits (matching finetune_biencoder.py seed=42)
    train_df = pd.read_csv(PUBLIC_DIR / "resume-job-description-fit" / "train.csv", encoding="latin1")
    test_df = pd.read_csv(PUBLIC_DIR / "resume-job-description-fit" / "test.csv", encoding="latin1")
    train_df["rel"] = train_df["label"].map(LABEL_MAP)
    test_df["rel"] = test_df["label"].map(LABEL_MAP)

    test_cv_set = set(test_df["resume_text"].unique())
    test_jd_set = set(test_df["job_description_text"].unique())

    clean_train_df = train_df[
        (~train_df["resume_text"].isin(test_cv_set))
        & (~train_df["job_description_text"].isin(test_jd_set))
    ].copy()

    all_clean_jds = sorted(clean_train_df["job_description_text"].unique().tolist())
    rng_split = np.random.RandomState(42)
    shuffled_jds = list(all_clean_jds)
    rng_split.shuffle(shuffled_jds)

    val_jds_list = shuffled_jds[:50]  # Unseen by Fine-Tuned Bi-Encoder!
    val_sub_df = clean_train_df[clean_train_df["job_description_text"].isin(set(val_jds_list))].copy()

    def build_split_arrays(df_sub: pd.DataFrame):
        u_jobs = df_sub["job_description_text"].unique().tolist()
        u_cvs = df_sub["resume_text"].unique().tolist()
        j2i = {j: i for i, j in enumerate(u_jobs)}
        c2i = {c: i for i, c in enumerate(u_cvs)}
        cv_p = [sections_to_dict(split_sections(c, is_jd=False)) for c in u_cvs]
        prs, y_l = [], []
        q_grp: Dict[int, List[int]] = defaultdict(list)
        for _, r in df_sub.iterrows():
            ji, ci = j2i[r["job_description_text"]], c2i[r["resume_text"]]
            pi = len(prs)
            q_grp[ji].append(pi)
            prs.append((ji, ci))
            y_l.append(int(r["rel"]))
        return u_jobs, u_cvs, cv_p, prs, np.array(y_l, dtype=int), q_grp

    # Fit Frozen LTR on clean train (280 JDs, 5,432 pairs)
    print("\n--> [Step 1/4] Extracting features & training LTR for Frozen MiniLM (280 Clean JDs, 5,432 pairs) ...", flush=True)
    tr_jobs, tr_cvs, tr_cv_p, tr_pairs, tr_y, tr_qgrp = build_split_arrays(clean_train_df)
    tr_bm25, tr_tmust, tr_tnice, tr_tfull = compute_lexical_and_taxonomy(tr_jobs, tr_cvs, tr_pairs, taxonomy)
    fr_tr_flat, fr_tr_sec, fr_tr_comp = compute_dense_and_components(frozen_model, tr_jobs, tr_cvs, tr_cv_p, tr_pairs)
    X_tr_frozen = extract_query_feature_matrix(
        fr_tr_flat, fr_tr_sec, fr_tr_comp, tr_bm25, tr_tmust, tr_tnice, tr_tfull, tr_qgrp
    )

    ltr_linear_frozen = MonotonicLinearLTR(temperature=15.0, l2_prior_reg=0.25).fit(X_tr_frozen, tr_y, tr_qgrp)
    ltr_mart_frozen = MonotonicLambdaMART(n_estimators=80, learning_rate=0.03, num_leaves=7, max_depth=3).fit(
        X_tr_frozen, tr_y, tr_qgrp
    )

    # Fit Fine-Tuned LTR strictly on the 50 Bi-Encoder-Unseen Validation JDs (866 pairs) to avoid stacking leakage!
    print("--> [Step 2/4] Extracting out-of-sample features & training LTR for Fine-Tuned MiniLM (50 Unseen Val JDs, 866 pairs) ...", flush=True)
    va_jobs, va_cvs, va_cv_p, va_pairs, va_y, va_qgrp = build_split_arrays(val_sub_df)
    va_bm25, va_tmust, va_tnice, va_tfull = compute_lexical_and_taxonomy(va_jobs, va_cvs, va_pairs, taxonomy)
    ft_va_flat, ft_va_sec, ft_va_comp = compute_dense_and_components(ft_model, va_jobs, va_cvs, va_cv_p, va_pairs)
    X_va_ft = extract_query_feature_matrix(
        ft_va_flat, ft_va_sec, ft_va_comp, va_bm25, va_tmust, va_tnice, va_tfull, va_qgrp
    )

    ltr_linear_ft = MonotonicLinearLTR(temperature=15.0, l2_prior_reg=0.30).fit(X_va_ft, va_y, va_qgrp)
    ltr_mart_ft = MonotonicLambdaMART(n_estimators=70, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=15).fit(
        X_va_ft, va_y, va_qgrp
    )

    print("\nLearned Feature Weights (MonotonicLinearLTR - Fine-Tuned Backbone):", flush=True)
    for k, v in sorted(ltr_linear_ft.weight_summary().items(), key=lambda x: -x[1]):
        print(f"  {k:<18}: {v:.4f}", flush=True)

    print("\nLearned Feature Importance Gain (MonotonicLambdaMART - Fine-Tuned Backbone):", flush=True)
    for k, v in sorted(ltr_mart_ft.importance_summary().items(), key=lambda x: -x[1]):
        print(f"  {k:<18}: {v:.4f}", flush=True)

    # ==============================================================================
    # 4. EVALUATE ON IN-DOMAIN TEST.CSV (1,759 PAIRS, 71 JDs)
    # ==============================================================================
    print("\n--> [Step 3/4] Evaluating LTR on Frozen Test Set (`test.csv`: 1,759 pairs, 71 JDs) ...", flush=True)
    te_jobs, te_cvs, te_cv_p, te_pairs, te_y, te_qgrp = build_split_arrays(test_df)
    train_cv_set = set(train_df["resume_text"].unique())
    overlap_cv_indices = set(i for i, c in enumerate(te_cvs) if c in train_cv_set)

    te_bm25, te_tmust, te_tnice, te_tfull = compute_lexical_and_taxonomy(te_jobs, te_cvs, te_pairs, taxonomy)
    fr_te_flat, fr_te_sec, fr_te_comp = compute_dense_and_components(frozen_model, te_jobs, te_cvs, te_cv_p, te_pairs)
    ft_te_flat, ft_te_sec, ft_te_comp = compute_dense_and_components(ft_model, te_jobs, te_cvs, te_cv_p, te_pairs)

    X_te_frozen = extract_query_feature_matrix(
        fr_te_flat, fr_te_sec, fr_te_comp, te_bm25, te_tmust, te_tnice, te_tfull, te_qgrp
    )
    X_te_ft = extract_query_feature_matrix(
        ft_te_flat, ft_te_sec, ft_te_comp, te_bm25, te_tmust, te_tnice, te_tfull, te_qgrp
    )

    fr_hyb_heur = build_heuristic_hybrid(fr_te_sec, te_bm25, te_tfull, te_qgrp)
    ft_hyb_heur = build_heuristic_hybrid(ft_te_sec, te_bm25, te_tfull, te_qgrp)

    fr_lin, fr_mart, fr_ens = build_ltr_ensemble_scores(ltr_linear_frozen, ltr_mart_frozen, X_te_frozen, te_qgrp)
    ft_lin, ft_mart, ft_ens = build_ltr_ensemble_scores(ltr_linear_ft, ltr_mart_ft, X_te_ft, te_qgrp)

    # Measure CPU ranking latency for N=25 candidate pool
    X_sample_25 = X_te_ft[:25]
    for _ in range(20):
        _ = ltr_linear_ft.predict(X_sample_25)
        _ = ltr_mart_ft.predict(X_sample_25)

    t0 = time.perf_counter()
    for _ in range(500):
        _ = ltr_linear_ft.predict(X_sample_25)
    t_rank_linear_ms = (time.perf_counter() - t0) * 1000.0 / 500.0

    t0 = time.perf_counter()
    for _ in range(200):
        _ = ltr_mart_ft.predict(X_sample_25)
    t_rank_mart_ms = (time.perf_counter() - t0) * 1000.0 / 200.0
    print(f"CPU LTR Scoring Latency (N=25): LTR-Linear = {t_rank_linear_ms:.4f} ms | LTR-LambdaMART = {t_rank_mart_ms:.3f} ms", flush=True)

    test_models: Dict[str, np.ndarray] = {
        "Section-Aware Dense (Frozen MiniLM)": fr_te_sec,
        "Calibrated Hybrid Heuristic (Frozen MiniLM)": fr_hyb_heur,
        "LTR-Linear Simplex (Frozen MiniLM)": fr_lin,
        "LTR-LambdaMART 5-Fold (Frozen MiniLM)": fr_mart,
        "LTR-Ensemble (Frozen MiniLM)": fr_ens,
        "Section-Aware Dense (Fine-Tuned MiniLM)": ft_te_sec,
        "Calibrated Hybrid Heuristic (Fine-Tuned MiniLM)": ft_hyb_heur,
        "LTR-Linear Simplex (Fine-Tuned MiniLM)": ft_lin,
        "LTR-LambdaMART 5-Fold (Fine-Tuned MiniLM)": ft_mart,
        "LTR-Ensemble (Fine-Tuned MiniLM)": ft_ens,
    }

    def eval_test_subset(target_q: List[int], filter_fn=None, bin_threshold: int = 1) -> Dict[str, Any]:
        rng = np.random.RandomState(42)
        res_dict = {}
        q_to_p = {}
        for q in target_q:
            plist = te_qgrp[q]
            if filter_fn:
                plist = [p for p in plist if filter_fn(te_pairs[p])]
            q_to_p[q] = plist

        for m_name, m_sc in test_models.items():
            q_m, q_rho = [], []
            for q in target_q:
                p_idx = q_to_p[q]
                if not p_idx:
                    continue
                sc = m_sc[p_idx]
                gr = te_y[p_idx].astype(float)
                bn = (te_y[p_idx] >= bin_threshold).astype(int)
                q_m.append(evaluate_query_tie_aware(sc, gr, bn, n_tie_shuffles=50, rng=rng))
                if len(p_idx) >= 2 and len(np.unique(gr)) > 1 and len(np.unique(sc)) > 1:
                    rv, _ = spearmanr(sc, gr)
                    if not np.isnan(rv):
                        q_rho.append(float(rv))
            avg_m = {k: round(float(np.mean([r[k] for r in q_m])), 4) for k in q_m[0].keys()}
            all_p = [p for q in target_q for p in q_to_p[q]]
            rmic, _ = spearmanr(m_sc[all_p], te_y[all_p])
            rmac = round(float(np.mean(q_rho)), 4) if q_rho else 0.0
            res_dict[m_name] = {
                "metrics": avg_m,
                "spearman_micro": round(float(rmic), 4),
                "spearman_macro": rmac,
                "per_query_n10": [r["ndcg@10"] for r in q_m],
                "per_query_map": [r["map"] for r in q_m],
            }
        return res_dict

    all_te_q = list(range(len(te_qgrp)))
    pos_q_th1 = [q for q, p_idx in te_qgrp.items() if any(te_y[p] >= 1 for p in p_idx)]
    pos_q_th2 = [q for q, p_idx in te_qgrp.items() if any(te_y[p] == 2 for p in p_idx)]

    te_setting_a = eval_test_subset(all_te_q, bin_threshold=1)
    te_setting_b_th1 = eval_test_subset(pos_q_th1, bin_threshold=1)
    te_setting_b_th2 = eval_test_subset(pos_q_th2, bin_threshold=2)
    te_clean_cvs = eval_test_subset(all_te_q, filter_fn=lambda pair: pair[1] not in overlap_cv_indices, bin_threshold=1)

    print("\n" + "-" * 108, flush=True)
    print(f"{'In-Domain Test Set (Setting A: N=71 JDs)':<48} | {'nDCG@5':>7} | {'nDCG@10':>7} | {'MRR':>7} | {'MAP':>7} | {'P@5':>7} | {'Spearman':>8}", flush=True)
    print("-" * 108, flush=True)
    for m_name, d in te_setting_a.items():
        mt = d["metrics"]
        print(
            f"{m_name:<48} | {mt['ndcg@5']:>7.4f} | {mt['ndcg@10']:>7.4f} | "
            f"{mt['mrr']:>7.4f} | {mt['map']:>7.4f} | {mt['p@5']:>7.4f} | {d['spearman_macro']:>8.4f}",
            flush=True,
        )
    print("-" * 108, flush=True)

    # Statistical Significance Tests on Setting A
    stat_pairs = [
        (
            "LTR-Linear (Frozen) vs Heuristic Hybrid (Frozen)",
            "LTR-Linear Simplex (Frozen MiniLM)",
            "Calibrated Hybrid Heuristic (Frozen MiniLM)",
        ),
        (
            "LTR-Ensemble (Frozen) vs Heuristic Hybrid (Frozen)",
            "LTR-Ensemble (Frozen MiniLM)",
            "Calibrated Hybrid Heuristic (Frozen MiniLM)",
        ),
        (
            "LTR-Linear (FT) vs Heuristic Hybrid (FT)",
            "LTR-Linear Simplex (Fine-Tuned MiniLM)",
            "Calibrated Hybrid Heuristic (Fine-Tuned MiniLM)",
        ),
        (
            "LTR-Ensemble (FT) vs Heuristic Hybrid (FT)",
            "LTR-Ensemble (Fine-Tuned MiniLM)",
            "Calibrated Hybrid Heuristic (Fine-Tuned MiniLM)",
        ),
        (
            "LTR-Linear (FT) vs Heuristic Hybrid (Frozen Baseline)",
            "LTR-Linear Simplex (Fine-Tuned MiniLM)",
            "Calibrated Hybrid Heuristic (Frozen MiniLM)",
        ),
        (
            "LTR-Ensemble (FT) vs Heuristic Hybrid (Frozen Baseline)",
            "LTR-Ensemble (Fine-Tuned MiniLM)",
            "Calibrated Hybrid Heuristic (Frozen MiniLM)",
        ),
    ]
    in_domain_stats = {}
    print("\nPaired Statistical Tests on In-Domain Test Set (Setting A):", flush=True)
    for label, k_a, k_b in stat_pairs:
        a_n10, b_n10 = te_setting_a[k_a]["per_query_n10"], te_setting_a[k_b]["per_query_n10"]
        dn10, cl10, ch10 = paired_bootstrap_ci(a_n10, b_n10)
        _, pt10 = ttest_rel(a_n10, b_n10)
        try:
            _, pw10 = wilcoxon(a_n10, b_n10)
        except ValueError:
            pw10 = 1.0

        a_mp, b_mp = te_setting_a[k_a]["per_query_map"], te_setting_a[k_b]["per_query_map"]
        dmp, clm, chm = paired_bootstrap_ci(a_mp, b_mp)
        _, ptm = ttest_rel(a_mp, b_mp)
        try:
            _, pwm = wilcoxon(a_mp, b_mp)
        except ValueError:
            pwm = 1.0

        in_domain_stats[label] = {
            "ndcg@10_delta": round(dn10, 4),
            "ndcg@10_ci95": [round(cl10, 4), round(ch10, 4)],
            "ndcg@10_ttest_p": round(float(pt10), 5),
            "ndcg@10_wilcoxon_p": round(float(pw10), 5),
            "map_delta": round(dmp, 4),
            "map_ci95": [round(clm, 4), round(chm, 4)],
            "map_ttest_p": round(float(ptm), 5),
            "map_wilcoxon_p": round(float(pwm), 5),
        }
        print(
            f"  * {label:<54}: d_nDCG@10 = {dn10:+.4f} (95% CI [{cl10:+.4f}, {ch10:+.4f}], p={pt10:.4f}) | "
            f"d_MAP = {dmp:+.4f} (95% CI [{clm:+.4f}, {chm:+.4f}], p={ptm:.4f})",
            flush=True,
        )

    # ==============================================================================
    # 5. ZERO-SHOT CROSS-DATASET EVALUATION OF LEARNED LTR MODELS
    # ==============================================================================
    print("\n--> [Step 4/4] Evaluating Zero-Shot Cross-Dataset Generalization of LTR Models ...", flush=True)

    def eval_external_ltr(
        ds_name: str,
        u_jobs: List[str],
        u_cvs: List[str],
        prs: List[Tuple[int, int]],
        gr_rels: np.ndarray,
        bn_rels: np.ndarray,
    ) -> Dict[str, Any]:
        q_grp: Dict[int, List[int]] = defaultdict(list)
        for pi, (ji, _) in enumerate(prs):
            q_grp[ji].append(pi)
        valid_q = [q for q, idx in q_grp.items() if len(idx) >= 2]
        cv_p = [sections_to_dict(split_sections(c, is_jd=False)) for c in u_cvs]

        bm25, tmust, tnice, tfull = compute_lexical_and_taxonomy(u_jobs, u_cvs, prs, taxonomy)
        fr_fl, fr_sc, fr_cp = compute_dense_and_components(frozen_model, u_jobs, u_cvs, cv_p, prs)
        ft_fl, ft_sc, ft_cp = compute_dense_and_components(ft_model, u_jobs, u_cvs, cv_p, prs)

        X_fr = extract_query_feature_matrix(fr_fl, fr_sc, fr_cp, bm25, tmust, tnice, tfull, q_grp)
        X_ft = extract_query_feature_matrix(ft_fl, ft_sc, ft_cp, bm25, tmust, tnice, tfull, q_grp)

        fr_heur = build_heuristic_hybrid(fr_sc, bm25, tfull, q_grp)
        ft_heur = build_heuristic_hybrid(ft_sc, bm25, tfull, q_grp)
        fr_l, fr_m, fr_e = build_ltr_ensemble_scores(ltr_linear_frozen, ltr_mart_frozen, X_fr, q_grp)
        ft_l, ft_m, ft_e = build_ltr_ensemble_scores(ltr_linear_ft, ltr_mart_ft, X_ft, q_grp)

        ext_models = {
            "Section-Aware Dense (Frozen MiniLM)": fr_sc,
            "Calibrated Hybrid Heuristic (Frozen MiniLM)": fr_heur,
            "LTR-Linear Simplex (Frozen MiniLM)": fr_l,
            "LTR-Ensemble (Frozen MiniLM)": fr_e,
            "Section-Aware Dense (Fine-Tuned MiniLM)": ft_sc,
            "Calibrated Hybrid Heuristic (Fine-Tuned MiniLM)": ft_heur,
            "LTR-Linear Simplex (Fine-Tuned MiniLM)": ft_l,
            "LTR-LambdaMART 5-Fold (Fine-Tuned MiniLM)": ft_m,
            "LTR-Ensemble (Fine-Tuned MiniLM)": ft_e,
        }

        print(f"\n--- {ds_name} (JDs={len(valid_q)}, Pairs={len(prs)}) ---", flush=True)
        rng = np.random.RandomState(42)
        ds_res = {}
        for m_name, m_sc in ext_models.items():
            qm, qrho = [], []
            for q in valid_q:
                p_idx = q_grp[q]
                sc, gr, bn = m_sc[p_idx], gr_rels[p_idx], bn_rels[p_idx]
                qm.append(evaluate_query_tie_aware(sc, gr, bn, n_tie_shuffles=30, rng=rng))
                if len(np.unique(gr)) > 1 and len(np.unique(sc)) > 1:
                    rv, _ = spearmanr(sc, gr)
                    if not np.isnan(rv):
                        qrho.append(float(rv))
            avg_m = {k: round(float(np.mean([r[k] for r in qm])), 4) for k in qm[0].keys()}
            rmac = round(float(np.mean(qrho)), 4) if qrho else 0.0
            auc = round(float(roc_auc_score(bn_rels, m_sc)), 4) if len(np.unique(bn_rels)) == 2 else 0.0
            ds_res[m_name] = {"metrics": avg_m, "spearman_macro": rmac, "roc_auc": auc}
            print(
                f"  {m_name:<48} | nDCG@5: {avg_m['ndcg@5']:.4f} | nDCG@10: {avg_m['ndcg@10']:.4f} | "
                f"MAP: {avg_m['map']:.4f} | Spearman: {rmac:.4f} | AUC: {auc:.4f}",
                flush=True,
            )
        return ds_res

    train_cvs_str = set(train_df["resume_text"].astype(str).str.strip())
    train_jds_str = set(train_df["job_description_text"].astype(str).str.strip())

    # External 1: med2425 IT zero-overlap
    med_df = pd.read_parquet(PUBLIC_DIR / "med2425" / "test.parquet")
    med_df["rel"] = med_df["label"].map(LABEL_MAP)
    it_dom = {"software", "data", "ai"}
    med_it = med_df[med_df["resume_domain"].isin(it_dom) & med_df["jd_domain"].isin(it_dom)].copy()
    med_clean = med_it[
        (~med_it["resume"].astype(str).str.strip().isin(train_cvs_str))
        & (~med_it["jd"].astype(str).str.strip().isin(train_jds_str))
    ].copy()
    m_jobs = med_clean["jd"].unique().tolist()
    m_cvs = med_clean["resume"].unique().tolist()
    mj2i, mc2i = {j: i for i, j in enumerate(m_jobs)}, {c: i for i, c in enumerate(m_cvs)}
    m_pairs = [(mj2i[r["jd"]], mc2i[r["resume"]]) for _, r in med_clean.iterrows()]
    m_gr = med_clean["rel"].to_numpy(dtype=float)
    m_bn = (m_gr >= 1.0).astype(int)
    ext_med_res = eval_external_ltr("1. med2425 IT (Strict Zero-Overlap)", m_jobs, m_cvs, m_pairs, m_gr, m_bn)

    # External 2: job_resume_fit
    jrf_df = pd.read_csv(PUBLIC_DIR / "job_resume_fit" / "train.csv", encoding_errors="replace").dropna(
        subset=["resume_text", "job_text", "ai_match_score"]
    )
    j_jobs = jrf_df["job_text"].astype(str).unique().tolist()
    j_cvs = jrf_df["resume_text"].astype(str).unique().tolist()
    jj2i, jc2i = {j: i for i, j in enumerate(j_jobs)}, {c: i for i, c in enumerate(j_cvs)}
    j_pairs = [(jj2i[str(r["job_text"])], jc2i[str(r["resume_text"])]) for _, r in jrf_df.iterrows()]
    j_raw = jrf_df["ai_match_score"].to_numpy(dtype=float)
    j_gr = np.clip(j_raw / 50.0, 0.0, 2.0)
    j_bn = (j_raw >= 60.0).astype(int)
    ext_jrf_res = eval_external_ltr("2. job_resume_fit (0% Overlap)", j_jobs, j_cvs, j_pairs, j_gr, j_bn)

    # External 3: resume-score-details
    rsd_rows = []
    for fp in sorted((PUBLIC_DIR / "resume-score-details").glob("*.json")):
        d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        inp, out = d.get("input", {}), d.get("output", {})
        if not out.get("valid_resume_and_jd"):
            continue
        rt, jt = str(inp.get("resume") or "").strip(), str(inp.get("job_description") or "").strip()
        if not rt or not jt:
            continue
        agg = (out.get("scores") or {}).get("aggregated_scores") or {}
        if agg.get("macro_scores") is None or agg.get("micro_scores") is None:
            continue
        avg_g = 0.5 * (float(agg["macro_scores"]) + float(agg["micro_scores"]))
        ism = 1 if fp.name.startswith("match_") else (0 if fp.name.startswith("mismatch_") else int(avg_g >= 3.0))
        rsd_rows.append((jt, rt, avg_g, ism))
    r_jobs = list(dict.fromkeys([r[0] for r in rsd_rows]))
    r_cvs = list(dict.fromkeys([r[1] for r in rsd_rows]))
    rj2i, rc2i = {j: i for i, j in enumerate(r_jobs)}, {c: i for i, c in enumerate(r_cvs)}
    r_pairs = [(rj2i[r[0]], rc2i[r[1]]) for r in rsd_rows]
    r_gr = np.clip(np.array([r[2] for r in rsd_rows], dtype=float) * (2.0 / 5.0), 0.0, 2.0)
    r_bn = np.array([r[3] for r in rsd_rows], dtype=int)
    ext_rsd_res = eval_external_ltr("3. resume-score-details (GPT-4o, 0% Overlap)", r_jobs, r_cvs, r_pairs, r_gr, r_bn)

    full_report = {
        "feature_names": list(FEATURE_NAMES),
        "learned_weights_frozen_linear": ltr_linear_frozen.weight_summary(),
        "learned_weights_finetuned_linear": ltr_linear_ft.weight_summary(),
        "learned_importance_frozen_lambdamart": ltr_mart_frozen.importance_summary(),
        "learned_importance_finetuned_lambdamart": ltr_mart_ft.importance_summary(),
        "cpu_latency_ms_per_query_N25": {
            "ltr_linear_ms": round(t_rank_linear_ms, 4),
            "ltr_lambdamart_ms": round(t_rank_mart_ms, 3),
        },
        "in_domain_statistical_tests_setting_A": in_domain_stats,
        "in_domain_setting_A_all_71": {
            k: {mk: mv for mk, mv in v.items() if not mk.startswith("per_query_")}
            for k, v in te_setting_a.items()
        },
        "in_domain_setting_B_rel_ge_1": {
            k: {mk: mv for mk, mv in v.items() if not mk.startswith("per_query_")}
            for k, v in te_setting_b_th1.items()
        },
        "in_domain_setting_B_rel_eq_2": {
            k: {mk: mv for mk, mv in v.items() if not mk.startswith("per_query_")}
            for k, v in te_setting_b_th2.items()
        },
        "in_domain_clean_cvs_409": {
            k: {mk: mv for mk, mv in v.items() if not mk.startswith("per_query_")}
            for k, v in te_clean_cvs.items()
        },
        "external_zeroshot": {
            "med2425_it_zero_overlap": ext_med_res,
            "job_resume_fit": ext_jrf_res,
            "resume_score_details_gpt4o": ext_rsd_res,
        },
    }

    out_json = RESULTS_DIR / "ltr_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved complete LTR benchmark results to: {out_json}", flush=True)


if __name__ == "__main__":
    main()
