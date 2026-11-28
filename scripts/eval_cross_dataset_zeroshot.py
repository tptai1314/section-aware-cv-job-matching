"""Zero-Shot Cross-Dataset Generalization Benchmark for Fine-Tuned Bi-Encoder (Direction #1).

Evaluates `models/minilm-cvjd-ft` (trained ONLY on clean `cnamuangtoun/resume-job-description-fit` train split)
against Frozen `sentence-transformers/all-MiniLM-L6-v2` with ZERO additional training on 3 independent external datasets:

1. `med2425/resume-job-fit-merged-v1` (IT domain: software/data/ai, Qwen2.5-32B labels)
   - Strictly removes all CVs overlapping with `cnamuangtoun/train.csv` (0% leakage -> 7,418 unseen pairs, 41 JDs).
2. `job_resume_fit/train.csv` (23 JDs, 2,385 pairs, 0% overlap, continuous `ai_match_score` 0-100).
3. `resume-score-details/` (26 JDs, 851 valid pairs, 0% overlap, GPT-4o macro/micro rubric scores + match/mismatch).

Outputs:
- results/cross_dataset_zeroshot_results.json
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
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

PUBLIC_DIR = REPO_ROOT / "data" / "public_benchmark"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FT_MODEL_DIR = REPO_ROOT / "models" / "minilm-cvjd-ft"

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
    n_tie_shuffles: int = 30,
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
        r_graded = graded_rels[order]
        r_bin = bin_rels[order]

        acc["ndcg@5"] += ndcg_at_k(r_graded, ideal_graded, 5)
        acc["ndcg@10"] += ndcg_at_k(r_graded, ideal_graded, 10)
        acc["mrr"] += reciprocal_rank(r_bin)
        acc["map"] += average_precision(r_bin)
        acc["p@5"] += precision_at_k(r_bin, 5)

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
# 2. SHARED EVALUATION ENGINE
# ==============================================================================

def compute_dense_scores(
    model: SentenceTransformer,
    unique_jobs: List[str],
    unique_cvs: List[str],
    cv_parsed: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
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
    for idx, (j, c) in enumerate(pairs):
        jv = job_embs[j]
        c_secs = cv_sec_vecs[c]
        w_sim, tot_w = 0.0, 0.0
        for sn, w in SEC_WEIGHTS.items():
            if sn in c_secs:
                w_sim += w * float(np.dot(jv, c_secs[sn]))
                tot_w += w
        sec_scores[idx] = (w_sim / tot_w) if tot_w > 0.0 else flat_scores[idx]

    return flat_scores, sec_scores


def run_benchmark_on_dataset(
    name: str,
    unique_jobs: List[str],
    unique_cvs: List[str],
    pairs: List[Tuple[int, int]],
    graded_rels: np.ndarray,
    bin_rels: np.ndarray,
    frozen_model: SentenceTransformer,
    ft_model: SentenceTransformer,
    taxonomy: Any,
    compute_auc: bool = False,
) -> Dict[str, Any]:
    print("\n" + "=" * 95, flush=True)
    print(f"EXTERNAL ZERO-SHOT BENCHMARK: {name}", flush=True)
    print(f"Unique JDs: {len(unique_jobs)} | Unique CVs: {len(unique_cvs)} | Pairs: {len(pairs)}", flush=True)
    print("=" * 95, flush=True)

    query_groups: Dict[int, List[int]] = defaultdict(list)
    for p_idx, (j_i, _) in enumerate(pairs):
        query_groups[j_i].append(p_idx)

    # Filter queries with >= 2 candidates for ranking evaluation
    valid_queries = [q for q, p_idx in query_groups.items() if len(p_idx) >= 2]
    print(f"Ranking queries (>=2 candidates): {len(valid_queries)} / {len(query_groups)}", flush=True)

    # Parse CV sections
    cv_parsed = [sections_to_dict(split_sections(c, is_jd=False)) for c in unique_cvs]
    fallback_cnt = sum(1 for sd in cv_parsed if not any(k in sd for k in SEC_KEYS))
    print(f"Section Parser CV Fallback: {fallback_cnt}/{len(unique_cvs)} ({fallback_cnt/max(1,len(unique_cvs))*100:.2f}%)", flush=True)

    # BM25 / TF-IDF
    vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    vec.fit(unique_jobs + unique_cvs)
    j_tfidf = vec.transform(unique_jobs)
    c_tfidf = vec.transform(unique_cvs)
    bm25_scores = np.array([float((j_tfidf[j].multiply(c_tfidf[c])).sum()) for j, c in pairs], dtype=float)

    # Taxonomy Asymmetric Coverage (L1+L2+L3 Must/Nice)
    j_skills = [taxonomy.classify_jd_skills(j) for j in unique_jobs]
    c_skills = [set(taxonomy.extract_skills_from_text(c)) for c in unique_cvs]
    tax_list = []
    for j, c in pairs:
        must, nice = j_skills[j]
        c_sk = c_skills[c]
        if not must and not nice:
            tax_list.append(1.0)
            continue
        cov_m = sum(
            1.0 if m in c_sk else max([taxonomy.get_relation_weight(cs, m) for cs in c_sk], default=0.0)
            for m in must
        )
        m_sc = cov_m / len(must) if must else 1.0
        n_sc = sum(1.0 for n in nice if n in c_sk) / len(nice) if nice else 0.0
        tax_list.append(0.85 * m_sc + 0.15 * n_sc)
    tax_scores = np.array(tax_list, dtype=float)

    # Dense scores (Frozen vs Fine-Tuned)
    fr_flat, fr_sec = compute_dense_scores(frozen_model, unique_jobs, unique_cvs, cv_parsed, pairs)
    ft_flat, ft_sec = compute_dense_scores(ft_model, unique_jobs, unique_cvs, cv_parsed, pairs)

    def build_hybrids(sec_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_sec = np.zeros_like(sec_arr)
        n_bm = np.zeros_like(bm25_scores)
        rrf_raw = np.zeros_like(sec_arr)
        for q, p_idx in query_groups.items():
            sub_s = sec_arr[p_idx]
            sub_b = bm25_scores[p_idx]
            n_sec[p_idx] = min_max(sub_s)
            n_bm[p_idx] = min_max(sub_b)

            r_s = np.empty_like(sub_s, dtype=int)
            r_s[np.argsort(-sub_s)] = np.arange(1, len(sub_s) + 1)
            r_b = np.empty_like(sub_b, dtype=int)
            r_b[np.argsort(-sub_b)] = np.arange(1, len(sub_b) + 1)
            rrf_raw[p_idx] = min_max((1.0 / (60 + r_s)) + (1.0 / (60 + r_b)))
        d_bm = 0.60 * n_sec + 0.40 * n_bm
        cal_h = 0.60 * d_bm + 0.40 * tax_scores
        rrf_h = 0.60 * rrf_raw + 0.40 * tax_scores
        return d_bm, cal_h, rrf_h

    fr_dbm, fr_hyb, fr_rrf = build_hybrids(fr_sec)
    ft_dbm, ft_hyb, ft_rrf = build_hybrids(ft_sec)

    models: Dict[str, np.ndarray] = {
        "BM25 / TF-IDF (Lexical)": bm25_scores,
        "Single-Vector Dense (Frozen MiniLM)": fr_flat,
        "Single-Vector Dense (Fine-Tuned MiniLM)": ft_flat,
        "Section-Aware Dense (Frozen MiniLM)": fr_sec,
        "Section-Aware Dense (Fine-Tuned MiniLM)": ft_sec,
        "Dense + BM25 (Frozen MiniLM)": fr_dbm,
        "Dense + BM25 (Fine-Tuned MiniLM)": ft_dbm,
        "Calibrated Hybrid (Frozen MiniLM)": fr_hyb,
        "Calibrated Hybrid (Fine-Tuned MiniLM)": ft_hyb,
        "Parameter-Free RRF + Tax (Frozen MiniLM)": fr_rrf,
        "Parameter-Free RRF + Tax (Fine-Tuned MiniLM)": ft_rrf,
    }

    rng = np.random.RandomState(42)
    results_summary: Dict[str, Any] = {}
    per_q_store: Dict[str, Dict[str, List[float]]] = {}

    header = f"{'Model / Pipeline':<45} | {'nDCG@5':>7} | {'nDCG@10':>7} | {'MRR':>7} | {'MAP':>7} | {'P@5':>7} | {'Spearman':>8}"
    if compute_auc:
        header += f" | {'ROC-AUC':>7}"
    print("-" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for m_name, m_scores in models.items():
        q_metrics = []
        q_rhos = []
        for q in valid_queries:
            p_idx = query_groups[q]
            q_sc = m_scores[p_idx]
            q_gr = graded_rels[p_idx]
            q_bn = bin_rels[p_idx]
            q_res = evaluate_query_tie_aware(q_sc, q_gr, q_bn, n_tie_shuffles=30, rng=rng)
            q_metrics.append(q_res)
            if len(np.unique(q_gr)) > 1 and len(np.unique(q_sc)) > 1:
                r_val, _ = spearmanr(q_sc, q_gr)
                if not np.isnan(r_val):
                    q_rhos.append(float(r_val))

        avg_m = {k: float(np.mean([r[k] for r in q_metrics])) for k in q_metrics[0].keys()}
        rho_mic, _ = spearmanr(m_scores, graded_rels)
        rho_mac = float(np.mean(q_rhos)) if q_rhos else 0.0

        entry: Dict[str, Any] = {
            "metrics": {k: round(v, 4) for k, v in avg_m.items()},
            "spearman_micro": round(float(rho_mic), 4),
            "spearman_macro": round(float(rho_mac), 4),
        }
        line = (
            f"{m_name:<45} | {avg_m['ndcg@5']:>7.4f} | {avg_m['ndcg@10']:>7.4f} | "
            f"{avg_m['mrr']:>7.4f} | {avg_m['map']:>7.4f} | {avg_m['p@5']:>7.4f} | {rho_mac:>8.4f}"
        )
        if compute_auc and len(np.unique(bin_rels)) == 2:
            auc = float(roc_auc_score(bin_rels, m_scores))
            entry["roc_auc"] = round(auc, 4)
            line += f" | {auc:>7.4f}"

        print(line, flush=True)
        results_summary[m_name] = entry
        per_q_store[m_name] = {
            "ndcg@10": [r["ndcg@10"] for r in q_metrics],
            "map": [r["map"] for r in q_metrics],
        }
    print("-" * len(header), flush=True)

    # Paired Statistical Tests (Fine-Tuned vs Frozen)
    pairs_to_test = [
        ("Single-Vector Dense", "Single-Vector Dense (Fine-Tuned MiniLM)", "Single-Vector Dense (Frozen MiniLM)"),
        ("Section-Aware Dense", "Section-Aware Dense (Fine-Tuned MiniLM)", "Section-Aware Dense (Frozen MiniLM)"),
        ("Dense + BM25", "Dense + BM25 (Fine-Tuned MiniLM)", "Dense + BM25 (Frozen MiniLM)"),
        ("Calibrated Hybrid", "Calibrated Hybrid (Fine-Tuned MiniLM)", "Calibrated Hybrid (Frozen MiniLM)"),
        ("Parameter-Free RRF + Tax", "Parameter-Free RRF + Tax (Fine-Tuned MiniLM)", "Parameter-Free RRF + Tax (Frozen MiniLM)"),
    ]
    stat_tests = {}
    print("Paired Statistical Comparison (Fine-Tuned vs. Frozen):", flush=True)
    for label, ft_k, fr_k in pairs_to_test:
        ft_n10, fr_n10 = per_q_store[ft_k]["ndcg@10"], per_q_store[fr_k]["ndcg@10"]
        dn10, cl10, ch10 = paired_bootstrap_ci(ft_n10, fr_n10)
        _, pt10 = ttest_rel(ft_n10, fr_n10)
        try:
            _, pw10 = wilcoxon(ft_n10, fr_n10)
        except ValueError:
            pw10 = 1.0

        ft_mp, fr_mp = per_q_store[ft_k]["map"], per_q_store[fr_k]["map"]
        dmp, clm, chm = paired_bootstrap_ci(ft_mp, fr_mp)
        _, ptm = ttest_rel(ft_mp, fr_mp)
        try:
            _, pwm = wilcoxon(ft_mp, fr_mp)
        except ValueError:
            pwm = 1.0

        stat_tests[label] = {
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
            f"  * {label:<26}: d_nDCG@10 = {dn10:+.4f} (95% CI [{cl10:+.4f}, {ch10:+.4f}], p={pt10:.4f}) | "
            f"d_MAP = {dmp:+.4f} (95% CI [{clm:+.4f}, {chm:+.4f}], p={ptm:.4f})",
            flush=True,
        )

    return {
        "counts": {
            "unique_jds": len(unique_jobs),
            "ranking_queries": len(valid_queries),
            "unique_cvs": len(unique_cvs),
            "pairs": len(pairs),
            "cv_fallback_pct": round(fallback_cnt / max(1, len(unique_cvs)) * 100.0, 2),
        },
        "models": results_summary,
        "paired_statistical_tests": stat_tests,
    }


# ==============================================================================
# 3. DATASET LOADERS (WITH STRICT ZERO-LEAKAGE CHECK AGAINST TRAIN.CSV)
# ==============================================================================

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on device: {device}", flush=True)
    frozen_model = SentenceTransformer(BASE_MODEL_NAME, device=device)
    frozen_model.max_seq_length = 256
    ft_model = SentenceTransformer(str(FT_MODEL_DIR), device=device)
    ft_model.max_seq_length = 256
    taxonomy = get_taxonomy()

    # Load cnamuangtoun train texts strictly to filter any overlap
    train_df = pd.read_csv(PUBLIC_DIR / "resume-job-description-fit" / "train.csv", encoding="latin1")
    train_cvs = set(train_df["resume_text"].astype(str).str.strip())
    train_jds = set(train_df["job_description_text"].astype(str).str.strip())

    report: Dict[str, Any] = {
        "base_model": BASE_MODEL_NAME,
        "finetuned_model": str(FT_MODEL_DIR),
        "benchmarks": {},
    }

    # ------------------------------------------------------------------
    # Dataset 1: med2425 (IT domain: software, data, ai) - Strict Zero-Overlap
    # ------------------------------------------------------------------
    med_df = pd.read_parquet(PUBLIC_DIR / "med2425" / "test.parquet")
    label_map_med = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
    med_df["rel"] = med_df["label"].map(label_map_med)
    it_domains = {"software", "data", "ai"}
    med_it = med_df[med_df["resume_domain"].isin(it_domains) & med_df["jd_domain"].isin(it_domains)].copy()

    # Strict zero-overlap filter against cnamuangtoun/train.csv
    med_it_clean = med_it[
        (~med_it["resume"].astype(str).str.strip().isin(train_cvs))
        & (~med_it["jd"].astype(str).str.strip().isin(train_jds))
    ].copy()

    med_jobs = med_it_clean["jd"].unique().tolist()
    med_cvs = med_it_clean["resume"].unique().tolist()
    j2i = {j: i for i, j in enumerate(med_jobs)}
    c2i = {c: i for i, c in enumerate(med_cvs)}
    med_pairs = [(j2i[r["jd"]], c2i[r["resume"]]) for _, r in med_it_clean.iterrows()]
    med_graded = med_it_clean["rel"].to_numpy(dtype=float)
    med_bin = (med_graded >= 1.0).astype(int)

    report["benchmarks"]["med2425_it_zero_overlap"] = run_benchmark_on_dataset(
        name="1. med2425 IT Domain (Qwen2.5-32B Judge, Strict Zero-Overlap with Train)",
        unique_jobs=med_jobs,
        unique_cvs=med_cvs,
        pairs=med_pairs,
        graded_rels=med_graded,
        bin_rels=med_bin,
        frozen_model=frozen_model,
        ft_model=ft_model,
        taxonomy=taxonomy,
        compute_auc=True,
    )

    # ------------------------------------------------------------------
    # Dataset 2: job_resume_fit (Continuous AI Match Score 0-100, 0% overlap)
    # ------------------------------------------------------------------
    jrf_df = pd.read_csv(PUBLIC_DIR / "job_resume_fit" / "train.csv", encoding_errors="replace")
    jrf_df = jrf_df.dropna(subset=["resume_text", "job_text", "ai_match_score"]).copy()
    jrf_df = jrf_df[
        (~jrf_df["resume_text"].astype(str).str.strip().isin(train_cvs))
        & (~jrf_df["job_text"].astype(str).str.strip().isin(train_jds))
    ].copy()

    jrf_jobs = jrf_df["job_text"].astype(str).unique().tolist()
    jrf_cvs = jrf_df["resume_text"].astype(str).unique().tolist()
    j2i = {j: i for i, j in enumerate(jrf_jobs)}
    c2i = {c: i for i, c in enumerate(jrf_cvs)}
    jrf_pairs = [(j2i[str(r["job_text"])], c2i[str(r["resume_text"])]) for _, r in jrf_df.iterrows()]
    # Map continuous score 0..100 into 0..2 continuous graded scale (compatible with 2^rel - 1 DCG)
    jrf_raw_score = jrf_df["ai_match_score"].to_numpy(dtype=float)
    jrf_graded = np.clip(jrf_raw_score / 50.0, 0.0, 2.0)
    jrf_bin = (jrf_raw_score >= 60.0).astype(int)

    report["benchmarks"]["job_resume_fit"] = run_benchmark_on_dataset(
        name="2. job_resume_fit (Continuous AI Match Score 0-100, 0% Overlap)",
        unique_jobs=jrf_jobs,
        unique_cvs=jrf_cvs,
        pairs=jrf_pairs,
        graded_rels=jrf_graded,
        bin_rels=jrf_bin,
        frozen_model=frozen_model,
        ft_model=ft_model,
        taxonomy=taxonomy,
        compute_auc=True,
    )

    # ------------------------------------------------------------------
    # Dataset 3: resume-score-details (GPT-4o Rubric Scores + Match/Mismatch)
    # ------------------------------------------------------------------
    rsd_dir = PUBLIC_DIR / "resume-score-details"
    rsd_rows = []
    for fp in sorted(rsd_dir.glob("*.json")):
        d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        inp = d.get("input", {})
        out = d.get("output", {})
        if not out.get("valid_resume_and_jd"):
            continue
        res_txt = str(inp.get("resume") or "").strip()
        jd_txt = str(inp.get("job_description") or "").strip()
        if not res_txt or not jd_txt:
            continue
        if res_txt in train_cvs or jd_txt in train_jds:
            continue
        agg = (out.get("scores") or {}).get("aggregated_scores") or {}
        macro = agg.get("macro_scores")
        micro = agg.get("micro_scores")
        if macro is None or micro is None:
            continue
        avg_gpt = 0.5 * (float(macro) + float(micro))  # scale 0..5
        if fp.name.startswith("match_"):
            is_match = 1
        elif fp.name.startswith("mismatch_"):
            is_match = 0
        else:
            is_match = 1 if avg_gpt >= 3.0 else 0
        rsd_rows.append((jd_txt, res_txt, avg_gpt, is_match))

    rsd_jobs = list(dict.fromkeys([r[0] for r in rsd_rows]))
    rsd_cvs = list(dict.fromkeys([r[1] for r in rsd_rows]))
    j2i = {j: i for i, j in enumerate(rsd_jobs)}
    c2i = {c: i for i, c in enumerate(rsd_cvs)}
    rsd_pairs = [(j2i[r[0]], c2i[r[1]]) for r in rsd_rows]
    # Scale GPT-4o 0..5 -> 0..2 graded scale
    rsd_graded = np.clip(np.array([r[2] for r in rsd_rows], dtype=float) * (2.0 / 5.0), 0.0, 2.0)
    rsd_bin = np.array([r[3] for r in rsd_rows], dtype=int)

    report["benchmarks"]["resume_score_details_gpt4o"] = run_benchmark_on_dataset(
        name="3. resume-score-details (GPT-4o Rubric Scores & Match/Mismatch, 0% Overlap)",
        unique_jobs=rsd_jobs,
        unique_cvs=rsd_cvs,
        pairs=rsd_pairs,
        graded_rels=rsd_graded,
        bin_rels=rsd_bin,
        frozen_model=frozen_model,
        ft_model=ft_model,
        taxonomy=taxonomy,
        compute_auc=True,
    )

    out_path = RESULTS_DIR / "cross_dataset_zeroshot_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full cross-dataset zero-shot report to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
