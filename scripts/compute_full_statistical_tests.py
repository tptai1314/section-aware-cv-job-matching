"""Compute full statistical test suite for the primary comparison family.

Includes:
- Paired differences per query for all key comparison pairs.
- Paired t-test (t-stat, raw p-value).
- Wilcoxon signed-rank test (raw p-value).
- Paired bootstrap 95% confidence interval for Delta (B=10,000 resamples).
- Holm-Bonferroni correction across the primary hypothesis family.
- Generates LaTeX table: paper/tables/tab_statistical_tests.tex.
- Saves results to: results/statistical_significance_family.json.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "public_benchmark" / "resume-job-description-fit"
CACHE_DIR = REPO_ROOT / "data" / "cache_embeddings"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


def dcg_at_k(rels: List[int], k: int) -> float:
    score = 0.0
    for i in range(min(k, len(rels))):
        score += (math.pow(2.0, float(rels[i])) - 1.0) / math.log2(i + 2.0)
    return score


def ndcg_at_k(ranked_rels: List[int], ideal_rels: List[int], k: int) -> float:
    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0.0:
        return 0.0
    return min(1.0, max(0.0, dcg_at_k(ranked_rels, k) / idcg))


def reciprocal_rank(ranked_rels: List[int], threshold: int = 1) -> float:
    for rank, r in enumerate(ranked_rels, start=1):
        if r >= threshold:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_rels: List[int], threshold: int = 1) -> float:
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


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        sample = diffs[rng.randint(0, n, size=n)]
        boot_means[b] = np.mean(sample)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return low, high


def holm_bonferroni_correction(raw_p_values: List[Tuple[str, float]], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    sorted_p = sorted(raw_p_values, key=lambda x: x[1])
    m = len(sorted_p)
    corrected = {}
    cum_max = 0.0

    for rank, (name, p_val) in enumerate(sorted_p, start=1):
        hb_threshold = alpha / (m - rank + 1)
        adj_p = min(1.0, p_val * (m - rank + 1))
        cum_max = max(cum_max, adj_p)
        corrected[name] = {
            "rank": rank,
            "raw_p": round(float(p_val), 5),
            "hb_threshold": round(float(hb_threshold), 5),
            "adj_p": round(float(cum_max), 5),
            "significant": bool(p_val <= hb_threshold),
        }
    return corrected


def main() -> None:
    # 1. Load Data
    test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="latin1")
    test_df["rel"] = test_df["label"].map(LABEL_MAP)

    unique_jobs = test_df["job_description_text"].unique().tolist()
    job_to_idx = {j: i for i, j in enumerate(unique_jobs)}
    unique_cvs = test_df["resume_text"].unique().tolist()
    cv_to_idx = {r: i for i, r in enumerate(unique_cvs)}

    pairs: List[Tuple[int, int]] = []
    y_true: List[int] = []
    query_groups: Dict[int, List[int]] = defaultdict(list)

    for _, row in test_df.iterrows():
        j_idx = job_to_idx[row["job_description_text"]]
        c_idx = cv_to_idx[row["resume_text"]]
        p_idx = len(pairs)
        query_groups[j_idx].append(p_idx)
        pairs.append((j_idx, c_idx))
        y_true.append(row["rel"])

    y_true_arr = np.array(y_true, dtype=int)
    n_queries = len(query_groups)

    # Filtered queries with >= 1 positive
    pos_queries_th1 = [q for q, p_idx in query_groups.items() if any(y_true_arr[p] >= 1 for p in p_idx)]

    # 2. Load benchmark predictions from results or recompute from caches
    with open(RESULTS_DIR / "scientific_benchmark_results.json", "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    # 3. Model score mappings
    # We can reconstruct per-query metrics for any metric name
    # Let's inspect the models we want to compare
    model_keys = {
        "BM25": "BM25 / TF-IDF (Lexical)",
        "Single-Vector": "MiniLM Single-Vector (Flat, 256 tok)",
        "Chunk-Mean": "MiniLM Chunk-Mean (128w, stride 64)",
        "Chunk-MaxSim": "MiniLM Chunk-MaxSim (ColBERT-style)",
        "Section-Aware": "MiniLM Section-Aware (Variant 1: Pure)",
        "BGE-M3 Flat": "BGE-M3 Single-Vector (Flat, 8192 tok)",
        "BGE-M3 Section": "BGE-M3 Section-Aware (8192 tok)",
        "Dense+BM25": "Dense + BM25 (L0: No Taxonomy)",
        "Calibrated Hybrid": "Calibrated Hybrid (Dense+BM25+Tax)",
        "RRF+Taxonomy": "Parameter-Free RRF + Taxonomy",
        "Cross-Encoder": "Full Cross-Encoder (ms-marco-MiniLM)",
        "2-Stage Top10": "2-Stage Rerank: Hybrid -> CE (Top-10)",
    }

    # Extract per-query ndcg@10 and map_th1 from scientific_benchmark_results.json
    per_query_data = {}
    for short_name, json_key in model_keys.items():
        per_query_data[f"{short_name}__ndcg10_all"] = np.array(bench_data["setting_A_all_71_queries"][json_key]["per_query_n10"])
        per_query_data[f"{short_name}__map_all"] = np.array(bench_data["setting_A_all_71_queries"][json_key]["per_query_map1"])
        per_query_data[f"{short_name}__ndcg10_fil"] = np.array(bench_data["setting_B_filtered_rel_ge_1"][json_key]["per_query_n10"])
        per_query_data[f"{short_name}__map_fil"] = np.array(bench_data["setting_B_filtered_rel_ge_1"][json_key]["per_query_map1"])

    # 4. Define Primary Hypothesis Family
    # Each entry: (Hypothesis Name, RQ, Model_A, Model_B, Metric, Description)
    hypotheses = [
        # RQ1: Section-Aware vs Baselines
        ("H1_rq1_sec_vs_single_ndcg", "RQ1", "Section-Aware", "Single-Vector", "ndcg10", "Section-Aware vs Single-Vector (nDCG@10)"),
        ("H2_rq1_sec_vs_single_map", "RQ1", "Section-Aware", "Single-Vector", "map", "Section-Aware vs Single-Vector (MAP)"),
        ("H3_rq1_bgem3_sec_vs_flat_ndcg", "RQ1", "BGE-M3 Section", "BGE-M3 Flat", "ndcg10", "BGE-M3 Section vs Flat (nDCG@10)"),
        ("H4_rq1_bgem3_sec_vs_flat_map", "RQ1", "BGE-M3 Section", "BGE-M3 Flat", "map", "BGE-M3 Section vs Flat (MAP)"),
        ("H5_rq1_chunk_mean_vs_single_ndcg", "RQ1", "Chunk-Mean", "Single-Vector", "ndcg10", "Chunk-Mean vs Single-Vector (nDCG@10)"),
        ("H6_rq1_chunk_maxsim_vs_single_ndcg", "RQ1", "Chunk-MaxSim", "Single-Vector", "ndcg10", "Chunk-MaxSim vs Single-Vector (nDCG@10)"),
        # RQ2: Lexical vs Dense vs Hybrid
        ("H7_rq2_single_vs_bm25_ndcg", "RQ2", "Single-Vector", "BM25", "ndcg10", "Single-Vector vs BM25 (nDCG@10)"),
        ("H8_rq2_single_vs_bm25_map", "RQ2", "Single-Vector", "BM25", "map", "Single-Vector vs BM25 (MAP)"),
        ("H9_rq2_dense_bm25_vs_single_map", "RQ2", "Dense+BM25", "Single-Vector", "map", "Dense+BM25 vs Single-Vector (MAP)"),
        ("H10_rq2_rrf_vs_dense_bm25_map", "RQ2", "RRF+Taxonomy", "Dense+BM25", "map", "RRF+Taxonomy vs Dense+BM25 (MAP)"),
        # RQ3: Structured Taxonomy
        ("H11_rq3_hybrid_vs_dense_bm25_map", "RQ3", "Calibrated Hybrid", "Dense+BM25", "map", "Calibrated Hybrid vs Dense+BM25 (MAP)"),
        ("H12_rq3_hybrid_vs_dense_bm25_ndcg", "RQ3", "Calibrated Hybrid", "Dense+BM25", "ndcg10", "Calibrated Hybrid vs Dense+BM25 (nDCG@10)"),
        # RQ4: Cross-Encoder & Cascaded Reranking
        ("H13_rq4_hybrid_vs_ce_ndcg", "RQ4", "Calibrated Hybrid", "Cross-Encoder", "ndcg10", "Calibrated Hybrid vs Cross-Encoder (nDCG@10)"),
        ("H14_rq4_hybrid_vs_ce_map", "RQ4", "Calibrated Hybrid", "Cross-Encoder", "map", "Calibrated Hybrid vs Cross-Encoder (MAP)"),
        ("H15_rq4_hybrid_vs_top10_ndcg", "RQ4", "Calibrated Hybrid", "2-Stage Top10", "ndcg10", "Calibrated Hybrid vs 2-Stage Top-10 (nDCG@10)"),
    ]

    results_family = {}

    for setting, suffix, label in [("Setting A (All 71)", "all", "all"), ("Setting B (Filtered 61)", "fil", "fil")]:
        setting_res = {}
        raw_p_list = []

        for h_id, rq, mod_a, mod_b, metric, desc in hypotheses:
            vec_a = per_query_data[f"{mod_a}__{metric}_{suffix}"]
            vec_b = per_query_data[f"{mod_b}__{metric}_{suffix}"]
            diff = vec_a - vec_b
            mean_diff = float(np.mean(diff))

            # t-test
            t_res = ttest_rel(vec_a, vec_b)
            raw_t_p = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 1.0

            # Wilcoxon
            try:
                w_res = wilcoxon(vec_a, vec_b, zero_method="pratt")
                raw_w_p = float(w_res.pvalue)
            except Exception:
                raw_w_p = 1.0

            # Paired Bootstrap 95% CI
            ci_low, ci_high = paired_bootstrap_ci(diff, n_resamples=10000, ci=0.95, seed=42)

            raw_p_list.append((h_id, raw_t_p))

            setting_res[h_id] = {
                "rq": rq,
                "description": desc,
                "model_a": mod_a,
                "model_b": mod_b,
                "metric": metric,
                "mean_diff": round(mean_diff, 4),
                "ci_95_low": round(ci_low, 4),
                "ci_95_high": round(ci_high, 4),
                "ci_str": f"[{ci_low:+.4f}, {ci_high:+.4f}]",
                "t_stat": round(float(t_res.statistic), 3) if not np.isnan(t_res.statistic) else 0.0,
                "raw_t_p": round(raw_t_p, 5),
                "raw_w_p": round(raw_w_p, 5),
            }

        # Holm-Bonferroni correction
        hb_corr = holm_bonferroni_correction(raw_p_list, alpha=0.05)
        for h_id, corr_data in hb_corr.items():
            setting_res[h_id]["hb_rank"] = corr_data["rank"]
            setting_res[h_id]["hb_threshold"] = corr_data["hb_threshold"]
            setting_res[h_id]["adj_p"] = corr_data["adj_p"]
            setting_res[h_id]["significant_after_hb"] = corr_data["significant"]

        results_family[setting] = setting_res

    # 5. Save JSON
    out_json = RESULTS_DIR / "statistical_significance_family.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_family, f, indent=2)
    print(f"Saved full statistical test family to {out_json}")

    # 6. Generate LaTeX Table
    # We output a publication-grade LaTeX table for Setting A (with notes on Setting B)
    set_a = results_family["Setting A (All 71)"]
    latex_lines = [
        r"% Auto-generated by scripts/compute_full_statistical_tests.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive Paired Statistical Significance Testing across Primary Hypothesis Family ($N=71$ queries, Setting A). Bootstrap 95\% CIs are computed over $10{,}000$ paired resamples. Multiplicity is controlled via Holm-Bonferroni correction ($\alpha=0.05$).}",
        r"\label{tab:statistical_tests}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrc}",
        r"\hline",
        r"\textbf{RQ} & \textbf{Hypothesis Comparison ($A$ vs. $B$)} & \textbf{Metric} & \textbf{Mean $\Delta$} & \textbf{Paired 95\% CI} & \textbf{$t$-stat} & \textbf{Raw $p$} & \textbf{Adj. $p$} & \textbf{Sig. ($\alpha=0.05$)} \\",
        r"\hline",
    ]

    current_rq = ""
    for h_id, rq, mod_a, mod_b, metric, desc in hypotheses:
        row = set_a[h_id]
        if rq != current_rq:
            if current_rq != "":
                latex_lines.append(r"\hline")
            current_rq = rq

        sig_mark = r"\textbf{Yes}" if row["significant_after_hb"] else "No"
        m_name = "nDCG@10" if row["metric"] == "ndcg10" else "MAP"
        comp_str = f"{row['model_a']} vs. {row['model_b']}"
        latex_lines.append(
            f"{row['rq']} & {comp_str} & {m_name} & {row['mean_diff']:+.4f} & {row['ci_str']} & {row['t_stat']:.2f} & {row['raw_t_p']:.4f} & {row['adj_p']:.4f} & {sig_mark} \\\\"
        )

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])

    out_tex = TABLES_DIR / "tab_statistical_tests.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"Generated LaTeX table at {out_tex}")

    # Print summary to console
    print("\n" + "=" * 90)
    print(f"{'RQ':4s} | {'Comparison':35s} | {'Metric':8s} | {'Mean Delta':10s} | {'95% Bootstrap CI':22s} | {'Raw p':8s} | {'Adj p':8s} | {'Sig'}")
    print("=" * 90)
    for h_id, rq, mod_a, mod_b, metric, desc in hypotheses:
        r = set_a[h_id]
        m_name = "nDCG@10" if r["metric"] == "ndcg10" else "MAP"
        c_str = f"{r['model_a']} vs {r['model_b']}"
        print(f"{r['rq']:4s} | {c_str:35s} | {m_name:8s} | {r['mean_diff']:+10.4f} | {r['ci_str']:22s} | {r['raw_t_p']:8.4f} | {r['adj_p']:8.4f} | {r['significant_after_hb']}")
    print("=" * 90)


if __name__ == "__main__":
    main()
