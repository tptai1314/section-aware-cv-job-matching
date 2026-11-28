"""Generate 3-panel publication figures (EN & VI) and evaluate CV -> JD reverse direction."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.data.taxonomy import get_taxonomy
from cvjd.ranking.engine import min_max
from sklearn.feature_extraction.text import TfidfVectorizer

CACHE_DIR = REPO_ROOT / "data" / "cache_embeddings"
RESULTS_DIR = REPO_ROOT / "results"
LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}

def dcg_at_k(rels, k):
    score = 0.0
    for i in range(min(k, len(rels))):
        score += (math.pow(2.0, float(rels[i])) - 1.0) / math.log2(i + 2.0)
    return score

def ndcg_at_k(ranked_rels, ideal_rels, k):
    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0.0:
        return 0.0
    return min(1.0, max(0.0, dcg_at_k(ranked_rels, k) / idcg))

def average_precision(ranked_rels, threshold=1):
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

def reciprocal_rank(ranked_rels, threshold=1):
    for rank, r in enumerate(ranked_rels, start=1):
        if r >= threshold:
            return 1.0 / rank
    return 0.0

def precision_at_k(ranked_rels, k, threshold=1):
    if k <= 0 or len(ranked_rels) == 0:
        return 0.0
    return sum(1 for r in ranked_rels[:k] if r >= threshold) / float(k)

def evaluate_query(scores, rels, n_tie_shuffles=50, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    n_items = len(scores)
    ideal_rels = np.sort(rels)[::-1]
    acc = defaultdict(float)
    for _ in range(n_tie_shuffles):
        jitter = rng.uniform(-1e-9, 1e-9, size=n_items)
        order = np.argsort(-(scores + jitter))
        ranked = rels[order]
        acc["ndcg@5"] += ndcg_at_k(ranked, ideal_rels, 5)
        acc["ndcg@10"] += ndcg_at_k(ranked, ideal_rels, 10)
        acc["mrr_th1"] += reciprocal_rank(ranked, 1)
        acc["map_th1"] += average_precision(ranked, 1)
        acc["p@5_th1"] += precision_at_k(ranked, 5, 1)
    return {k: v / n_tie_shuffles for k, v in acc.items()}

def bootstrap_ci(diffs, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(diffs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = np.mean(diffs[rng.randint(0, n, size=n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def run_bidirectional_eval():
    print("--> [1/2] Evaluating Reverse Direction (CV -> JD Job Recommendation) on med2425 ...")
    med_file = REPO_ROOT / "data" / "public_benchmark" / "med2425" / "test.parquet"
    df_med = pd.read_parquet(med_file)
    df_med["rel"] = df_med["label"].map(LABEL_MAP)
    target_domains = {"software", "data", "ai"}
    sub_df = df_med[df_med["resume_domain"].isin(target_domains) & df_med["jd_domain"].isin(target_domains)].copy()

    u_jobs_med = sub_df["jd"].unique().tolist()
    u_cvs_med = sub_df["resume"].unique().tolist()
    j2i_med = {j: i for i, j in enumerate(u_jobs_med)}
    c2i_med = {r: i for i, r in enumerate(u_cvs_med)}

    pairs_med, y_med = [], []
    cv_query_groups = defaultdict(list)
    for _, row in sub_df.iterrows():
        j_i = j2i_med[row["jd"]]
        c_i = c2i_med[row["resume"]]
        p_i = len(pairs_med)
        cv_query_groups[c_i].append(p_i)
        pairs_med.append((j_i, c_i))
        y_med.append(row["rel"])
    y_med_arr = np.array(y_med, dtype=int)

    # Filter CVs with >= 5 candidate JDs and >= 1 positive JD (Setting B standard)
    valid_cv_queries = [
        c_i for c_i, p_idx in cv_query_groups.items()
        if len(p_idx) >= 5 and any(y_med_arr[p] >= 1 for p in p_idx)
    ]
    print(f"Total Unique CVs: {len(u_cvs_med)}, Valid CV Queries (>=1 positive JD): {len(valid_cv_queries)}")

    vec_med = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    vec_med.fit(u_jobs_med + u_cvs_med)
    j_tf_med = vec_med.transform(u_jobs_med)
    c_tf_med = vec_med.transform(u_cvs_med)
    bm25_med = np.array([float((j_tf_med[j].multiply(c_tf_med[c])).sum()) for j, c in pairs_med])

    j_flat_med = np.load(CACHE_DIR / "med2425_job_flat.npy")
    c_flat_med = np.load(CACHE_DIR / "med2425_cv_flat.npy")
    c_sec_vecs_flat_med = np.load(CACHE_DIR / "med2425_cv_sec_flat.npy")

    from cvjd.data.preprocessing import sections_to_dict, split_sections
    cv_parsed_med = [sections_to_dict(split_sections(c, is_jd=False)) for c in u_cvs_med]
    sec_meta_med = []
    for c_idx, sd in enumerate(cv_parsed_med):
        for s_name, s_obj in sd.items():
            if len(s_obj.text.strip()) > 5:
                sec_meta_med.append((c_idx, s_name))

    cv_sec_vecs_med = [{} for _ in range(len(u_cvs_med))]
    for (c_idx, s_name), v in zip(sec_meta_med, c_sec_vecs_flat_med):
        cv_sec_vecs_med[c_idx][s_name] = v

    flat_med = np.array([float(np.dot(j_flat_med[j], c_flat_med[c])) for j, c in pairs_med])
    sec_weights = {"skills": 0.40, "experience": 0.35, "education": 0.15, "summary": 0.10}
    sec_mixed_med = []
    for idx, (j, c) in enumerate(pairs_med):
        jv = j_flat_med[j]
        c_secs = cv_sec_vecs_med[c]
        w_sim, tot_w = 0.0, 0.0
        for sn, w in sec_weights.items():
            if sn in c_secs:
                w_sim += w * float(np.dot(jv, c_secs[sn]))
                tot_w += w
        sc = w_sim / tot_w if tot_w > 0 else flat_med[idx]
        sec_mixed_med.append(0.65 * sc + 0.35 * flat_med[idx])
    sec_mixed_med = np.array(sec_mixed_med)

    taxonomy = get_taxonomy()
    j_tax_med = [taxonomy.classify_jd_skills(j) for j in u_jobs_med]
    c_tax_med = [taxonomy.extract_skills_from_text(c) for c in u_cvs_med]

    tax_med = np.array([
        taxonomy.score_skills_coverage(
            j_tax_med[j][0],
            j_tax_med[j][1],
            c_tax_med[c],
            must_weight=0.85,
            nice_weight=0.15,
            neutral_score=1.0,
            use_layer2=True,
            use_layer3=False,
        )
        for j, c in pairs_med
    ])

    # Per-CV-query Min-Max normalization for CV -> JD direction
    norm_sec_cv = np.zeros_like(sec_mixed_med)
    norm_bm25_cv = np.zeros_like(bm25_med)
    norm_tax_cv = np.zeros_like(tax_med)
    for c_i, p_idx in cv_query_groups.items():
        norm_sec_cv[p_idx] = min_max(sec_mixed_med[p_idx])
        norm_bm25_cv[p_idx] = min_max(bm25_med[p_idx])
        norm_tax_cv[p_idx] = min_max(tax_med[p_idx])

    l0_cv = 0.60 * norm_sec_cv + 0.40 * norm_bm25_cv
    hybrid_cv = 0.60 * l0_cv + 0.40 * norm_tax_cv

    rng_rand = np.random.RandomState(42)
    rand_cv = rng_rand.uniform(0, 1, size=len(pairs_med))

    models_cv = {
        "Random Ranking (Floor)": rand_cv,
        "BM25 / TF-IDF (Lexical)": bm25_med,
        "MiniLM Single-Vector (Flat)": flat_med,
        "MiniLM Section-Aware Mixed (Ours)": sec_mixed_med,
        "Dense + BM25 (L0: No Taxonomy)": l0_cv,
        "Calibrated Hybrid (Dense+BM25+Tax)": hybrid_cv,
    }

    res_cv = {}
    per_q_n10 = {}
    for m_name, arr in models_cv.items():
        rng_eval = np.random.RandomState(42)
        per_q = [evaluate_query(arr[cv_query_groups[c_i]], y_med_arr[cv_query_groups[c_i]], rng=rng_eval) for c_i in valid_cv_queries]
        avg_m = {k: round(float(np.mean([r[k] for r in per_q])), 4) for k in per_q[0].keys()}
        res_cv[m_name] = avg_m
        per_q_n10[m_name] = np.array([r["ndcg@10"] for r in per_q])
        print(f"CV->JD {m_name:35s}: nDCG@10={avg_m['ndcg@10']:.4f} | MAP={avg_m['map_th1']:.4f} | MRR={avg_m['mrr_th1']:.4f} | P@5={avg_m['p@5_th1']:.4f}")

    diff_hyb_single = per_q_n10["Calibrated Hybrid (Dense+BM25+Tax)"] - per_q_n10["MiniLM Single-Vector (Flat)"]
    ci_lo, ci_hi = bootstrap_ci(diff_hyb_single)
    t_st, p_val = ttest_rel(per_q_n10["Calibrated Hybrid (Dense+BM25+Tax)"], per_q_n10["MiniLM Single-Vector (Flat)"])
    print(f"CV->JD Statistical Test (Calibrated Hybrid vs Single-Vector): Delta={np.mean(diff_hyb_single):+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], t={t_st:.2f}, p={p_val:.6f}")

    diff_sec_single = per_q_n10["MiniLM Section-Aware Mixed (Ours)"] - per_q_n10["MiniLM Single-Vector (Flat)"]
    ci_s_lo, ci_s_hi = bootstrap_ci(diff_sec_single)
    t_s, p_s = ttest_rel(per_q_n10["MiniLM Section-Aware Mixed (Ours)"], per_q_n10["MiniLM Single-Vector (Flat)"])
    print(f"CV->JD Statistical Test (Section-Aware Mixed vs Single-Vector): Delta={np.mean(diff_sec_single):+.4f}, 95% CI=[{ci_s_lo:+.4f}, {ci_s_hi:+.4f}], t={t_s:.2f}, p={p_s:.6f}")

    out_bi = {
        "dataset": "med2425/resume-job-fit-merged-v1 (Reverse Direction: CV -> JD Job Recommendation)",
        "n_valid_cv_queries": len(valid_cv_queries),
        "n_total_cvs": len(u_cvs_med),
        "n_candidate_jds": len(u_jobs_med),
        "metrics": res_cv,
        "statistical_tests": {
            "hybrid_vs_single_ndcg10": {
                "mean_diff": round(float(np.mean(diff_hyb_single)), 4),
                "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                "t_stat": round(float(t_st), 3),
                "p_val": float(p_val),
            },
            "section_mixed_vs_single_ndcg10": {
                "mean_diff": round(float(np.mean(diff_sec_single)), 4),
                "ci_95": [round(ci_s_lo, 4), round(ci_s_hi, 4)],
                "t_stat": round(float(t_s), 3),
                "p_val": float(p_s),
            },
        },
    }
    with open(RESULTS_DIR / "bidirectional_cv_to_jd_results.json", "w", encoding="utf-8") as f:
        json.dump(out_bi, f, indent=2)

def generate_3panel_figure(lang="en", out_path=None):
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.titlesize": 11.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), dpi=300)

    # Panel (a): Hyperparameter Sensitivity (alpha & beta)
    ax1 = axes[0]
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    a_ndcg = [0.6626, 0.6884, 0.6946, 0.6951, 0.6994, 0.6882, 0.6870, 0.6601]
    betas = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    b_ndcg = [0.7070, 0.7074, 0.7078, 0.7054, 0.7009, 0.6994]

    if lang == "en":
        ax1.plot(alphas, a_ndcg, "o-", color="#1f77b4", lw=2.0, markersize=5.5, label=r"Retrieval Blend $\alpha$ (Dense vs. BM25)")
        ax1.plot(betas, b_ndcg, "s--", color="#d62728", lw=2.0, markersize=5.5, label=r"Final Blend $\beta$ (Retrieval vs. Tax.)")
        ax1.axvline(0.60, color="#2ca02c", linestyle=":", lw=1.5, label=r"Calibrated Optimum ($\alpha=\beta=0.60$)")
        ax1.set_title("(a) Hyperparameter Sensitivity Analysis", fontweight="bold")
        ax1.set_xlabel(r"Hyperparameter Value ($\alpha, \beta \in [0, 1]$)")
        ax1.set_ylabel("nDCG@10 (50-Query Train Sample)")
    else:
        ax1.plot(alphas, a_ndcg, "o-", color="#1f77b4", lw=2.0, markersize=5.5, label=r"Tỷ lệ $\alpha$ (Dense vs. BM25)")
        ax1.plot(betas, b_ndcg, "s--", color="#d62728", lw=2.0, markersize=5.5, label=r"Tỷ lệ $\beta$ (Retrieval vs. Taxonomy)")
        ax1.axvline(0.60, color="#2ca02c", linestyle=":", lw=1.5, label=r"Đỉnh hiệu chuẩn ($\alpha=\beta=0.60$)")
        ax1.set_title("(a) Độ nhạy Siêu tham số Hòa trộn", fontweight="bold")
        ax1.set_xlabel(r"Giá trị tham số ($\alpha, \beta \in [0, 1]$)")
        ax1.set_ylabel("nDCG@10 (Tập khảo sát 50 JD)")
    ax1.set_ylim(0.650, 0.720)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower center", frameon=True)

    # Panel (b): Length-Stratified Truncation Mitigation
    ax2 = axes[1]
    if lang == "en":
        strata = ["Short/Med\n(<=350w)", "Long CVs\n(>350w)", "Very Long\n(>600w)"]
        l_flat = "Single-Vector Flat (256-tok)"
        l_sec = "Section-Aware Dense (Ours)"
        l_hyb = "Calibrated Hybrid (Ours)"
        ax2.set_title("(b) Truncation Mitigation by Resume Length", fontweight="bold")
        ax2.set_ylabel("nDCG@10 (Setting B Filtered)")
    else:
        strata = ["CV Ngắn/TB\n(<=350 từ)", "CV Dài\n(>350 từ)", "CV Rất dài\n(>600 từ)"]
        l_flat = "Single-Vector Phẳng (256-tok)"
        l_sec = "Section-Aware Dense (Đề xuất)"
        l_hyb = "Calibrated Hybrid (Đề xuất)"
        ax2.set_title("(b) Khắc phục Cắt cụt theo Độ dài CV", fontweight="bold")
        ax2.set_ylabel("nDCG@10 (Setting B)")

    v_flat = [0.8536, 0.7158, 0.7465]
    v_sec = [0.8357, 0.7285, 0.7776]
    v_hyb = [0.8693, 0.7353, 0.7585]

    x = np.arange(len(strata))
    width = 0.25
    b1 = ax2.bar(x - width, v_flat, width, label=l_flat, color="#8c8c8c", edgecolor="black", linewidth=0.6)
    b2 = ax2.bar(x, v_sec, width, label=l_sec, color="#1f77b4", edgecolor="black", linewidth=0.6)
    b3 = ax2.bar(x + width, v_hyb, width, label=l_hyb, color="#2ca02c", edgecolor="black", linewidth=0.6)

    ax2.annotate("+0.0127", xy=(1, 0.7285), xytext=(0.88, 0.742), fontsize=7.8, fontweight="bold", color="#1f77b4")
    ax2.annotate("+0.0195", xy=(1 + width, 0.7353), xytext=(1.13, 0.750), fontsize=7.8, fontweight="bold", color="#1b7837")
    ax2.annotate("+0.0311", xy=(2, 0.7776), xytext=(1.85, 0.790), fontsize=8.0, fontweight="bold", color="#1f77b4")

    ax2.set_xticks(x)
    ax2.set_xticklabels(strata)
    ax2.set_ylim(0.68, 0.895)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True)

    # Panel (c): Quality vs. Online Latency Pareto Frontier
    ax3 = axes[2]
    # Models: (T_online_ms, MAP_SettingA, label, color, marker)
    if lang == "en":
        pts = [
            (0.88, 0.5930, "BM25 (0.9ms)", "#7f7f7f", "^"),
            (29.79, 0.6210, "Single-Vector (29.8ms)", "#ff7f0e", "o"),
            (29.90, 0.6328, "Section-Aware (29.9ms)", "#1f77b4", "s"),
            (33.44, 0.6351, "Calibrated Hybrid (33.4ms)", "#2ca02c", "*"),
            (33.57, 0.6307, "RRF + Tax. (33.6ms)", "#17becf", "D"),
            (457.63, 0.5786, "2-Stage CE Top-10 (458ms)", "#9467bd", "v"),
            (1178.05, 0.5650, "Full Cross-Encoder (1178ms)", "#d62728", "X"),
            (1646.25, 0.6169, "BGE-M3 Sec. (1646ms)", "#8c564b", "P"),
        ]
        ax3.set_title("(c) Accuracy vs. Online Latency Pareto Frontier", fontweight="bold")
        ax3.set_xlabel(r"Total End-to-End Online Latency $T_{\mathrm{online}}$ (ms, Log Scale)")
        ax3.set_ylabel("Mean Average Precision (MAP, Setting A)")
    else:
        pts = [
            (0.88, 0.5930, "BM25 (0.9ms)", "#7f7f7f", "^"),
            (29.79, 0.6210, "Single-Vector (29.8ms)", "#ff7f0e", "o"),
            (29.90, 0.6328, "Section-Aware (29.9ms)", "#1f77b4", "s"),
            (33.44, 0.6351, "Calibrated Hybrid (33.4ms)", "#2ca02c", "*"),
            (33.57, 0.6307, "RRF + Tax. (33.6ms)", "#17becf", "D"),
            (457.63, 0.5786, "2-Stage CE Top-10 (458ms)", "#9467bd", "v"),
            (1178.05, 0.5650, "Full Cross-Encoder (1178ms)", "#d62728", "X"),
            (1646.25, 0.6169, "BGE-M3 Sec. (1646ms)", "#8c564b", "P"),
        ]
        ax3.set_title("(c) Biên Pareto Chất lượng vs. Độ trễ Trực tuyến", fontweight="bold")
        ax3.set_xlabel(r"Tổng độ trễ trực tuyến $T_{\mathrm{online}}$ (ms, Thang Log)")
        ax3.set_ylabel("Độ chính xác trung bình (MAP, Setting A)")

    for t_ms, map_v, lbl, col, mk in pts:
        sz = 130 if mk == "*" else 65
        ax3.scatter(t_ms, map_v, color=col, marker=mk, s=sz, zorder=5, label=lbl, edgecolors="black", linewidths=0.6)

    # Pareto frontier line connecting BM25 -> Section-Aware -> Calibrated Hybrid
    ax3.plot([0.88, 29.90, 33.44], [0.5930, 0.6328, 0.6351], "--", color="#2ca02c", lw=1.6, alpha=0.8, zorder=4)
    ax3.set_xscale("log")
    ax3.set_xlim(0.5, 3000)
    ax3.set_ylim(0.555, 0.648)
    ax3.grid(True, which="both", linestyle="--", alpha=0.4)
    ax3.legend(loc="lower left", fontsize=7.2, ncol=1, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3-panel figure to {out_path}")

if __name__ == "__main__":
    run_bidirectional_eval()
    generate_3panel_figure("en", REPO_ROOT / "paper" / "figures" / "fig_empirical_analysis.png")
    generate_3panel_figure("vi", REPO_ROOT / "paper_vi" / "figures" / "fig_empirical_analysis.png")
