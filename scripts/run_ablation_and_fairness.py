"""Ablation Study, Statistical Significance Testing, and Fairness Perturbation Evaluation.

This script directly addresses the reviewer's critiques:
1. Component ablation for S_hybrid (w/o skills, w/o exp, w/o budget, w/o pref, semantic-only, rules-only).
2. Paired t-tests and Wilcoxon signed-rank tests with bootstrap 95% confidence intervals.
3. Fairness perturbation testing (gender/name and university prestige invariance).
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cvjd.io import load_documents
from cvjd.ranking.metrics import ndcg_at_k, reciprocal_rank, average_precision, precision_at_k
from cvjd.ranking.hybrid import (
    hybrid_score,
    job_from_document,
    profile_from_document,
    DEFAULT_WEIGHTS,
)
from cvjd.models.embedding import create_embedder
from cvjd.models.matching import embed_corpus, score_pair


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate empirical bootstrap confidence interval."""
    arr = np.array(data)
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2)))
    return lower, upper


def evaluate_query_list(ranked_cands_per_query: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, List[float]]:
    """Compute per-query metric lists for statistical testing."""
    ndcg5_list = []
    ndcg10_list = []
    mrr_list = []
    map_list = []

    for qid, ranked_ids in ranked_cands_per_query.items():
        q_qrels = qrels.get(qid, {})
        if not q_qrels:
            continue
        ndcg5_list.append(ndcg_at_k(ranked_ids, q_qrels, k=5))
        ndcg10_list.append(ndcg_at_k(ranked_ids, q_qrels, k=10))
        mrr_list.append(reciprocal_rank(ranked_ids, q_qrels, threshold=2))
        map_list.append(average_precision(ranked_ids, q_qrels, threshold=2))

    return {
        "ndcg5": ndcg5_list,
        "ndcg10": ndcg10_list,
        "mrr": mrr_list,
        "map": map_list,
    }


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    qrels_path = os.path.join(base_dir, "data", "benchmark_qrels.json")
    raw_dir = os.path.join(base_dir, "data", "raw")

    print("Loading benchmark qrels from", qrels_path)
    with open(qrels_path, encoding="utf-8") as f:
        qrels_data = json.load(f)

    # Load CVs and JDs
    cvs = load_documents(os.path.join(raw_dir, "cvs.jsonl"))
    jds = load_documents(os.path.join(raw_dir, "jds.jsonl"))
    print(f"Loaded {len(cvs)} CVs, {len(jds)} JDs.")

    jd2cv_qrels = qrels_data["jd2cv"]
    queries = list(jd2cv_qrels.keys())

    # Create embedder and embed corpus
    print("Embedding documents for semantic matching...")
    embedder = create_embedder("mock")
    cv_embs = embed_corpus(cvs.values(), embedder)
    jd_embs = embed_corpus(jds.values(), embedder)

    # Precompute JobRequirements and CandidateProfiles
    job_reqs = {jid: job_from_document(doc) for jid, doc in jds.items()}
    cand_profs = {cid: profile_from_document(doc) for cid, doc in cvs.items()}

    # 1. COMPONENT ABLATION STUDY
    print("\n--- 1. Running Component Ablation on Hybrid Matching ---")
    ablation_configs = {
        "Full Hybrid (Semantic + Rules)": {
            "semantic": 0.5, "skills": 0.25, "experience": 0.15, "budget": 0.05, "preferences": 0.05
        },
        "w/o S_skills": {
            "semantic": 0.65, "skills": 0.0, "experience": 0.25, "budget": 0.05, "preferences": 0.05
        },
        "w/o S_exp": {
            "semantic": 0.65, "skills": 0.25, "experience": 0.0, "budget": 0.05, "preferences": 0.05
        },
        "w/o S_budget": {
            "semantic": 0.55, "skills": 0.25, "experience": 0.15, "budget": 0.0, "preferences": 0.05
        },
        "w/o S_pref": {
            "semantic": 0.55, "skills": 0.25, "experience": 0.15, "budget": 0.05, "preferences": 0.0
        },
        "Semantic Only (Section-Aware)": {
            "semantic": 1.0, "skills": 0.0, "experience": 0.0, "budget": 0.0, "preferences": 0.0
        },
        "Rules Only (w/o Semantic)": {
            "semantic": 0.0, "skills": 0.5, "experience": 0.3, "budget": 0.1, "preferences": 0.1
        },
    }

    ablation_results = {}
    metric_distributions = {}

    for name, w_dict in ablation_configs.items():
        ranked_per_query = {}

        for qid in queries:
            jd_doc = jds.get(qid)
            j_req = job_reqs.get(qid)
            if not jd_doc or not j_req:
                continue

            scored_candidates = []
            for cv_id, cv_doc in cvs.items():
                c_prof = cand_profs[cv_id]
                s_sem = score_pair(jd_embs[qid], cv_embs[cv_id], method="section_aware", pairing="cross", aggregation="weighted")
                j_req_with_sem = replace(j_req, semantic_base=s_sem)
                h_res = hybrid_score(j_req_with_sem, c_prof, weights=w_dict)
                scored_candidates.append((cv_id, h_res["overall"]))

            scored_candidates.sort(key=lambda x: -x[1])
            ranked_per_query[qid] = [c[0] for c in scored_candidates]

        metrics = evaluate_query_list(ranked_per_query, jd2cv_qrels)
        metric_distributions[name] = metrics
        ablation_results[name] = {
            "nDCG@5": float(np.mean(metrics["ndcg5"])),
            "nDCG@10": float(np.mean(metrics["ndcg10"])),
            "MRR": float(np.mean(metrics["mrr"])),
            "MAP": float(np.mean(metrics["map"])),
        }

    # 2. STATISTICAL SIGNIFICANCE & BOOTSTRAP CI
    print("\n--- 2. Computing Statistical Significance & Confidence Intervals ---")
    full_m = metric_distributions["Full Hybrid (Semantic + Rules)"]
    sem_m = metric_distributions["Semantic Only (Section-Aware)"]
    rules_m = metric_distributions["Rules Only (w/o Semantic)"]

    t_sem, p_sem = stats.ttest_rel(full_m["ndcg5"], sem_m["ndcg5"])
    w_sem, p_w_sem = stats.wilcoxon(full_m["ndcg5"], sem_m["ndcg5"])

    t_rules, p_rules = stats.ttest_rel(full_m["ndcg5"], rules_m["ndcg5"])
    w_rules, p_w_rules = stats.wilcoxon(full_m["ndcg5"], rules_m["ndcg5"])

    ci_ndcg5 = bootstrap_ci(sem_m["ndcg5"])
    ci_ndcg10 = bootstrap_ci(sem_m["ndcg10"])
    ci_mrr = bootstrap_ci(sem_m["mrr"])
    ci_map = bootstrap_ci(sem_m["map"])

    significance_results = {
        "FullHybrid_vs_RulesOnly": {
            "t_statistic": float(t_rules),
            "p_value_ttest": float(p_rules),
            "wilcoxon_stat": float(w_rules),
            "p_value_wilcoxon": float(p_w_rules),
            "significant_at_0.01": bool(p_w_rules < 0.01),
        },
        "FullHybrid_vs_SemanticOnly": {
            "t_statistic": float(t_sem),
            "p_value_ttest": float(p_sem),
            "wilcoxon_stat": float(w_sem),
            "p_value_wilcoxon": float(p_w_sem),
            "significant_at_0.01": bool(p_w_sem < 0.01),
        },
        "SectionAware_95_CI": {
            "nDCG@5": ci_ndcg5,
            "nDCG@10": ci_ndcg10,
            "MRR": ci_mrr,
            "MAP": ci_map,
        }
    }

    # 3. FAIRNESS & BIAS PERTURBATION TEST
    print("\n--- 3. Running Fairness & Bias Perturbation Testing ---")
    sample_cv_ids = list(cvs.keys())[:30]
    sample_jd_ids = list(jds.keys())[:5]

    diffs_gender = []
    diffs_school = []

    male_names = ["Nguyễn Văn An", "Trần Đức Minh", "Lê Văn Tuấn", "Phạm Hoàng Nam"]
    female_names = ["Nguyễn Thị Mai", "Trần Thị Lan", "Lê Thị Hương", "Phạm Ánh Linh"]

    for cv_id in sample_cv_ids:
        orig_doc = cvs[cv_id]

        gender_pert_doc = copy.deepcopy(orig_doc)
        if "summary" in gender_pert_doc.sections:
            old_sec = gender_pert_doc.sections["summary"]
            old_text = old_sec.text
            for m, f in zip(male_names, female_names):
                old_text = old_text.replace(m, f)
            gender_pert_doc.sections["summary"] = replace(old_sec, text=old_text)

        school_pert_doc = copy.deepcopy(orig_doc)
        if "education" in school_pert_doc.sections:
            old_sec = school_pert_doc.sections["education"]
            old_edu = old_sec.text
            old_edu = old_edu.replace("Đại học Bách Khoa", "Cao đẳng Nghề Tỉnh")
            old_edu = old_edu.replace("Đại học Quốc gia", "Đại học Địa phương")
            school_pert_doc.sections["education"] = replace(old_sec, text=old_edu)

        # Embed perturbed docs
        orig_emb = cv_embs[cv_id]
        gender_emb = embed_corpus([gender_pert_doc], embedder)[gender_pert_doc.doc_id]
        school_emb = embed_corpus([school_pert_doc], embedder)[school_pert_doc.doc_id]

        for jid in sample_jd_ids:
            s_orig = score_pair(jd_embs[jid], orig_emb, method="section_aware", pairing="cross", aggregation="weighted")
            s_gender = score_pair(jd_embs[jid], gender_emb, method="section_aware", pairing="cross", aggregation="weighted")
            s_school = score_pair(jd_embs[jid], school_emb, method="section_aware", pairing="cross", aggregation="weighted")

            diffs_gender.append(abs(s_orig - s_gender))
            diffs_school.append(abs(s_orig - s_school))

    fairness_results = {
        "Gender_Perturbation": {
            "mean_absolute_difference": float(np.mean(diffs_gender)),
            "max_absolute_difference": float(np.max(diffs_gender)),
            "invariance_rate_within_1pct": float(np.mean(np.array(diffs_gender) < 0.01)),
        },
        "School_Prestige_Perturbation": {
            "mean_absolute_difference": float(np.mean(diffs_school)),
            "max_absolute_difference": float(np.max(diffs_school)),
            "invariance_rate_within_2pct": float(np.mean(np.array(diffs_school) < 0.02)),
        }
    }

    # Save to JSON
    output_data = {
        "ablation_study": ablation_results,
        "statistical_significance": significance_results,
        "fairness_evaluation": fairness_results,
    }

    results_dir = os.path.join(base_dir, "results")
    out_json = os.path.join(results_dir, "ablation_and_fairness.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Save LaTeX table for paper/tables/tab_ablation.tex
    paper_tables_dir = os.path.join(base_dir, "paper", "tables")
    os.makedirs(paper_tables_dir, exist_ok=True)
    out_tex = os.path.join(paper_tables_dir, "tab_ablation.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by scripts/run_ablation_and_fairness.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Ablation analysis of hybrid matching components on \\texttt{Frevia-Bench} ($N=47$).}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("\\textbf{Configuration} & \\textbf{nDCG@5} & \\textbf{nDCG@10} & \\textbf{MRR} & \\textbf{MAP} \\\\ \\hline\n")
        for k, v in ablation_results.items():
            f.write(f"{k} & {v['nDCG@5']:.4f} & {v['nDCG@10']:.4f} & {v['MRR']:.4f} & {v['MAP']:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"\nSaved all results to {out_json} and {out_tex}")
    print("\n--- Ablation Results Summary ---")
    for k, v in ablation_results.items():
        print(f"| {k:30s} | nDCG@5: {v['nDCG@5']:.4f} | MRR: {v['MRR']:.4f} | MAP: {v['MAP']:.4f} |")

    print("\n--- Fairness Summary ---")
    print(f"Gender mean delta: {fairness_results['Gender_Perturbation']['mean_absolute_difference']:.6f}")
    print(f"School mean delta: {fairness_results['School_Prestige_Perturbation']['mean_absolute_difference']:.6f}")


if __name__ == "__main__":
    main()
