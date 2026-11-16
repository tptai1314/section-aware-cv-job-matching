"""Evaluate the frozen section parser (commit 41e1d27) against independent ground-truth annotations on 50 held-out documents.

Computes:
1. Precision, Recall, F1 for each section type (CV & JD separated).
2. False positive counts and rate.
3. Multi-section document rate (documents with >= 2 recognized sections).
4. Mean section word lengths (verifying JD Education phình to issue is resolved).
5. Checks against pre-established acceptance criteria:
   - Skills F1 >= 0.85
   - Experience F1 >= 0.85
   - Overall Precision >= 0.90
   - False positive rate < 5%

Saves report to results/frozen_parser_eval_results.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = REPO_ROOT / "data" / "audit_annotation"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
from cvjd.data.preprocessing import split_sections

with open(ANNOTATION_DIR / "raw_eval_samples.json", "r", encoding="utf-8") as f:
    raw_samples = {s["doc_id"]: s for s in json.load(f)}

with open(ANNOTATION_DIR / "ground_truth_eval_50.json", "r", encoding="utf-8") as f:
    gt_samples = {s["doc_id"]: s for s in json.load(f)}

print(f"Loaded {len(raw_samples)} raw samples and {len(gt_samples)} ground truth records.")

# Pre-established Acceptance Criteria
ACCEPTANCE_THRESHOLDS = {
    "min_skills_f1": 0.85,
    "min_experience_f1": 0.85,
    "min_overall_precision": 0.90,
    "max_false_positive_rate": 0.05,
}


def evaluate_doc_type(doc_type_filter: str) -> dict:
    tp_counts = defaultdict(int)
    fp_counts = defaultdict(int)
    fn_counts = defaultdict(int)

    sec_lengths = defaultdict(list)
    multi_sec_ge2 = 0
    multi_sec_ge3 = 0
    total_docs = 0

    detailed_evals = []

    for doc_id, raw_item in raw_samples.items():
        if raw_item["doc_type"] != doc_type_filter:
            continue
        total_docs += 1
        gt_item = gt_samples[doc_id]

        is_jd = (doc_type_filter == "JD")
        parsed_sections = split_sections(raw_item["text"], is_jd=is_jd)

        # Ground-truth section names (excluding preamble)
        gt_sec_names = set(ts["section"] for ts in gt_item["true_sections"] if ts["section"] != "preamble")
        # Parsed section names (excluding preamble and other)
        parsed_sec_names = set(ps.name for ps in parsed_sections if ps.name not in ["preamble", "other"])

        for ps in parsed_sections:
            if ps.name not in ["preamble", "other"]:
                sec_lengths[ps.name].append(len(ps.text.split()))

        # Calculate TP, FP, FN
        for name in parsed_sec_names:
            if name in gt_sec_names:
                tp_counts[name] += 1
            else:
                fp_counts[name] += 1

        for name in gt_sec_names:
            if name not in parsed_sec_names:
                fn_counts[name] += 1

        if len(parsed_sec_names) >= 2:
            multi_sec_ge2 += 1
        if len(parsed_sec_names) >= 3:
            multi_sec_ge3 += 1

        detailed_evals.append({
            "doc_id": doc_id,
            "word_count": raw_item["word_count"],
            "gt_sections": list(gt_sec_names),
            "parsed_sections": list(parsed_sec_names),
            "is_multi_ge2": len(parsed_sec_names) >= 2,
        })

    all_sec_types = sorted(list(set(list(tp_counts.keys()) + list(fp_counts.keys()) + list(fn_counts.keys()))))
    metrics_by_section = {}

    total_tp = sum(tp_counts.values())
    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())

    for st in all_sec_types:
        tp = tp_counts[st]
        fp = fp_counts[st]
        fn = fn_counts[st]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        mean_w = float(np.mean(sec_lengths[st])) if st in sec_lengths and sec_lengths[st] else 0.0
        metrics_by_section[st] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "mean_words": round(mean_w, 1),
        }

    macro_p = float(np.mean([m["precision"] for m in metrics_by_section.values()])) if metrics_by_section else 0.0
    macro_r = float(np.mean([m["recall"] for m in metrics_by_section.values()])) if metrics_by_section else 0.0
    macro_f1 = float(np.mean([m["f1"] for m in metrics_by_section.values()])) if metrics_by_section else 0.0
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "doc_type": doc_type_filter,
        "total_docs": total_docs,
        "multi_section_ge2_pct": round(multi_sec_ge2 / total_docs * 100.0, 2),
        "multi_section_ge3_pct": round(multi_sec_ge3 / total_docs * 100.0, 2),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "false_positive_rate": round(total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0, 4),
        "metrics_by_section": metrics_by_section,
    }


cv_eval = evaluate_doc_type("CV")
jd_eval = evaluate_doc_type("JD")

# Verify against pre-established acceptance thresholds
overall_tp = cv_eval["total_tp"] + jd_eval["total_tp"]
overall_fp = cv_eval["total_fp"] + jd_eval["total_fp"]
overall_fn = cv_eval["total_fn"] + jd_eval["total_fn"]
overall_p = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
overall_r = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
overall_fp_rate = overall_fp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0

skills_f1 = cv_eval["metrics_by_section"].get("skills", {}).get("f1", 0.0)
exp_f1 = cv_eval["metrics_by_section"].get("experience", {}).get("f1", 0.0)

criteria_checks = {
    "skills_f1_ge_0.85": {
        "actual": skills_f1, "target": ACCEPTANCE_THRESHOLDS["min_skills_f1"],
        "passed": bool(skills_f1 >= ACCEPTANCE_THRESHOLDS["min_skills_f1"])
    },
    "experience_f1_ge_0.85": {
        "actual": exp_f1, "target": ACCEPTANCE_THRESHOLDS["min_experience_f1"],
        "passed": bool(exp_f1 >= ACCEPTANCE_THRESHOLDS["min_experience_f1"])
    },
    "overall_precision_ge_0.90": {
        "actual": round(overall_p, 4), "target": ACCEPTANCE_THRESHOLDS["min_overall_precision"],
        "passed": bool(overall_p >= ACCEPTANCE_THRESHOLDS["min_overall_precision"])
    },
    "fp_rate_le_0.05": {
        "actual": round(overall_fp_rate, 4), "target": ACCEPTANCE_THRESHOLDS["max_false_positive_rate"],
        "passed": bool(overall_fp_rate <= ACCEPTANCE_THRESHOLDS["max_false_positive_rate"])
    },
}

all_passed = all(c["passed"] for c in criteria_checks.values())

final_report = {
    "git_commit": "41e1d27",
    "parser_status": "FROZEN",
    "acceptance_criteria_checks": criteria_checks,
    "all_criteria_passed": all_passed,
    "cv_evaluation": cv_eval,
    "jd_evaluation": jd_eval,
    "overall_summary": {
        "total_docs_evaluated": cv_eval["total_docs"] + jd_eval["total_docs"],
        "total_tp": overall_tp,
        "total_fp": overall_fp,
        "total_fn": overall_fn,
        "overall_precision": round(overall_p, 4),
        "overall_recall": round(overall_r, 4),
        "overall_f1": round(overall_f1, 4),
        "overall_fp_rate": round(overall_fp_rate, 4),
    }
}

report_path = RESULTS_DIR / "frozen_parser_eval_results.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2)

print("\n" + "=" * 80)
print(f"FROZEN PARSER EVALUATION REPORT (Commit: 41e1d27) - ALL CRITERIA PASSED: {all_passed}")
print("=" * 80)
print(f"CV Multi-Section (>= 2): {cv_eval['multi_section_ge2_pct']}% (was 1.0% in broken version)")
print(f"CV Skills F1:      {skills_f1:.4f} (Target: >= 0.85)")
print(f"CV Experience F1:  {exp_f1:.4f} (Target: >= 0.85)")
print(f"Overall Precision: {overall_p:.4f} (Target: >= 0.90)")
print(f"Overall FP Rate:   {overall_fp_rate:.4f} (Target: <= 0.05)")
print("\nSection-Level Metrics on CV:")
for s_name, m in cv_eval["metrics_by_section"].items():
    print(f"  {s_name:12s} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f} | Mean Words: {m['mean_words']}")

print("\nSection-Level Metrics on JD:")
for s_name, m in jd_eval["metrics_by_section"].items():
    print(f"  {s_name:18s} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f} | Mean Words: {m['mean_words']}")

print(f"\nSaved full evaluation report to: {report_path}")
