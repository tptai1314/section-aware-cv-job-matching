"""Recompute Unified Corpus Statistics using the frozen section parser (commit 41e1d27).

Generates:
1. Exact section presence and length statistics (CV & JD separated) using split_sections().
2. Taxonomy skill coverage and extraction statistics.
3. Updates SOURCES.md with honest provenance (License: Unspecified, LiveCareer formatting).
4. Generates paper/tables/tab_section_skills_stats.tex.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "public_benchmark" / "resume-job-description-fit"
TABLES_DIR = REPO_ROOT / "paper" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
from cvjd.data.preprocessing import split_sections
from cvjd.data.taxonomy import get_taxonomy

def analyze_corpus(texts: list[str], is_jd: bool = False) -> dict:
    total_docs = len(texts)
    section_counts = defaultdict(int)
    section_lengths = defaultdict(list)
    multi_ge2 = 0
    multi_ge3 = 0
    multi_ge4 = 0

    for text in texts:
        sections = split_sections(text, is_jd=is_jd)
        sec_names = set(s.name for s in sections if s.name not in ["preamble", "other"])

        for s_name in sec_names:
            section_counts[s_name] += 1

        for s in sections:
            if s.name not in ["preamble", "other"]:
                section_lengths[s.name].append(len(s.text.split()))

        if len(sec_names) >= 2:
            multi_ge2 += 1
        if len(sec_names) >= 3:
            multi_ge3 += 1
        if len(sec_names) >= 4:
            multi_ge4 += 4

    stats = {
        "total_docs": total_docs,
        "multi_ge2_pct": round(multi_ge2 / total_docs * 100.0, 1),
        "multi_ge3_pct": round(multi_ge3 / total_docs * 100.0, 1),
        "multi_ge4_pct": round(multi_ge4 / total_docs * 100.0, 1),
        "sections": {},
    }

    all_keys = sorted(list(section_counts.keys()))
    for k in all_keys:
        cnt = section_counts[k]
        lens = section_lengths[k]
        stats["sections"][k] = {
            "count": cnt,
            "presence_pct": round(cnt / total_docs * 100.0, 1),
            "mean_words": round(float(np.mean(lens)), 1) if lens else 0.0,
            "median_words": round(float(np.median(lens)), 1) if lens else 0.0,
            "min_words": int(np.min(lens)) if lens else 0,
            "max_words": int(np.max(lens)) if lens else 0,
            "est_mean_tokens": round(float(np.mean(lens)) * 1.33, 1) if lens else 0.0,
        }
    return stats


def main() -> None:
    print("=" * 80)
    print("RECOMPUTING UNIFIED CORPUS STATISTICS USING FROZEN PARSER (41e1d27)")
    print("=" * 80)

    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="latin1")
    test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="latin1")

    train_cvs = train_df["resume_text"].unique().tolist()
    train_jds = train_df["job_description_text"].unique().tolist()
    test_cvs = test_df["resume_text"].unique().tolist()
    test_jds = test_df["job_description_text"].unique().tolist()

    # Overlap CVs
    train_cv_set = set(train_cvs)
    overlap_cvs = [c for c in test_cvs if c in train_cv_set]
    clean_test_cvs = [c for c in test_cvs if c not in train_cv_set]

    print(f"Test CVs: {len(test_cvs)} total (Clean: {len(clean_test_cvs)}, Overlap: {len(overlap_cvs)})")
    print(f"Test JDs: {len(test_jds)} total")

    cv_stats = analyze_corpus(test_cvs, is_jd=False)
    jd_stats = analyze_corpus(test_jds, is_jd=True)

    # Taxonomy analysis
    print("\n--> Analyzing Skill Taxonomy Coverage on Test Set ...")
    taxonomy = get_taxonomy()
    jd_skills = [taxonomy.classify_jd_skills(j) for j in test_jds]
    cv_skills = [taxonomy.extract_skills_from_text(c) for c in test_cvs]

    jd_has_skills = sum(1 for m, n in jd_skills if len(m) + len(n) > 0)
    cv_has_skills = sum(1 for s in cv_skills if len(s) > 0)

    jd_must_lens = [len(m) for m, n in jd_skills]
    jd_nice_lens = [len(n) for m, n in jd_skills]
    cv_skill_lens = [len(s) for s in cv_skills]

    active_skills = set()
    for m, n in jd_skills:
        active_skills.update(m)
        active_skills.update(n)
    for s in cv_skills:
        active_skills.update(s)

    tax_stats = {
        "jd_with_skills_pct": round(jd_has_skills / len(test_jds) * 100.0, 1),
        "cv_with_skills_pct": round(cv_has_skills / len(test_cvs) * 100.0, 1),
        "jd_mean_must_skills": round(float(np.mean(jd_must_lens)), 1),
        "jd_mean_nice_skills": round(float(np.mean(jd_nice_lens)), 1),
        "jd_mean_total_skills": round(float(np.mean([m + n for m, n in zip(jd_must_lens, jd_nice_lens)])), 1),
        "cv_mean_skills": round(float(np.mean(cv_skill_lens)), 1),
        "active_skills_benchmark": len(active_skills),
        "total_canonical_taxonomy": len(taxonomy.entries),
    }

    full_results = {
        "parser_commit": "41e1d27",
        "cv_statistics": cv_stats,
        "jd_statistics": jd_stats,
        "taxonomy_statistics": tax_stats,
    }

    results_path = RESULTS_DIR / "corpus_unified_statistics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved full corpus statistics to: {results_path}")

    # Generate LaTeX table
    # Columns: Section Name & CV Presence (%) & CV Mean Words & JD Presence (%) & JD Mean Words
    core_sections = [
        ("skills", "Technical Skills / Requirements"),
        ("experience", "Work Experience / Responsibilities"),
        ("education", r"Education \& Qualifications"),
        ("summary", "Summary / Profile / Overview"),
        ("benefits", r"Benefits \& Compensation"),
        ("projects", "Projects / Portfolio"),
    ]

    lines = []
    lines.append("% Auto-generated by scripts/recompute_corpus_stats.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Unified corpus statistics across CV and JD sections extracted by the frozen section parser (commit \\texttt{41e1d27}) on public benchmark \\texttt{cnamuangtoun/resume-job-description-fit}.}")
    lines.append("\\label{tab:section_skills_stats}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\hline")
    lines.append("\\textbf{Section Entity} & \\textbf{CV Pres. (\\%)} & \\textbf{CV Mean Words} & \\textbf{JD Pres. (\\%)} & \\textbf{JD Mean Words} \\\\")
    lines.append("\\hline")

    for key, display_name in core_sections:
        cv_s = cv_stats["sections"].get(key, {})
        # Map JD key: skills -> requirements, experience -> responsibilities
        jd_key = key
        if key == "skills":
            jd_key = "requirements"
        elif key == "experience":
            jd_key = "responsibilities"
        jd_s = jd_stats["sections"].get(jd_key, {})

        cv_p = f"{cv_s.get('presence_pct', 0.0):.1f}\\%" if cv_s else "--"
        cv_w = f"{cv_s.get('mean_words', 0.0):.1f}" if cv_s else "--"
        jd_p = f"{jd_s.get('presence_pct', 0.0):.1f}\\%" if jd_s else "--"
        jd_w = f"{jd_s.get('mean_words', 0.0):.1f}" if jd_s else "--"
        lines.append(f"{display_name} & {cv_p} & {cv_w} & {jd_p} & {jd_w} \\\\")

    lines.append("\\hline")
    lines.append(f"Multi-Section ($\\ge 2$ sections) & {cv_stats['multi_ge2_pct']:.1f}\\% & -- & {jd_stats['multi_ge2_pct']:.1f}\\% & -- \\\\")
    lines.append(f"Multi-Section ($\\ge 3$ sections) & {cv_stats['multi_ge3_pct']:.1f}\\% & -- & {jd_stats['multi_ge3_pct']:.1f}\\% & -- \\\\")
    lines.append(f"Taxonomy Skill Coverage & {tax_stats['cv_with_skills_pct']:.1f}\\% & {tax_stats['cv_mean_skills']:.1f} skills/CV & {tax_stats['jd_with_skills_pct']:.1f}\\% & {tax_stats['jd_mean_total_skills']:.1f} skills/JD \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_content = "\n".join(lines) + "\n"
    latex_path = TABLES_DIR / "tab_section_skills_stats.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"Generated LaTeX table at: {latex_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("CORPUS AUDIT SUMMARY")
    print("=" * 80)
    print(f"CVs: {len(test_cvs)} total | Multi-Section (>=2): {cv_stats['multi_ge2_pct']}% | Multi-Section (>=3): {cv_stats['multi_ge3_pct']}%")
    print(f"JDs: {len(test_jds)} total | Multi-Section (>=2): {jd_stats['multi_ge2_pct']}% | Multi-Section (>=3): {jd_stats['multi_ge3_pct']}%")
    for k, v in cv_stats["sections"].items():
        print(f"  CV Section [{k:12s}]: {v['presence_pct']:5.1f}% | Mean words: {v['mean_words']:5.1f} (~{v['est_mean_tokens']} tokens)")
    for k, v in jd_stats["sections"].items():
        print(f"  JD Section [{k:16s}]: {v['presence_pct']:5.1f}% | Mean words: {v['mean_words']:5.1f} (~{v['est_mean_tokens']} tokens)")
    print(f"\nTaxonomy Coverage: CV {tax_stats['cv_with_skills_pct']}% ({tax_stats['cv_mean_skills']} skills/doc), JD {tax_stats['jd_with_skills_pct']}% ({tax_stats['jd_mean_total_skills']} skills/doc)")
    print(f"Active unique taxonomy skills in benchmark: {tax_stats['active_skills_benchmark']} / {tax_stats['total_canonical_taxonomy']}")


if __name__ == "__main__":
    main()
