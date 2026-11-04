"""Generate gold-standard relevance judgments (qrels) for benchmark evaluation.

Uses the 4-level rubric from plan.cv.md:
  0 = Không phù hợp (Irrelevant)
  1 = Hơi phù hợp (Partially relevant)
  2 = Phù hợp (Relevant)
  3 = Rất phù hợp (Highly relevant)

Ground truth is constructed based on:
  - Exact & normalized skill coverage (required & optional)
  - Domain / category semantic alignment
  - Experience level compatibility

Exports:
  data/benchmark_qrels.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from cvjd.io import load_documents
from cvjd.ranking.hybrid import job_from_document, normalize_skill, profile_from_document

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def extract_skills_set(items) -> Set[str]:
    return {normalize_skill(s.name.lstrip("- ").strip()) for s in items if s.name.strip()}


def compute_pair_grade(job_doc, cv_doc) -> int:
    jp = job_from_document(job_doc)
    cp = profile_from_document(cv_doc)

    req_skills = extract_skills_set(jp.required_skills)
    opt_skills = extract_skills_set(jp.optional_skills)
    all_job_skills = req_skills | opt_skills

    cv_skills = extract_skills_set(cp.skills)

    # 1. Text overlap tokens for domain
    job_words = set(job_doc.raw_text.lower().split())
    cv_words = set(cv_doc.raw_text.lower().split())
    common_words = job_words & cv_words

    if not all_job_skills:
        # Fallback to general keyword/domain overlap
        overlap_ratio = len(common_words) / max(len(job_words), 1)
        if overlap_ratio >= 0.35:
            return 3
        elif overlap_ratio >= 0.22:
            return 2
        elif overlap_ratio >= 0.12:
            return 1
        return 0

    # Skills matching
    matched_req = req_skills & cv_skills
    matched_opt = opt_skills & cv_skills
    req_ratio = len(matched_req) / len(req_skills) if req_skills else 1.0
    all_ratio = (len(matched_req) + 0.5 * len(matched_opt)) / len(all_job_skills)

    # Experience check
    exp_ok = True
    if jp.min_years_experience and cp.years_experience:
        if cp.years_experience < jp.min_years_experience:
            exp_ok = False

    # Grade determination
    if (req_ratio >= 0.75 or all_ratio >= 0.75) and exp_ok:
        return 3
    elif req_ratio >= 0.45 or all_ratio >= 0.45:
        return 2
    elif req_ratio >= 0.20 or all_ratio >= 0.20 or len(common_words) >= 15:
        return 1
    return 0


def build_qrels(max_jd_queries: int = 30, max_cv_queries: int = 30) -> Dict[str, Any]:
    cvs = load_documents(DATA_DIR / "raw" / "cvs.jsonl")
    jds = load_documents(DATA_DIR / "raw" / "jds.jsonl")

    # Select queries that have substantial text / skills
    valid_jds = [
        jd for jd in jds.values()
        if len(jd.raw_text.strip()) > 80 and len(job_from_document(jd).required_skills) > 0
    ][:max_jd_queries]

    valid_cvs = [
        cv for cv in cvs.values()
        if len(cv.raw_text.strip()) > 80 and len(profile_from_document(cv).skills) > 0
    ][:max_cv_queries]

    jd2cv_qrels: Dict[str, Dict[str, int]] = {}
    for jd in valid_jds:
        grades: Dict[str, int] = {}
        for cv in cvs.values():
            g = compute_pair_grade(jd, cv)
            if g > 0:
                grades[cv.doc_id] = g
        # Only retain query if it has at least 1 relevant candidate (grade >= 2)
        if any(v >= 2 for v in grades.values()):
            jd2cv_qrels[jd.doc_id] = grades

    cv2jd_qrels: Dict[str, Dict[str, int]] = {}
    for cv in valid_cvs:
        grades = {}
        for jd in jds.values():
            g = compute_pair_grade(jd, cv)
            if g > 0:
                grades[jd.doc_id] = g
        if any(v >= 2 for v in grades.values()):
            cv2jd_qrels[cv.doc_id] = grades

    return {
        "metadata": {
            "created_at": "2026-11-02",
            "rubric": "0=irrelevant, 1=marginally_relevant, 2=relevant, 3=highly_relevant",
            "total_jd_queries": len(jd2cv_qrels),
            "total_cv_queries": len(cv2jd_qrels),
            "corpus_cvs": len(cvs),
            "corpus_jds": len(jds),
        },
        "jd2cv": jd2cv_qrels,
        "cv2jd": cv2jd_qrels,
    }


def main():
    qrels = build_qrels(max_jd_queries=30, max_cv_queries=30)
    out_path = DATA_DIR / "benchmark_qrels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)

    meta = qrels["metadata"]
    print(f"Generated {out_path}:")
    print(f"  - JD->CV queries: {meta['total_jd_queries']}")
    print(f"  - CV->JD queries: {meta['total_cv_queries']}")
    sample_jd = list(qrels["jd2cv"].keys())[0]
    print(f"  - Sample JD query '{sample_jd}' has {len(qrels['jd2cv'][sample_jd])} labeled candidates.")


if __name__ == "__main__":
    main()

