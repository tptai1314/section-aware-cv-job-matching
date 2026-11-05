"""Build the VietResume-Bench public benchmark from Hugging Face datasets.

Inputs:
  - data/public_benchmark/resumes-raw-pdf-for-ocr/data/train-00000-of-00001.parquet (1,585 CVs)
  - data/public_benchmark/vietnamese-job-descriptions/data.parquet (606k Vietnamese JDs)

Outputs:
  - data/public_benchmark/vietresume/cvs.jsonl (1,585 CVs with sectioning & lang)
  - data/public_benchmark/vietresume/jds.jsonl (60 JDs across 10 IT categories)
  - data/public_benchmark/vietresume/benchmark_qrels.json (3-level relevance: 0, 1, 2)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Add src to python path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.data.loader import detect_lang
from cvjd.data.preprocessing import sections_to_dict, split_sections

OUT_DIR = REPO_ROOT / "data" / "public_benchmark" / "vietresume"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Technical skill profiles for 10 core IT categories
ROLE_PROFILES = [
    {
        "category": "Java",
        "pattern": r"java\b",
        "must": ["java", "spring", "hibernate", "jvm"],
        "opt": ["sql", "mysql", "postgresql", "maven", "microservice", "restful api", "oop", "kafka"],
    },
    {
        "category": "Python / AI",
        "pattern": r"python|machine learning|ai engineer|computer vision|nlp",
        "must": ["python", "machine learning", "deep learning", "ai", "computer vision", "pytorch", "tensorflow", "opencv"],
        "opt": ["nlp", "pandas", "numpy", "scikit-learn", "keras", "hugging face", "llm", "data"],
    },
    {
        "category": "React / Frontend",
        "pattern": r"react|frontend|front-end|vue|angular",
        "must": ["react", "frontend", "front end", "vue", "angular", "javascript", "typescript"],
        "opt": ["html", "css", "redux", "tailwind", "bootstrap", "nextjs", "webpack", "responsive"],
    },
    {
        "category": "NodeJS / Backend",
        "pattern": r"node|backend|back-end|golang",
        "must": ["node", "nodejs", "express", "nest", "backend", "back end", "golang", "go"],
        "opt": ["mongodb", "redis", "postgresql", "restful api", "docker", "microservice", "mysql"],
    },
    {
        "category": "PHP / Web",
        "pattern": r"php|laravel|wordpress|yii",
        "must": ["php", "laravel", "wordpress", "yii", "codeigniter"],
        "opt": ["mysql", "html", "css", "javascript", "apache", "mvc", "jquery"],
    },
    {
        "category": "Mobile Developer",
        "pattern": r"flutter|react native|android|ios|swift|mobile developer",
        "must": ["flutter", "android", "ios", "swift", "kotlin", "react native", "mobile"],
        "opt": ["dart", "xcode", "gradle", "app store", "google play"],
    },
    {
        "category": "Data Engineer / Analyst",
        "pattern": r"data engineer|data analyst|bi developer|\bdata\b",
        "must": ["data analyst", "data engineer", "sql", "power bi", "tableau", "etl", "data warehouse"],
        "opt": ["python", "pandas", "spark", "bigquery", "bi", "reporting", "dashboard"],
    },
    {
        "category": "DevOps / Cloud",
        "pattern": r"devops|cloud|aws|docker|kubernetes",
        "must": ["devops", "docker", "kubernetes", "aws", "ci/cd", "cloud", "linux"],
        "opt": ["jenkins", "terraform", "git", "azure", "ansible", "monitoring", "prometheus"],
    },
    {
        "category": "QA / Tester",
        "pattern": r"tester|qa|qc|automation test|test engineer",
        "must": ["tester", "qa", "qc", "test case", "manual test", "automation test"],
        "opt": ["selenium", "jmeter", "postman", "jira", "bug", "test plan", "regression"],
    },
    {
        "category": "Business Analyst",
        "pattern": r"business analyst|\bba\b|product owner",
        "must": ["business analyst", "ba", "product owner", "requirement", "user story"],
        "opt": ["jira", "uml", "wireframe", "agile", "scrum", "sql", "flowchart"],
    },
]


def build_cvs(cv_parquet_path: Path) -> list[dict]:
    """Process and export all 1,585 CVs with section splitting and lang detection."""
    print("Loading CV parquet...", flush=True)
    table = pq.read_table(cv_parquet_path, columns=["label", "text"])
    df = table.to_pandas()
    print(f"Total CVs in parquet: {len(df)}")

    cv_docs = []
    out_file = OUT_DIR / "cvs.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            doc_id = f"cv_{idx + 1}"
            raw_text = str(row["text"]).strip()
            lang = detect_lang(raw_text)
            domain_label = int(row["label"])  # 1: IT, 0: All-domains

            # Section splitting
            raw_sections = split_sections(raw_text)
            sec_dict = sections_to_dict(raw_sections)
            sections_json = {
                name: {
                    "name": sec.name,
                    "text": sec.text,
                    "heading_raw": sec.heading_raw,
                    "order": sec.order,
                }
                for name, sec in sec_dict.items()
            }

            doc = {
                "doc_id": doc_id,
                "doc_type": "cv",
                "lang": lang,
                "raw_text": raw_text,
                "sections": sections_json,
                "metadata": {
                    "domain_label": domain_label,
                    "is_it": (domain_label == 1),
                },
                "schema_version": "1.0.0",
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            cv_docs.append(doc)

    print(f"Exported {len(cv_docs)} CVs to {out_file}")
    return cv_docs


def build_jds(jd_parquet_path: Path) -> list[dict]:
    """Sample and export 60 representative IT JDs across 10 categories."""
    print("Loading JD parquet...", flush=True)
    table = pq.read_table(
        jd_parquet_path,
        columns=[
            "id",
            "job_title",
            "company_name",
            "job_industry",
            "job_description",
            "requirements",
            "benefits",
            "salary",
            "experience_level",
        ],
    )
    df = table.to_pandas()
    it_jds = df[
        df["job_industry"].str.contains("IT|Công nghệ thông tin|Phần mềm", case=False, na=False)
        & (df["job_description"].str.len() > 100)
        & (df["requirements"].str.len() > 100)
    ].copy()

    selected_jds = []
    jd_counter = 1
    out_file = OUT_DIR / "jds.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for profile in ROLE_PROFILES:
            cat = profile["category"]
            pat = profile["pattern"]
            matches = it_jds[it_jds["job_title"].str.contains(pat, case=False, na=False)]
            # Pick 6 distinct JDs per category = 60 total
            sample = matches.drop_duplicates(subset=["job_title"]).head(6)

            for _, row in sample.iterrows():
                doc_id = f"jd_{jd_counter}"
                jd_counter += 1

                title = str(row["job_title"]).strip()
                desc = str(row["job_description"]).strip()
                req = str(row["requirements"]).strip()
                benefits = str(row.get("benefits", "")).strip()

                raw_text = f"{title}\n\nMÔ TẢ CÔNG VIỆC\n{desc}\n\nYÊU CẦU CÔNG VIỆC\n{req}"
                if benefits:
                    raw_text += f"\n\nQUYỀN LỢI\n{benefits}"

                lang = detect_lang(raw_text)

                raw_sections = split_sections(raw_text)
                sec_dict = sections_to_dict(raw_sections)
                sections_json = {
                    name: {
                        "name": sec.name,
                        "text": sec.text,
                        "heading_raw": sec.heading_raw,
                        "order": sec.order,
                    }
                    for name, sec in sec_dict.items()
                }

                doc = {
                    "doc_id": doc_id,
                    "doc_type": "jd",
                    "lang": lang,
                    "raw_text": raw_text,
                    "sections": sections_json,
                    "metadata": {
                        "original_id": int(row["id"]),
                        "job_title": title,
                        "company_name": str(row.get("company_name", "")),
                        "role_category": cat,
                        "salary": str(row.get("salary", "")),
                        "experience_level": str(row.get("experience_level", "")),
                    },
                    "schema_version": "1.0.0",
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                selected_jds.append(doc)

    print(f"Exported {len(selected_jds)} JDs to {out_file}")
    return selected_jds


def grade_pair(jd_doc: dict, cv_doc: dict, role_profile: dict) -> int:
    """Evaluate pair relevance: 0 (No Fit), 1 (Potential Fit), 2 (Good Fit)."""
    # 1. Non-IT CVs are immediate No Fit (Hard Negatives)
    if cv_doc["metadata"]["domain_label"] == 0:
        return 0

    cv_text = cv_doc["raw_text"].lower()
    must_pats = role_profile["must"]
    opt_pats = role_profile["opt"]

    must_hits = sum(1 for p in must_pats if re.search(r"\b" + re.escape(p) + r"\b", cv_text))
    opt_hits = sum(1 for p in opt_pats if re.search(r"\b" + re.escape(p) + r"\b", cv_text))

    # Good Fit: strong core stack alignment (>= 2 must-have tech skills or 1 must-have + 2 optional skills)
    if must_hits >= 2 or (must_hits >= 1 and opt_hits >= 2):
        return 2

    # Potential Fit: adjacent software engineering background with partial skill overlap
    if must_hits >= 1 or opt_hits >= 2:
        return 1

    return 0


def build_qrels(cv_docs: list[dict], jd_docs: list[dict]) -> dict:
    """Build 3-level relevance qrels for both jd2cv and cv2jd."""
    print("Computing relevance matrix across pairs...", flush=True)

    # Build profile lookup by category
    profile_map = {p["category"]: p for p in ROLE_PROFILES}

    jd2cv_qrels: dict[str, dict[str, int]] = {}
    pair_grade_cache: dict[tuple[str, str], int] = {}

    for jd in jd_docs:
        cat = jd["metadata"]["role_category"]
        p = profile_map[cat]
        grades: dict[str, int] = {}

        for cv in cv_docs:
            g = grade_pair(jd, cv, p)
            pair_grade_cache[(jd["doc_id"], cv["doc_id"])] = g
            if g > 0:
                grades[cv["doc_id"]] = g

        # Ensure query has candidates
        if any(v == 2 for v in grades.values()):
            jd2cv_qrels[jd["doc_id"]] = grades

    # Select 60 representative IT CV queries (6 per category) for cv2jd
    cv2jd_qrels: dict[str, dict[str, int]] = {}
    it_cvs = [cv for cv in cv_docs if cv["metadata"]["domain_label"] == 1]

    # Sample 60 IT CVs with highest relevance coverage
    cv_scores = []
    for cv in it_cvs:
        rel_count = sum(1 for jd in jd_docs if pair_grade_cache.get((jd["doc_id"], cv["doc_id"]), 0) == 2)
        cv_scores.append((cv, rel_count))
    cv_scores.sort(key=lambda x: -x[1])
    selected_cv_queries = [cv for cv, _ in cv_scores[:60]]

    for cv in selected_cv_queries:
        grades = {}
        for jd in jd_docs:
            g = pair_grade_cache.get((jd["doc_id"], cv["doc_id"]), 0)
            if g > 0:
                grades[jd["doc_id"]] = g
        if any(v == 2 for v in grades.values()):
            cv2jd_qrels[cv["doc_id"]] = grades

    # Track languages
    query_languages_jd = {jd["doc_id"]: jd["lang"] for jd in jd_docs if jd["doc_id"] in jd2cv_qrels}
    query_languages_cv = {cv["doc_id"]: cv["lang"] for cv in selected_cv_queries if cv["doc_id"] in cv2jd_qrels}

    qrels_data = {
        "metadata": {
            "benchmark_name": "VietResume-Bench",
            "created_at": time.strftime("%Y-%m-%d"),
            "rubric": "0=No Fit, 1=Potential Fit, 2=Good Fit",
            "total_jd_queries": len(jd2cv_qrels),
            "total_cv_queries": len(cv2jd_qrels),
            "corpus_cvs": len(cv_docs),
            "corpus_jds": len(jd_docs),
            "jd_query_languages": query_languages_jd,
            "cv_query_languages": query_languages_cv,
        },
        "jd2cv": jd2cv_qrels,
        "cv2jd": cv2jd_qrels,
    }

    out_file = OUT_DIR / "benchmark_qrels.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(qrels_data, f, ensure_ascii=False, indent=2)

    print(f"Exported benchmark qrels to {out_file}:")
    print(f"  - JD->CV queries: {len(jd2cv_qrels)}")
    print(f"  - CV->JD queries: {len(cv2jd_qrels)}")
    return qrels_data


def export_pairs_csv(cv_docs: list[dict], jd_docs: list[dict], qrels: dict) -> Path:
    """Export a balanced 3,000-pair CSV file matching the format of resume-job-description-fit."""
    import csv
    import random

    random.seed(42)
    cv_map = {c["doc_id"]: c for c in cv_docs}
    jd_map = {j["doc_id"]: j for j in jd_docs}
    all_cv_ids = list(cv_map.keys())

    pairs = []
    for jd_id, jd in jd_map.items():
        labeled = qrels["jd2cv"].get(jd_id, {})
        good_fit_ids = [cid for cid, score in labeled.items() if score == 2]
        pot_fit_ids = [cid for cid, score in labeled.items() if score == 1]

        sampled_good = good_fit_ids[:15]
        sampled_pot = random.sample(pot_fit_ids, min(len(pot_fit_ids), 15))

        no_fit_candidates = [cid for cid in all_cv_ids if cid not in labeled]
        sampled_no_fit = random.sample(no_fit_candidates, min(len(no_fit_candidates), 20))

        jd_lang = jd["lang"].upper()

        for cid in sampled_good:
            cv_lang = cv_map[cid]["lang"].upper()
            pairs.append({
                "job_id": jd_id,
                "resume_id": cid,
                "job_description_text": jd["raw_text"],
                "resume_text": cv_map[cid]["raw_text"],
                "label": 2,
                "label_text": "Good Fit",
                "lang_pair": f"{jd_lang}-{cv_lang}",
            })
        for cid in sampled_pot:
            cv_lang = cv_map[cid]["lang"].upper()
            pairs.append({
                "job_id": jd_id,
                "resume_id": cid,
                "job_description_text": jd["raw_text"],
                "resume_text": cv_map[cid]["raw_text"],
                "label": 1,
                "label_text": "Potential Fit",
                "lang_pair": f"{jd_lang}-{cv_lang}",
            })
        for cid in sampled_no_fit:
            cv_lang = cv_map[cid]["lang"].upper()
            pairs.append({
                "job_id": jd_id,
                "resume_id": cid,
                "job_description_text": jd["raw_text"],
                "resume_text": cv_map[cid]["raw_text"],
                "label": 0,
                "label_text": "No Fit",
                "lang_pair": f"{jd_lang}-{cv_lang}",
            })

    csv_out = OUT_DIR / "pairs.csv"
    fieldnames = ["job_id", "resume_id", "job_description_text", "resume_text", "label", "label_text", "lang_pair"]

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Exported {len(pairs)} pairs to {csv_out}")
    return csv_out


def main():
    t0 = time.time()
    cv_path = REPO_ROOT / "data" / "public_benchmark" / "resumes-raw-pdf-for-ocr" / "data" / "train-00000-of-00001.parquet"
    jd_path = REPO_ROOT / "data" / "public_benchmark" / "vietnamese-job-descriptions" / "data.parquet"

    if not cv_path.exists():
        print(f"Error: {cv_path} not found!", file=sys.stderr)
        return 1
    if not jd_path.exists():
        print(f"Error: {jd_path} not found!", file=sys.stderr)
        return 1

    cv_docs = build_cvs(cv_path)
    jd_docs = build_jds(jd_path)
    qrels = build_qrels(cv_docs, jd_docs)
    export_pairs_csv(cv_docs, jd_docs, qrels)

    print(f"\nAll benchmark files generated successfully in {time.time()-t0:.2f}s!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

