"""Generate mock CV/JD data matching src/cvjd/schema.py (English documents).

Hidden signal: label ~ skill overlap + experience match, plus label noise.
Usage: python scripts/make_mock_data.py --out-dir data --n-cvs 20 --n-jds 8
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.io import build_queries, save_documents, save_pairs, save_queries
from cvjd.schema import Document, Pair, Query, Section

SKILLS = [
    "python", "sql", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "machine learning", "deep learning", "nlp", "computer vision",
    "data visualization", "statistics", "excel", "spark", "aws", "docker",
    "kubernetes", "git", "rest api", "java", "javascript", "react",
    "node.js", "fastapi", "flask", "linux", "communication", "teamwork",
    "english",
]

TITLES = [
    "Data Scientist", "Machine Learning Engineer", "NLP Engineer",
    "Backend Engineer", "Data Analyst", "DevOps Engineer",
]

COMPANIES = [
    "FPT Software", "Viettel", "VNG", "Grab", "Shopee", "Tiki",
    "VNPT", "Batdongsan", "Zalo", "MoMo",
]

EDUCATION = ["Bachelor", "Master", "PhD"]

HEADINGS = {
    "summary": ["SUMMARY", "PROFILE"],
    "skills": ["SKILLS", "TECHNICAL SKILLS"],
    "experience": ["EXPERIENCE", "WORK EXPERIENCE"],
    "education": ["EDUCATION", "ACADEMIC BACKGROUND"],
}


def make_cv(
    idx: int,
    skills: Sequence[str],
    years: int,
    edu_level: int,
    rng: random.Random,
) -> Document:
    title = rng.choice(TITLES)
    company = rng.choice(COMPANIES)
    edu = EDUCATION[edu_level]
    start = max(0, 2025 - years)
    summary = (
        f"{title} with {years} years of experience, "
        "seeking new opportunities."
    )
    experience = f"- {start}-{start + max(1, years // 3)}: {title}, {company}"
    education = f"- {edu} degree, Vietnam National University"

    sections = {
        "summary": Section("summary", summary, rng.choice(HEADINGS["summary"]), 0),
        "skills": Section("skills", ", ".join(skills), rng.choice(HEADINGS["skills"]), 1),
        "experience": Section("experience", experience, rng.choice(HEADINGS["experience"]), 2),
        "education": Section("education", education, rng.choice(HEADINGS["education"]), 3),
    }
    raw_text = "\n\n".join(f"{s.heading_raw}\n{s.text}" for s in sections.values())
    return Document(
        doc_id=f"cv_{idx:04d}", doc_type="cv", lang="en",
        raw_text=raw_text, sections=sections,
    )


def make_jd(
    idx: int,
    skills: Sequence[str],
    years: int,
    rng: random.Random,
) -> Document:
    title = rng.choice(TITLES)
    summary = f"Hiring {title}, {years}+ years of experience required."
    experience = f"- Minimum {years} years of relevant experience."
    education = "- Bachelor's degree in a relevant field."

    sections = {
        "summary": Section("summary", summary, rng.choice(HEADINGS["summary"]), 0),
        "skills": Section("skills", ", ".join(skills), rng.choice(HEADINGS["skills"]), 1),
        "experience": Section("experience", experience, rng.choice(HEADINGS["experience"]), 2),
        "education": Section("education", education, rng.choice(HEADINGS["education"]), 3),
    }
    raw_text = "\n\n".join(f"{s.heading_raw}\n{s.text}" for s in sections.values())
    return Document(
        doc_id=f"jd_{idx:04d}", doc_type="jd", lang="en",
        raw_text=raw_text, sections=sections,
    )


def _label(overlap: float, years_ok: bool, rng: random.Random, noise: float) -> int:
    score = 0.6 * overlap + 0.4 * (1.0 if years_ok else 0.0)
    if score >= 0.8:
        label = 3
    elif score >= 0.55:
        label = 2
    elif score >= 0.3:
        label = 1
    else:
        label = 0
    if rng.random() < noise:
        label = min(3, max(0, label + rng.choice((-1, 1))))
    return label


def build_mock(
    n_cvs: int = 20,
    n_jds: int = 8,
    seed: int = 42,
    noise: float = 0.15,
    coverage: float = 1.0,
) -> Tuple[List[Document], List[Document], List[Pair], List[Query]]:
    rng = random.Random(seed)

    cv_skills = [rng.sample(SKILLS, k=8) for _ in range(n_cvs)]
    cv_years = [rng.randint(0, 12) for _ in range(n_cvs)]
    cv_edu = [rng.randint(0, 2) for _ in range(n_cvs)]
    jd_skills = [rng.sample(SKILLS, k=rng.randint(5, 7)) for _ in range(n_jds)]
    jd_years = [rng.randint(0, 10) for _ in range(n_jds)]

    cvs = [make_cv(i + 1, cv_skills[i], cv_years[i], cv_edu[i], rng) for i in range(n_cvs)]
    jds = [make_jd(i + 1, jd_skills[i], jd_years[i], rng) for i in range(n_jds)]

    combos = [(c, j) for c in range(n_cvs) for j in range(n_jds)]
    n_labeled = max(1, round(len(combos) * coverage))
    labeled = set(rng.sample(combos, n_labeled))

    pairs: List[Pair] = []
    for c, j in sorted(labeled):
        overlap = len(set(cv_skills[c]) & set(jd_skills[j])) / len(jd_skills[j])
        years_ok = cv_years[c] >= jd_years[j]
        pairs.append(Pair(
            cv_id=cvs[c].doc_id,
            jd_id=jds[j].doc_id,
            label=_label(overlap, years_ok, rng, noise),
        ))

    queries = build_queries(pairs, [d.doc_id for d in jds], [d.doc_id for d in cvs])
    return cvs, jds, pairs, queries


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--n-cvs", type=int, default=20)
    parser.add_argument("--n-jds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--coverage", type=float, default=1.0)
    args = parser.parse_args(argv)

    cvs, jds, pairs, queries = build_mock(
        n_cvs=args.n_cvs, n_jds=args.n_jds, seed=args.seed,
        noise=args.noise, coverage=args.coverage,
    )
    save_documents(args.out_dir / "raw" / "cvs.jsonl", cvs)
    save_documents(args.out_dir / "raw" / "jds.jsonl", jds)
    save_pairs(args.out_dir / "labels" / "pairs.jsonl", pairs)
    save_queries(args.out_dir / "labels" / "queries.jsonl", queries)
    print(
        f"wrote {len(cvs)} CVs, {len(jds)} JDs, {len(pairs)} pairs, "
        f"{len(queries)} queries -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
