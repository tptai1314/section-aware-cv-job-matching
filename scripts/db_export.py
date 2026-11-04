"""Export real CV/JD documents from the Frevia PostgreSQL database.

Connects with DATABASE_URL (from .env or the environment — see
src/cvjd/utils/config.py) and writes cvjd-schema documents to

    data/raw/cvs.jsonl    CV documents (doc_id = "cv_<FreelancerProfile.id>")
    data/raw/jds.jsonl    JD documents (doc_id = "jd_<Job.id>")
    data/raw/manifest.json  doc_id -> database id mapping (for labels/feedback)

CV source of truth (priority order):
  1. Latest CONFIRMED "CvAnalysis" row -> JSON "payload"
     (skills / experience / education / projects / summary; defined in
     frevia-clone/docs/frevia-ai-cv-service-spec.md). Confidence/evidence and
     other meta are dropped so no extraction artifacts leak into the corpus.
  2. Structured fallback: "FreelancerSkill" -> "Skill"["name"],
     "FreelancerExperience", "FreelancerEducation".
  3. Legacy fallback: "FreelancerProfile"["experience"] / ["education"]
     (free-text string arrays).

The script introspects the database and only uses tables that actually exist,
so it works on databases where the CvAnalysis / FreelancerExperience /
FreelancerEducation migrations are not applied yet.

JD source: "Job" (non-deleted, with description) + "JobSkill" -> "Skill"["name"]
and "JobJobCategory" -> "JobCategory"["name"].

Privacy: personal identifiers (name, email, phone, address, website) are never
selected. Ids are opaque and only reachable through data/raw/manifest.json.

Usage:
    python scripts/db_export.py
    python scripts/db_export.py --output-dir data/raw --limit 50
    python scripts/db_export.py --skip-job-status DRAFT,CLOSED
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.data.loader import document_from_text  # noqa: E402
from cvjd.io import save_documents  # noqa: E402
from cvjd.schema import Document  # noqa: E402
from cvjd.utils.config import get_database_url  # noqa: E402

_JOB_STATUSES = ("DRAFT", "OPEN", "IN_PROGRESS", "COMPLETED", "CLOSED", "CANCELLED")


def _collapse(value: Optional[str]) -> Optional[str]:
    """Collapse whitespace/newlines; return None for empty strings."""
    if value is None:
        return None
    parts = value.split()
    return " ".join(parts) or None


def _fmt_date(value: Optional[str]) -> str:
    return value.strip() if value else ""


def _fmt_year(value: Optional[int]) -> str:
    return str(value) if value else ""


def _fmt_decimal(value: Any) -> Optional[str]:
    if value is None:
        return None
    f = float(value)
    return str(int(f)) if f == int(f) else str(value)


def _join(names: Sequence[str]) -> str:
    return ", ".join(dict.fromkeys(n.strip() for n in names if n.strip()))


def _pick(data: Dict[str, Any], *keys: str) -> Optional[Any]:
    """First non-None value among keys (dict rows keep only selected cols)."""
    for key in keys:
        if data.get(key) is not None:
            return data[key]
    return None


# --------------------------------------------------------------------------- #
# CV builders
# --------------------------------------------------------------------------- #

def _build_payload_text(payload: Dict[str, Any]) -> str:
    blocks: List[str] = []

    summary = _collapse(payload.get("summary"))
    if summary:
        blocks.append(f"SUMMARY\n- {summary}")

    skills = [item.get("name") for item in payload.get("skills", [])]
    joined = _join(skills)
    if joined:
        blocks.append(f"SKILLS\n- {joined}")

    experience = []
    for item in payload.get("experience", []):
        title = _collapse(item.get("title"))
        company = _collapse(item.get("company"))
        start = _fmt_date(item.get("start_date"))
        end = _fmt_date(item.get("end_date"))
        description = _collapse(item.get("description"))
        head = " - ".join(p for p in (title, company) if p)
        if not head and description:
            head = description
            description = None
        if start or end:
            head += f" ({start} -> {end})"
        lines = [head]
        if description:
            lines.append(f"    {description}")
        experience.append("\n".join(lines))
    if experience:
        blocks.append("EXPERIENCE\n" + "\n".join(experience))

    education = []
    for item in payload.get("education", []):
        degree = _collapse(item.get("degree"))
        school = _collapse(item.get("school"))
        start = _fmt_year(item.get("start_year"))
        end = _fmt_year(item.get("end_year"))
        head = " - ".join(p for p in (degree, school) if p)
        if start or end:
            head += f" ({start}-{end})"
        if head:
            education.append(head)
    if education:
        blocks.append("EDUCATION\n" + "\n".join(education))

    projects = []
    for item in payload.get("projects", []):
        name = _collapse(item.get("name"))
        description = _collapse(item.get("description"))
        technologies = _join(item.get("technologies", []))
        head = name or ""
        if description:
            head = f"{head}: {description}" if head else description
        if technologies:
            head = f"{head} [{technologies}]" if head else f"[{technologies}]"
        if head:
            projects.append(head)
    if projects:
        blocks.append("PROJECTS\n" + "\n".join(projects))

    return "\n\n".join(blocks)


def _build_structured_text(
    title: Optional[str],
    skills: Sequence[str],
    experiences: Sequence[Dict[str, Any]],
    educations: Sequence[Dict[str, Any]],
) -> str:
    blocks: List[str] = []

    if title:
        blocks.append(f"SUMMARY\n- {title}")

    joined = _join(skills)
    if joined:
        blocks.append(f"SKILLS\n- {joined}")

    lines = []
    for item in experiences:
        head = " - ".join(
            p for p in (_collapse(item.get("title")), _collapse(item.get("company"))) if p
        )
        start = _fmt_date(item.get("startDate"))
        end = _fmt_date(item.get("endDate"))
        if start or end:
            head += f" ({start} -> {end})"
        description = _collapse(item.get("description"))
        if description:
            lines.append(f"{head}\n    {description}")
        elif head:
            lines.append(head)
    if lines:
        blocks.append("EXPERIENCE\n" + "\n".join(lines))

    edu_lines = []
    for item in educations:
        degree = _collapse(item.get("degree"))
        school = _collapse(item.get("school"))
        head = " - ".join(p for p in (degree, school) if p)
        start = _fmt_year(item.get("startYear"))
        end = _fmt_year(item.get("endYear"))
        if start or end:
            head += f" ({start}-{end})"
        if head:
            edu_lines.append(head)
    if edu_lines:
        blocks.append("EDUCATION\n" + "\n".join(edu_lines))

    return "\n\n".join(blocks)


def _build_legacy_text(
    title: Optional[str],
    skills: Sequence[str],
    experience_free: Sequence[str],
    education_free: Sequence[str],
) -> str:
    """FreelancerProfile.experience/education are free-text string arrays."""
    blocks: List[str] = []

    if title:
        blocks.append(f"SUMMARY\n- {title}")

    joined = _join(skills)
    if joined:
        blocks.append(f"SKILLS\n- {joined}")

    if experience_free:
        blocks.append("EXPERIENCE\n" + "\n".join(str(x).strip() for x in experience_free if str(x).strip()))
    if education_free:
        blocks.append("EDUCATION\n" + "\n".join(str(x).strip() for x in education_free if str(x).strip()))

    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# JD builder
# --------------------------------------------------------------------------- #

def _build_jd_text(row: Dict[str, Any], skill_names: Sequence[str],
                   category_names: Sequence[str]) -> str:
    blocks: List[str] = []

    title = _collapse(row.get("title"))
    description = _collapse(row.get("description"))
    if title:
        blocks.append(title)
    if description:
        blocks.append(description)

    if category_names:
        blocks.append("CATEGORIES\n- " + _join(category_names))

    joined = _join(skill_names)
    if joined:
        blocks.append(f"SKILLS\n- {joined}")

    budget_min = _fmt_decimal(row.get("budgetMin"))
    budget_max = _fmt_decimal(row.get("budgetMax"))
    budget_parts = [p for p in (budget_min, budget_max) if p]
    if budget_parts:
        line = " - ".join(budget_parts)
        if row.get("budgetType"):
            line += f" ({row['budgetType']})"
        blocks.append(f"BUDGET\n{line}")

    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# DB catalog + access
# --------------------------------------------------------------------------- #

def _load_catalog(cur: Any) -> Set[Tuple[str, str]]:
    cur.execute(
        "SELECT table_name, column_name FROM information_schema.columns"
        " WHERE table_schema = 'public'"
    )
    return {(r["table_name"], r["column_name"]) for r in cur.fetchall()}


class Catalog:
    """Availability of tables/columns; unknown things are simply absent."""

    def __init__(self, columns: Set[Tuple[str, str]]):
        self._cols = columns
        by_table: Dict[str, Set[str]] = {}
        for table, col in columns:
            by_table.setdefault(table.lower(), set()).add(col.lower())
        self._by_table = by_table

    def table(self, name: str) -> bool:
        return name.lower() in self._by_table

    def has(self, table: str, *cols: str) -> bool:
        row = self._by_table.get(table.lower())
        return bool(row) and all(c.lower() in row for c in cols)


def _fetch_jds(cur: Any, skip_statuses: set[str], limit: int) -> List[Dict[str, Any]]:
    """Return rows (id, title, description, budgetMin, budgetMax, budgetType)."""
    placeholders = ", ".join(f"'{s}'" for s in skip_statuses) or "'__never__'"
    sql = (
        "SELECT j.\"id\", j.\"title\", j.\"description\", j.\"budgetMin\","
        " j.\"budgetMax\", j.\"budgetType\" FROM \"Job\" j"
        " WHERE j.\"deletedAt\" IS NULL"
        " AND j.\"description\" IS NOT NULL AND length(btrim(j.\"description\")) > 0"
        f" AND j.\"status\" NOT IN ({placeholders})"
        " ORDER BY j.\"id\""
    )
    if limit > 0:
        sql += f" LIMIT {limit}"
    cur.execute(sql)
    return cur.fetchall()


def _fetch_jd_extras(cur: Any, job_id: int) -> Tuple[List[str], List[str]]:
    cur.execute(
        "SELECT s.\"name\" FROM \"JobSkill\" js"
        ' JOIN "Skill" s ON s."id" = js."skillId"'
        ' WHERE js."jobId" = %s ORDER BY s."name"',
        (job_id,),
    )
    skills = [r["name"] for r in cur.fetchall() if r["name"]]
    cur.execute(
        "SELECT c.\"name\" FROM \"JobJobCategory\" jc"
        ' JOIN "JobCategory" c ON c."id" = jc."categoryId"'
        ' WHERE jc."jobId" = %s ORDER BY c."name"',
        (job_id,),
    )
    categories = [r["name"] for r in cur.fetchall() if r["name"]]
    return skills, categories


def _fetch_latest_confirmed_payload(cur: Any, profile_id: int) -> Optional[str]:
    cur.execute(
        "SELECT \"payload\" FROM \"CvAnalysis\""
        ' WHERE "freelancerProfileId" = %s AND "status" = \'CONFIRMED\''
        ' ORDER BY "createdAt" DESC LIMIT 1',
        (profile_id,),
    )
    row = cur.fetchone()
    return row["payload"] if row else None


def _fetch_structured_cv(
    cur: Any, profile_id: int, catalog: Catalog,
) -> Dict[str, Any]:
    title = None
    if catalog.has("FreelancerProfile", "title"):
        cur.execute(
            "SELECT \"title\" FROM \"FreelancerProfile\" WHERE \"id\" = %s",
            (profile_id,),
        )
        row = cur.fetchone()
        title = row["title"] if row else None

    skills: List[str] = []
    if catalog.table("FreelancerSkill") and catalog.table("Skill"):
        cur.execute(
            "SELECT s.\"name\" FROM \"FreelancerSkill\" fs"
            ' JOIN "Skill" s ON s."id" = fs."skillId"'
            ' WHERE fs."freelancerProfileId" = %s ORDER BY s."name"',
            (profile_id,),
        )
        skills = [r["name"] for r in cur.fetchall() if r["name"]]

    experiences: List[Dict[str, Any]] = []
    if catalog.has("FreelancerExperience", "freelancerProfileId"):
        cur.execute(
            "SELECT \"title\", \"company\", \"startDate\", \"endDate\", \"description\""
            ' FROM "FreelancerExperience" WHERE "freelancerProfileId" = %s'
            ' AND "deletedAt" IS NULL ORDER BY "id"',
            (profile_id,),
        )
        experiences = [
            {"title": r["title"], "company": r["company"], "startDate": r["startDate"],
             "endDate": r["endDate"], "description": r["description"]}
            for r in cur.fetchall()
        ]

    educations: List[Dict[str, Any]] = []
    if catalog.has("FreelancerEducation", "freelancerProfileId"):
        cur.execute(
            "SELECT \"degree\", \"school\", \"startYear\", \"endYear\""
            ' FROM "FreelancerEducation" WHERE "freelancerProfileId" = %s'
            ' AND "deletedAt" IS NULL ORDER BY "id"',
            (profile_id,),
        )
        educations = [
            {"degree": r["degree"], "school": r["school"],
             "startYear": r["startYear"], "endYear": r["endYear"]}
            for r in cur.fetchall()
        ]

    return {"title": title, "skills": skills, "experiences": experiences,
            "educations": educations}


def _fetch_profiles(cur: Any, limit: int) -> List[Dict[str, Any]]:
    """FreelancerProfile ids (legacy arrays are fetched separately)."""
    sql = 'SELECT "id" FROM "FreelancerProfile" ORDER BY "id"'
    if limit > 0:
        sql += f" LIMIT {limit}"
    cur.execute(sql)
    return [{"id": r["id"]} for r in cur.fetchall()]


def _fetch_legacy_texts(cur: Any, profile_id: int, catalog: Catalog) -> Tuple[List[str], List[str]]:
    if not catalog.has("FreelancerProfile", "experience", "education"):
        return [], []
    cur.execute(
        'SELECT "experience", "education" FROM "FreelancerProfile" WHERE "id" = %s',
        (profile_id,),
    )
    row = cur.fetchone()
    if row is None:
        return [], []
    return list(row["experience"] or []), list(row["education"] or [])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build_cv_document(
    payload_text: Optional[str],
    structured: Dict[str, Any],
    legacy: Tuple[List[str], List[str]],
    doc_id: str,
) -> Optional[Document]:
    """First usable text wins; None when the profile has no CV content."""
    if payload_text:
        return document_from_text(doc_id, "cv", payload_text)
    text = _build_structured_text(
        structured.get("title"),
        structured.get("skills", []),
        structured.get("experiences", []),
        structured.get("educations", []),
    )
    if text:
        return document_from_text(doc_id, "cv", text)
    exp_free, edu_free = legacy
    text = _build_legacy_text(
        structured.get("title"), structured.get("skills", []),
        exp_free, edu_free,
    )
    if not text:
        return None
    return document_from_text(doc_id, "cv", text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data" / "raw",
        help="where to write cvs.jsonl / jds.jsonl / manifest.json",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="max rows per collection (0 = no limit)",
    )
    parser.add_argument(
        "--skip-job-status", default="DRAFT",
        help="comma-separated Job statuses to exclude (default: DRAFT)",
    )
    args = parser.parse_args(argv)

    skip_statuses = {
        s.strip().upper() for s in args.skip_job_status.split(",") if s.strip()
    }
    unknown = skip_statuses - set(_JOB_STATUSES)
    if unknown:
        parser.error(f"unknown Job statuses: {', '.join(sorted(unknown))}")

    import psycopg
    from psycopg.rows import dict_row

    cv_docs: List[Document] = []
    jd_docs: List[Document] = []
    manifest_cvs: List[Dict[str, Any]] = []
    manifest_jds: List[Dict[str, Any]] = []
    notes: List[str] = []

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            catalog = Catalog(_load_catalog(cur))

            cv_source_available = catalog.has(
                "CvAnalysis", "payload", "status", "freelancerProfileId")
            if not cv_source_available:
                notes.append("CvAnalysis table absent -> fallback to structured/legacy CVs")
            if not catalog.has("FreelancerExperience", "freelancerProfileId"):
                notes.append("FreelancerExperience absent -> legacy experience array used")
            if not catalog.has("FreelancerEducation", "freelancerProfileId"):
                notes.append("FreelancerEducation absent -> legacy education array used")

            # ---- JDs -------------------------------------------------------
            for row in _fetch_jds(cur, skip_statuses, args.limit):
                job_id = row["id"]
                skills, categories = _fetch_jd_extras(cur, job_id)
                text = _build_jd_text(row, skills, categories)
                if not text:
                    continue
                doc_id = f"jd_{job_id}"
                jd_docs.append(document_from_text(doc_id, "jd", text))
                manifest_jds.append({"doc_id": doc_id, "jobId": job_id})

            # ---- CVs -------------------------------------------------------
            used_sources = {"CvAnalysis.payload": 0, "structured": 0, "legacy": 0}
            for profile in _fetch_profiles(cur, args.limit):
                profile_id = profile["id"]
                payload_text = None
                if cv_source_available:
                    payload_raw = _fetch_latest_confirmed_payload(cur, profile_id)
                    if payload_raw:
                        try:
                            payload = json.loads(payload_raw)
                            if isinstance(payload, dict):
                                payload_text = _build_payload_text(payload)
                        except ValueError:
                            payload_text = None
                structured = _fetch_structured_cv(cur, profile_id, catalog) \
                    if not payload_text else {}
                legacy = _fetch_legacy_texts(cur, profile_id, catalog) \
                    if not payload_text else ([], [])
                doc = build_cv_document(payload_text, structured, legacy,
                                        f"cv_{profile_id}")
                if doc is None:
                    continue
                cv_docs.append(doc)
                if payload_text:
                    source = "CvAnalysis.payload"
                elif _build_structured_text(
                    structured.get("title"), structured.get("skills", []),
                    structured.get("experiences", []), structured.get("educations", []),
                ):
                    source = "structured"
                else:
                    source = "legacy"
                used_sources[source] += 1
                manifest_cvs.append({
                    "doc_id": doc.doc_id,
                    "freelancerProfileId": profile_id,
                    "source": source,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_documents(args.output_dir / "cvs.jsonl", cv_docs)
    save_documents(args.output_dir / "jds.jsonl", jd_docs)

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "database": "frevia-clone (PostgreSQL, public schema)",
        "document_schema": "cvjd.Document",
        "privacy": "no personal identifiers exported; mapping in this file",
        "notes": notes,
        "cvs": manifest_cvs,
        "jds": manifest_jds,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"exported {len(cv_docs)} CVs, {len(jd_docs)} JDs -> {args.output_dir}")
    if notes:
        for note in notes:
            print(f"  note: {note}")
    print(f"  cv sources: {used_sources}")
    return 0


if __name__ == "__main__":
    sys.exit(main())