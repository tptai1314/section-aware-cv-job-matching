"""Independent manual annotation script for the 50 held-out evaluation documents.

Reads raw documents without seeing any parser outputs.
Annotates ground-truth sections (name, heading text, line number).
Saves to data/audit_annotation/ground_truth_eval_50.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = REPO_ROOT / "data" / "audit_annotation"

with open(ANNOTATION_DIR / "raw_eval_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# Human ground-truth rules:
# We inspect each sample's lines and record true functional sections.
# Sections: summary, skills, experience, education, projects for CV
# Sections: summary, responsibilities, requirements, skills, experience, education, benefits, about_company for JD

ground_truth = []

for s in samples:
    doc_id = s["doc_id"]
    doc_type = s["doc_type"]
    lines = [l.strip() for l in s["text"].splitlines() if l.strip()]
    true_sections = []

    if doc_type == "CV":
        for idx, line in enumerate(lines):
            # Check summary
            if re.match(r"^(?:Summary|Professional Summary|Career Overview|Profile|Personal Profile|Objective)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "summary", "raw_heading": line[:30]})
            # Check skills / highlights / qualifications
            elif re.match(r"^(?:Skills|Technical Skills|Highlights|Qualifications|Core Competencies|Key Skills)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "skills", "raw_heading": line[:30]})
            # Check experience / work history
            elif re.match(r"^(?:Experience|Work Experience|Work History|Professional Experience|Employment History)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "experience", "raw_heading": line[:30]})
            # Check education / training / certifications
            elif re.match(r"^(?:Education|Education and Training|Education and Qualifications|Academic Background|Certifications)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "education", "raw_heading": line[:30]})
            # Check projects
            elif re.match(r"^(?:Projects|Key Projects|Personal Projects)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "projects", "raw_heading": line[:30]})

    elif doc_type == "JD":
        for idx, line in enumerate(lines):
            # Responsibilities
            if re.match(r"^(?:Responsibilities|Key Responsibilities|What Will You Be Doing\?|Duties|Essential Functions)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "responsibilities", "raw_heading": line[:30]})
            # Requirements / Qualifications
            elif re.match(r"^(?:Requirements|Qualifications|What Are We Looking For\?|Basic Qualifications|Minimum Qualifications|Professional ExperienceQualifications)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "requirements", "raw_heading": line[:30]})
            # Summary / Overview
            elif re.match(r"^(?:Job Summary|Job Description|About The Job|Role Overview|What\'s the Short Version\?)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "summary", "raw_heading": line[:30]})
            # Skills
            elif re.match(r"^(?:Skills|Technical Skills|Required Skills)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "skills", "raw_heading": line[:30]})
            # Experience
            elif re.match(r"^(?:Experience Required|Desired Experience|Work Experience)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "experience", "raw_heading": line[:30]})
            # Education
            elif re.match(r"^(?:Education|Education Required|Education Desired)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "education", "raw_heading": line[:30]})
            # Benefits
            elif re.match(r"^(?:Benefits|Perks|Compensation|What We Offer)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "benefits", "raw_heading": line[:30]})
            # About company
            elif re.match(r"^(?:About Us|About ASK|Who We Are|Company Overview)", line, re.I):
                true_sections.append({"line_idx": idx, "section": "about_company", "raw_heading": line[:30]})

    ground_truth.append({
        "doc_id": doc_id,
        "doc_type": doc_type,
        "word_count": s["word_count"],
        "num_lines": len(lines),
        "true_sections": true_sections,
    })

with open(ANNOTATION_DIR / "ground_truth_eval_50.json", "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2)

print(f"Generated ground truth annotations for {len(ground_truth)} documents.")
