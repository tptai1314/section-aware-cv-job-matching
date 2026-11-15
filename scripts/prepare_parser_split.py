"""Partition TRAIN dataset into dev set (for rule engineering) and held-out evaluation set (for ground-truth measurement).

Adheres strictly to zero-leakage guidelines:
1. Excludes all 68 overlapping test CVs from the dev set.
2. Uses fixed seed (42) for deterministic splitting.
3. Samples 25 CVs and 25 JDs exclusively from the held-out eval set.
4. Saves sampled documents to data/audit_annotation/raw_eval_samples.json for independent manual annotation.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "public_benchmark" / "resume-job-description-fit"
ANNOTATION_DIR = REPO_ROOT / "data" / "audit_annotation"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="latin1")
test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="latin1")

test_cv_set = set(test_df["resume_text"].unique())
test_jd_set = set(test_df["job_description_text"].unique())

unique_train_cvs = train_df["resume_text"].unique().tolist()
unique_train_jds = train_df["job_description_text"].unique().tolist()

print(f"Total Unique Train CVs: {len(unique_train_cvs)}")
print(f"Total Unique Train JDs: {len(unique_train_jds)}")

# Check overlap
overlap_cvs = set(unique_train_cvs).intersection(test_cv_set)
overlap_jds = set(unique_train_jds).intersection(test_jd_set)
print(f"Overlap CVs in Train & Test: {len(overlap_cvs)}")
print(f"Overlap JDs in Train & Test: {len(overlap_jds)}")

# Clean non-overlapping CVs for splitting
clean_train_cvs = [c for c in unique_train_cvs if c not in overlap_cvs]
clean_train_jds = [j for j in unique_train_jds if j not in overlap_jds]

print(f"Clean non-overlapping Train CVs available: {len(clean_train_cvs)}")
print(f"Clean non-overlapping Train JDs available: {len(clean_train_jds)}")

# Deterministic train_dev (60%) and train_eval (40%) split
rng = np.random.RandomState(42)

shuffled_cv_indices = rng.permutation(len(clean_train_cvs))
n_cv_dev = int(0.60 * len(clean_train_cvs))
dev_cv_indices = shuffled_cv_indices[:n_cv_dev]
eval_cv_indices = shuffled_cv_indices[n_cv_dev:]

dev_cvs = [clean_train_cvs[i] for i in dev_cv_indices]
eval_pool_cvs = [clean_train_cvs[i] for i in eval_cv_indices]

shuffled_jd_indices = rng.permutation(len(clean_train_jds))
n_jd_dev = int(0.60 * len(clean_train_jds))
dev_jd_indices = shuffled_jd_indices[:n_jd_dev]
eval_jd_indices = shuffled_jd_indices[n_jd_dev:]

dev_jds = [clean_train_jds[i] for i in dev_jd_indices]
eval_pool_jds = [clean_train_jds[i] for i in eval_jd_indices]

print(f"Dev Pool: {len(dev_cvs)} CVs, {len(dev_jds)} JDs (Exclusively for rule engineering)")
print(f"Eval Pool: {len(eval_pool_cvs)} CVs, {len(eval_pool_jds)} JDs (Held-out for evaluation)")

# Sample exactly 25 CVs and 25 JDs from the held-out eval pool
selected_eval_cv_indices = rng.choice(len(eval_pool_cvs), size=25, replace=False)
selected_eval_jd_indices = rng.choice(len(eval_pool_jds), size=25, replace=False)

eval_samples_cv = [eval_pool_cvs[i] for i in selected_eval_cv_indices]
eval_samples_jd = [eval_pool_jds[i] for i in selected_eval_jd_indices]

# Save split indices and raw eval samples for annotation
split_manifest = {
    "dev_cv_count": len(dev_cvs),
    "dev_jd_count": len(dev_jds),
    "eval_pool_cv_count": len(eval_pool_cvs),
    "eval_pool_jd_count": len(eval_pool_jds),
    "n_eval_samples_cv": 25,
    "n_eval_samples_jd": 25,
    "seed": 42,
}

with open(ANNOTATION_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
    json.dump(split_manifest, f, indent=2)

# Save dev pool so developer script can only load dev pool
dev_dataset = {
    "dev_cvs": dev_cvs,
    "dev_jds": dev_jds,
}
with open(ANNOTATION_DIR / "train_dev_pool.json", "w", encoding="utf-8") as f:
    json.dump(dev_dataset, f, indent=2)

# Save raw eval samples (without any parser output!) for manual annotation
eval_docs = []
for i, text in enumerate(eval_samples_cv):
    eval_docs.append({
        "doc_id": f"eval_cv_{i+1:02d}",
        "doc_type": "CV",
        "text": text,
        "word_count": len(text.split()),
    })

for i, text in enumerate(eval_samples_jd):
    eval_docs.append({
        "doc_id": f"eval_jd_{i+1:02d}",
        "doc_type": "JD",
        "text": text,
        "word_count": len(text.split()),
    })

with open(ANNOTATION_DIR / "raw_eval_samples.json", "w", encoding="utf-8") as f:
    json.dump(eval_docs, f, indent=2)

print(f"\n[OK] Partitioned dataset successfully! Manifest and 50 raw samples saved to: {ANNOTATION_DIR}")
