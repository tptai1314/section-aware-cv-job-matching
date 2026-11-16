import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "public_benchmark" / "resume-job-description-fit"
ANNOTATION_DIR = REPO_ROOT / "data" / "audit_annotation"

train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="latin1")
test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="latin1")

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
train_df["target"] = train_df["label"].map(LABEL_MAP)

unique_train_jds = train_df["job_description_text"].unique().tolist()
print(f"Total Unique Train JDs: {len(unique_train_jds)}")

# 1. Dev JDs from parser
dev_jds_set = set()
if (ANNOTATION_DIR / "train_dev_pool.json").exists():
    with open(ANNOTATION_DIR / "train_dev_pool.json", "r", encoding="utf-8") as f:
        d = json.load(f)
        dev_jds_set = set(d.get("dev_jds", []))
print(f"Parser Dev JDs (looked at during rule dev): {len(dev_jds_set)}")

# 2. Eval sample JDs from parser
eval_jds_set = set()
if (ANNOTATION_DIR / "raw_eval_samples.json").exists():
    with open(ANNOTATION_DIR / "raw_eval_samples.json", "r", encoding="utf-8") as f:
        d = json.load(f)
        eval_jds_set = set(item["text"] for item in d if item.get("doc_type") == "JD")
print(f"Parser Eval JDs (measured for ground-truth): {len(eval_jds_set)}")

# 3. Sensitivity analysis queries
query_counts = train_df["job_description_text"].value_counts()
valid_queries = query_counts[query_counts >= 10].index.tolist()
np.random.seed(42)
selected_queries_sensitivity = set(np.random.choice(valid_queries, size=min(50, len(valid_queries)), replace=False))
print(f"Sensitivity Analysis JDs: {len(selected_queries_sensitivity)}")

# Combine touched queries
all_touched_jds = dev_jds_set.union(eval_jds_set).union(selected_queries_sensitivity)
print(f"\nTotal Touched JDs in Train: {len(all_touched_jds)} / {len(unique_train_jds)} ({len(all_touched_jds)/len(unique_train_jds)*100:.1f}%)")

# Untouched clean queries
clean_train_jds = [j for j in unique_train_jds if j not in all_touched_jds]
print(f"Clean (Untouched / Completely Unseen) JDs in Train: {len(clean_train_jds)} ({len(clean_train_jds)/len(unique_train_jds)*100:.1f}%)")

# Filter train_df to clean JDs
clean_train_df = train_df[train_df["job_description_text"].isin(clean_train_jds)]
print(f"Clean Train Pairs: {len(clean_train_df)}")

# Check candidate counts per query in clean train
clean_counts = clean_train_df["job_description_text"].value_counts()
print(f"Clean JDs candidate count stats: min={clean_counts.min()}, median={clean_counts.median()}, mean={clean_counts.mean():.1f}, max={clean_counts.max()}")
print(f"Clean JDs with >= 5 candidates: {sum(clean_counts >= 5)}")
print(f"Clean JDs with >= 10 candidates: {sum(clean_counts >= 10)}")

# Check positive counts
pos_th1 = clean_train_df.groupby("job_description_text")["target"].apply(lambda s: (s >= 1).sum())
pos_th2 = clean_train_df.groupby("job_description_text")["target"].apply(lambda s: (s == 2).sum())
print(f"Clean JDs with >= 1 positive (rel >= 1): {sum(pos_th1 >= 1)} / {len(clean_train_jds)}")
print(f"Clean JDs with >= 1 positive (rel == 2): {sum(pos_th2 >= 1)} / {len(clean_train_jds)}")

# Save clean query list
output_data = {
    "total_train_jds": len(unique_train_jds),
    "dev_jds_count": len(dev_jds_set),
    "eval_jds_count": len(eval_jds_set),
    "sensitivity_jds_count": len(selected_queries_sensitivity),
    "total_touched_jds": len(all_touched_jds),
    "clean_train_jds_count": len(clean_train_jds),
    "clean_pairs_count": len(clean_train_df),
    "clean_jds_ge_5_candidates": int(sum(clean_counts >= 5)),
    "clean_jds_ge_10_candidates": int(sum(clean_counts >= 10)),
    "clean_jds_pos_th1": int(sum(pos_th1 >= 1)),
    "clean_jds_pos_th2": int(sum(pos_th2 >= 1)),
}

with open(REPO_ROOT / "results" / "train_clean_queries_audit.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print("\nSaved audit to results/train_clean_queries_audit.json")
