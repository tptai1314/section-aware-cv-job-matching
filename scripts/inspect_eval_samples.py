import json
from pathlib import Path

p = Path("data/audit_annotation/raw_eval_samples.json")
with open(p, "r", encoding="utf-8") as f:
    samples = json.load(f)

print(f"Total samples: {len(samples)}")
cv_count = sum(1 for s in samples if s["doc_type"] == "CV")
jd_count = sum(1 for s in samples if s["doc_type"] == "JD")
print(f"CVs: {cv_count}, JDs: {jd_count}")

# Check ground_truth_eval_50.json
gt_p = Path("data/audit_annotation/ground_truth_eval_50.json")
with open(gt_p, "r", encoding="utf-8") as f:
    gt = json.load(f)
print(f"Total ground truth entries: {len(gt)}")
