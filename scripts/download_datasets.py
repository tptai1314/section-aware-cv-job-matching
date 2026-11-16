"""Download and cache batuhanmtl/job_resume_fit and IoannaDi05/resume-score-details from Hugging Face."""

import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_1 = REPO_ROOT / "data" / "public_benchmark" / "job_resume_fit"
OUT_DIR_2 = REPO_ROOT / "data" / "public_benchmark" / "resume-score-details"

OUT_DIR_1.mkdir(parents=True, exist_ok=True)
OUT_DIR_2.mkdir(parents=True, exist_ok=True)

print("=" * 70, flush=True)
print("1. Downloading batuhanmtl/job_resume_fit ...", flush=True)
print("=" * 70, flush=True)
try:
    ds1 = load_dataset("batuhanmtl/job_resume_fit")
    print("Downloaded ds1 splits:", list(ds1.keys()), flush=True)
    for split_name, split_data in ds1.items():
        csv_path = OUT_DIR_1 / f"{split_name}.csv"
        df = split_data.to_pandas()
        df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} ({len(df)} rows) to: {csv_path}", flush=True)
        print(f"Columns: {df.columns.tolist()}", flush=True)
except Exception as e:
    print(f"Error downloading batuhanmtl/job_resume_fit: {e}", flush=True)

print("\n" + "=" * 70, flush=True)
print("2. Downloading IoannaDi05/resume-score-details ...", flush=True)
print("=" * 70, flush=True)
try:
    ds2 = load_dataset("IoannaDi05/resume-score-details")
    print("Downloaded ds2 splits:", list(ds2.keys()), flush=True)
    for split_name, split_data in ds2.items():
        jsonl_path = OUT_DIR_2 / f"{split_name}.jsonl"
        df2 = split_data.to_pandas()
        df2.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        print(f"Saved {split_name} ({len(df2)} rows) to: {jsonl_path}", flush=True)
        print(f"Columns: {df2.columns.tolist()}", flush=True)
except Exception as e:
    print(f"Error downloading IoannaDi05/resume-score-details: {e}", flush=True)

print("\nFinished downloading datasets successfully!", flush=True)

