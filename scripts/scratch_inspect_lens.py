from pathlib import Path
import pandas as pd

p = Path("data/public_benchmark/resume-job-description-fit/test.csv")
df = pd.read_csv(p, encoding="latin1")
cvs = df["resume_text"].unique()
jds = df["job_description_text"].unique()

cv_lens = [len(c.split()) for c in cvs]
jd_lens = [len(j.split()) for j in jds]

print(f"CV count: {len(cvs)}, words min={min(cv_lens)}, median={sorted(cv_lens)[len(cv_lens)//2]}, max={max(cv_lens)}, mean={sum(cv_lens)/len(cv_lens):.1f}")
print(f"JD count: {len(jds)}, words min={min(jd_lens)}, median={sorted(jd_lens)[len(jd_lens)//2]}, max={max(jd_lens)}, mean={sum(jd_lens)/len(jd_lens):.1f}")
