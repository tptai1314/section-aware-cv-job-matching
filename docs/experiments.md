# Experimental results

Every table must be marked **mock** or **real**.

## Setup (mock)

- Data: 20 CVs x 8 JDs, 160 labeled pairs (0-3), seed 42, English.
- Embedder: `mock` (deterministic hashing, 384d) unless stated otherwise.
- Query = JD, candidate = CV. Metric mean over 8 queries.

## Main comparison — mock

Run: `python scripts/benchmark.py --repeats 3` (results/benchmark.csv)

| method | params | query ms (mean) | NDCG@10 | MRR | Spearman |
|---|---|---|---|---|---|
| bm25 | | 0.29 | 0.7454 | 1.0000 | 0.3807 |
| single_vector | | 0.06 | 0.6641 | 0.8750 | 0.2271 |
| section_aware | cross, mean | 1.16 | 0.7611 | 0.9375 | 0.3424 |
| chunk_max_sim | | 0.43 | 0.6641 | 0.8750 | 0.2278 |
| single_pca | pca_dim=64 | 0.02 | 0.7460 | 1.0000 | 0.3606 |
| single_int8 | | 0.03 | 0.6636 | 0.8750 | 0.2236 |
| two_stage | top_k=10 | 0.71 | 0.7363 | 0.9375 | 0.1301 |

## Cross-encoder — mock

Run: `python scripts/rerank.py --mode ...` (results/rerank.csv)

| mode | NDCG@10 | query ms (mean) |
|---|---|---|
| full (all pairs) | 0.6485 | 371.7 |
| two-stage (section_aware, top-10) | 0.7052 | 124.9 |

## Ablation — mock

Run: `python scripts/ablation.py` (results/ablation.csv)

| variant | NDCG@10 |
|---|---|
| full (cross, mean) | 0.7611 |
| pairing_fixed | 0.7196 |
| agg_max | 0.6502 |
| drop_skills | 0.5986 |
| drop_experience | 0.6959 |
| drop_summary | 0.7456 |
| drop_education | 0.7893 |

## Artifacts

- results/pareto.png — quality vs latency, front: `single_pca`, `section_aware`
- results/heatmap_jd_0002_cv_0018.png — section-to-section similarity
- results/benchmark.csv, results/rerank.csv, results/ablation.csv

## Notes

- On **mock** the section-aware ranker beats the cross-encoder; the mock signal is
  lexical (token overlap), which favors it. Re-run everything on real data before
  drawing conclusions (week 9-10).
- `drop_education` improving NDCG is a mock artifact worth revisiting on real data.
