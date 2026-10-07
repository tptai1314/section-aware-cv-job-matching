# Phương pháp (Methodology)

- Query = JD, candidate = CV.
- Nhãn 0–3 cho mỗi cặp (JD, CV).
- Baselines: BM25, single-vector, chunk max-sim; upper bound: cross-encoder.
- Section-aware multi-vector: tách section rule-based, embed từng section, ghép cặp fixed/cross, tổng hợp điểm.
- Không train model; embedding frozen, 384d, CPU.
