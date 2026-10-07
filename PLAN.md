# KẾ HOẠCH THỰC HIỆN ĐỀ TÀI NGHIÊN CỨU KHOA HỌC

## Đề tài
**Đánh đổi giữa chất lượng xếp hạng và hiệu năng của so khớp CV–JD đa vector theo section trên dữ liệu của hệ thống tuyển dụng**

---

## I. THÔNG TIN ĐỀ TÀI

| Mục | Nội dung |
|---|---|
| Tên đề tài | Đánh đổi chất lượng xếp hạng – hiệu năng của so khớp CV–JD đa vector theo section |
| Lĩnh vực | NLP, Information Retrieval, Efficient AI |
| Loại nghiên cứu | Ứng dụng – thực nghiệm (không đề xuất thuật toán mới) |
| Dữ liệu | Nội bộ đồ án (không dùng dataset bên ngoài) |
| Mô hình | Embedding có sẵn, frozen, chạy CPU |
| Thời gian dự kiến | 12–14 tuần |
| Sản phẩm | Bài báo/báo cáo NCKH + code + bộ nhãn |

---

## II. MỤC TIÊU

### Mục tiêu chung
Đánh giá thực nghiệm biểu diễn đa vector theo section trong so khớp CV–JD, xét đồng thời **chất lượng xếp hạng**, **hiệu năng tính toán** và **khả năng giải thích**.

### Mục tiêu cụ thể
1. So sánh section-aware với baseline (BM25, single-vector, chunk max-sim).
2. Xác định mức chất lượng và chi phí so với cross-encoder/LLM.
3. Đo mức chất lượng còn lại khi áp dụng kỹ thuật tăng tốc.
4. Kiểm chứng giả định ghép cặp cố định bằng ghép cặp chéo.

---

## III. CÂU HỎI NGHIÊN CỨU VÀ GIẢ THUYẾT

| RQ | Câu hỏi | Giả thuyết |
|---|---|---|
| RQ1 | Section-aware có xếp hạng tốt hơn single-vector và chunk max-sim? | H1: NDCG@10 không thấp hơn, có điểm thành phần giải thích được |
| RQ2 | Section-aware đạt gần mô hình nặng đến mức nào, rẻ hơn bao nhiêu? | H2: Nhanh hơn rõ rệt, chất lượng chênh nhỏ |
| RQ3 | Kỹ thuật tăng tốc làm giảm chất lượng bao nhiêu? | H3: Pipeline hai tầng giảm latency đáng kể, mất ít chất lượng |
| RQ4 | Ghép cặp chéo có tốt hơn ghép cặp cố định? | H4: Có, khi section CV–JD không tương ứng một-một |

---

## IV. PHẠM VI

**Trong phạm vi:**
- Embedding có sẵn (384 chiều), frozen, chạy CPU.
- Dữ liệu nội bộ đồ án.
- Đánh giá thực nghiệm, không train model mới.

**Ngoài phạm vi:**
- Xây dataset quy mô lớn.
- Benchmark nhiều mô hình.
- Đánh giá fairness/bias.

---

## V. TIẾN ĐỘ THỰC HIỆN (12–14 tuần)

### Giai đoạn 1: Nền tảng (Tuần 1–2)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 1 | Chốt đề bài, RQ, giả thuyết. Dựng repo, schema, mock data. | README, schema.py, mock data |
| 2 | IO, embedder mock, matcher, aggregate, evaluate + test | Pipeline chạy end-to-end bằng mock |

**Mốc 1 (cuối tuần 2):** Pipeline chạy được với mock, metric đúng toán, test pass.

---

### Giai đoạn 2: Hoàn thiện phương pháp (Tuần 3–5)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 3 | Pipeline hoàn chỉnh 3 phương pháp. Benchmark latency/RAM. | run_experiment.py, benchmark.py, CSV |
| 4 | Split section rule-based. Baseline BM25. | split.py, baseline_bm25.py |
| 5 | Chunk max-sim. Embedder thật (SentenceTransformer). | chunking.py, so sánh mock vs thật |

**Mốc 2 (cuối tuần 5):** Có đủ baseline + section-aware + chunking, chạy trên embedding thật.

---

### Giai đoạn 3: Nâng cao (Tuần 6–8)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 6 | Cross-encoder reranker. Upper bound. | rerank.py, kết quả heavy baseline |
| 7 | Pipeline hai tầng. Nén embedding (quantization/PCA). | Kết quả RQ3 |
| 8 | Ablation: bỏ section, heatmap, phân tích lỗi. Vẽ Pareto. | Bảng ablation, biểu đồ Pareto |

**Mốc 3 (cuối tuần 8):** Trả lời được RQ1–RQ4 trên mock + embedding thật.

---

### Giai đoạn 4: Data thật (Tuần 9–10)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 9 | Kiểm kê data. Viết loader. Hoàn thiện rule tách section. | Loader, section splitter |
| 10 | Gán nhãn 100–150 cặp. Đo đồng thuận. Chạy lại pipeline. | Bộ nhãn, kết quả thực nghiệm thật |

**Mốc 4 (cuối tuần 10):** Kết quả thực nghiệm trên data thật, không sửa code lõi.

---

### Giai đoạn 5: Viết báo cáo (Tuần 11–14)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 11 | Viết mở đầu, cơ sở nghiên cứu (bổ sung tài liệu thật). | Chương 1–2 |
| 12 | Viết phương pháp, mô tả data, metric. | Chương 3 |
| 13 | Viết kết quả, bảng biểu, Pareto, ablation. | Chương 4 |
| 14 | Thảo luận, giới hạn, kết luận. Chỉnh sửa, nộp. | Báo cáo hoàn chỉnh |

**Mốc 5 (cuối tuần 14):** Nộp báo cáo + code + bộ nhãn.

---

## VI. PHÂN CÔNG CÔNG VIỆC

| Vai trò | Trách nhiệm |
|---|---|
| Thành viên 1 (chính) | Code pipeline, embedding, matching, benchmark |
| Thành viên 2 | Data, gán nhãn, split section, annotation guideline |
| Thành viên 3 (nếu có) | Đánh giá, viết báo cáo, vẽ biểu đồ |
| GVHD | Định hướng, review, kiểm tra tính khoa học |

*Nếu làm một mình: gộp tất cả, ưu tiên pipeline trước, data sau.*

---

## VII. CÔNG CỤ VÀ TÀI NGUYÊN

**Phần mềm:**
- Python 3.10+, numpy, pytest, pyyaml.
- sentence-transformers, rank_bm25.
- matplotlib/seaborn (vẽ biểu đồ).
- Git + GitHub.

**Phần cứng:**
- CPU (đủ cho embedding 384 chiều).
- RAM tối thiểu 8GB.
- Không cần GPU.

**Mô hình dự kiến:**
- `all-MiniLM-L6-v2` (384d, nhẹ, CPU tốt).
- Hoặc `keepitreal/vietnamese-sbert` nếu data chủ yếu tiếng Việt.
- Cross-encoder: `ms-marco-MiniLM-L-6-v2` cho upper bound.

---

## VIII. RỦI RO VÀ BIỆN PHÁP

| Rủi ro | Mức độ | Biện pháp |
|---|---|---|
| Data thật ít, nhãn nhỏ | Cao | Bổ sung bootstrap CI, phân tích định tính |
| Rule tách section kém | Trung bình | Test trên mẫu, ghi rõ giới hạn |
| Mock overfit | Trung bình | Mock có tín hiệu ẩn + nhiễu, không random thuần |
| Embedding tiếng Việt kém | Trung bình | Thử 2–3 model, chọn tốt nhất |
| Cross-encoder quá chậm | Thấp | Chỉ chạy subset nhỏ, dùng làm upper bound |
| Kết quả mock khác thật | Cao | Không sửa code lõi, chỉ đổi loader + embedder |

---

## IX. SẢN PHẨM ĐẦU RA

1. **Bài báo/báo cáo NCKH** hoàn chỉnh (14–20 trang).
2. **Code repo** công khai: pipeline, baseline, benchmark, test.
3. **Bộ nhãn** 100–150 cặp CV–JD đã gán.
4. **Biểu đồ Pareto** chất lượng–độ trễ.
5. **Bảng kết quả** NDCG, MRR, Spearman, latency, RAM.
6. **Slide thuyết trình** bảo vệ.

---

## X. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG

| Tiêu chí | Ngưỡng |
|---|---|
| Pipeline chạy end-to-end | Bắt buộc |
| Metric đúng toán (test pass) | Bắt buộc |
| Có ít nhất 3 baseline | Bắt buộc |
| Đo cả chất lượng lẫn hiệu năng | Bắt buộc |
| Trả lời đủ 4 RQ | Bắt buộc |
| Biểu đồ Pareto | Bắt buộc |
| Bộ nhãn có đo đồng thuận | Nên có |
| Ablation đầy đủ | Nên có |

---

## XI. ĐÓNG GÓP DỰ KIẾN

1. Đánh giá có kiểm soát **tách yếu tố section khỏi chunking**.
2. Kiểm chứng thực nghiệm **giả định ghép cặp cố định**.
3. Phân tích **đánh đổi chất lượng–hiệu năng** trên data thực.
4. **Cơ chế điểm thành phần** giúp giải thích xếp hạng.

Đóng góp nằm ở **phân tích thực nghiệm**, không phải thuật toán mới.

---

## XII. LỊCH TRÌNH TỔNG HỢP

```
Tuần 1–2   ████ Nền tảng + mock pipeline
Tuần 3–5   ████ Baseline + embedding thật
Tuần 6–8   ████ Rerank + 2 tầng + ablation
Tuần 9–10  ████ Data thật + gán nhãn
Tuần 11–14 ████ Viết báo cáo + nộp
```

**Mốc kiểm tra:**
- Cuối tuần 2: pipeline mock chạy.
- Cuối tuần 5: đủ baseline + embedding thật.
- Cuối tuần 8: trả lời RQ trên mock.
- Cuối tuần 10: kết quả data thật.
- Cuối tuần 14: nộp báo cáo.

---

## XIII. GHI CHÚ KHI VIẾT BÁO CÁO

- Phần **Cơ sở nghiên cứu** phải bổ sung tài liệu thật đã đọc (Li et al. 2020, Resume2Vec, v.v.).
- Các số liệu (số CV, JD, nhãn) **điền sau khi kiểm kê data**.
- Ghi rõ **mock hay real** trong mọi bảng kết quả.
- Nêu rõ **giới hạn** ở cuối báo cáo.
