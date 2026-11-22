# Kế Hoạch Đánh Giá Thực Nghiệm Cố Định Trên Tập Query Sạch Của Train
## (Pre-Registered Evaluation Plan on Held-Out Clean Train Queries)

**Ngày lập**: 2026-11-17  
**Mã commit kế hoạch**: *(Được ghi nhận tại commit trước khi chạy thực nghiệm)*  
**Trạng thái**: ĐÓNG BĂNG TRƯỚC THỰC NGHIỆM (PRE-REGISTERED & FROZEN)

---

### 1. Bối Cảnh & Động Lực (Audit Context)
Để kiểm tra tính tổng quát hóa (generalization), đo lường khoảng cách giữa Train và Test (Train-Test Gap / Overfitting analysis), và trả lời câu hỏi: *"Các kết luận khoa học trên tập Test có nhất quán khi đánh giá trên tập Huấn luyện hay không?"*, nghiên cứu tiến hành đánh giá thực nghiệm có đối chứng trên tập Huấn luyện (`train.csv`).

Tuy nhiên, để đảm bảo tính toàn vẹn khoa học tuyệt đối và chống rò rỉ (leakage):
* Không được đánh giá trên các query JD đã từng được soi khi viết luật tách section (`dev_jds`: 168 JDs).
* Không được đánh giá trên các query JD đã dùng để gắn nhãn đo độ chính xác bộ tách (`eval_jds`: 25 JDs).
* Không được đánh giá trên các query JD đã dùng để quét tham số độ nhạy (`sensitivity_jds`: 50 JDs).

---

### 2. Định Nghĩa Tập Dữ Liệu Đánh Giá (Clean Held-Out Train Split)
* **Tổng số JD trong `train.csv`**: 280 JDs (6.241 cặp CV–JD).
* **Số JD đã từng bị tiếp xúc (Touched JDs)**: 204 JDs (72.9%).
* **Số JD Sạch Hoàn Toàn (Clean / Untouched JDs)**: **76 JDs** (27.1%).
* **Số cặp CV–JD tương ứng**: **1.829 cặp** (quy mô tương đương tập kiểm thử chính thức `test.csv` gồm 1.759 cặp).
* **Đặc tính Candidate Pool**:
  * Tối thiểu: 4 CVs/query; Trung vị: 11.5 CVs/query; Trung bình: 24.1 CVs/query; Tối đa: 111 CVs/query.
  * Số query có $\ge 5$ ứng viên: 72 / 76 queries (94.7%).
  * Số query có $\ge 10$ ứng viên: 45 / 76 queries (59.2%).

---

### 3. Hai Phương Án Đánh Giá Query (Dual Query Evaluation Protocol)
1. **Setting A (All 76 Clean Queries - Toàn diện)**:
   * Tính trung bình trên toàn bộ 76 queries sạch.
   * Các query không có positive candidate nhận điểm $0.0$.
2. **Setting B (Filtered TREC Standard - Có positive)**:
   * **Ngưỡng $\text{rel} \ge 1$**: Đánh giá trên **$N=71$ queries** có ít nhất 1 ứng viên đạt `Potential Fit` hoặc `Good Fit` (chỉ 5 query zero-positive).
   * **Ngưỡng $\text{rel} = 2$**: Đánh giá trên **$N=31$ queries** có ít nhất 1 ứng viên đạt `Good Fit` (45 query zero-positive).
   * Điểm Oracle đạt tuyệt đối $1.0000$.

---

### 4. Danh Sách Mô Hình Đánh Giá Đối Sánh (Model Suite)
1. **Floor: Random Ranking** (Sàn ngẫu nhiên kỳ vọng).
2. **BM25 / TF-IDF** (Baseline từ vựng chuẩn).
3. **MiniLM Single-Vector (Flat, 256 tokens)** (Biểu diễn vector toàn văn truyền thống).
4. **MiniLM Chunk-Mean (128 words, stride 64)** (Trung bình pooling các đoạn trượt).
5. **MiniLM Chunk-MaxSim (ColBERT-style)** (Tương tác muộn cực đại giữa các đoạn).
6. **MiniLM Section-Aware Dense (Đề xuất)** (Phân rã 4 section có chuẩn hóa trọng số động).
7. **Dense + BM25 ($L_0$ Baseline)** (Hòa trộn $\alpha=0.60$ không dùng taxonomy).
8. **Calibrated Hybrid ($L_0$ + Taxonomy Coverage)** ($\alpha=0.60, \beta=0.60$).
9. **Parameter-Free RRF + Taxonomy** (Reciprocal Rank Fusion $k=60$).
10. **Ceiling: Oracle Ranking** (Trần lý thuyết sắp xếp theo nhãn thật).

---

### 5. Chỉ Số Đánh Giá & Giao Thức Thống Kê (Metrics & Statistical Protocol)
* **Chỉ số chính (Primary Metric)**: **nDCG@10** (phản ánh chất lượng thứ hạng ứng viên top đầu).
* **Chỉ số phụ (Secondary Metrics)**: MRR, MAP, P@5.
* **Cơ chế phá đồng hạng**: Trung bình qua **50 lần xáo trộn ngẫu nhiên (jittered tie-breaking)** để triệt tiêu hoàn toàn thiên vị index file.
* **Họ giả thuyết kiểm định chính**:
  1. $H_1$: Section-Aware Dense vs. Single-Vector Dense (Kiểm chứng RQ1 trên train sạch).
  2. $H_2$: Lexical BM25 vs. Single-Vector Dense (Kiểm chứng RQ2 trên train sạch).
  3. $H_3$: Calibrated Hybrid vs. Dense + BM25 (Kiểm chứng RQ3 trên train sạch).
  4. $H_4$: Parameter-Free RRF vs. Calibrated Hybrid.
* **Quy chuẩn kiểm định**:
  * Hiệu số cặp trung bình $\bar{\Delta}$.
  * Khoảng tin cậy 95% Bootstrap CI (1.000 lần lấy mẫu lại có hoàn lại).
  * Paired two-sided $t$-test (kiểm định tham số).
  * Wilcoxon signed-rank test (kiểm định phi tham số).
  * Hiệu chỉnh đa giả thuyết **Holm-Bonferroni** ở mức ý nghĩa $\alpha = 0.05$.
* **Đo lường độ bao phủ ký tự chức năng (Functional Character Coverage)**:
  * Đo tỷ lệ ký tự rơi vào các section chức năng (`skills`, `experience`, `education`, `summary`, v.v.) so với đoạn mở đầu (`preamble`) trên toàn bộ văn bản CV và JD.

---

### 6. Cam Kết Tách Biệt & Minh Bạch (Independence Commitment)
* Kết quả trên 76 clean train queries được báo cáo **hoàn toàn riêng biệt** với bảng kết quả kiểm thử chính thức `test.csv`.
* Không dùng kết quả này để hồi cứu hay thay đổi bất kỳ hằng số nào đã đóng băng.
