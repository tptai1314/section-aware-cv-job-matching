# Kế Hoạch Đánh Giá Thực Nghiệm Đóng Băng (Pre-Registered Evaluation Protocol)

> **Trạng thái**: ĐÃ ĐÓNG BĂNG (FROZEN / PRE-REGISTERED)  
> **Cam kết liêm chính học thuật**: Toàn bộ tiêu chí, metric, họ kiểm định giả thuyết thống kê, ngưỡng phân loại và bộ siêu tham số trong tài liệu này được xác lập và cam kết cố định trước mọi đợt chạy thực nghiệm bổ sung, nhằm loại trừ tuyệt đối rủi ro cherry-picking, p-hacking hoặc đổi tiêu chí sau khi biết kết quả (HARKing - Hypothesizing After the Results are Known).

---

## 1. Thang Nhãn và Ngưỡng Phân Khối Nhị Phân

### 1.1. Thang Nhãn Phân Cấp Gốc
Dữ liệu chuẩn hóa sử dụng thang nhãn rời rạc 3 mức $y_{q,p} \in \{0, 1, 2\}$:
- **Mức 0 (Không phù hợp / Unfit)**: Ứng viên thiếu hầu hết kỹ năng tiên quyết, kinh nghiệm không liên quan.
- **Mức 1 (Phù hợp tiềm năng / Potential Fit)**: Đáp ứng một phần yêu cầu kỹ năng nền tảng nhưng thiếu kinh nghiệm chuyên sâu hoặc kỹ năng ngách.
- **Mức 2 (Phù hợp tốt / Good Fit)**: Đáp ứng toàn diện các kỹ năng cốt lõi và số năm kinh nghiệm theo yêu cầu của bài đăng tuyển dụng (JD).

### 1.2. Quy Tắc Chuyển Đổi Nhị Phân (Binarization Thresholds $\tau$)
Đối với các thước đo xếp hạng nhị phân yêu cầu định nghĩa tập ứng viên liên quan ($R_q = \{p \in \mathcal{C}_q \mid y_{q,p} \ge \tau\}$):
- **Ngưỡng Nới Lỏng (Permissive Threshold, $\tau = 1$)**: 
  $$\mathbb{I}(y_{q,p} \ge 1)$$
  Ứng viên ở cả Mức 1 và Mức 2 đều được coi là liên quan. Phản ánh kịch bản sàng lọc vòng sơ loại (coarse screening / sourcing).
- **Ngưỡng Nghiêm Ngặt (Strict Threshold, $\tau = 2$)**: 
  $$\mathbb{I}(y_{q,p} \ge 2)$$
  Chỉ duy nhất ứng viên ở Mức 2 được tính là liên quan, Mức 1 bị coi là không đạt. Phản ánh kịch bản giới thiệu ứng viên vào vòng phỏng vấn chuyên sâu (shortlisting).

---

## 2. Danh Mục Thước Đo Hiệu Năng (Evaluation Metrics)

### 2.1. Thước Đo Chính (Primary Metric)
- **$\text{nDCG@10}$ (Normalized Discounted Cumulative Gain tại Top 10)**:
  - Khai thác trọn vẹn thang nhãn đa mức $\{0, 1, 2\}$ thông qua hàm mức độ thỏa mãn lũy thừa:
    $$\text{DCG@10}(q) = \sum_{i=1}^{\min(10, N_q)} \frac{2^{y_{q, \pi(i)}} - 1}{\log_2(i + 1)}$$
    $$\text{nDCG@10}(q) = \frac{\text{DCG@10}(q)}{\text{IDCG@10}(q)}$$
  - **Quy tắc giải quyết đồng hạng (Tie-breaking protocol)**: Khi các ứng viên có điểm số dự đoán bằng nhau trong danh sách xếp hạng, hệ thống thực hiện $50$ lần xáo trộn ngẫu nhiên (jittered tie-breaking with random permutations) và lấy giá trị kỳ vọng trung bình $\mathbb{E}[\text{nDCG@10}]$, triệt tiêu hiện tượng lạm phát điểm do thứ tự xuất hiện ngẫu nhiên trong bộ nhớ.

### 2.2. Các Thước Đo Phụ (Secondary Metrics)
- **$\text{MAP@10}$ (Mean Average Precision tại Top 10)**: Đo lường độ chuẩn xác trung bình trên cả hai ngưỡng $\tau \in \{1, 2\}$.
- **$\text{MRR}$ (Mean Reciprocal Rank)**: Đo lường vị trí nghịch đảo của ứng viên liên quan đầu tiên:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\min \{i \mid y_{q, \pi(i)} \ge \tau\}}$$
- **$\text{P@5}$ (Precision at Rank 5)**: Tỷ lệ ứng viên đạt chuẩn trong Top 5 kết quả đầu bảng.
- **Hệ số tương quan hạng Spearman $\rho$**: Tính tương quan giữa hoán vị dự đoán và thứ tự nhãn thực tế, sử dụng **tied midranks** và tự động loại trừ các truy vấn có phương sai nhãn bằng 0 ($\sigma_y^2 = 0$) để tránh chia cho 0.

---

## 3. Không Gian Đánh Giá Kép (Dual Evaluation Settings)

Mọi báo cáo thực nghiệm bắt buộc phải công bố song song kết quả trên cả hai môi trường:
1. **Setting A (Toàn bộ truy vấn / Unfiltered Macro Evaluation)**:
   - Giữ nguyên toàn bộ các truy vấn kiểm thử ($N_q = 71$ trên tập test, $N_q = 76$ trên tập train-clean).
   - Đối với các truy vấn không có bất kỳ ứng viên nào đạt ngưỡng liên quan ($|R_q| = 0$), gán điểm $0.0$ cho MAP, MRR, P@5. Điều này phản ánh trung thực năng lực của hệ thống trong thực tế khi gặp bài đăng tuyển dụng khó tuyển hoặc không có ứng viên phù hợp.
2. **Setting B (Tập truy vấn hợp lệ / Filtered Valid Positive Pool)**:
   - Chỉ tính toán trên các truy vấn có ít nhất một ứng viên đạt nhãn dương ($|R_q| \ge 1$ theo ngưỡng $\tau$).
   - Loại bỏ thiên lệch mẫu số khi mẫu số của Precision/Recall bằng 0, cho phép đo lường thuần túy chất lượng tái xếp hạng.

---

## 4. Họ Kiểm Định Thống Kê và Kiểm Soát Sai Số Đa Giả Thuyết

### 4.1. Họ Kiểm Định Cặp Cố Định ($L = 15$ Hypothesis Family)
Chúng tôi định nghĩa trước họ gồm đúng $15$ cặp so sánh đối đầu giữa mô hình đề xuất (**Section-Aware Hybrid**) và các đường cơ sở:
1. Hybrid vs. BM25 Baseline
2. Hybrid vs. Single-Vector Dense (all-MiniLM-L6-v2)
3. Hybrid vs. Chunk-MaxSim (ColBERT-style Late Interaction)
4. Hybrid vs. RRF Baseline (Dense + BM25 không trọng số)
5. Hybrid vs. Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
6. Hybrid vs. BGE-M3 Dense
7. Hybrid vs. Section-Dense Alone (không có BM25 và Taxonomy)
8. Hybrid vs. Section-BM25 Alone (không có Dense và Taxonomy)
9. Hybrid vs. Taxonomy Coverage Alone
10. Hybrid (Cross Pairing) vs. Diagonal Pairing (không ghép chéo)
11. Hybrid (Full Taxonomy L1-L3) vs. Flat Keyword Jaccard
12. Hybrid vs. BGE-Reranker-v2-m3
13. Setting A vs. Setting B (Tác động của việc lọc query)
14. Permissive ($\tau=1$) vs. Strict ($\tau=2$) trên MAP
15. Full Candidate Pool vs. Clean Candidate Pool (Kiểm tra rò rỉ dữ liệu)

### 4.2. Hiệu Chỉnh Đa Giả Thuyết (Holm-Bonferroni Step-Down)
Để kiểm soát tỷ lệ sai sót loại I toàn gia đình (Family-Wise Error Rate - FWER) ở mức nghiêm ngặt $\alpha_{\text{global}} = 0.05$:
- Sắp xếp $L = 15$ giá trị $p$-value theo thứ tự tăng dần: $p_{(1)} \le p_{(2)} \le \dots \le p_{(L)}$.
- Bác bỏ giả thuyết vô hiệu $H_{(k)}$ nếu và chỉ nếu:
  $$p_{(k)} \le \frac{\alpha}{L - k + 1}$$
- Dừng ngay tại giá trị $k$ đầu tiên không thỏa mãn bất đẳng thức; toàn bộ các giả thuyết còn lại được giữ nguyên.

### 4.3. Khoảng Tin Cậy Phi Tham Số (Bootstrap Resampling)
- Số lần lặp tái lấy mẫu có hoàn lại ở cấp độ truy vấn (Query-level Bootstrap): $B = 10{,}000$.
- Báo cáo khoảng tin cậy $95\%$ Confidence Intervals ($\text{CI}_{95\%}$) theo phương pháp phân vị (Percentile / BCa Bootstrap) cho mọi chênh lệch hiệu năng $\Delta = \text{Metric}_{\text{Ours}} - \text{Metric}_{\text{Baseline}}$.

---

## 5. Bảng Siêu Tham Số Đóng Băng Tuyệt Đối

Mọi mô hình đánh giá phải tuân thủ nghiêm ngặt các hằng số cấu hình sau:

| Tên Tham Số | Giá Trị Cố Định | Ý Nghĩa / Nguồn Gốc | Commit Git Gốc |
|---|---|---|---|
| $w_{\text{skills}}$ | `0.40` | Trọng số phần Kỹ năng | `55c22ae` |
| $w_{\text{experience}}$ | `0.35` | Trọng số phần Kinh nghiệm làm việc | `55c22ae` |
| $w_{\text{education}}$ | `0.15` | Trọng số phần Học vấn | `55c22ae` |
| $w_{\text{summary}}$ | `0.10` | Trọng số phần Tóm tắt mục tiêu | `55c22ae` |
| $\alpha$ | `0.60` | Tỷ lệ hòa trộn Dense trong Retrieval | `results/train_sensitivity_analysis.json` |
| $\beta$ | `0.60` | Tỷ lệ hòa trộn Retrieval trong Final Hybrid | `results/train_sensitivity_analysis.json` |
| Trọng số hiệu dụng Dense | `0.36` | $0.60 \times 0.60$ | Tính từ $\alpha, \beta$ |
| Trọng số hiệu dụng BM25 | `0.24` | $0.60 \times 0.40$ | Tính từ $\alpha, \beta$ |
| Trọng số hiệu dụng Taxonomy | `0.40` | $1.0 - \beta$ | Tính từ $\beta$ |
| $w_{\text{must}}$ | `0.85` | Trọng số kỹ năng bắt buộc (Must-have) | Rule-based prior |
| $w_{\text{nice}}$ | `0.15` | Trọng số kỹ năng ưu tiên (Nice-to-have) | Rule-based prior |
| Điều kiện biên $|K_q| = 0$ | $S_{\text{tax}} = 1.0$ | Fallback trung tính khi JD thiếu kỹ năng | Code `run_scientific_benchmark.py:461` |
| Hằng số RRF $k$ | `60` | Hằng số làm mượt xếp hạng nghịch đảo | Cormack et al. (2009) |
| Ngưỡng phạt kỹ năng | Không dùng trong benchmark | Tùy chọn runtime product serving ($\gamma = \text{Cov}(\text{Must})$ khi độ phủ $< 0.50$, tắt trong benchmark) | `src/cvjd/ranking/hybrid.py:197` |
| Cây kỹ năng Taxonomy | `615 concepts` | IT Skills Ontology v1.0.0-frozen | `9664f83` (SHA256: `b0ceddac...`) |
