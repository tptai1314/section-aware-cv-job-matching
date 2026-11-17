# Sổ tay Ký hiệu Toán học và Công thức Toàn văn
## Hệ thống Ghép cặp Hai chiều CV--Job Đa Vector Nhận biết Phân mục (Section-Aware Multi-Vector Matching)

---

## 1. Bảng Ký hiệu Toán học & Thuật ngữ (Mathematical Notations)

### 1.1. Tập hợp và Không gian Dữ liệu
| Ký hiệu | Ý nghĩa toán học / kỹ thuật | Phạm vi / Giá trị thực tế |
| :--- | :--- | :--- |
| $\mathcal{D}$ | Không gian toàn thể các tài liệu tuyển dụng | Toàn bộ văn bản thô |
| $\mathcal{J} \subset \mathcal{D}$ | Tập hợp các bài đăng mô tả công việc (Job Descriptions - JD) | 351 JD độc nhất trên benchmark |
| $\mathcal{C} \subset \mathcal{D}$ | Tập hợp các hồ sơ ứng viên (Curriculum Vitae - CV) | 1.051 CV độc nhất trên benchmark |
| $D_q \in \mathcal{D}$ | Tài liệu truy vấn (Query document: JD trong tuyển dụng, CV trong gợi ý việc) | $1$ tài liệu đầu vào |
| $\mathcal{C}_{cand}$ | Tập hợp các ứng viên đối sánh (Candidate candidate pool) | $\{D_1, D_2, \dots, D_N\}$ |
| $N$ | Kích thước tập ứng viên đối sánh cho mỗi truy vấn | $N=25$ (cnamuangtoun); $N \approx 209$ (med2425) |
| $\mathbf{y}_q$ | Vector nhãn phù hợp thực tế (Ground-truth relevance labels) | $\mathbf{y}_q \in \{0, 1, 2\}^N$ |
| $y_{q, i}$ | Nhãn độ phù hợp giữa truy vấn $q$ và ứng viên $i$ | $0$: No Fit, $1$: Potential Fit, $2$: Good Fit |
| $\pi$ | Hoán vị thứ tự xếp hạng của hệ thống | Hoán vị trên $\{1, 2, \dots, N\}$ |
| $\pi^*$ | Hoán vị thứ hạng tối ưu tối đa hóa nDCG@K | Nghiệm tối ưu của bài toán |
| $\mathcal{S}_{domain}$ | Tập các nhãn phân mục chức năng chuẩn mực | $\{\text{skills}, \text{exp}, \text{edu}, \text{sum}, \text{proj}\}$ |
| $M$ | Số lượng phân mục chức năng có mặt trong một tài liệu | $1 \le M \le 5$ (trung bình 3 mục/CV) |
| $\mathcal{S}(D)$ | Tập các phân mục chức năng thực sự có mặt trong tài liệu $D$ | $\mathcal{S}(D) \subseteq \mathcal{S}_{domain}$ |
| $\mathcal{S}_{\text{matched}}$ | Tập các phân mục giao nhau thực sự được ghép cặp | $\mathcal{S}(D_q) \cap \mathcal{S}_{\text{active}}$ |
| $\mathcal{T}$ | Tập hợp kỹ năng trích xuất từ bài đăng tuyển dụng (JD) | Trung bình 4.4 kỹ năng/JD |
| $\mathcal{C}_{skills}$ | Tập hợp kỹ năng của ứng viên (CV) sau khi mở rộng qua cây phân loại | Trung bình 10.3 kỹ năng/CV |
| $\mathcal{M}$ | Tập các danh sách xếp hạng thành phần trong RRF | $\mathcal{M} = \{\text{Dense}, \text{BM25}\}$ |
| $r_m(d)$ | Thứ hạng (rank index, 1-based) của tài liệu $d$ trong danh sách $m$ | $1 \le r_m(d) \le N$ |

---

### 1.2. Biểu diễn Vector và Đại số Tuyến tính
| Ký hiệu | Ý nghĩa toán học / kỹ thuật | Phạm vi / Giá trị thực tế |
| :--- | :--- | :--- |
| $d$ | Số chiều của không gian vector nhúng ngữ nghĩa | $d=384$ (MiniLM); $d=1024$ (BGE-M3) |
| $T_m$ | Chuỗi ký tự văn bản của phân mục chức năng thứ $m$ | Đoạn văn bản thô |
| $T_{raw}$ | Toàn văn chuỗi ký tự của tài liệu gốc | Văn bản đầy đủ chưa cắt xén |
| $\mathcal{E}(\cdot)$ | Mô hình Bi-Encoder đóng băng ánh xạ văn bản thành vector | $\mathbb{R}^{|T|} \rightarrow \mathbb{R}^d$ |
| $\mathbf{v}_m$ | Vector đặc biểu diễn phân mục chức năng thứ $m$ | $\mathbf{v}_m = \mathcal{E}(T_m) \in \mathbb{R}^d$ |
| $\mathbf{v}_{full}$ | Vector đặc phẳng biểu diễn toàn văn tài liệu | $\mathbf{v}_{full} = \mathcal{E}(T_{raw}) \in \mathbb{R}^d$ |
| $\mathbf{v}_q^{s_m}$ | Vector của phân mục $s_m$ thuộc tài liệu truy vấn $D_q$ | Vector đơn vị $\in \mathbb{R}^d$ |
| $\mathbf{v}_c^{s_k}$ | Vector của phân mục $s_k$ thuộc tài liệu ứng viên $D_c$ | Vector đơn vị $\in \mathbb{R}^d$ |
| $\|\mathbf{v}\|_2$ | Chuẩn Euclidean ($L_2$-norm) của vector $\mathbf{v}$ | $\sqrt{\sum_{i=1}^d v_i^2} = 1.0$ |
| $\text{sim}(\mathbf{u}, \mathbf{w})$ | Độ tương đồng Cosine giữa hai vector $\mathbf{u}$ và $\mathbf{w}$ | $[-1.0, 1.0]$ |
| $c(s_m)$ | Điểm căn chỉnh cực đại của phân mục $s_m$ qua ghép chéo | $\max_{s_k} \text{sim}(\mathbf{v}_q^{s_m}, \mathbf{v}_c^{s_k}) \in [-1.0, 1.0]$ |

---

### 1.3. Các Hằng số Thực nghiệm và Trọng số Hệ thống (Constants)
| Hằng số | Giá trị | Ý nghĩa nghiệp vụ và lý do thiết kế |
| :--- | :---: | :--- |
| $w_{\text{skills}}$ | $0.40$ ($0.35$) | Trọng số phân mục Kỹ năng kỹ thuật cốt lõi (điều kiện cần thiết nhất) |
| $w_{\text{exp}}$ | $0.35$ | Trọng số phân mục Kinh nghiệm làm việc thực tế (chứng minh năng lực) |
| $w_{\text{proj}}$ | $0.15$ | Trọng số phân mục Dự án / Portfolio (minh chứng bổ sung) |
| $w_{\text{edu}}$ | $0.10$ | Trọng số phân mục Học vấn & Bằng cấp (điều kiện nền tảng) |
| $w_{\text{sum}}$ | $0.05$ | Trọng số phân mục Tóm tắt bản thân (khi có xuất hiện) |
| $\alpha$ | $0.60$ | Tỷ lệ ưu tiên cho ngữ nghĩa đặc ($60\%$ Dense, $40\%$ BM25) |
| $1 - \alpha$ | $0.40$ | Tỷ lệ cho điểm từ vựng BM25 (chống bỏ sót mã hiệu / từ khóa hiếm) |
| $\beta$ | $0.60$ | Hệ số điều phối đóng góp của điểm độ phủ cây phân loại (Taxonomy) |
| $k$ | $60$ | Hằng số làm mịn kinh điển trong công thức RRF (ngăn lấn át top-1) |
| $w_{\text{must}}$ | $0.80$ | Tỷ trọng điểm dành cho kỹ năng bắt buộc (Must-Have skills) |
| $w_{\text{nice}}$ | $0.20$ | Tỷ trọng điểm dành cho kỹ năng khuyến khích (Nice-To-Have skills) |
| $\theta_{\text{must}}$ | $0.50$ | Ngưỡng tỷ lệ Must-Have tối thiểu; dưới $50\%$ bị phạt nhân tử $\gamma$ |
| $\epsilon$ | $10^{-8}$ | Hằng số làm mịn mẫu số trong chuẩn hóa Min-Max (chống chia cho 0) |
| $K$ | $5, 10$ | Ngưỡng cắt vị trí trong các thước đo xếp hạng (nDCG@K, P@K) |
| $B$ | $10.000$ | Số lần lấy mẫu lặp ghép cặp có hoàn lại trong Bootstrap CI |
| $\alpha_{\text{stat}}$ | $0.05$ | Mức ý nghĩa thống kê danh định cho kiểm định giả thuyết |
| $m$ | $15$ | Số lượng phép thử so sánh trong họ giả thuyết chính kiểm soát FWER |

---

## 2. Hệ thống Công thức Toán học Toàn văn

### 2.1. Tối ưu Hóa Thứ tự Xếp hạng Tổng quát
Mục tiêu của hệ thống là tìm hoán vị $\pi^*$ trên tập $N$ ứng viên đối sánh:
$$\pi^* = \arg\max_{\pi} \text{nDCG}@K\big( \pi(\mathcal{C}_{cand}), \mathbf{y}_q \big)$$

---

### 2.2. Biểu thức Chính quy Tách Ranh giới Tiêu đề Dính chữ
Quy tắc Regex nhận diện ranh giới ký tự viết hoa đứng liền sau chữ thường:
$$\text{Pattern}_{\text{split}} = \texttt{r'(?<=[a-z])(?=[A-Z][a-z]+)'}$$

---

### 2.3. Chuẩn hóa Vector Đơn vị ($L_2$-norm)
Mỗi vector đầu ra từ mô hình Bi-Encoder được chiếu về mặt cầu đơn vị:
$$\mathbf{v} \leftarrow \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^d v_i^2}}, \quad \text{sao cho } \|\mathbf{v}\|_2 = 1$$

---

### 2.4. Độ tương đồng Cosine giữa hai Vector
Khi vector đã được chuẩn hóa $L_2$, độ tương đồng Cosine tương đương tích vô hướng:
$$\text{sim}(\mathbf{u}, \mathbf{w}) = \frac{\mathbf{u} \cdot \mathbf{w}}{\|\mathbf{u}\|_2 \|\mathbf{w}\|_2} = \sum_{i=1}^d u_i w_i \in [-1.0, 1.0]$$

---

### 2.5. Ghép cặp Chéo Cực đại (Cross-Section Pairing)
Mỗi mục $s_m$ trong truy vấn $D_q$ tìm mục có độ tương đồng lớn nhất trong ứng viên $D_c$:
$$c(s_m) = \max_{s_k \in \mathcal{S}(D_c)} \text{sim}\big(\mathbf{v}_q^{s_m}, \mathbf{v}_c^{s_k}\big)$$

---

### 2.6. Tái Chuẩn hóa Trọng số Động (Dynamic Weight Renormalization)
Điểm tương đồng ngữ nghĩa phân mục có bù trừ cho hồ sơ khuyết mục:
$$S_{sem}(D_q, D_c) = \frac{\sum_{m \in \mathcal{S}_{\text{matched}}} w_m \cdot c(s_m)}{\sum_{m \in \mathcal{S}_{\text{matched}}} w_m}$$
*Tính chất*: $\sum_{m \in \mathcal{S}_{\text{matched}}} w_m' = 1.0$ với $w_m' = \frac{w_m}{\sum_{k} w_k}$, bảo đảm $S_{sem} \in [0, 1]$.

---

### 2.7. Chuẩn hóa Tuyến tính Min-Max Theo Từng Truy vấn
Áp dụng đưa điểm số các nhánh về cùng thang đo đoạn $[0, 1]$ trước khi kết hợp:
$$S^{\text{norm}}(c, j) = \frac{S(c, j) - \min_{k \in \mathcal{C}_{cand}} S(k, j)}{\max_{k \in \mathcal{C}_{cand}} S(k, j) - \min_{k \in \mathcal{C}_{cand}} S(k, j) + \epsilon}$$

---

### 2.8. Điểm Xếp hạng Lai Hiệu chuẩn (Calibrated Hybrid Scoring)
Hợp nhất điểm ngữ nghĩa đặc, điểm từ vựng BM25 và độ phủ cây phân loại:
$$S_{hybrid}(c, j) = \alpha \cdot S_{sem}^{\text{norm}}(c, j) + (1 - \alpha) \cdot S_{lex}^{\text{norm}}(c, j) + \beta \cdot S_{tax}(c, j)$$
Với giá trị hiệu chuẩn mặc định:
$$S_{hybrid}(c, j) = 0.60 \cdot S_{sem}^{\text{norm}}(c, j) + 0.40 \cdot S_{lex}^{\text{norm}}(c, j) + 0.60 \cdot S_{tax}(c, j)$$

---

### 2.9. Phối hợp Xếp hạng Nghịch đảo Không Tham số (Parameter-Free RRF)
Kết hợp danh sách thứ hạng mà không cần huấn luyện hay dò tìm tham số:
$$S_{RRF}(d) = \sum_{m \in \mathcal{M}} \frac{1}{k + r_m(d)} = \frac{1}{60 + r_{dense}(d)} + \frac{1}{60 + r_{bm25}(d)}$$

---

### 2.10. Độ phủ Kỹ năng Bất đối xứng (Asymmetric Skill Coverage)
Đo tỷ lệ đáp ứng của ứng viên đối với tập yêu cầu của bài đăng tuyển dụng:
$$\text{Cov}(\mathcal{T}) = \frac{\sum_{t \in \mathcal{T}} \mathbb{I}(t \in \mathcal{C}_{skills}) \cdot w_t}{\sum_{t \in \mathcal{T}} w_t}$$
trong đó $\mathbb{I}(\cdot)$ là hàm chỉ thị (indicator function, nhận giá trị $1$ nếu điều kiện đúng, ngược lại $0$).

---

### 2.11. Điểm Kỹ năng 2 Nhóm và Hệ số Phạt Kỹ năng Sống còn
$$S_{skills} = 0.80 \cdot \text{Cov}(\text{MustHave}) + 0.20 \cdot \text{Cov}(\text{NiceToHave})$$
Nếu ứng viên đáp ứng dưới một nửa số kỹ năng bắt buộc:
$$\text{Nếu } \text{Cov}(\text{MustHave}) < 0.50 \implies S_{skills} \leftarrow S_{skills} \cdot \text{Cov}(\text{MustHave})$$

---

### 2.12. Tương tác Trễ ColBERT MaxSim trên Sliding Chunks
Điểm tương tác trễ cực đại qua các đoạn chia lát:
$$S_{\text{MaxSim}}(Q, D) = \frac{1}{|Q|} \sum_{q \in Q} \max_{d \in D} \cos(q, d)$$

---

### 2.13. Độ lợi Tích lũy Giảm dần Chuẩn hóa (nDCG@K)
$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{y_i} - 1}{\log_2(i + 1)}$$
$$\text{IDCG}@K = \sum_{i=1}^{\min(K, |R_q|)} \frac{2^{y_i^*} - 1}{\log_2(i + 1)}$$
$$\text{nDCG}@K = \begin{cases} \dfrac{\text{DCG}@K}{\text{IDCG}@K} & \text{nếu } \text{IDCG}@K > 0 \\ 0.0 & \text{nếu } \text{IDCG}@K = 0 \text{ (Kịch bản Setting A)} \end{cases}$$

---

### 2.14. Điểm Nghịch đảo Trung bình của Vị trí Đầu tiên (MRR)
$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$
với $\text{rank}_q = \min \{ i \in \{1, \dots, N\} \mid y_{q, \pi(i)} \ge 1 \}$.

---

### 2.15. Độ chính xác Trung bình Trung bình (MAP)
$$\text{AP}(q) = \frac{1}{\sum_{k=1}^N \mathbb{I}(y_{q, k} \ge 1)} \sum_{k=1}^N P@k \cdot \mathbb{I}(y_{q, \pi(k)} \ge 1)$$
$$\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}(q)$$

---

### 2.16. Hệ số Tương quan Hạng Spearman ($\rho$)
$$\rho = 1 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}$$
với $d_i = \text{rank}_{\text{model}}(D_i) - \text{rank}_{\text{label}}(D_i)$.
- **Macro-Spearman**: $\rho_{\text{mac}} = \frac{1}{|Q|} \sum_{q \in Q} \rho_q$
- **Micro-Spearman**: $\rho_{\text{mic}}$ tính gộp trên toàn thể $|P| = 1.759$ cặp.

---

### 2.17. Hiệu chỉnh Đa Giả thuyết Bậc thang lùi Holm--Bonferroni
Kiểm soát tỷ lệ sai số loại I trên toàn họ (FWER $\le \alpha_{\text{stat}} = 0.05$):
Sắp xếp $m=15$ giá trị $p$ theo thứ tự tăng dần: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
Tại mỗi bước $i \in \{1, \dots, m\}$:
$$p_{(i)} \le \frac{\alpha_{\text{stat}}}{m - i + 1}$$
Giá trị $p$ hiệu chỉnh báo cáo:
$$p_{\text{adj},(i)} = \min \left( 1.0, \, \max_{k \le i} \Big( (m - k + 1) \cdot p_{(k)} \Big) \right)$$

---

### 2.18. Cấu hình Phân rã Độ trễ Chuẩn hóa 3 Pha
Tổng thời gian xử lý trực tuyến toàn trình của một truy vấn:
$$T_{\text{online}} = T_{\text{query}} + T_{\text{rank}}$$
trong đó:
- $T_{\text{index}}$: Thời gian bóc tách và trích xuất vector ngoại tuyến (Offline Indexing, ms/tài liệu).
- $T_{\text{query}}$: Thời gian mã hóa truy vấn trực tuyến (Online Query Encoding, ms).
- $T_{\text{rank}}$: Thời gian quét ma trận vector và xếp hạng ứng viên (Online Candidate Scoring, ms cho $N=25$).

---

### 2.19. Độ lệch Tuyệt đối Trung bình trong Kiểm toán Công bằng (MAD)
Đo lường mức độ biến động điểm số khi thay đổi định danh nhân khẩu học (tên, trường đại học):
$$\text{MAD} = \frac{1}{|P|} \sum_{(c, j) \in P} \Big| S(c, j) - S_{\text{perturbed}}(c', j) \Big|$$
*Kết quả thực nghiệm*: $\text{MAD} = 0.000000$ (Bất biến $100\%$).
