# Sổ tay Chuẩn hóa Ký hiệu Toán học và Công thức Toàn văn
## Hệ thống Ghép cặp Hai chiều CV--Job Đa Vector Nhận biết Phân mục (Section-Aware Multi-Vector Matching)

> **Ghi chú chuẩn hóa**: Sổ tay này được rà soát và đối chiếu nguyên văn trực tiếp với mã nguồn triển khai (`src/cvjd/data/preprocessing.py`, `src/cvjd/models/matching.py`, `src/cvjd/ranking/engine.py`, `src/cvjd/data/taxonomy.py`, và `scripts/run_scientific_benchmark.py`). Toàn bộ ký hiệu đã được chuẩn hóa để loại bỏ triệt để xung đột danh định.

---

## 1. Bảng Ký hiệu Toán học & Thuật ngữ Hệ thống (Mathematical Notations)

### 1.1. Tập hợp và Không gian Dữ liệu
| Ký hiệu | Định nghĩa toán học / kỹ thuật | Ý nghĩa nghiệp vụ & Giá trị thực tế |
| :--- | :--- | :--- |
| $\mathcal{D}$ | $\mathcal{D} = \mathcal{J} \sqcup \mathcal{C}$ | Không gian tài liệu tuyển dụng, là hợp rời của tập JD và tập CV |
| $\mathcal{J}$ | Tập hợp các bài đăng mô tả công việc (Job Descriptions) | Toàn tập: $351$ JD độc nhất; Tập test: $71$ JD; Train sạch: $76$ JD |
| $\mathcal{C}$ | Tập hợp các hồ sơ ứng viên (Curriculum Vitae) | Toàn tập: $1.051$ CV độc nhất; Tập test: $477$ CV (gồm $409$ CV sạch) |
| $q \in \mathcal{J}$ | Tài liệu truy vấn (Query document) | Một bài đăng tuyển dụng đưa vào hệ thống matching |
| $\mathcal{P}_q \subset \mathcal{C}$ | Tập ứng viên đối sánh của truy vấn $q$ (Candidate pool) | Tập các CV được ghép cặp đánh giá với $q$ trong tập dữ liệu |
| $N_q$ | Kích thước tập ứng viên của truy vấn $q$ ($N_q = \|\mathcal{P}_q\|$) | Biến thiên từng query: $N_q \in [2, 89]$, trung bình $\bar{N}_q = 24.77 \approx 25$ |
| $p \in \mathcal{P}_q$ | Một ứng viên cụ thể trong tập đối sánh của $q$ | Hồ sơ CV đang được chấm điểm và xếp hạng |
| $\mathbf{y}_q$ | Vector nhãn mức độ phù hợp thực tế của pool $\mathcal{P}_q$ | $\mathbf{y}_q = (y_{q, 1}, y_{q, 2}, \dots, y_{q, N_q}) \in \{0, 1, 2\}^{N_q}$ |
| $y_{q, i}$ | Nhãn độ phù hợp giữa truy vấn $q$ và ứng viên $p_i$ | Thang 3 mức: $0$ (No Fit), $1$ (Potential Fit), $2$ (Good Fit) |
| $\tau$ | Ngưỡng nhị phân hóa nhãn phù hợp khi tính MRR, MAP | $\tau \in \{1, 2\}$ (chuẩn chính: $\tau = 1$, tức $y \ge 1$ là phù hợp) |
| $s_\theta(q, p)$ | Hàm tính điểm tương thích giữa truy vấn $q$ và ứng viên $p$ | Giá trị số thực sinh bởi mô hình cấu hình $\theta$ |
| $\pi_{s_\theta}$ | Hoán vị thứ tự xếp hạng sinh bởi hàm điểm $s_\theta$ trên $\mathcal{P}_q$ | Sắp xếp giảm dần theo điểm số: $s_\theta(q, p_{\pi(1)}) \ge \dots \ge s_\theta(q, p_{\pi(N_q)})$ |
| $\pi^*$ | Hoán vị lý tưởng sắp xếp giảm dần theo nhãn thực tế $\mathbf{y}_q$ | Dùng để tính điểm chuẩn hóa tối đa $\text{IDCG}@K$ |
| $\mathcal{S}_{domain}$ | Tập 4 nhãn phân mục chức năng chuẩn mực | $\mathcal{S}_{domain} = \{\text{skills}, \text{experience}, \text{education}, \text{summary}\}$ |
| $\mathcal{S}(D)$ | Tập các phân mục chức năng có mặt trong tài liệu $D$ | $\mathcal{S}(D) \subseteq \mathcal{S}_{domain}$ |
| $M$ | Số lượng phân mục chức năng thực tế của một CV | $1 \le M \le 4$, trung bình $3.65$ mục/CV ($340$ CV đủ $4$ mục, $107$ có $3$, $28$ có $2$, $2$ có $1$) |
| $\mathcal{K}_q$ | Tập hợp các kỹ năng trích xuất từ JD $q$ | Gồm tập bắt buộc $\mathcal{K}_{q,\text{must}}$ và ưu tiên $\mathcal{K}_{q,\text{nice}}$ (trung bình $4.4$ kỹ năng/JD) |
| $\mathcal{K}_p$ | Tập hợp các kỹ năng trích xuất từ CV $p$ | Tập kỹ năng chuẩn hóa của ứng viên (trung bình $10.3$ kỹ năng/CV) |
| $\mathcal{M}$ | Tập các danh sách xếp hạng thành phần trong RRF | $\mathcal{M} = \{\text{Dense}, \text{BM25}\}$ |
| $r_m(p)$ | Vị trí thứ hạng (1-based) của ứng viên $p$ trong danh sách $m$ | $1 \le r_m(p) \le N_q$ |

---

### 1.2. Biểu diễn Không gian Vector và Ký hiệu Đại số
| Ký hiệu | Định nghĩa toán học | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- |
| $\Sigma^*$ | Tập hợp tất cả các chuỗi ký tự hữu hạn | Miền không gian văn bản thô đầu vào |
| $h$ | Chiều không gian vector nhúng ngữ nghĩa (embedding dimension) | $h = 384$ (MiniLM); $h = 1024$ (BGE-M3) |
| $T_\sigma$ | Chuỗi ký tự văn bản của phân mục $\sigma \in \mathcal{S}(D)$ | Trích đoạn nội dung của phân mục |
| $T_{raw}$ | Toàn văn chuỗi ký tự của tài liệu | Văn bản đầy đủ của CV hoặc JD |
| $\mathcal{E}$ | Mô hình Bi-Encoder đóng băng: $\mathcal{E}: \Sigma^* \rightarrow \mathbb{R}^h$ | Ánh xạ chuỗi văn bản thành vector đặc |
| $\mathbf{v}_q^\sigma$ | Vector biểu diễn phân mục $\sigma$ của truy vấn $q$ | $\mathbf{v}_q^\sigma = \mathcal{E}(T_\sigma) \in \mathbb{R}^h$, $\|\mathbf{v}_q^\sigma\|_2 = 1.0$ |
| $\mathbf{v}_p^s$ | Vector biểu diễn phân mục $s$ của ứng viên $p$ | $\mathbf{v}_p^s = \mathcal{E}(T_s) \in \mathbb{R}^h$, $\|\mathbf{v}_p^s\|_2 = 1.0$ |
| $\mathbf{v}_q^{full}, \mathbf{v}_p^{full}$ | Vector biểu diễn toàn văn không chia mục (Single-Vector) | $\mathbf{v}^{full} = \mathcal{E}(T_{raw}) \in \mathbb{R}^h$, $\|\mathbf{v}^{full}\|_2 = 1.0$ |
| $\|\mathbf{v}\|_2$ | Chuẩn Euclidean: $\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^h v_i^2}$ | Dùng để chiếu mọi vector về mặt cầu đơn vị |
| $\text{sim}(\mathbf{u}, \mathbf{w})$ | Tích vô hướng Cosine: $\mathbf{u} \cdot \mathbf{w} = \sum_{i=1}^h u_i w_i$ | Đo khoảng cách góc ngữ nghĩa, miền giá trị $[-1.0, 1.0]$ |
| $c_\sigma(q, p)$ | Điểm ghép cặp cực đại của phân mục $\sigma \in \mathcal{S}(q)$ với ứng viên $p$ | $c_\sigma(q, p) = \max_{s \in \mathcal{S}(p)} \text{sim}(\mathbf{v}_q^\sigma, \mathbf{v}_p^s) \in [-1.0, 1.0]$ |

---

### 1.3. Bảng Hằng số và Trọng số Hệ thống Thực tế (Constants)
| Hằng số | Giá trị | Tệp mã nguồn định nghĩa | Ý nghĩa nghiệp vụ và lý do thiết kế |
| :--- | :---: | :--- | :--- |
| $w_{\text{skills}}$ | $0.40$ | `src/cvjd/models/matching.py:23` | Trọng số phân mục Kỹ năng kỹ thuật cốt lõi (điều kiện quan trọng nhất) |
| $w_{\text{experience}}$ | $0.35$ | `src/cvjd/models/matching.py:24` | Trọng số phân mục Kinh nghiệm làm việc thực chiến (chứng minh năng lực) |
| $w_{\text{education}}$ | $0.15$ | `src/cvjd/models/matching.py:25` | Trọng số phân mục Học vấn & Bằng cấp (điều kiện nền tảng) |
| $w_{\text{summary}}$ | $0.10$ | `src/cvjd/models/matching.py:26` | Trọng số phân mục Tóm tắt bản thân / Mục tiêu nghề nghiệp |
| $\sum w_\sigma$ | $1.00$ | `src/cvjd/models/matching.py` | Tổng 4 trọng số mặc định bằng đúng $1.00$ ($0.40+0.35+0.15+0.10$) |
| $\alpha$ | $0.60$ | `scripts/run_scientific_benchmark.py:505` | Trọng số ưu tiên ngữ nghĩa đặc trong chặng tìm kiếm ($60\%$ Dense, $40\%$ BM25) |
| $1 - \alpha$ | $0.40$ | `scripts/run_scientific_benchmark.py:505` | Trọng số từ vựng BM25 (chống bỏ sót mã hiệu / tên thư viện hiếm) |
| $\beta$ | $0.60$ | `scripts/run_scientific_benchmark.py:508` | Hệ số phối hợp chặng tìm kiếm ($60\%$ Retrieval, $40\%$ Taxonomy Coverage) |
| $1 - \beta$ | $0.40$ | `scripts/run_scientific_benchmark.py:508` | Tỷ trọng đóng góp của độ phủ kỹ năng cấu trúc taxonomy ($S_{tax}$) |
| $k$ | $60$ | `scripts/run_scientific_benchmark.py:519` | Hằng số làm mịn kinh điển trong Reciprocal Rank Fusion (ngăn lấn át top-1) |
| $\lambda_{\text{must}}$ | $0.85$ | `scripts/run_scientific_benchmark.py:492` | Tỷ trọng độ phủ kỹ năng bắt buộc (Must-Have) trong đánh giá khoa học |
| $\lambda_{\text{nice}}$ | $0.15$ | `scripts/run_scientific_benchmark.py:492` | Tỷ trọng độ phủ kỹ năng khuyến khích (Nice-To-Have) |
| $K$ | $5, 10$ | `scripts/run_scientific_benchmark.py:97` | Ngưỡng cắt vị trí hàng đầu trong các chỉ số nDCG@K, P@K |
| $N_{\text{shuffles}}$ | $50$ | `scripts/run_scientific_benchmark.py:99` | Số lần xáo trộn ngẫu nhiên để phá đồng hạng công bằng (Fair tie-breaking) |
| $B$ | $10.000$ | `scripts/compute_full_statistical_tests.py` | Số lần lấy mẫu lặp ghép cặp có hoàn lại trong Bootstrap 95% CI |
| $\alpha_{\text{stat}}$ | $0.05$ | `scripts/compute_full_statistical_tests.py` | Mức ý nghĩa thống kê danh định (Type I error rate) |
| $L$ | $15$ | `paper/tables/tab_statistical_tests.tex` | Số lượng phép so sánh trong họ giả thuyết chính để kiểm soát FWER |

---

## 2. Hệ thống Công thức Toán học Toàn văn

### 2.1. Định thức Bài toán Tối ưu Hóa Hệ thống Xếp hạng
Hệ thống không tìm kiếm hoán vị dựa trên nhãn thực tế (vì nhãn thực tế là thông tin che giấu trong pha dự đoán). Thay vào đó, bài toán là xác định cấu hình tham số $\theta^*$ cho hàm chấm điểm $s_\theta(q, p)$ sao cho thứ tự sắp xếp cảm ứng bởi hàm này tối đa hóa kỳ vọng chỉ số đánh giá trên toàn bộ tập truy vấn $Q$:
$$\theta^* = \arg\max_\theta \frac{1}{|Q|} \sum_{q \in Q} \text{nDCG}@K\Big( \pi_{s_\theta}(q, \mathcal{P}_q), \, \mathbf{y}_q \Big)$$
trong đó $\pi_{s_\theta}(q, \mathcal{P}_q)$ là hoán vị thu được bằng cách sắp xếp các ứng viên $p \in \mathcal{P}_q$ theo thứ tự giảm dần của điểm số $s_\theta(q, p)$.

---

### 2.2. Biểu thức Chính quy Bóc tách Tiêu đề Dính chữ Thực tế
Trong mã nguồn triển khai (`src/cvjd/data/preprocessing.py`), bộ bóc tách mục không sử dụng các biểu thức cắt ranh giới ngẫu nhiên mà sử dụng biểu thức chính quy với vị trí bắt đầu dòng (`^`), từ khóa không phân biệt hoa thường (`(?i:...)`), và lookahead khẳng định (`(?=[A-Z0-9\-])`):
$$\text{Pattern}_{\text{glued}} = \texttt{\textasciicircum(?i:}\langle\text{keyword}\rangle\texttt{)(?=[A-Z0-9\textbackslash-])}$$
Cụ thể với từng phân mục:
- **Tóm tắt (Summary)**: `r"^(?i:professional\s+summary|career\s+profile|summary|profile|overview)(?=[A-Z0-9\-])"`
- **Kỹ năng (Skills)**: `r"^(?i:technical\s+skills\s+and\s+qualifications|technical\s+skills|core\s+competencies|key\s+skills|skills?)(?=[A-Z0-9\-])"`
- **Kinh nghiệm (Experience)**: `r"^(?i:professional\s+experience|work\s+experience|work\s+history|employment\s+history|experience)(?=[A-Z0-9\-])"`
- **Học vấn (Education)**: `r"^(?i:education\s+and\s+training|education\s+and\s+qualifications|academic\s+background|education|certifications?)(?=[A-Z0-9\-])"`

*Tác dụng*: Quy tắc này nhận diện chính xác các lỗi dính chữ từ HTML như `SummaryHighly qualified...` hoặc `SkillsPython` mà không cắt nhầm các danh từ riêng như `JavaScript`, `GitHub`, hay `PowerPoint`.

---

### 2.3. Chuẩn hóa Vector Đơn vị ($L_2$-norm)
Mỗi đoạn văn bản phân mục $T_\sigma$ được ánh xạ qua mô hình Bi-Encoder và chuẩn hóa Euclid về mặt cầu đơn vị $S^{h-1} \subset \mathbb{R}^h$:
$$\mathbf{v} \leftarrow \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^h v_i^2}}, \quad \text{đảm bảo } \|\mathbf{v}\|_2 = 1.0$$

---

### 2.4. Độ tương đồng Cosine giữa hai Vector
Nhờ chuẩn hóa $L_2$, độ tương đồng Cosine tương đương với tích vô hướng thông thường:
$$\text{sim}(\mathbf{u}, \mathbf{w}) = \frac{\mathbf{u} \cdot \mathbf{w}}{\|\mathbf{u}\|_2 \|\mathbf{w}\|_2} = \sum_{i=1}^h u_i w_i \in [-1.0, 1.0]$$

---

### 2.5. Ghép cặp Chéo Cực đại (Cross-Section Pairing)
Để không bỏ sót các kỹ năng được mô tả lồng ghép trong quá trình làm việc, mỗi phân mục $\sigma$ của truy vấn $q$ tìm phân mục có độ tương đồng lớn nhất trên toàn bộ các phân mục hiện có của ứng viên $p$:
$$c_\sigma(q, p) = \max_{s \in \mathcal{S}(p)} \text{sim}\big(\mathbf{v}_q^\sigma, \mathbf{v}_p^s\big) \in [-1.0, 1.0]$$

*Tính chất toán học*: Do lấy toán tử $\max$, hồ sơ ứng viên $p$ sở hữu càng nhiều phân mục chức năng thì cơ hội đạt điểm cực đại $c_\sigma(q, p)$ càng cao. Trường hợp ứng viên khuyết một mục cụ thể (ví dụ không có mục kỹ năng riêng), toán tử $\max$ sẽ tự động ghép với mục kinh nghiệm làm việc hoặc dự án, do đó ứng viên không bị phạt vì khuyết mục.

---

### 2.6. Tái Chuẩn hóa Trọng số Động (Dynamic Weight Renormalization)
Điểm tương đồng ngữ nghĩa phân mục tổng hợp giữa truy vấn $q$ và ứng viên $p$ được tính theo công thức:
$$S_{sem}(q, p) = \frac{\sum_{\sigma \in \mathcal{S}(q)} w_\sigma \cdot c_\sigma(q, p)}{\sum_{\sigma \in \mathcal{S}(q)} w_\sigma}$$
trong đó $\mathcal{S}(q)$ là tập các phân mục thực sự có mặt trong bài đăng tuyển dụng $q$, và $w_\sigma$ là các trọng số cơ sở: $w_{\text{skills}}=0.40, w_{\text{exp}}=0.35, w_{\text{edu}}=0.15, w_{\text{sum}}=0.10$.

*Cơ chế thực chất*: 
1. Mẫu số $\sum_{\sigma \in \mathcal{S}(q)} w_\sigma$ giải quyết vấn đề **truy vấn $q$ (JD) bị thiếu mục** (ví dụ JD không có yêu cầu bằng cấp hay mục tóm tắt), đảm bảo tổng trọng số hiệu dụng của các mục tham gia tính điểm luôn được chuẩn hóa về đúng $1.0$.
2. Miền giá trị: Do $c_\sigma(q, p) \in [-1.0, 1.0]$, điểm $S_{sem}(q, p)$ có miền giá trị lý thuyết là $[-1.0, 1.0]$ (không bị cắt cưỡng bức về $[0, 1]$ trước khi vào khâu chuẩn hóa min-max).
3. **Quy ước Biên (Fallback Mechanism)**: Nếu $\mathcal{S}(q) = \emptyset$ hoặc $\mathcal{S}(p) = \emptyset$ (tức không bóc tách được phân mục nào), hệ thống tự động kích hoạt cơ chế dự phòng:
$$S_{sem}(q, p) = \text{sim}\big(\mathbf{v}_q^{full}, \mathbf{v}_p^{full}\big)$$

---

### 2.7. Chuẩn hóa Tuyến tính Min-Max Theo Từng Query Pool
Trước khi hợp nhất các nguồn điểm có thang đo khác nhau, điểm số của từng ứng viên $p$ được chuẩn hóa trên tập ứng viên $\mathcal{P}_q$ của truy vấn $q$:
$$\tilde{S}(q, p) = \begin{cases} \dfrac{S(q, p) - \min_{p' \in \mathcal{P}_q} S(q, p')}{\max_{p' \in \mathcal{P}_q} S(q, p') - \min_{p' \in \mathcal{P}_q} S(q, p')} & \text{nếu } \max \ne \min \\ 1.0 & \text{nếu } \max = \min \text{ (dãy hằng)} \end{cases}$$

*Lưu ý về mặt tính toán*: 
- Với các truy vấn có kích thước pool rất nhỏ (ví dụ $N_q = 2$ ứng viên), điểm chuẩn hóa của hai ứng viên luôn bị kéo dãn tuyệt đối về $0.0$ và $1.0$.
- Tính bất đối xứng: Điểm Dense ($S_{sem}$) và BM25 ($S_{bm25}$) đều trải qua chuẩn hóa Min-Max theo pool. Ngược lại, điểm Taxonomy ($S_{tax}$) vốn dĩ đã nằm trong đoạn $[0, 1]$ nên không qua Min-Max. Do đó, độ phân tán thực tế của $S_{tax}$ trong pool $\mathcal{P}_q$ sẽ quyết định mức độ chi phối thứ hạng của nó.

---

### 2.8. Điểm Xếp hạng Lai Hiệu chuẩn (Calibrated Hybrid Scoring)
Trong pipeline thực nghiệm khoa học (`scripts/run_scientific_benchmark.py:505-508`), điểm xếp hạng lai được tính theo cấu trúc lồng 2 chặng:

**Chặng 1 — Hợp nhất Tìm kiếm (Retrieval Fusion)**:
$$S_{retrieval}(q, p) = \alpha \cdot \tilde{S}_{dense}(q, p) + (1 - \alpha) \cdot \tilde{S}_{bm25}(q, p)$$
với $\alpha = 0.60$, tức $60\%$ Dense Semantic và $40\%$ BM25 Lexical.

**Chặng 2 — Tích hợp Cây Phân loại (Taxonomy Fusion)**:
$$S_{hybrid}(q, p) = \beta \cdot S_{retrieval}(q, p) + (1 - \beta) \cdot S_{tax}(q, p)$$
với $\beta = 0.60$.

**Dạng Khai triển Tuyến tính Hiệu dụng**:
$$S_{hybrid}(q, p) = (\beta \cdot \alpha) \, \tilde{S}_{dense}(q, p) + \big(\beta \cdot (1 - \alpha)\big) \, \tilde{S}_{bm25}(q, p) + (1 - \beta) \, S_{tax}(q, p)$$
Thay số $\alpha = 0.60, \beta = 0.60$, ta thu được tổ hợp lồi với các trọng số hiệu dụng hoàn hảo:
$$S_{hybrid}(q, p) = 0.36 \cdot \tilde{S}_{dense}(q, p) + 0.24 \cdot \tilde{S}_{bm25}(q, p) + 0.40 \cdot S_{tax}(q, p)$$
*Kiểm tra tổng trọng số*: $0.36 + 0.24 + 0.40 = 1.00$.

---

### 2.9. Phối hợp Xếp hạng Nghịch đảo Không Trọng số Trộn (Weights-Free RRF)
Thuật toán Reciprocal Rank Fusion kết hợp thứ tự xếp hạng từ nhánh Dense và BM25 mà không cần dò tìm hệ số hòa trộn $\alpha$:
$$S_{RRF}(q, p) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{k + r_m(p)} = \frac{1}{60 + r_{dense}(p)} + \frac{1}{60 + r_{bm25}(p)}$$
trong đó $k = 60$ là siêu tham số làm mịn (smoothing constant), và $r_m(p) \in \{1, \dots, N_q\}$ là vị trí thứ tự của ứng viên $p$ trong danh sách $m$.

Để tạo cấu hình **RRF + Taxonomy** trong bảng thực nghiệm (`scripts/run_scientific_benchmark.py:511-521`), điểm $S_{RRF}$ sau đó được chuẩn hóa Min-Max theo từng pool và hợp nhất cùng taxonomy:
$$\tilde{S}_{RRF}(q, p) = \text{MinMax}\big( S_{RRF}(q, p) \big)$$
$$S_{RRF+Tax}(q, p) = 0.60 \cdot \tilde{S}_{RRF}(q, p) + 0.40 \cdot S_{tax}(q, p)$$

---

### 2.10. Hàm Mức độ Tương quan Kỹ năng (Skill Relation Degree)
Khác với hàm chỉ thị nhị phân $\{0, 1\}$, hệ thống tính toán mức độ đáp ứng của ứng viên đối với từng kỹ năng mục tiêu $t \in \mathcal{K}_q$ dựa trên hàm tương quan liên tục $\mu(t, \mathcal{K}_p) \in [0, 1]$ (`src/cvjd/data/taxonomy.py:81-100`):
$$\mu(t, \mathcal{K}_p) = \begin{cases} 1.00 & \text{nếu } t \in \mathcal{K}_p \text{ (khớp chính xác hoặc trùng bí danh alias)} \\ \max_{u \in \mathcal{K}_p} \text{rel}(u, t) & \text{nếu tồn tại quan hệ trong taxonomy} \\ 0.00 & \text{ngược lại} \end{cases}$$
trong đó trọng số quan hệ $\text{rel}(u, t)$ giữa kỹ năng ứng viên $u$ và yêu cầu $t$ được định nghĩa trong cây phân loại:
- $\text{rel} = 0.80$: Quan hệ bao hàm tập cha ($\texttt{is\_superset\_of}$, ví dụ: TypeScript $\rightarrow$ JavaScript).
- $\text{rel} = 0.60$: Quan hệ khung phát triển ($\texttt{framework\_of}$, ví dụ: Next.js $\rightarrow$ React).
- $\text{rel} \in [0.25, 0.50]$: Quan hệ công nghệ đồng hành trong hệ sinh thái ($\texttt{ecosystem\_companion}$, ví dụ: Redux $\rightarrow$ React).
- $\text{rel} = 0.00$: Không có liên kết trực tiếp trong taxonomy.

---

### 2.11. Độ phủ Kỹ năng Bất đối xứng và Điểm Taxonomy Tổng hợp
Độ phủ bất đối xứng trên một tập kỹ năng mục tiêu $\mathcal{T} \subseteq \mathcal{K}_q$ được định nghĩa:
$$\text{Cov}(\mathcal{T}, \mathcal{K}_p) = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \mu(t, \mathcal{K}_p)$$

Điểm Taxonomy tổng hợp ($S_{tax}$) trong thực nghiệm khoa học (`scripts/run_scientific_benchmark.py:483-492`) kết hợp độ phủ của hai nhóm kỹ năng:
$$S_{tax}(q, p) = 0.85 \cdot \text{Cov}(\mathcal{K}_{q,\text{must}}, \mathcal{K}_p) + 0.15 \cdot \text{Cov}(\mathcal{K}_{q,\text{nice}}, \mathcal{K}_p)$$

**Các Quy ước Xử lý Ca Biên (Boundary Conditions) và Phân tích Bất Đối xứng**:
1. **JD không trích xuất được kỹ năng nào** ($|\mathcal{K}_q| = 0$, chiếm $12.7\%$ JD test và train): Gán $S_{tax}(q, p) = 1.0$ cho mọi ứng viên để đóng vai trò phần tử trung tính, không làm biến dạng thứ tự do nhánh vector tạo ra.
2. **JD không có kỹ năng bắt buộc** ($|\mathcal{K}_{q,\text{must}}| = 0$, chiếm $16.9\%$ JD test): Gán $\text{Cov}(\mathcal{K}_{q,\text{must}}, \mathcal{K}_p) = 1.0$. Điểm tối đa có thể đạt $1.0$ nếu ứng viên thỏa mãn mọi kỹ năng ưu tiên.
3. **JD không có kỹ năng ưu tiên** ($|\mathcal{K}_{q,\text{nice}}| = 0$, chiếm $43.7\%$ JD test): Gán $\text{Cov}(\mathcal{K}_{q,\text{nice}}, \mathcal{K}_p) = 0.0$. Hệ quả thực tế: điểm $S_{tax}$ của ứng viên cho nhóm JD này bị giới hạn trần ở mức $0.85$ (dù ứng viên đạt $100\%$ kỹ năng bắt buộc). Mặc dù thứ hạng trong nội bộ một truy vấn không bị thay đổi (do mọi ứng viên cùng chịu chung thang đo), nhưng giá trị tuyệt đối của $S_{tax}$ giữa các truy vấn khác nhau sẽ dao động, gây ảnh hưởng cục bộ khi hòa trộn với điểm retrieval chuẩn hóa Min-Max.
4. **Phạm vi của biến thể phạt nhân tử $\gamma$**: Trong mô hình phục vụ sản phẩm (`src/cvjd/ranking/hybrid.py:194`), khi cờ `must_have_penalty=True` được bật, nếu $\text{Cov}(\text{Must}) < 0.50$, điểm sẽ nhân với hệ số phạt $\gamma = \text{Cov}(\text{Must})$. Cần lưu ý rõ: **trong pipeline đo đạc khoa học công bố (`scripts/run_scientific_benchmark.py`), cờ này không được bật**, và hệ thống sử dụng dạng tổ hợp lồi tuyến tính thuần túy $0.36 \tilde{S}_{\text{dense}} + 0.24 \tilde{S}_{\text{bm25}} + 0.40 S_{\text{tax}}$.

---

### 2.12. Tương tác Trễ Mức Chunk (Chunk Max-Sim)
Để đánh giá cơ chế tương tác trễ kiểu ColBERT nhưng ở cấp độ khối văn bản (sliding window $128$ từ, bước trượt $64$ từ), cả bài đăng $q$ và ứng viên $p$ được chia thành các tập vector chunk $\mathcal{U}_q = \{\mathbf{u}_1, \dots, \mathbf{u}_{|\mathcal{U}_q|}\}$ và $\mathcal{W}_p = \{\mathbf{w}_1, \dots, \mathbf{w}_{|\mathcal{W}_p|}\}$ với $\mathbf{u}_i, \mathbf{w}_j \in \mathbb{R}^h, \|\mathbf{u}_i\|_2 = \|\mathbf{w}_j\|_2 = 1.0$:
$$S_{\text{Chunk-MaxSim}}(q, p) = \frac{1}{|\mathcal{U}_q|} \sum_{\mathbf{u} \in \mathcal{U}_q} \max_{\mathbf{w} \in \mathcal{W}_p} (\mathbf{u} \cdot \mathbf{w})$$

---

### 2.13. Độ lợi Tích lũy Giảm dần Chuẩn hóa (nDCG@K)
Chỉ số đánh giá độ chính xác xếp hạng đa mức tại ngưỡng $K \in \{5, 10\}$:
$$\text{DCG}@K(q, \pi) = \sum_{i=1}^{\min(K, N_q)} \frac{2^{y_{q, \pi(i)}} - 1}{\log_2(i + 1)}$$
trong đó $y_{q, \pi(i)} \in \{0, 1, 2\}$ là nhãn thực tế của ứng viên đứng tại vị trí thứ $i$ trong hoán vị $\pi$.

Điểm lý tưởng $\text{IDCG}@K$ thu được từ hoán vị lý tưởng $\pi^*$ (sắp xếp giảm dần theo $\mathbf{y}_q$):
$$\text{IDCG}@K(q) = \sum_{i=1}^{\min(K, N_q)} \frac{2^{y_{q, \pi^*(i)}} - 1}{\log_2(i + 1)}$$

**Quy ước Hai Kịch bản**:
$$\text{nDCG}@K(q, \pi) = \begin{cases} \dfrac{\text{DCG}@K(q, \pi)}{\text{IDCG}@K(q)} & \text{nếu } \text{IDCG}@K(q) > 0 \\ 0.0 & \text{nếu } \text{IDCG}@K(q) = 0 \text{ (Kịch bản Setting A)} \\ \text{Loại bỏ khỏi mẫu} & \text{nếu } \text{IDCG}@K(q) = 0 \text{ (Kịch bản Setting B)} \end{cases}$$

---

### 2.14. Điểm Nghịch đảo Vị trí Đầu tiên (MRR)
Đo lường tốc độ tìm thấy ứng viên phù hợp đầu tiên đạt ngưỡng $\tau \in \{1, 2\}$:
$$\text{RR}_\tau(q, \pi) = \begin{cases} \dfrac{1}{\min \{ i \in \{1, \dots, N_q\} \mid y_{q, \pi(i)} \ge \tau \}} & \text{nếu } \exists i: y_{q, \pi(i)} \ge \tau \\ 0.0 & \text{ngược lại (Setting A)} \end{cases}$$
$$\text{MRR}_\tau = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}_\tau(q, \pi)$$

---

### 2.15. Độ chính xác Trung bình (Mean Average Precision - MAP)
Đo lường toàn diện diện tích dưới đường cong Precision-Recall tại ngưỡng $\tau \in \{1, 2\}$:
$$\text{AP}_\tau(q, \pi) = \begin{cases} \dfrac{1}{R_{q, \tau}} \sum_{k=1}^{N_q} P_\tau@k(q, \pi) \cdot \mathbb{I}(y_{q, \pi(k)} \ge \tau) & \text{nếu } R_{q, \tau} > 0 \\ 0.0 & \text{nếu } R_{q, \tau} = 0 \text{ (Setting A)} \end{cases}$$
trong đó:
- $R_{q, \tau} = \sum_{i=1}^{N_q} \mathbb{I}(y_{q, i} \ge \tau)$ là tổng số ứng viên phù hợp thực tế trong pool của query $q$.
- $P_\tau@k(q, \pi) = \frac{1}{k} \sum_{j=1}^k \mathbb{I}(y_{q, \pi(j)} \ge \tau)$ là độ chính xác tại ngưỡng cắt $k$.
$$\text{MAP}_\tau = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}_\tau(q, \pi)$$

---

### 2.16. Kỳ vọng Phá Đồng hạng Công bằng (Fair Tie-Breaking Expectation)
Khi nhiều ứng viên có điểm số mô hình bằng nhau ($s(q, p_i) = s(q, p_j)$), thứ tự sắp xếp cố định theo ID có thể gây sai lệch lạc quan hoặc bi quan. Hệ thống tính toán kỳ vọng của mọi thước đo thông qua $N_{\text{shuffles}} = 50$ lần xáo trộn ngẫu nhiên có thêm nhiễu vi mô $z_i \sim \text{Uniform}(-10^{-9}, 10^{-9})$:
$$\mathbb{E}[\text{Metric}(q)] = \frac{1}{50} \sum_{s=1}^{50} \text{Metric}\Big( \pi \big( \mathbf{s}_q + \mathbf{z}^{(s)} \big) \Big)$$

---

### 2.17. Hệ số Tương quan Hạng Spearman với Đồng hạng (Midrank Spearman)
Do nhãn thực tế chỉ có 3 mức phân loại rời rạc $\{0, 1, 2\}$, hiện tượng đồng hạng (ties) xảy ra rất phổ biến. Vì vậy, hệ số tương quan hạng Spearman được tính chính xác bằng tương quan Pearson trên vector hạng trung bình (midranks) $\mathbf{r}_{score}$ và $\mathbf{r}_{label}$:
$$\rho_q = \frac{\sum_{i=1}^{N_q} \big( r_{\text{score}}(p_i) - \bar{r}_{\text{score}} \big) \big( r_{\text{label}}(p_i) - \bar{r}_{\text{label}} \big)}{\sqrt{\sum_{i=1}^{N_q} \big( r_{\text{score}}(p_i) - \bar{r}_{\text{score}} \big)^2 \sum_{i=1}^{N_q} \big( r_{\text{label}}(p_i) - \bar{r}_{\text{label}} \big)^2}}$$

**Quy ước Biên và Báo cáo**:
- Nếu toàn bộ nhãn trong pool của query $q$ bằng nhau (ví dụ $100\%$ là No Fit), phương sai nhãn bằng $0$, mẫu số bằng $0$, $\rho_q$ không xác định (NaN) và được loại bỏ khỏi mẫu tính trung bình.
- **Macro-Spearman**: $\rho_{\text{mac}} = \frac{1}{|Q_{\text{valid}}|} \sum_{q \in Q_{\text{valid}}} \rho_q$.
- **Micro-Spearman**: $\rho_{\text{mic}}$ tính tương quan Pearson gộp trên toàn bộ $1.759$ cặp điểm số và nhãn.

---

### 2.18. Hiệu chỉnh Đa Giả thuyết Bậc thang lùi Holm--Bonferroni
Để kiểm soát nguy cơ sai số loại I trên toàn họ kiểm định (FWER $\le \alpha_{\text{stat}} = 0.05$) đối với $L = 15$ cặp so sánh chính:
1. Tính giá trị $p$ thô (raw $p$-value) từ kiểm định tham số ghép cặp Student's paired $t$-test trên biến ngẫu nhiên sai biệt $\Delta_q = M_A(q) - M_B(q)$.
2. Sắp xếp $L = 15$ giá trị $p$ theo thứ tự tăng dần: $p_{(1)} \le p_{(2)} \le \dots \le p_{(L)}$.
3. Với mỗi chỉ số $i \in \{1, 2, \dots, L\}$, kiểm tra điều kiện bác bỏ giả thuyết không $H_{0,(i)}$:
$$p_{(i)} \le \frac{\alpha_{\text{stat}}}{L - i + 1}$$
Nếu thỏa mãn, bác bỏ $H_{0,(i)}$ và tiếp tục với $i+1$. Khi gặp chỉ số đầu tiên không thỏa mãn, dừng lại và chấp nhận toàn bộ các giả thuyết còn lại.

Giá trị $p$ hiệu chỉnh báo cáo trong bảng thực nghiệm:
$$p_{\text{adj},(i)} = \min \left( 1.0, \, \max_{k \le i} \Big( (L - k + 1) \cdot p_{(k)} \Big) \right)$$

*Bản chất thống kê*: Do họ 15 giả thuyết này được xác định post-hoc nhằm bao quát các so sánh của RQ1--RQ4, các suy luận hiệu chỉnh Holm cần được hiểu là phân tích khám phá (exploratory) thay vì thử nghiệm khẳng định có đăng ký trước.

---

### 2.19. Cấu hình Phân rã Độ trễ Chuẩn hóa 3 Pha
Độ trễ toàn trình trực tuyến cho một truy vấn mới đến hệ thống:
$$T_{\text{online}} = T_{\text{query}} + T_{\text{rank}}$$
trong đó:
- $T_{\text{index}}$: Thời gian bóc tách và mã hóa vector ngoại tuyến (Offline Indexing, ms/tài liệu).
- $T_{\text{query}}$: Thời gian mô hình Bi-Encoder mã hóa nội dung truy vấn trực tuyến (Online Query Encoding, ms).
- $T_{\text{rank}}$: Thời gian tính ma trận tương đồng và xếp hạng ứng viên trong pool chuẩn hóa $N=25$ ứng viên trên CPU (Online Scoring Latency, ms).
