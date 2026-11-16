# Thẩm Định Dữ Liệu Nghiên Cứu (Dataset Audit Report - SOURCES.md)

Tài liệu này ghi lại kết quả thẩm định kỹ thuật, thống kê chi tiết và đánh giá tính toàn vẹn (integrity audit) của bộ dữ liệu nghiên cứu chuẩn công khai được sử dụng cho bài báo khoa học và đồ án tốt nghiệp.

---

## 1. Thông Tin Chung & Nguồn Gốc (Dataset Provenance)

| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Tên bộ dữ liệu** | `cnamuangtoun/resume-job-description-fit` |
| **Nguồn phát hành** | Hugging Face Datasets ([Link](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)) |
| **Tác giả / Uploader** | `cnamuangtoun` (Chapwit Na Muangtoun) |
| **Giấy phép (License)** | **Không khai báo (Unspecified / Unknown)** trên Hugging Face API. Văn bản có dấu hiệu trích xuất tự động (HTML scraping) từ các trang mẫu CV công khai như LiveCareer / Kaggle (các thẻ heading bị dính chữ không có khoảng trắng như `SummaryHighly`, `Experience08/2014`, `EducationMaster`). Nghiên cứu sử dụng thuần túy cho mục đích đối sánh khoa học mở. |
| **Ngôn ngữ** | 100% Tiếng Anh (English Monolingual) |
| **Ghi chú về Dataset Card** | Uploader chỉ tải lên 2 file `train.csv` và `test.csv`, **không kèm theo Dataset Card hay văn bản quy định gán nhãn chính thức**. Trong nghiên cứu, 3 nhãn chuỗi được xử lý khách quan theo thang đo Ordinal IR: `No Fit` $\rightarrow 0$, `Potential Fit` $\rightarrow 1$, `Good Fit` $\rightarrow 2$. |
| **Định dạng lưu trữ** | `train.csv` (6.241 dòng, 1,29 MB) và `test.csv` (1.759 dòng, 1,41 MB) |
| **Lịch sử Git log hằng số** | Trọng số section ($0.40/0.35/0.15/0.10$) được đưa vào cấu hình tại commit `55c22ae` (03/11/2026). Tham số hòa trộn $\alpha=0.60$ và $\beta=0.60$ được chọn theo trực giác ban đầu và sau đó được kiểm chứng độc lập trên tập train. |
| **Nguồn gốc Taxonomy** | Bộ từ điển 615 khái niệm kỹ năng kỹ thuật chuẩn hóa (`skills_taxonomy.json`) là ontology độc lập, không gắn với nhãn của train hay test. |

---

## 2. Cấu Trúc Cột & Dữ Liệu Mẫu (Schema & Column Mapping)

Bộ dữ liệu bao gồm 3 cột chuẩn hóa:

| Tên Cột | Kiểu Dữ Liệu | Vai Trò trong Hệ Thống | Mô Tả Chi Tiết |
| :--- | :---: | :---: | :--- |
| **`resume_text`** | `string` | **Ứng viên (CV / Resume)** | Toàn văn bản CV của ứng viên. |
| **`job_description_text`** | `string` | **Yêu cầu (Job Description - JD)** | Toàn văn bản mô tả vị trí công việc tuyển dụng. |
| **`label`** | `string` | **Nhãn Phù Hợp (Relevance Grade)** | Nhãn phân loại 3 mức: `No Fit` ($0$), `Potential Fit` ($1$), `Good Fit` ($2$). |

### Hai Ngưỡng Đánh Giá Nhãn Trong Nghiên Cứu:
1. **Ngưỡng nới lỏng ($\text{rel} \ge 1$)**: Coi cả `Potential Fit` và `Good Fit` là ứng viên liên quan (chiếm 51.3% pool).
2. **Ngưỡng nghiêm ngặt ($\text{rel} = 2$)**: Chỉ coi `Good Fit` là ứng viên liên quan (chiếm 26.0% pool).

---

## 3. Thống Kê Số Lượng & Thực Thể Duy Nhất (Corpus Scale & Diversity)

| Tập Dữ Liệu | Tổng Số Cặp (Pairs) | Số JD Duy Nhất | Số CV Duy Nhất | Tỷ lệ Cặp / JD |
| :--- | :---: | :---: | :---: | :---: |
| **Tập Huấn luyện (Train)** | **6.241** | 280 | 642 | ~22.3 CVs / JD |
| **Tập Kiểm thử (Test)** | **1.759** | 71 | 477 | ~24.8 CVs / JD |
| **Toàn bộ Corpus (Tổng)** | **8.000** | **351** | **1.051** | -- |

---

## 4. Phân Bố Số Lượng Ứng Viên Trên Mỗi JD Query

Trong bài toán IR/Ranking, mỗi JD đóng vai trò là một **Query** và hệ thống cần xếp hạng danh sách các CV ứng viên (**Candidates pool**) cho query đó:

| Chỉ số Thống Kê | Tập Train (CVs / JD) | Tập Test (CVs / JD) |
| :--- | :---: | :---: |
| **Số lượng truy vấn (Count)** | 280 JDs | 71 JDs |
| **Trung bình (Mean $\pm$ Std)** | 22.3 $\pm$ 20.7 | 24.8 $\pm$ 21.7 |
| **Tối thiểu (Min)** | 1 CVs | 2 CVs |
| **Trung vị (Median / 50%)** | 14 CVs | 20 CVs |
| **Phân vị 75%** | 29 CVs | 33 CVs |
| **Tối đa (Max)** | 111 CVs | 89 CVs |

---

## 5. Phân Bố Nhãn Phù Hợp (Label Distribution)

| Mức Độ Phù Hợp | Tập Train (Số lượng & Tỷ lệ) | Tập Test (Số lượng & Tỷ lệ) |
| :--- | :---: | :---: |
| **No Fit (0)** | 3.143 (50.36%) | 857 (48.72%) |
| **Potential Fit (1)** | 1.556 (24.93%) | 444 (25.24%) |
| **Good Fit (2)** | 1.542 (24.71%) | 458 (26.04%) |
| **Tổng cộng** | **6.241 (100.0%)** | **1.759 (100.0%)** |

---

## 6. Kiểm Thử Rò Rỉ Dữ Liệu Giữa Train & Test (Data Leakage Audit)

| Loại Thực Thể | Số lượng Trùng Chính xác (Exact Overlap) | Tỷ lệ Trùng lặp | So khớp Gần trùng (90% - 99% Jaccard) |
| :--- | :---: | :---: | :---: |
| **Cặp (Job, Resume) Pair** | **0 / 1.759** | **0.0% (Tuyệt đối không rò rỉ cặp)** | 0 |
| **Văn bản Job Description (JD)** | 0 / 71 | 0.0% | 1 |
| **Văn bản Resume (CV)** | 68 / 477 | 14.3% | 0 |

---

## 7. Khảo Sát Cấu Trúc Section & Độ Dài Từng Phần (Bộ Tách Đóng Băng Commit `41e1d27`)

Thống kê thực hiện bởi bộ tách section chuẩn hóa duy nhất (`cvjd.data.preprocessing.split_sections`), đã qua kiểm định độc lập trên tập mẫu dev và đóng băng tại commit `41e1d27`:

| Khối Section Chức Năng | Tỷ lệ trên CVs (477 CVs) | Độ dài TB CV (Từ / Tokens) | Tỷ lệ trên JDs (71 JDs) | Độ dài TB JD (Từ / Tokens) |
| :--- | :---: | :---: | :---: | :---: |
| **Technical Skills / Requirements** | **86.4%** | 49.8 từ (~66 tokens) | **19.7%** | 250.6 từ (~333 tokens) |
| **Work Experience / Responsibilities** | **100.0%** | 527.0 từ (~701 tokens) | **31.0%** | 205.6 từ (~273 tokens) |
| **Education & Qualifications** | **94.3%** | 38.9 từ (~52 tokens) | **8.5%** | 253.8 từ (~338 tokens) |
| **Summary / Overview** | **83.9%** | 86.3 từ (~115 tokens) | -- | -- |
| **Benefits & Compensation** | -- | -- | **15.5%** | 172.6 từ (~230 tokens) |
| **About Company** | -- | -- | **8.5%** | 226.9 từ (~302 tokens) |
| **Đa Section ($\ge 2$ Section)** | **99.6%** (475/477) | -- | **28.2%** (20/71) | -- |
| **Đa Section ($\ge 3$ Section)** | **94.3%** (450/477) | -- | **7.0%** (5/71) | -- |

### Phân Bố Số Lượng Section & Kiểm Tra Tỷ Lệ Fallback 0.00%
Khảo sát toàn bộ 477 CV kiểm thử duy nhất cho thấy:
* **0 section chức năng**: **0 CVs (0.00%)**.
* **1 section chức năng**: **2 CVs (0.42%)** (cả 2 CV này chỉ có mục `experience`).
* **2 section chức năng**: **28 CVs (5.87%)**.
* **3 section chức năng**: **107 CVs (22.43%)**.
* **4 section chức năng đầy đủ**: **340 CVs (71.28%)**.

*Phần mở đầu (preamble)*: Có 72 CV (15.09%) chứa đoạn văn bản mở đầu trước heading đầu tiên, nhưng 100% trong số này đều có ít nhất 1 section chức năng sau đó; **không có bất kỳ CV nào chỉ chứa duy nhất preamble**.

*Cơ chế Dynamic Weight Renormalization*: Khi một CV chỉ có 1 section chức năng (ví dụ `experience` với trọng số chuẩn 0.35), hệ thống tự động chuẩn hóa mẫu số: $\frac{0.35 \cdot S_{\text{exp}}}{0.35} = S_{\text{exp}}$. Điểm số được tính trực tiếp từ vector section đó mà không cần kích hoạt fallback về single-vector toàn văn. Do đó, tỷ lệ fallback trên tập test chính thức là **chính xác 0.00%**.

> **Lưu ý về mâu thuẫn số liệu trước đây**:
> Các bản thảo sơ khởi trước đây từng ghi nhận tỷ lệ xuất hiện từ khóa section dựa trên tìm kiếm regex toàn văn (tìm chữ "skills", "experience" rải rác trong văn bản mà không xét cấu trúc tiêu đề). Khi đưa vào pipeline trích xuất thực tế, lỗi dính heading HTML (`SummaryHighly`, `SkillsAccounting`) từng khiến 89.7% CV rơi vào fallback. Bộ tách mới tại commit `41e1d27` đã khắc phục lỗi này: **99.6% CV nhận diện thành công $\ge 2$ section chức năng**, đưa tỷ lệ fallback về chính xác 0.00%.
>
> **Hiện tượng Cắt cụt Token (Truncation)**:
> Model `all-MiniLM-L6-v2` có `max_seq_length` mặc định là **256 tokens** (không phải 512). Toàn văn CV trung bình dài ~966 tokens $\rightarrow$ nếu dùng Single-Vector thông thường, hơn 95% CV sẽ bị cắt cụt phần sau (thường là học vấn và dự án). Phân rã Section-Aware giúp cô lập mục Kỹ năng (49.8 từ) và Học vấn (38.9 từ) vào các vector riêng biệt.

---

## 8. Khảo Sát Độ Bao Phủ Kỹ Năng Của Ontology Taxonomy

Sử dụng bộ từ điển chuẩn hóa 615 kỹ năng kỹ thuật (`src/cvjd/data/taxonomy/skills_taxonomy.json`):
* **Tỷ lệ JD trích xuất được kỹ năng**: **87.3%** (Trung bình 4.4 kỹ năng/JD, trong đó có 3.0 kỹ năng bắt buộc Must-have).
* **Tỷ lệ CV trích xuất được kỹ năng**: **96.4%** (Trung bình 10.3 kỹ năng/CV).
* **Số kỹ năng duy nhất được kích hoạt trong test set**: **297** / 615 khái niệm ontology.

---

## 9. Hai Phương Án Xử Lý Query Không Có Positive (Zero-positive Queries)

Trong tập kiểm thử gồm 71 JDs:
* Ở ngưỡng $\ge 1$: có **10 / 71 queries (14.1%)** hoàn toàn không có bất kỳ ứng viên nào đạt chuẩn (`No Fit` 100%).
* Ở ngưỡng $= 2$: có **43 / 71 queries (60.6%)** không có bất kỳ ứng viên nào đạt `Good Fit`.

### Đối chiếu kết quả giữa 2 phương án:

#### 1. Phương án A (All 71 Queries - Đánh giá toàn diện):
* Tính trung bình qua toàn bộ 71 queries; các query không có positive nhận điểm $0.0$.
* Trần Oracle đạt **0.8592** (ở ngưỡng $\ge 1$) và **0.3944** (ở ngưỡng $= 2$).
* Sàn Random: MRR **0.5721**, MAP **0.4941**.
* Single-Vector MiniLM: nDCG@10 **0.6146**, MRR **0.7055**, MAP **0.6210**.
* Section-Aware MiniLM: nDCG@10 **0.6170**, MRR **0.7091**, MAP **0.6328**.
* Calibrated Hybrid: nDCG@10 **0.6162**, MRR **0.6764**, MAP **0.6366**.

#### 2. Phương án B (Filtered Queries - Chuẩn TREC Standard):
Chỉ đánh giá trên các queries có ít nhất 1 positive candidate ($N=61$ cho $\text{rel} \ge 1$ và $N=28$ cho $\text{rel} = 2$). Trần Oracle đạt tuyệt đối **1.0000** cho nDCG@10, MRR, MAP.

| Phương Pháp / Cấu Hình | Ngưỡng $\text{rel} \ge 1$ ($N=61$) nDCG@10 | $\text{rel} \ge 1$ MRR | $\text{rel} \ge 1$ MAP | Ngưỡng $\text{rel} = 2$ ($N=28$) nDCG@10 | $\text{rel} = 2$ MRR | $\text{rel} = 2$ MAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Floor: Random Ranking** | 0.5743 | 0.6639 | 0.5741 | 0.6094 | 0.6566 | 0.5561 |
| **BM25 / TF-IDF (Lexical)** | 0.6720 | 0.7441 | 0.6903 | 0.7011 | 0.7423 | 0.6779 |
| **MiniLM Single-Vector (Flat)** | 0.7153 | 0.8211 | 0.7228 | 0.7443 | 0.8214 | 0.6798 |
| **MiniLM Chunk-Mean** | 0.7208 | 0.8126 | 0.7398 | 0.7631 | 0.8107 | 0.7316 |
| **MiniLM Chunk-MaxSim** | 0.6974 | 0.7898 | 0.7122 | 0.7224 | 0.7474 | 0.6822 |
| **MiniLM Section-Aware (Pure)** | 0.7182 | 0.8253 | 0.7365 | 0.7770 | 0.8869 | 0.7411 |
| **BGE-M3 Single-Vector** | 0.6910 | 0.7630 | 0.7178 | 0.7170 | 0.7381 | 0.6704 |
| **BGE-M3 Section-Aware** | 0.7084 | 0.8294 | 0.7181 | 0.7740 | 0.9405 | 0.7101 |
| **Dense + BM25 ($L_0$ Baseline)** | 0.7127 | 0.7916 | 0.7376 | 0.7689 | 0.8801 | 0.7422 |
| **Calibrated Hybrid** | 0.7173 | 0.7873 | 0.7410 | 0.7794 | 0.8492 | 0.7522 |
| **Parameter-Free RRF** | 0.7109 | 0.7996 | 0.7332 | 0.7632 | 0.8456 | 0.7383 |
| **2-Stage Rerank (Top-10 CE)** | 0.6711 | 0.7459 | 0.6712 | 0.7168 | 0.7560 | 0.6492 |
| **Full Cross-Encoder ($N=25$)** | 0.6437 | 0.7473 | 0.6577 | 0.6715 | 0.7351 | 0.6222 |
| **Ceiling: Oracle Ranking** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 10. Kiểm Định Ý Nghĩa Thống Kê & Ma Trận Triệt Tiêu Taxonomy 2 Chiều (2D Ablation)

* **Cơ chế phá đồng hạng**: Sử dụng kỳ vọng toán học qua $50$ lần xáo trộn ngẫu nhiên các phần tử có điểm số bằng nhau (jittered uniform perturbation) để loại bỏ hoàn toàn thiên vị thứ tự index trong file CSV.
* **Kiểm định Cặp Bootstrap 95% CI & Paired $t$-test (Section-Aware vs Single-Vector nDCG@10)**: $\bar{\Delta} = +0.0024$, 95% Bootstrap CI: $[-0.0296, +0.0333]$, $t = 0.150$, raw $p = 0.8811$, Holm-Bonferroni adjusted $p = 1.0000$ (Wilcoxon signed-rank $p = 0.4184$). Khoảng tin cậy bao gồm 0 $\implies$ tương đương thống kê (statistical parity) về thứ hạng ứng viên top đầu trên tập test đầy đủ.
* **Kiểm định Cặp Bootstrap 95% CI & Paired $t$-test (Calibrated Hybrid vs Dense+BM25 MAP)**: $\bar{\Delta} = +0.0029$, 95% Bootstrap CI: $[-0.0055, +0.0127]$, $t = 0.611$, raw $p = 0.5430$, Holm-Bonferroni adjusted $p = 1.0000$ (Wilcoxon signed-rank $p = 0.3402$). Khoảng tin cậy bao gồm 0 $\implies$ việc bổ sung Taxonomy không tạo ra mức tăng thứ hạng có ý nghĩa thống kê; giá trị thực tiễn cốt lõi nằm ở tính minh bạch, khả năng giải trình (explainability) và đảm bảo các điều kiện tiên quyết (hard constraints) cho nhà tuyển dụng.

### Ma Trận Triệt Tiêu Taxonomy 2 Chiều (Hàm Điểm $\times$ Tầng Quan Hệ):
So khớp đối chiếu với baseline $L_0$ (Dense + BM25, không dùng taxonomy):

| Cấu Hình Tầng Taxonomy | Hàm Điểm (Scoring Function) | nDCG@10 | MAP | $\Delta$ MAP | Raw $p$-val | Holm $p$-val | Ý Nghĩa ($\alpha=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline $L_0$** | Dense + BM25 (No Taxonomy) | 0.6123 | 0.6337 | -- | -- | -- | -- |
| **Layer $L_1$** | Symmetric Jaccard | 0.5910 | 0.5977 | -0.0360 | 0.1651 | 0.6393 | Không |
| **Layer $L_1+L_2$** | Symmetric Jaccard | 0.5913 | 0.5959 | -0.0378 | 0.1542 | 0.6393 | Không |
| **Layer $L_1$** | Asymmetric Coverage | 0.5773 | 0.5888 | -0.0449 | 0.1077 | 0.6393 | Không |
| **Layer $L_1+L_2$** | Asymmetric Coverage | 0.5783 | 0.5896 | -0.0441 | 0.1080 | 0.6393 | Không |
| **Layer $L_1+L_2+L_3$** | Asymmetric Coverage (Must/Nice) | 0.5775 | 0.5892 | -0.0446 | 0.1066 | 0.6393 | Không |
| **Calibrated Hybrid** | 0.60 Retrieval + 0.40 Coverage | 0.6162 | 0.6366 | +0.0029 | 0.5430 | 0.6393 | Không |

---

## 11. Phân Tích Độ Nhạy Siêu Tham Số Trên Tập Huấn Luyện (Train Sensitivity Analysis)

Nhằm đảm bảo tính minh bạch phương pháp luận và tránh rò rỉ tập kiểm thử (test leakage), toàn bộ khảo sát độ nhạy siêu tham số được thực thi độc lập và độc quyền trên **`train.csv`** (50 queries, 479 CVs, 1.627 cặp CV–JD, `seed=42`):

### 1. Độ nhạy Trọng số Section (Section Weights):
Khảo sát phân bổ trọng số giữa các section chức năng (Skills / Experience / Education / Summary):
* **Default (0.40 / 0.35 / 0.15 / 0.10)**: nDCG@10 = **0.6601**, MAP = 0.6930.
* **Equal (0.25 / 0.25 / 0.25 / 0.25)**: nDCG@10 = **0.6613**, MAP = 0.6921.
* **Skills-Heavy (0.60 / 0.20 / 0.10 / 0.10)**: nDCG@10 = **0.6578**, MAP = 0.6869.
* **Experience-Heavy (0.20 / 0.60 / 0.10 / 0.10)**: nDCG@10 = **0.6636**, MAP = 0.6937.
* **Core Only (0.50 / 0.50 / 0.00 / 0.00)**: nDCG@10 = **0.6578**, MAP = 0.6898.

*Nhận xét*: Biên độ dao động của nDCG@10 nằm trong phạm vi cực hẹp ($\pm 0.005$), chứng minh mô hình Section-Aware không bị phụ thuộc vào một cấu hình trọng số heuristically duy nhất.

### 2. Quét Tham số Kết hợp Dense--Lexical ($\alpha$ Sweep):
Hòa trộn $S_{\text{retrieval}} = \alpha S_{\text{dense}} + (1-\alpha) S_{\text{bm25}}$:
* $\alpha = 0.0$ (Thuần BM25): nDCG@10 = 0.6626, MAP = 0.6952.
* $\alpha = 0.2$: nDCG@10 = 0.6884, MAP = 0.7186.
* $\alpha = 0.4$: nDCG@10 = 0.6946, MAP = 0.7224.
* $\alpha = 0.5$: nDCG@10 = 0.6951, MAP = 0.7193.
* **$\alpha = 0.6$ (Đỉnh thực nghiệm trên train)**: nDCG@10 = **0.6994**, MAP = 0.7193.
* $\alpha = 0.7$: nDCG@10 = 0.6882, MAP = 0.7127.
* $\alpha = 0.8$: nDCG@10 = 0.6870, MAP = 0.7122.
* $\alpha = 1.0$ (Thuần Dense): nDCG@10 = 0.6601, MAP = 0.6930.

### 3. Quét Tham số Trọng số Retrieval--Taxonomy ($\beta$ Sweep):
Hòa trộn $S_{\text{final}} = \beta S_{\text{retrieval}} + (1-\beta) S_{\text{taxonomy}}$:
* $\beta = 0.4$: nDCG@10 = 0.7070, MAP = 0.7210.
* $\beta = 0.5$: nDCG@10 = 0.7074, MAP = 0.7205.
* **$\beta = 0.6$ (Đỉnh thực nghiệm trên train)**: nDCG@10 = **0.7078**, MAP = 0.7209.
* $\beta = 0.7$: nDCG@10 = 0.7054, MAP = 0.7238.
* $\beta = 0.8$: nDCG@10 = 0.7009, MAP = 0.7238.
* $\beta = 1.0$ (Không dùng Taxonomy): nDCG@10 = 0.6994, MAP = 0.7193.

---

## 12. Đo Lường Độ Trễ Chuẩn Hóa Đa Lần Lặp (Standardized Latency Profile)

Đo đạc trên môi trường Intel Core i7 CPU (Multi-core, 16GB RAM), PyTorch CPU (không GPU), pool size $N=25$ ứng viên / query. Giao thức đo: **10 lần khởi động (warm-up)**, **15--20 lần lặp độc lập**, các phép toán ma trận nhanh được lấy trung bình qua 100 lần lặp nội bộ nhằm triệt tiêu nhiễu phân giải đồng hồ hệ thống.

*Định nghĩa độ trễ*:
* $T_{\text{index}}$: Thời gian offline mã hóa/chỉ mục trước 1 CV.
* $T_{\text{query}}$: Thời gian online mã hóa văn bản truy vấn JD lúc nhận request.
* $T_{\text{rank}}$: Thời gian tính ma trận tương đồng và sắp xếp $N=25$ ứng viên.
* $T_{\text{online}}$: Tổng độ trễ phục vụ người dùng cuối trực tuyến ($T_{\text{online}} = T_{\text{query}} + T_{\text{rank}}$).

| Phương Pháp / Cấu Hình | $T_{\text{index}}$ (ms/CV) | $T_{\text{query}}$ (ms) [IQR] | $T_{\text{rank}}$ (ms, $N=25$) [IQR] | $T_{\text{online}}$ (ms) | Đặc Điểm Vận Hành |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BM25 / TF-IDF** | 0.64 | 0.72 [0.52, 1.03] | 0.16 [0.16, 0.17] | 0.88 | Sparse inverted dot-product |
| **Single-Vector MiniLM** | 19.25 | 29.78 [28.46, 30.11] | 0.004* [0.004, 0.004] | 29.79 | Dense 384-dim matrix multiply |
| **Section-Aware MiniLM** | 84.85 | 29.78 [28.46, 30.11] | 0.12 [0.10, 0.13] | 29.90 | Dynamic cross-section cosine |
| **BGE-M3 Flat** | 3012.90 | 1646.12 [972.63, 2965.03] | 0.007* [0.007, 0.008] | 1646.13 | 1024-dim dense matrix multiply |
| **BGE-M3 Section-Aware** | 3314.19 | 1646.12 [972.63, 2965.03] | 0.13 [0.09, 0.16] | 1646.25 | 1024-dim section weighted cosine |
| **Calibrated Hybrid** | 85.49 | 30.75 [30.13, 31.14] | 2.69 [2.63, 2.71] | 33.44 | MinMax fusion + taxonomy match |
| **Parameter-Free RRF** | 85.49 | 30.75 [30.13, 31.14] | 2.83 [2.77, 2.85] | 33.57 | RRF rank sort + taxonomy match |
| **2-Stage Top-10 CE** | -- | 30.75 [30.13, 31.14] | 426.88 [424.88, 427.43] | 457.63 | Stage 1 candidate gen + Top-10 CE |
| **Full Cross-Encoder** | -- | -- | 1178.05 [1173.02, 1189.00] | 1178.05 | Full $N=25$ neural cross-attention |

*\*Lưu ý: Các giá trị $T_{\text{rank}} < 0.10$ ms phản ánh phép nhân ma trận thuần túy trong RAM được tối ưu hóa vectorized. Độ trễ trực tuyến thực tế $T_{\text{online}}$ bị chi phối $98\%$ bởi thời gian mã hóa truy vấn $T_{\text{query}}$ ($\approx 30$ ms với MiniLM và $\approx 1646$ ms với BGE-M3).*

---

## 13. Đánh Giá Đối Chứng Trên 76 Query Sạch Của Tập Huấn Luyện (Clean Train Benchmark)

Thực hiện theo kế hoạch phân tích cố định đã đăng ký trước (Pre-registered plan tại `docs/PRE_REGISTERED_TRAIN_EVAL_PLAN.md`, commit `d22eb66`). Loại trừ toàn bộ 204 JDs đã dùng cho dev-luật parser (168 JDs), đo ground-truth parser (25 JDs), và quét độ nhạy tham số (50 JDs), giữ lại **76 queries sạch hoàn toàn (1.829 cặp CV--JD)**:

### 1. Tỷ lệ ký tự rơi vào section chức năng (Functional Character Coverage):
* **Văn bản CVs (547 CVs)**: Trung bình **95.32%** ký tự (Trung vị: **99.00%**) thuộc về các section chức năng (`skills`, `experience`, `education`, `summary`). Tỷ lệ đoạn mở đầu (`preamble`) chỉ chiếm trung bình **2.00%**. Tỷ lệ Fallback đạt **0.00%** (0 / 547 CVs).
* **Văn bản JDs (76 JDs)**: Trung bình **30.71%** ký tự (Trung vị: **0.00%**) có heading chức năng; phần lớn JDs mở đầu bằng văn bản mô tả doanh nghiệp/vai trò chung chiếm trung bình **67.72%** (`preamble`).

### 2. Kết quả xếp hạng trên 76 Query Sạch:
* **Setting A (Toàn bộ 76 queries)**:
  * Single-Vector Flat: nDCG@10 = 0.6540, MRR = 0.7269, MAP = 0.6503, P@5 = 0.4500.
  * Section-Aware MiniLM: nDCG@10 = 0.6890, MRR = 0.7492, MAP = 0.6892, P@5 = 0.4763.
  * Calibrated Hybrid: nDCG@10 = 0.6914, MRR = 0.7632, MAP = 0.6822, P@5 = 0.4553.
  * Dense + BM25 ($L_0$): nDCG@10 = 0.7043, MRR = 0.7794, MAP = 0.6924, P@5 = 0.4895.
  * Parameter-Free RRF: nDCG@10 = 0.6227, MRR = 0.6942, MAP = 0.6048, P@5 = 0.4605.
* **Setting B (Filtered $\text{rel}\ge 1$, $N=71$ queries)**:
  * Single-Vector Flat: nDCG@10 = 0.7001, MRR = 0.7781, MAP = 0.6961.
  * Section-Aware MiniLM: nDCG@10 = 0.7376, MRR = 0.8020, MAP = 0.7377.
  * Calibrated Hybrid: nDCG@10 = 0.7401, MRR = 0.8170, MAP = 0.7303.
  * Dense + BM25: nDCG@10 = 0.7539, MRR = 0.8343, MAP = 0.7411.
* **Kiểm định thống kê cặp (Setting A & B)**:
  * RQ1 (Section-Aware vs Single-Vector): Setting A có $\bar{\Delta} = +0.0351$, 95% CI $[+0.0024, +0.0658]$, raw $p=0.0379$, nhưng sau hiệu chỉnh Holm thì $p_{\text{adj}} = 0.1517 > 0.05 \implies$ **tương đương thống kê**, hoàn toàn nhất quán với kết luận trên tập test.
  * RRF vs Calibrated Hybrid: $\bar{\Delta} = -0.0687$, 95% CI $[-0.0996, -0.0387]$, raw $p < 0.0001$, Holm $p = 0.0003 \implies$ **RRF kém hơn Calibrated Hybrid có ý nghĩa thống kê**.

---

## 14. Đánh Giá Ngoại Vi Trên Bộ Dữ Liệu `med2425/resume-job-fit-merged-v1`

* **Nguồn gốc & Bản chất**: Dataset do Mohamed Douali phát hành (2026), kết hợp `cnamuangtoun` và `kens1ang` để tạo các cặp ghép ứng viên dense candidate pool và được gán nhãn tự động bởi **Qwen2.5-32B** (LLM Recruiter ở temperature = 0).
* **Tập kiểm chứng phụ (Miền IT chuyên biệt)**: Lọc các cặp thỏa mãn `resume_domain` $\in \{\text{software}, \text{data}, \text{ai}\}$ và `jd_domain` $\in \{\text{software}, \text{data}, \text{ai}\}$:
  * Quy mô: **41 JDs, 277 CVs, 8.575 cặp CV--JD** (mật độ pool rất dày: ~209 ứng viên/query).
* **Kết quả đối sánh dưới nhãn LLM Recruiter**:
  * Floor Random: nDCG@10 = 0.2350, MRR = 0.5748, MAP = 0.4179.
  * BM25 / TF-IDF: nDCG@10 = 0.5427, MRR = 0.9129, MAP = 0.6052.
  * MiniLM Single-Vector: nDCG@10 = 0.5393, MRR = 0.8189, MAP = 0.6323.
  * MiniLM Section-Aware: nDCG@10 = 0.5136, MRR = 0.7859, MAP = 0.6132 (Kiểm định cặp với Single-Vector: $\bar{\Delta} = -0.0257$, 95% Bootstrap CI $[-0.0780, +0.0242]$, paired $t$ $p=0.3588 \implies$ tương đương thống kê).
  * Dense + BM25 ($L_0$): nDCG@10 = 0.5879, MRR = 0.8892, MAP = 0.6438.
  * **Calibrated Hybrid ($L_0$ + Taxonomy)**: nDCG@10 = **0.6242**, MRR = **0.9142**, MAP = **0.6814**, P@5 = **0.7805** ($\Delta = +0.0849$ so với Single-Vector, $\Delta = +0.0363$ so với Dense+BM25).
  * *Nhận định*: Trong môi trường candidate pool rất lớn (~209 ứng viên/query) với nhãn do mô hình LLM tuyển dụng gán, việc kết hợp Dense + BM25 + Taxonomy Coverage thể hiện sức mạnh vượt trội rõ nét trong việc cô lập các ứng viên đạt chuẩn kỹ năng chuyên môn.

---

*Báo cáo được thực thi và xác nhận hoàn toàn trên bộ dữ liệu công khai bởi các script `scripts/run_scientific_benchmark.py`, `scripts/measure_rigorous_latency.py`, `scripts/run_clean_train_benchmark.py`, và `scripts/evaluate_med2425_llm_benchmark.py`.*
