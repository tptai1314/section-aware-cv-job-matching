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
1. **Phương án A (All 71 Queries - Đánh giá toàn diện)**:
   * Tính trung bình qua toàn bộ 71 queries; các query không có positive nhận điểm $0.0$.
   * Trần Oracle đạt **0.8592** (ở ngưỡng $\ge 1$) và **0.3944** (ở ngưỡng $= 2$).
   * Sàn Random: MRR **0.5721**, MAP **0.4941**.
   * Single-Vector: nDCG@10 **0.6146**, MRR **0.7055**, MAP **0.6210**.
   * Section-Aware Dense: nDCG@10 **0.6170**, MRR **0.7091**, MAP **0.6328**.
   * Calibrated Hybrid: nDCG@10 **0.6162**, MRR **0.6764**, MAP **0.6366**.
2. **Phương án B (Filtered Queries - Chuẩn TREC Standard)**:
   * Chỉ tính trên các queries có ít nhất 1 positive ($N=61$ cho $\ge 1$, $N=28$ cho $= 2$).
   * Trần Oracle đạt tuyệt đối **1.0000** cho nDCG@10, MRR, MAP.
   * Single-Vector: nDCG@10 **0.7153**, MRR **0.8211**, MAP **0.7228**.
   * Section-Aware Dense: nDCG@10 **0.7182**, MRR **0.8253**, MAP **0.7365**.
   * Calibrated Hybrid: nDCG@10 **0.7173**, MRR **0.7873**, MAP **0.7410**.

---

## 10. Kiểm Định Ý Nghĩa Thống Kê & Phá Đồng Hạng (Fair Tie-breaking)

* **Cơ chế phá đồng hạng**: Sử dụng kỳ vọng toán học qua $50$ lần xáo trộn ngẫu nhiên các phần tử có điểm số bằng nhau (jittered uniform perturbation) để loại bỏ hoàn toàn thiên vị thứ tự index trong file CSV.
* **Kiểm định Cặp Bootstrap 95% CI & Paired $t$-test (Section-Aware vs Single-Vector nDCG@10)**: $\bar{\Delta} = +0.0024$, 95% Bootstrap CI: $[-0.0296, +0.0333]$, $t = 0.150$, raw $p = 0.8811$, Holm-Bonferroni adjusted $p = 1.0000$ (Wilcoxon signed-rank $p = 0.4184$). Khoảng tin cậy bao gồm 0 $\implies$ tương đương thống kê (statistical parity) về thứ hạng ứng viên top đầu trên tập test đầy đủ.
* **Kiểm định Cặp Bootstrap 95% CI & Paired $t$-test (Calibrated Hybrid vs Dense+BM25 MAP)**: $\bar{\Delta} = +0.0029$, 95% Bootstrap CI: $[-0.0055, +0.0127]$, $t = 0.611$, raw $p = 0.5430$, Holm-Bonferroni adjusted $p = 1.0000$ (Wilcoxon signed-rank $p = 0.3402$). Khoảng tin cậy bao gồm 0 $\implies$ việc bổ sung Taxonomy không tạo ra mức tăng thứ hạng có ý nghĩa thống kê; giá trị thực tiễn cốt lõi nằm ở tính minh bạch, khả năng giải trình (explainability) và đảm bảo các điều kiện tiên quyết (hard constraints) cho nhà tuyển dụng.

---

*Báo cáo được thực thi và xác nhận hoàn toàn trên bộ dữ liệu công khai bởi script `scripts/run_comprehensive_audit.py`.*
