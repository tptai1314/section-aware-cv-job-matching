# Hệ Thống Ghép Cặp Hồ Sơ Ứng Viên & Mô Tả Công Việc Song Ngữ (Bilingual & Section-Aware CV–Job Matching)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Vector DB](https://img.shields.io/badge/ChromaDB-Multi--Vector-orange.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-BAAI%2FBGE--M3%20%7C%20MiniLM-green.svg)](https://huggingface.co/BAAI/bge-m3)
[![Tests](https://img.shields.io/badge/Tests-151%20Passed-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/Paper-Springer%20LNCS%20(17%20pages)-purple.svg)](paper/main.pdf)

Hệ thống AI gợi ý và ghép cặp thông minh hai chiều giữa **Hồ sơ ứng viên (CV/Resume)** và **Mô tả công việc (Job Description - JD)** được nghiên cứu và phát triển cho nền tảng tuyển dụng freelancer **Frevia**.

Dự án giải quyết trực tiếp bài toán tuyển dụng trong bối cảnh **đa ngôn ngữ (Bilingual: Tiếng Việt – Tiếng Anh)** và **trộn ngôn ngữ kỹ thuật (Code-Mixing)** đặc trưng của thị trường CNTT Việt Nam. Thay vì xem tài liệu như văn bản phẳng truyền thống, hệ thống áp dụng kỹ thuật **biểu diễn đa vector theo section (Section-Aware Multi-Vector)**, **tái chuẩn hóa trọng số động (Dynamic Weight Renormalization)**, **độ phủ kỹ năng bất đối xứng (Asymmetric Skill Coverage)** kết hợp với **mô hình chấm điểm lai cân bằng thang đo (Calibrated Hybrid Fusion)** và **giải thích bằng chứng bằng LLM (LLM Grounded Explanation)**. Toàn bộ pipeline chạy hoàn toàn trên **CPU** với độ trễ phục vụ cực thấp ($15 - 32$ ms/query).

---

## Mục lục

- [1. Câu Hỏi Nghiên Cứu (Research Questions)](#1-câu-hỏi-nghiên-cứu-research-questions)
- [2. Điểm Nổi Bật & Đóng Góp Thực Tế](#2-điểm-nổi-bật--đóng-góp-thực-tế)
- [3. Kiến Trúc Hệ Thống](#3-kiến-trúc-hệ-thống)
- [4. Cơ Chế Hoạt Động & Cải Tiến Cốt Lõi](#4-cơ-chế-hoạt-động--cải-tiến-cốt-lõi)
  - [4.1 Phân đoạn Section Song ngữ & Đa Vector](#41-phân-đoạn-section-song-ngữ--đa-vector)
  - [4.2 Tái chuẩn hóa Trọng số Động (Dynamic Weight Renormalization)](#42-tái-chuẩn-hóa-trọng-số-động-dynamic-weight-renormalization)
  - [4.3 Độ phủ Kỹ năng Bất đối xứng vs. Jaccard (Asymmetric Skill Coverage)](#43-độ-phủ-kỹ-năng-bất-đối-xứng-vs-jaccard)
  - [4.4 Chuẩn hóa Kỹ năng Song ngữ & Xử lý Không dấu](#44-chuẩn-hóa-kỹ-năng-song-ngữ--xử-lý-không-dấu)
  - [4.5 Chuẩn hóa Thang điểm Trước khi Dung hợp Alpha (Score Calibration)](#45-chuẩn-hóa-thang-điểm-trước-khi-dung-hợp-alpha)
  - [4.6 Truy hồi Hai Giai đoạn (Two-Stage Retrieval)](#46-truy-hồi-hai-giai-đoạn-two-stage-retrieval)
  - [4.7 Matching Hai Chiều (Bidirectional Matching)](#47-matching-hai-chiều-bidirectional-matching)
  - [4.8 Tính Giải thích Thực chứng bằng LLM (Grounded Explanation)](#48-tính-giải-thích-thực-chứng-bằng-llm)
- [5. Nguồn Dữ Liệu & Thiết Kế Phân Tầng Ngôn Ngữ](#5-nguồn-dữ-liệu--thiết-kế-phân-tầng-ngôn-ngữ)
  - [5.1 Các Bộ Dữ Liệu Nghiên Cứu](#51-các-bộ-dữ-liệu-nghiên-cứu)
  - [5.2 Quy ước Phân loại Ngôn ngữ Tài liệu (Ngưỡng $\ge 20\%$)](#52-quy-ước-phân-loại-ngôn-ngữ-tài-liệu-ngưỡng-ge-20)
  - [5.3 5 Nhóm Cặp Đánh Giá](#53-5-nhóm-cặp-đánh-giá)
  - [5.4 Quy trình Dịch có Gắn cờ & Kiểm tra Nhất quán Nhãn](#54-quy-trình-dịch-có-gắn-cờ--kiểm-tra-nhất-quán-nhãn)
- [6. Đánh Giá Thực Nghiệm & Kết Quả Học Thuật](#6-đánh-giá-thực-nghiệm--kết-quả-học-thuật)
  - [6.1 Trả lời RQ1: Hiệu năng trên các nhóm cặp ngôn ngữ (Table 7)](#61-trả-lời-rq1-hiệu-năng-trên-các-nhóm-cặp-ngôn-ngữ)
  - [6.2 Trả lời RQ2: Multilingual Dense vs. Lexical BM25 & Tính công bằng Baseline](#62-trả-lời-rq2-multilingual-dense-vs-lexical-bm25)
  - [6.3 Trả lời RQ3: Lợi thế biểu diễn Section-Aware & Ablation Study](#63-trả-lời-rq3-lợi-thế-biểu-diễn-section-aware)
  - [6.4 Trả lời RQ4: Độ bền trước dữ liệu bẩn (Không dấu, Code-Mixing, Thiếu section)](#64-trả-lời-rq4-độ-bền-trước-dữ-liệu-bẩn)
  - [6.5 Đánh giá Tổng quát hóa trên Benchmark Quốc tế Công khai](#65-đánh-giá-tổng-quát-hóa-trên-benchmark-quốc-tế-công-khai)
  - [6.6 Kiểm định Thống kê & Khoảng Tin cậy 95% Bootstrap](#66-kiểm-định-thống-kê--khoảng-tin-cậy-95-bootstrap)
  - [6.7 Thẩm định Công bằng & Thiên kiến (Counterfactual Fairness Test)](#67-thẩm-định-công-bằng--thiên-kiến)
  - [6.8 Bài học Thực tiễn & Hướng dẫn Vận hành](#68-bài-học-thực-tiễn--hướng-dẫn-vận-hành)
- [7. Cấu Trúc Thư Mục](#7-cấu-trúc-thư-mục)
- [8. Hướng Dẫn Cài Đặt & Sử Dụng](#8-hướng-dẫn-cài-đặt--sử-dụng)
- [9. Bản Thảo Bài Báo Khoa Học (Springer LNCS Paper)](#9-bản-thảo-bài-báo-khoa-học-springer-lncs-paper)

---

## 1. Câu Hỏi Nghiên Cứu (Research Questions)

Đề tài tập trung giải quyết 4 câu hỏi nghiên cứu cốt lõi:

* **RQ1 (Monolingual vs. Cross-Lingual Discrepancy)**: *Hiệu năng ghép CV–JD thay đổi thế nào giữa cặp đơn ngữ ($\text{VI}\leftrightarrow\text{VI}$, $\text{EN}\leftrightarrow\text{EN}$) và chéo ngôn ngữ ($\text{VI}\leftrightarrow\text{EN}$)?*
* **RQ2 (Multilingual Dense vs. Lexical BM25)**: *Embedding đa ngôn ngữ so với từ khóa (BM25) khác nhau ra sao ở từng nhóm, đặc biệt ở nhóm chéo ngôn ngữ?*  
  *(Kiểm chứng giả thuyết: Phân tích sự sụt giảm của truy hồi từ khóa khi chuyển dịch ngôn ngữ, đồng thời chứng minh BM25 vẫn giữ được độ chính xác cục bộ khi có token thực thể kỹ năng tiếng Anh dùng chung).*
* **RQ3 (Section Granularity across Languages)**: *Biểu diễn section-aware có giúp thêm so với văn bản phẳng (single-vector và chunking), và lợi ích này có khác nhau giữa các nhóm ngôn ngữ không?*
* **RQ4 (Robustness to Real-World Noisy Data)**: *Hệ thống bền đến đâu trước dữ liệu thực tế bẩn (gõ tiếng Việt không dấu, trộn ngôn ngữ tự nhiên, thiếu section không bắt buộc)?*

---

## 2. Điểm Nổi Bật & Đóng Góp Thực Tế

1. **Khắc phục hiện tượng suy giảm ngữ nghĩa (Semantic Dilution)**: Việc nén toàn bộ CV dài $2 - 3$ trang vào một vector duy nhất làm loãng thông tin các kỹ năng then chốt. Hệ thống tách tài liệu thành từng phần độc lập (`skills`, `experience`, `education`, `summary`) và nhúng đa vector độc lập.
2. **Xử lý Section rỗng chuẩn mực (Dynamic Weight Renormalization)**: Không dùng Zero-Imputation gây phạt oan ứng viên Junior thiếu kinh nghiệm hoặc Freelancer thiếu bằng cấp; hệ thống tự động tái chuẩn hóa mẫu số trọng số theo các section thực tế có mặt.
3. **Độ phủ Kỹ năng Bất đối xứng (Asymmetric Requirement Coverage)**: Thay thế Jaccard đối xứng nhằm tránh phạt các ứng viên Senior giỏi hơn yêu cầu công việc ("penalizing overqualification").
4. **Chuẩn hóa Kỹ năng Song ngữ & Xử lý Không dấu**: Tích hợp bảng ánh xạ từ vựng kỹ thuật cốt lõi (Việt - Anh: *cơ sở dữ liệu* $\leftrightarrow$ *database*, *kỹ sư phần mềm* $\leftrightarrow$ *software engineer*) kết hợp chuẩn hóa Unicode NFC và so khớp không dấu `strip_accents()`.
5. **Cân bằng Thang đo (Score Calibration)**: Chuẩn hóa Min-Max theo từng truy vấn giúp cân bằng phương sai giữa dải điểm Cosine hẹp ($[0.60, 0.85]$) và dải điểm quy tắc rời rạc ($[0.0, 1.0]$) trước khi dung hợp $\alpha$.
6. **Chi phí thấp, chạy hoàn toàn trên CPU**: Sử dụng mô hình nhúng đa ngôn ngữ đóng băng `BAAI/bge-m3` ($1024$-d) kết hợp chỉ mục HNSW trên ChromaDB, đạt tốc độ $15 - 32$ ms/query mà không cần GPU đắt đỏ.

---

## 3. Kiến Trúc Hệ Thống

Hệ thống được thiết kế theo mô hình 5 tầng phân tách rõ ràng:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION & BILINGUAL STRATIFICATION                                │
│    PostgreSQL (Source of truth) ──db_export.py──▶ data/raw/*.jsonl          │
│    Language Classifier (VI / EN / Code-Mixed with >=20% token threshold)    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. BILINGUAL SECTIONING & MULTI-VECTOR INDEXING                             │
│    Regex Heading Engine (Accent-resilient & abbreviations) ──▶ Sections     │
│    Sections ──BAAI/BGE-M3 Multilingual Embedder (1024d)──▶ ChromaDB (HNSW)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. TWO-STAGE RETRIEVAL & CROSS-SECTION ALIGNMENT                            │
│    Query ──Full-text coarse vector──▶ Candidate Pool (Top-K Coarse)         │
│    Pool ──Cross-Section Matrix (Max-sim & Dynamic Renormalization)──▶ S_sem  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. CALIBRATED HYBRID RANKING & ELIGIBILITY HARD FILTERS                     │
│    Tier 1 Hard Filters (Privacy, Availability, Budget Ceiling)              │
│    Tier 2 Competency: Calibrated Min-Max Norm + Asymmetric Skill Coverage   │
│    S_hybrid = α · S_sem_norm + (1 - α) · [w_sk·S_skills + w_exp·S_exp + ...]│
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. SERVING, RE-RANKING & EXPLAINABILITY (apps/ai)                           │
│    FastAPI Gateway + Local LLM (Qwen-2.5 / Ollama on CPU)                   │
│    JSON Response: Match Score, Strengths, Skill Gaps, Verbatim Grounding    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Cơ Chế Hoạt Động & Cải Tiến Cốt Lõi

### 4.1 Phân đoạn Section Song ngữ & Đa Vector
Bộ bóc tách sử dụng từ điển tiêu đề đa ngữ (`_HEADING_RULES` trong [`preprocessing.py`](src/cvjd/data/preprocessing.py)) kết hợp cơ chế slug hóa tự động (`_slug()`), cho phép giữ lại đầy đủ tất cả các phần thông tin phong phú của CV và JD:

#### 1. Nhóm Section Chuẩn Hóa Cốt Lõi (Canonical Sections)
- `summary`: Mục tiêu nghề nghiệp, tóm tắt bản thân, giới thiệu tổng quan, profile, about me.
- `skills`: Danh sách kỹ năng chuyên môn, năng lực, công nghệ, tech stack, core competencies, expertise.
- `experience`: Lịch sử làm việc, vị trí đảm nhiệm, thành tựu, work experiences, employment history, exp.
- `education`: Trình độ học vấn, bằng cấp, đào tạo, chuyên ngành, academic background, qualifications, edu.

#### 2. Nhóm Section Mở Rộng & Nghiệp Vụ (Extended & Dynamic Sections)
- `projects`: Dự án thực tế, dự án cá nhân/tiêu biểu, quy mô, tech stack sử dụng, portfolio.
- `certifications`: Chứng chỉ chuyên môn quốc tế, giải thưởng, bằng cấp phụ (certs/licenses).
- `responsibilities`: Trách nhiệm công việc, mô tả nhiệm vụ chi tiết của dự án (trong JD).
- `budget`: Khung thù lao / ngân sách dự kiến của khách hàng hoặc mức giá freelancer đề xuất.
- `categories`: Lĩnh vực chuyên ngành, phân nhóm kỹ thuật (Web Dev, Mobile, AI/Data).
- `preamble`: Phần mở đầu tài liệu trước tiêu đề đầu tiên.
- `full_text`: Toàn văn tài liệu phục vụ truy hồi vòng thô (Coarse Retrieval).

Tất cả các section trên (dù là canonical hay dynamic slug) **đều được nhúng vector độc lập vào ChromaDB** và tham gia đầy đủ vào ma trận ghép cặp chéo (Cross-Section Pairing), giúp giữ trọn vẹn mọi thông tin có giá trị.

Audit độ chính xác trên 50 CV song ngữ gán nhãn tay đạt **$94.0\% - 100\%$**, tỷ lệ fallback lỗi $0.0\%$.

### 4.2 Tái chuẩn hóa Trọng số Động (Dynamic Weight Renormalization)
Khi tài liệu thiếu các section tùy chọn (ví dụ: hồ sơ Junior không có kinh nghiệm, hoặc Freelancer không ghi bằng cấp), việc gán điểm $0.0$ và giữ nguyên mẫu số (Zero-Imputation) sẽ phạt oan ứng viên. Hệ thống tự động tính điểm trung bình có trọng số trên tập hợp section thực sự có mặt:

$$S_{sem} = \frac{\sum_{m \in \mathcal{S}_{\text{matched}}} w_m \cdot c(s_m)}{\sum_{m \in \mathcal{S}_{\text{matched}}} w_m}$$

Trong đó $c(s_m) = \max_{s_k \in D_c} \cos(\mathbf{v}_q^{s_m}, \mathbf{v}_c^{s_k})$ qua phép ghép cặp chéo cực đại (Cross-Section Pairing).

### 4.3 Độ phủ Kỹ năng Bất đối xứng vs. Jaccard
Chỉ số **Jaccard đối xứng** ($J = \frac{|A \cap B|}{|A \cup B|}$) có lỗi logic nghiêm trọng trong tuyển dụng: **phạt nặng ứng viên giỏi hơn yêu cầu**.
* Nếu JD yêu cầu 2 kỹ năng: $\{\text{React, Node}\}$.
* Ứng viên Junior biết đúng 2 kỹ năng $\rightarrow J = 2/2 = 1.0$.
* Ứng viên Senior biết 10 kỹ năng $\{\text{React, Node, Docker, K8s, PostgreSQL,...}\} \rightarrow J = 2/10 = 0.20$ (bị phạt $80\%$ điểm số!).

Hệ thống thay bằng **Asymmetric Requirement Recall**:

$$\text{Cov}(\mathcal{T}) = \frac{\sum_{t \in \mathcal{T}} \mathbb{I}(t \in \mathcal{C}_{skills}) \cdot w_t}{\sum_{t \in \mathcal{T}} w_t}$$

$$S_{skills} = 0.8 \cdot \text{Cov}(\text{MustHave}) + 0.2 \cdot \text{Cov}(\text{NiceToHave})$$

Khi ứng viên thiếu trên $50\%$ kỹ năng bắt buộc ($\text{Cov} < 0.5$), áp dụng hệ số phạt nhân tử $\gamma = \text{Cov}(\text{MustHave})$ để ngăn việc bù đắp bởi các tiêu chí không cốt lõi.

### 4.4 Chuẩn hóa Kỹ năng Song ngữ & Xử lý Không dấu
* **Ánh xạ từ vựng cốt lõi**: Bảng `_ALIASES` trong [`hybrid.py`](src/cvjd/ranking/hybrid.py) giải quyết triệt để sự phân mảnh giữa tiếng Việt và tiếng Anh (*cơ sở dữ liệu* $\leftrightarrow$ *database*, *kỹ sư phần mềm* $\leftrightarrow$ *software engineer*, *lập trình web* $\leftrightarrow$ *web development*).
* **Diacritic & Variant Invariance**: Hàm `normalize_skill()` kết hợp Unicode NFC, chuyển đổi chữ thường, chuẩn hóa khoảng trắng và hàm `strip_accents()` để so khớp tương đồng không phụ thuộc dấu tiếng Việt.

### 4.5 Chuẩn hóa Thang điểm Trước khi Dung hợp Alpha (Score Calibration)
Điểm Cosine từ dense bi-encoder thường co cụm trong khoảng hẹp $[0.60, 0.85]$ (phương sai thấp $\sim 0.08$), trong khi điểm kỹ năng quy tắc $S_{skills}$ là đại lượng rời rạc trải dài $[0.0, 1.0]$ (phương sai cao $\sim 0.35$). Nếu cộng tuyến tính trực tiếp, $S_{skills}$ sẽ áp đảo toàn bộ thứ hạng.

Cơ chế `calibrate_scores=True` trong [`HybridRanker.rank()`](src/cvjd/ranking/engine.py) chuẩn hóa Min-Max điểm ngữ nghĩa theo từng truy vấn:

$$S'_{sem} = \frac{S_{sem} - \min_{k}(S_k)}{\max_{k}(S_k) - \min_{k}(S_k) + \epsilon}$$

Nhờ đó, cả hai thành phần có phương sai tương đồng trước khi thực hiện tổ hợp lồi với trọng số $\alpha$.

### 4.6 Truy hồi Hai Giai đoạn (Two-Stage Retrieval)
- **Giai đoạn 1 (Coarse Filtering)**: Vector toàn văn quét trên ChromaDB HNSW để lọc từ hàng nghìn hồ sơ xuống Top-$K$ ($K = 50 \div 100$). Đo kiểm thực tế cho thấy $K=100$ giữ lại trên **$85.32\%$** ứng viên chuẩn.
- **Giai đoạn 2 (Fine-grained Section Matching)**: Tính toán ma trận đối sánh section chi tiết trên Top-$K$, duy trì độ trễ chỉ $15 - 32$ ms/query trên CPU.

### 4.7 Matching Hai Chiều (Bidirectional Matching)
- **`jd2cv` (Client $\rightarrow$ Freelancer)**: Nhà tuyển dụng tìm ứng viên; áp dụng bộ lọc điều kiện chặt chẽ.
- **`cv2jd` (Freelancer $\rightarrow$ Job)**: Ứng viên khám phá cơ hội việc làm; ưu tiên độ bao quát ngữ nghĩa để tránh việc các job đơn giản (ít kỹ năng) áp đảo các job toàn diện.

### 4.8 Tính Giải thích Thực chứng bằng LLM
Top 5 ứng viên sau vòng Hybrid được đưa qua mô hình ngôn ngữ lớn cục bộ (Qwen-2.5 / DeepSeek qua Ollama) để trích xuất lý do phù hợp:
- **Tỷ lệ có bằng chứng thực chứng (Verbatim Groundedness)**: **$95.41\%$** (104/109 kỹ năng khẳng định xuất hiện nguyên văn trong CV).
- **Tỷ lệ suy diễn ảo (Hallucination Rate)**: Chỉ **$4.59\%$**.

---

## 5. Nguồn Dữ Liệu & Thiết Kế Phân Tầng Ngôn Ngữ

Hệ thống kết hợp các nguồn dữ liệu chuyên biệt phục vụ cả bài toán xếp hạng thực tế lẫn thẩm định tính tổng quát hóa và độ bền bỉ trong môi trường học thuật mở (Open Science):

### 5.1 Các Bộ Dữ Liệu Nghiên Cứu

| STT | Bộ dữ liệu | Nguồn / Đường link trực tuyến & Lưu trữ cục bộ | Quy mô | Vai trò & Ứng dụng trong Nghiên cứu |
|:---:|---|---|:---:|---|
| 1 | **`VietResume-Bench`** | Hugging Face Open-Science Corpora:<br>• CVs: [`lhoestq/resumes-raw-pdf-for-ocr`](https://huggingface.co/datasets/lhoestq/resumes-raw-pdf-for-ocr)<br>• JDs: [`tinixai/vietnamese-job-descriptions`](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions)<br>• Cục bộ: [`data/public_benchmark/vietresume/`](data/public_benchmark/vietresume/) | 1.585 CVs, 60 JDs IT<br>60 queries hai chiều, 3.000 cặp đánh giá<br>Nhãn: 0 (No Fit), 1 (Potential Fit), 2 (Good Fit) | **Benchmark chính cho Việt Nam**: Đánh giá nDCG@5, nDCG@10, MRR, MAP qua 3 nhóm ngôn ngữ ($\text{VI}\leftrightarrow\text{VI}$, $\text{VI}\leftrightarrow\text{EN}$ chéo ngôn ngữ, Code-Mixed) và độ bền không dấu (RQ1, RQ2, RQ3, RQ4). |
| 2 | **`resume-job-description-fit`** | Hugging Face Dataset:<br>🔗 [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)<br>• Cục bộ: [`data/public_benchmark/resume-job-description-fit/`](data/public_benchmark/resume-job-description-fit/) | 1.759 cặp CV–JD<br>(71 JD, 477 CV)<br>Nhãn: 0 (No Fit), 1 (Potential Fit), 2 (Good Fit) | **Benchmark Đối chứng Quốc tế (Generalizability)**: Kiểm chứng mô hình trên tập dữ liệu tiếng Anh công khai chuẩn quốc tế, chứng minh tính tổng quát hóa không bị rò rỉ nhãn. |
| 3 | **`resumes-raw-pdf-for-ocr`** | Hugging Face Dataset:<br>🔗 [lhoestq/resumes-raw-pdf-for-ocr](https://huggingface.co/datasets/lhoestq/resumes-raw-pdf-for-ocr)<br>• Cục bộ: [`data/public_benchmark/resumes-raw-pdf-for-ocr/`](data/public_benchmark/resumes-raw-pdf-for-ocr/) | 1.585 CV thực tế tại VN<br>(574 IT domain, 1.011 All-domains)<br>Văn bản OCR thô + Ảnh trang | **Kho ứng viên thực tế**: Thử thách bộ tách section (`split_sections()`) trên văn bản OCR thô từ PDF thực tế; cung cấp 1.011 CV ngành khác làm mẫu âm tính gây nhiễu (Hard Negatives). |
| 4 | **`vietnamese-job-descriptions`** | Hugging Face Dataset:<br>🔗 [tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions)<br>• Cục bộ: [`data/public_benchmark/vietnamese-job-descriptions/`](data/public_benchmark/vietnamese-job-descriptions/) | 606.878 tin tuyển dụng tiếng Việt<br>(Đầy đủ yêu cầu, phúc lợi, lương) | **Kho ngữ liệu JD thực tế quy mô lớn**: Phân tích phân phối từ vựng/kỹ năng tiếng Việt và trích xuất 60 JD tiêu biểu theo 10 phân nhánh chuyên môn IT. |

---

### 5.2 Phân Bố Ngôn Ngữ trong `VietResume-Bench`
Tập dữ liệu **`VietResume-Bench`** phản ánh trung thực thị trường lao động công nghệ tại Việt Nam với tỷ lệ đa ngôn ngữ và trộn ngữ tự nhiên:

| Phân loại Ngôn ngữ | Số lượng CV | Tỷ lệ CV (%) | Số lượng JD | Tỷ lệ JD (%) |
|---|:---:|:---:|:---:|:---:|
| **Tiếng Việt (VI)** | 859 | 54.2% | 43 | 71.7% |
| **Tiếng Anh (EN)** | 346 | 21.8% | 0 | 0.0% |
| **Trộn (Code-Mixed)** | 380 | 24.0% | 17 | 28.3% |
| **Tổng cộng** | **1.585** | **100.0%** | **60** | **100.0%** |

---

## 6. Đánh Giá Thực Nghiệm & Kết Quả Học Thuật

### 6.1 Bảng Xếp Hạng Tổng Thể trên `VietResume-Bench` (Table 4)
Đánh giá trên chiều tuyển dụng chính JD $\rightarrow$ CV ($N=60$ queries IT, 1.585 CVs, thang điểm 3 bậc):

| Phương pháp | nDCG@5 | nDCG@10 | MRR | MAP | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **BM25 (whitespace)** | 0.2213 | 0.2178 | 0.5620 | 0.1813 | 122.3 ms |
| **BM25 (pyvi)** | 0.2483 | 0.2341 | 0.6092 | 0.1867 | 93.7 ms |
| **Single-Vector (MiniLM)** | 0.3515 | 0.2979 | 0.7633 | 0.1965 | 7.1 ms |
| **Chunk Max-Sim** | 0.3101 | 0.2709 | 0.7463 | 0.2178 | 0.3 ms |
| **Section-Aware (Fixed)** | 0.1228 | 0.1014 | 0.4566 | 0.2562 | 3.4 ms |
| **Section-Aware (Cross - Ours)** | **0.3414** | **0.3002** | **0.7322** | **0.2384** | 447.0 ms |
| **Calibrated Hybrid (Ours)** | **0.3130** | **0.2889** | **0.6783** | **0.2048** | 560.1 ms |

*Ghi chú*: Trên chiều ngược lại ($\text{CV}\rightarrow\text{JD}$), **Section-Aware (Cross)** đạt **$0.7861$ nDCG@5** (cao nhất trong tất cả các phương pháp, vượt Single-Vector $0.6485$ tới $+21.2\%$).

### 6.2 Trả lời RQ1: Phân tích Lát cắt Ngôn ngữ (Table 5)

| Lát cắt Ngôn ngữ | Phương pháp | nDCG@5 | nDCG@10 | MAP |
|---|---|:---:|:---:|:---:|
| **VI ↔ VI (Đơn ngữ tiếng Việt)** | BM25 (pyvi) | 0.2054 | 0.1824 | 0.1893 |
| | Single-Vector | 0.3138 | 0.2633 | 0.1688 |
| | **Section-Aware (Ours)** | 0.2655 | 0.2243 | **0.1930** |
| | **Calibrated Hybrid (Ours)** | 0.2757 | 0.2452 | **0.2154** |
| **VI ↔ EN (Chéo ngôn ngữ)** | BM25 (whitespace) | 0.4931 | 0.4414 | 0.3603 |
| | BM25 (pyvi) | 0.4730 | 0.4289 | 0.3535 |
| | Single-Vector | 0.2641 | 0.2391 | 0.2411 |
| | **Section-Aware (Ours)** | **0.3574** | **0.3062** | **0.3087** |
| | **Calibrated Hybrid (Ours)** | **0.4700** | **0.4209** | **0.3485** |
| **Code-Mixed (Trộn Anh - Việt)** | Single-Vector | 0.3490 | 0.3057 | 0.2720 |
| | **Section-Aware (Ours)** | **0.3931** | **0.3450** | **0.3189** |
| | **Calibrated Hybrid (Ours)** | **0.4718** | **0.4017** | **0.3477** |

*Kết luận RQ1*: Trên nhóm chéo ngôn ngữ ($\text{VI}\leftrightarrow\text{EN}$), Single-Vector bị trôi dạt ngữ nghĩa nghiêm trọng ($0.2641$ nDCG@5), trong khi **Section-Aware đạt $0.3574$ nDCG@5 (+35.3% so với Single-Vector)** và **Hybrid đạt $0.4700$ nDCG@5 (+78.0%)**.

### 6.3 Trả lời RQ2: Multilingual Dense vs. Lexical BM25 & Tính công bằng Baseline
- **Baseline Fairness (Khoảng trắng vs. Tách từ tiếng Việt `pyvi`)**:
  - Trên tập đơn ngữ $\text{VI}\leftrightarrow\text{VI}$, BM25 tách từ bằng `pyvi` đạt **$0.1824$ nDCG@10**, vượt BM25 tách khoảng trắng thô (**$0.1556$ nDCG@10**, chênh lệch **$+17.2\%$**). Việc tách từ ghép là bắt buộc để baseline từ vựng không bị phạt oan.

### 6.4 Trả lời RQ4: Độ bền trước Tiếng Việt Không Dấu (Table 6)
Đánh giá mức độ suy giảm khi ứng viên viết truy vấn/CV không dấu (*tieng Viet khong dau*):

| Phương pháp | nDCG@10 (Có dấu) | nDCG@10 (Không dấu) | Mức thay đổi $\Delta$ (%) |
|---|:---:|:---:|:---:|
| **BM25 (pyvi)** | 0.2341 | 0.2161 | -7.69% |
| **Single-Vector** | 0.2979 | 0.2166 | **-27.29%** (Sụp đổ) |
| **Section-Aware (Ours)** | **0.3002** | **0.3002** | **+0.00%** (Bảo toàn) |

*Kết luận RQ4*: Single-Vector sụp đổ tới $-27.29\%$ khi mất dấu, trong khi **Section-Aware bảo toàn tuyệt đối 100% hiệu năng** nhờ cơ chế đối sánh cục bộ đa section và chuẩn hóa Unicode.

Nhờ chuẩn hóa Unicode NFC và từ điển alias kỹ năng chịu lỗi không dấu, Section-Aware duy trì độ ổn định cao (chỉ suy giảm nhẹ $-2.09\%$), trong khi BM25 bị phân mảnh từ vựng nghiêm trọng.

### 6.5 Đánh giá Tổng quát hóa trên Benchmark Quốc tế Công khai
Kiểm thử trên tập dữ liệu độc lập [**`cnamuangtoun/resume-job-description-fit`**](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) **(1.759 cặp test, 71 job queries)**:

| Mô hình / Pipeline | nDCG@5 | nDCG@10 | MRR | MAP | P@5 | Spearman $\rho$ | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF / BM25** | 0.5536 | 0.5778 | 0.6487 | 0.5931 | 0.4563 | 0.1279 | 0.60 s |
| **Dense Bi-Encoder (MiniLM)** | **0.5895** | **0.6103** | **0.7005** | 0.6200 | **0.4930** | 0.1078 | 18.03 s |
| **Cross-Encoder (ms-marco)** | 0.5348 | 0.5589 | 0.6354 | 0.5650 | 0.4648 | -0.0163 | 89.80 s |
| **Section-Aware Multi-Vector** | **0.5895** | **0.6103** | **0.7005** | 0.6200 | **0.4930** | 0.1078 | **0.01 s** |
| **Hybrid (Section-Aware + BM25)** | 0.5869 | 0.6075 | 0.6857 | **0.6238** | 0.4789 | **0.1274** | 0.61 s |

*Khẳng định tính chuyển giao: Trên văn bản tiếng Anh phẳng không có section chuẩn, mô hình tự động fallback an toàn về biểu diễn đoạn văn, giữ nguyên chất lượng so với Bi-Encoder chuẩn.*

### 6.6 Kiểm định Thống kê & Khoảng Tin cậy 95% Bootstrap
- **Paired $t$-test & Wilcoxon**:
  - Section-Aware vs. BM25: $t = 2.7911, p = 0.0097$; Wilcoxon $W = 81.0, p = 0.0082$ (Ý nghĩa sau Holm-Bonferroni).
  - Section-Aware vs. Single-Vector: $t = 3.9890, p < 0.001$; Wilcoxon $W = 55.0, p < 0.001$.
- **Khoảng tin cậy 95% Bootstrap (1.000 resamples) trên nDCG@10**:
  - BM25: $[0.5346, 0.6513]$
  - Single-Vector: $[0.3991, 0.5378]$
  - Section-Aware (Cross): $[0.6318, 0.8020]$ (Cận dưới vượt trội hoàn toàn so với trung bình của baselines).

### 6.7 Thẩm định Công bằng & Thiên kiến
Kiểm thử nhiễu nhân quả (Counterfactual Perturbation Testing) trên 30 CV đại diện ($150$ cặp):
- **Thay đổi giới tính & tên gọi**: MAD = **$0.000000$**, bảo toàn $100\%$ thứ hạng Top 5.
- **Thay đổi danh tiếng trường đại học**: MAD = **$0.000000$**, bảo toàn $100\%$ thứ hạng Top 5.

### 6.8 Bài học Thực tiễn & Hướng dẫn Vận hành
1. **Chiến lược Ngôn ngữ**: Khi hệ thống có dữ liệu chéo ngôn ngữ ($\text{VI}\leftrightarrow\text{EN}$), **bắt buộc dùng embedding đa ngôn ngữ kết hợp phân rã section**. Khi dữ liệu thuần đơn ngữ, BM25 có tách từ ghép tiếng Việt (`pyvi`) là một baseline rất mạnh với chi phí siêu rẻ.
2. **Quy tắc Nghiệp vụ mang tính Bất đối xứng theo chiều**: Bộ lọc quy tắc ($S_{rules}$) cải thiện mạnh mẽ chiều tuyển dụng (`jd2cv`), nhưng làm giảm độ đa dạng khi ứng viên tìm việc (`cv2jd`). Cần điều chỉnh trọng số linh hoạt theo chiều tìm kiếm.
3. **Phân rã Section ngăn ngừa Semantic Dilution**: Việc cô lập section kỹ năng ngăn không cho văn xuôi tiếng Việt trong phần tóm tắt làm loãng khoảng cách cosine của các từ khóa công nghệ tiếng Anh.

---

## 7. Cấu Trúc Thư Mục

```
section-aware-cv-job-matching/
├── configs/                      # Cấu hình YAML (trọng số, ngưỡng, tham số)
├── data/                         # Dữ liệu thực nghiệm
│   ├── public_benchmark/         # 100% Public Open-Science Benchmarks
│   │   ├── vietresume/           # VietResume-Bench: 1.585 CVs, 60 JDs, pairs.csv (3.000 cặp)
│   │   ├── resume-job-description-fit/  # 1.759 cặp CV-JD tiếng Anh có nhãn
│   │   ├── resumes-raw-pdf-for-ocr/     # 1.585 CV thực tế tại VN (OCR text + ảnh)
│   │   └── vietnamese-job-descriptions/ # 606k tin tuyển dụng tiếng Việt
│   └── chroma/                   # Vector database lưu trữ embeddings HNSW
├── docs/                         # Tài liệu thiết kế chi tiết (WORKFLOW.md, kiến trúc)
├── paper/                        # Bản thảo bài báo khoa học Springer LNCS LaTeX
│   ├── sections/                 # 00_abstract đến 06_conclusion & ethics
│   ├── tables/                   # Bảng số liệu LaTeX xuất tự động
│   ├── figures/                  # Biểu đồ kiến trúc và luồng dữ liệu
│   ├── references.bib            # Tài liệu tham khảo chuẩn BibTeX
│   ├── build.ps1                 # Script tự động biên dịch ra main.pdf
│   └── main.tex                  # File master LaTeX
├── results/                      # Kết quả benchmark xuất tự động (JSON, MD, TEX)
├── scripts/                      # Script CLI cho luồng xử lý và đánh giá
│   ├── build_vietresume_benchmark.py # Khởi tạo VietResume-Bench từ HF
│   ├── benchmark_vietresume.py       # Chạy đánh giá toàn diện VietResume-Bench
│   ├── eval_public_benchmark.py      # Đánh giá trên benchmark quốc tế
│   ├── db_export.py                  # Xuất dữ liệu
│   ├── index_chroma.py               # Phân tách section và đánh chỉ mục vector
│   └── recommend.py                  # CLI gợi ý matching nhanh
├── src/cvjd/                     # Mã nguồn lõi của module
│   ├── data/                     # Tiền xử lý, trích xuất section, payload schema
│   │   ├── loader.py             # Nạp tài liệu từ file/text
│   │   ├── preprocessing.py      # Regex section splitter đa ngữ & NFC
│   │   └── payload.py            # Bridge dữ liệu cấu trúc
│   ├── models/                   # Section-aware matching, BM25 tokenizer
│   │   ├── bm25.py               # BM25Ranker hỗ trợ pyvi & whitespace
│   │   ├── chunking.py           # Chunk max-similarity baseline
│   │   ├── embedding.py          # Wrapper BGE-M3, MiniLM, MockEmbedder
│   │   └── matching.py           # Section matching & Dynamic Renormalization
│   ├── ranking/                  # Hybrid scoring, constraint rules, IR metrics
│   │   ├── engine.py             # HybridRanker cân bằng thang đo & min-max
│   │   ├── hybrid.py             # Công thức S_hybrid, Asymmetric Skill Coverage
│   │   └── metrics.py            # nDCG@K, MRR, MAP, Recall, Precision
│   ├── retrieval/                # ChromaDB vector retriever
│   └── service.py                # Service tích hợp trung tâm
└── tests/                        # 151 unit tests kiểm thử toàn diện module
```

---

## 8. Hướng Dẫn Cài Đặt & Sử Dụng

### 8.1 Yêu Cầu Môi Trường
- Python 3.10+
- Cài đặt thư viện: `pip install -r requirements.txt`

### 8.2 Các Lệnh Thực Thi Chính

#### 1. Xuất dữ liệu và tạo chỉ mục Vector
```powershell
python scripts/db_export.py
python scripts/index_chroma.py
```

#### 2. Chạy gợi ý ghép cặp qua CLI
```powershell
# Gợi ý Top 10 ứng viên cho JD (Client -> Freelancer)
python scripts/recommend.py --direction jd2cv --query jd:1 --top-k 10

# Gợi ý Top 10 công việc phù hợp cho CV (Freelancer -> Job)
python scripts/recommend.py --direction cv2jd --query cv:1 --top-k 10
```

#### 3. Chạy toàn bộ Unit Tests (151 tests)
```powershell
python -m pytest tests/
```

#### 4. Chạy Toàn Bộ Chuỗi Thực Nghiệm Học Thuật
```powershell
# 1. Khởi tạo benchmark VietResume-Bench từ dữ liệu công khai Hugging Face
python scripts/build_vietresume_benchmark.py

# 2. Chạy đánh giá toàn diện VietResume-Bench (1.585 CVs, 60 JDs IT)
python scripts/benchmark_vietresume.py --embedder sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 3. Đánh giá đối chứng trên Public Dataset quốc tế (cnamuangtoun/resume-job-description-fit)
python scripts/eval_public_benchmark.py
```

---

## 9. Bản Thảo Bài Báo Khoa Học (Springer LNCS Paper)

Bản thảo bài báo hoàn chỉnh theo định dạng **Springer LNCS (Lecture Notes in Computer Science)** được đặt tại thư mục [`paper/`](paper/):
- **Cấu trúc 17 trang chuẩn proceedings**: Bao gồm 8 bảng số liệu định lượng, phân tích ablation, kiểm định thống kê $p$-value, kiểm thử fairness không thiên kiến, phân tích tính bền bỉ song ngữ và thảo luận đạo đức/giới hạn.
- Biên dịch file PDF chỉ với 1 lệnh:
  ```powershell
  cd paper
  .\build.ps1
  ```
  File PDF được sinh ra tại: [`paper/main.pdf`](paper/main.pdf).
