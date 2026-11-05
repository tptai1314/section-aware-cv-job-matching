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
| 1 | **Frevia-Bench** | Nền tảng nội bộ Frevia<br>• Cục bộ: [`data/raw/`](data/raw/), [`data/benchmark_qrels.json`](data/benchmark_qrels.json) | 169 CV, 210 JD<br>47 Query có nhãn graded relevance (0–3) | **Benchmark chính**: Đo lường nDCG@10, MRR, MAP qua 5 nhóm ngôn ngữ ($\text{VI}\leftrightarrow\text{VI}$, $\text{EN}\leftrightarrow\text{EN}$, $\text{VI}\leftrightarrow\text{EN}$, Code-mixed) trả lời RQ1, RQ2, RQ3. |
| 2 | **`resume-job-description-fit`** | Hugging Face Dataset:<br>🔗 [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)<br>• Cục bộ: [`data/public_benchmark/resume-job-description-fit/`](data/public_benchmark/resume-job-description-fit/) | 1.759 cặp CV–JD<br>(71 JD, 477 CV)<br>Nhãn: 0 (No), 1 (Potential), 2 (Good) | **Benchmark Đối chứng Quốc tế (Generalizability)**: Kiểm chứng mô hình trên tập dữ liệu tiếng Anh công khai chuẩn quốc tế, chứng minh không bị rò rỉ nhãn (Bảng 4 trong paper). |
| 3 | **`resumes-raw-pdf-for-ocr`** | Hugging Face Dataset:<br>🔗 [lhoestq/resumes-raw-pdf-for-ocr](https://huggingface.co/datasets/lhoestq/resumes-raw-pdf-for-ocr)<br>• Cục bộ: [`data/public_benchmark/resumes-raw-pdf-for-ocr/`](data/public_benchmark/resumes-raw-pdf-for-ocr/) | 1.585 CV thực tế tại VN<br>(574 IT domain, 1.011 All-domains)<br>Văn bản OCR thô + Ảnh trang | **Thực nghiệm Độ bền Dữ liệu Bẩn (Robustness - RQ4)**: Kiểm thử tính bền bỉ của bộ tách section (`split_sections()`) trên văn bản OCR thô từ PDF thực tế; mở rộng Candidate Pool cho truy hồi thô (Coarse Retrieval) và lọc âm tính (Negative Filtering). |
| 4 | **`vietnamese-job-descriptions`** | Hugging Face Dataset:<br>🔗 [tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions)<br>• Cục bộ: [`data/public_benchmark/vietnamese-job-descriptions/`](data/public_benchmark/vietnamese-job-descriptions/) | 606.878 tin tuyển dụng tiếng Việt<br>(Đầy đủ yêu cầu, phúc lợi, lương) | **Kho ngữ liệu JD thực tế quy mô lớn**: Phân tích phân phối từ vựng/kỹ năng tiếng Việt và mở rộng không gian tìm kiếm quy mô lớn. |

---

### 5.2 Quy ước Phân loại Ngôn ngữ Tài liệu (Ngưỡng $\ge 20\%$)
Tập dữ liệu kiểm thử thực tế **Frevia-Bench** được phân tầng nghiêm ngặt: Một tài liệu được xếp vào nhóm **Trộn (Code-Mixed)** nếu các token tiếng Việt và tiếng Anh mỗi bên chiếm $\ge 20\%$ tổng số token ngôn ngữ mang nghĩa ($\rho_{vi} \ge 0.20$ và $\rho_{en} \ge 0.20$); ngược lại được tính là **Tiếng Việt** ($\rho_{vi} \ge 0.80$) hoặc **Tiếng Anh** ($\rho_{en} \ge 0.80$).

| Phân loại Ngôn ngữ | Số lượng CV | Tỷ lệ CV (%) | Số lượng JD | Tỷ lệ JD (%) |
|---|:---:|:---:|:---:|:---:|
| **Tiếng Việt (VI)** | 50 | 29.6% | 70 | 33.3% |
| **Tiếng Anh (EN)** | 94 | 55.6% | 110 | 52.4% |
| **Trộn (Code-Mixed)** | 25 | 14.8% | 30 | 14.3% |
| **Tổng cộng** | **169** | **100.0%** | **210** | **100.0%** |

### 5.3 5 Nhóm Cặp Đánh Giá
1. **$\text{VI}\leftrightarrow\text{VI}$**: CV tiếng Việt $\leftrightarrow$ JD tiếng Việt (Đơn ngữ tiếng Việt).
2. **$\text{EN}\leftrightarrow\text{EN}$**: CV tiếng Anh $\leftrightarrow$ JD tiếng Anh (Đơn ngữ tiếng Anh).
3. **$\text{VI}\leftrightarrow\text{EN}$**: CV tiếng Việt $\leftrightarrow$ JD tiếng Anh (Trọng tâm chéo ngôn ngữ: Freelancer Việt tìm job quốc tế).
4. **$\text{EN}\leftrightarrow\text{VI}$**: CV tiếng Anh $\leftrightarrow$ JD tiếng Việt (Chéo ngôn ngữ chiều ngược).
5. **Code-Mixed**: Tài liệu trộn lẫn tự nhiên giữa tiếng Việt và tiếng Anh kỹ thuật.

*Mỗi nhóm chính có $\approx 10$ truy vấn chuẩn để đảm bảo tính hợp lệ thống kê.*

### 5.4 Quy trình Dịch có Gắn cờ & Kiểm tra Nhất quán Nhãn (Label Consistency Check)
- Dữ liệu dịch máy phục vụ làm giàu tập chéo ngôn ngữ được gắn cờ `translated` và kiểm duyệt bởi kỹ sư IT song ngữ.
- **Phép kiểm tra tính nhất quán nhãn**: Đối chiếu nhãn ground-truth giữa bản gốc và bản dịch đạt độ tương đồng **$94.6\%$** ($\kappa = 0.88$), xác nhận sự thay đổi điểm số chéo ngôn ngữ là do năng lực mô hình chứ không phải do sai lệch ngữ nghĩa khi dịch.

---

## 6. Đánh Giá Thực Nghiệm & Kết Quả Học Thuật

### 6.1 Trả lời RQ1: Hiệu năng trên các nhóm cặp ngôn ngữ (Table 7)
Đánh giá trên chiều tuyển dụng JD $\rightarrow$ CV ($N=27$ truy vấn):

| Nhóm Cặp Ngôn ngữ | Số Query | Phương pháp | nDCG@5 | nDCG@10 | MAP |
|---|:---:|---|:---:|:---:|:---:|
| **Đơn ngữ EN ↔ EN** | 10 | BM25 | 0.3840 | 0.5119 | 0.1038 |
| | | Single-Vector (BGE-M3) | 0.3712 | 0.4147 | 0.1040 |
| | | **Section-Aware (Ours)** | **0.7245** | **0.7553** | **0.6204** |
| **Đơn ngữ VI ↔ VI** | 9 | BM25 (pyvi) | 0.5374 | 0.5326 | 0.4372 |
| | | Single-Vector (BGE-M3) | 0.4485 | 0.4500 | 0.3022 |
| | | **Section-Aware (Ours)** | **0.5731** | **0.5989** | **0.5247** |
| **Chéo Ngôn ngữ VI ↔ EN** | 8 | BM25 | 0.7374 | 0.6547 | 0.5044 |
| | | Single-Vector (BGE-M3) | 0.3525 | 0.3251 | 0.1582 |
| | | **Section-Aware (Ours)** | **0.7020** | **0.7180** | **0.5973** |

*Kết luận RQ1*: Ở cả 3 phân vùng ngôn ngữ, **Section-Aware** đều vượt trội hoàn toàn so với Single-Vector, đặc biệt đạt hiệu năng cao và ổn định trên nhóm chéo ngôn ngữ ($0.7180$ nDCG@10).

### 6.2 Trả lời RQ2: Multilingual Dense vs. Lexical BM25 & Tính công bằng Baseline
- **Kiểm chứng giả thuyết suy giảm BM25**: Khi tài liệu có các thực thể kỹ thuật tiếng Anh dùng chung (*React, Docker, PostgreSQL*), BM25 giữ được độ chính xác ($0.6547$). Tuy nhiên, khi yêu cầu công việc được diễn đạt bằng văn xuôi không trùng lặp từ khóa (*lập trình giao diện* vs *frontend development*), BM25 thất bại hoàn toàn. Ngược lại, dense Section-Aware duy trì độ tương đồng ngữ nghĩa vượt trội ($+9.7\%$ nDCG@10 so với BM25 trên tập chéo).
- **Tính công bằng Baseline (Tokenize Khoảng trắng vs. Tách từ tiếng Việt `pyvi`)**:
  - BM25 có tách từ ghép (`pyvi`: *lap_trinh_vien, co_so_du_lieu*): **$0.4907$ nDCG@10**.
  - BM25 tách khoảng trắng thô (naive whitespace): **$0.4604$ nDCG@10** (bị phạt oan $-6.2\%$).
  $\implies$ Cần dùng tách từ tiếng Việt để đảm bảo so sánh công bằng.

### 6.3 Trả lời RQ3: Lợi thế biểu diễn Section-Aware & Ablation Study
So sánh tổng thể trên toàn bộ benchmark:
- **Single-Vector Baseline**: nDCG@10 = $0.4664$, MRR = $0.4441$.
- **Chunk Max-Sim**: nDCG@10 = $0.4547$, độ trễ cao ($59.8$ ms).
- **Section-Aware (Cross)**: nDCG@10 = **$0.7224$** ($+25.6\%$ so với Single-Vector, $p < 0.001$).
- **Hybrid (Section + BM25 + Rules)**: nDCG@10 = **$0.9542$**, MRR = **$1.0000$**.

#### Phân tích Ablation Study từng thành phần
| Cấu hình Thử nghiệm | nDCG@5 | nDCG@10 | MRR | MAP | Vai trò |
|---|:---:|:---:|:---:|:---:|---|
| **Full Hybrid** | **0.9409** | **0.9406** | **1.0000** | **0.9263** | Hiệu năng toàn diện cao nhất |
| **w/o $S_{skills}$** | 0.6796 | 0.6943 | 0.7312 | 0.5993 | Giảm mạnh nhất ($-26.13\%$), kỹ năng là trụ cột |
| **w/o $S_{exp}$** | 0.9409 | 0.9394 | 1.0000 | 0.9250 | Suy giảm nhẹ thứ hạng sâu |
| **w/o $S_{budget}$** | 0.9409 | 0.9396 | 1.0000 | 0.9260 | Tác dụng chủ yếu ở vòng lọc cứng |
| **Semantic Only** | 0.6796 | 0.6943 | 0.7312 | 0.5993 | Không phụ thuộc quy tắc nghiệp vụ |
| **Rules Only** | 0.8494 | 0.8728 | 0.7840 | 0.8392 | Dễ gặp lỗi biến thể từ vựng |

### 6.4 Trả lời RQ4: Độ bền trước dữ liệu bẩn
Thực nghiệm loại bỏ toàn bộ dấu tiếng Việt trên hồ sơ ứng viên (*tieng Viet khong dau*):

| Phương pháp | nDCG@10 (Có dấu) | nDCG@10 (Không dấu) | Mức thay đổi $\Delta$ (%) |
|---|:---:|:---:|:---:|
| BM25 (pyvi) | 0.4907 | 0.4412 | -10.09% |
| Single-Vector (BGE-M3) | 0.4502 | 0.4345 | -3.49% |
| **Section-Aware (Ours)** | **0.7071** | **0.6923** | **-2.09%** |

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
│   ├── raw/                      # Dữ liệu xuất từ DB PostgreSQL (cvs.jsonl, jds.jsonl)
│   ├── benchmark_qrels.json      # Ground truth labels cho 47 queries Frevia-Bench
│   ├── public_benchmark/         # Các tập benchmark công khai nguồn mở
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
│   ├── db_export.py              # Xuất dữ liệu từ PostgreSQL
│   ├── index_chroma.py           # Phân tách section và đánh chỉ mục vector
│   ├── build_benchmark_qrels.py  # Xây dựng tập nhãn ground-truth
│   ├── benchmark_paper.py        # Chạy kiểm thử toàn diện các cấu hình
│   ├── bilingual_experiment.py   # Thực nghiệm song ngữ, chéo ngôn ngữ & bỏ dấu
│   ├── eval_public_benchmark.py  # Đánh giá trên benchmark công khai Hugging Face
│   ├── run_ablation_and_fairness.py # Ablation study, kiểm định thống kê & fairness
│   └── recommend.py              # CLI gợi ý matching nhanh
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
# 1. Thực nghiệm Song ngữ, Chéo ngôn ngữ & Bỏ dấu tiếng Việt
python scripts/bilingual_experiment.py

# 2. Benchmark chính trên Frevia-Bench
python scripts/benchmark_paper.py

# 3. Đánh giá đối chứng trên Public Dataset (Hugging Face)
python scripts/eval_public_benchmark.py

# 4. Chạy Ablation Study, Kiểm định thống kê & Fairness
python scripts/run_ablation_and_fairness.py
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
