# Hệ Thống Ghép Cặp Hồ Sơ Ứng Viên & Mô Tả Công Việc (Section-Aware CV–Job Matching)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Vector DB](https://img.shields.io/badge/ChromaDB-Multi--Vector-orange.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-BAAI%2FBGE--M3-green.svg)](https://huggingface.co/BAAI/bge-m3)
[![Tests](https://img.shields.io/badge/Tests-146%20Passed-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/Paper-Springer%20LNCS-purple.svg)](paper/)

Hệ thống AI gợi ý và ghép cặp thông minh giữa **Hồ sơ ứng viên (CV/Resume)** và **Mô tả công việc (Job Description - JD)** được phát triển cho nền tảng tuyển dụng freelancer **Frevia**.

Khác với các hướng tiếp cận truyền thống xem CV/JD như một khối văn bản phẳng (flat text), hệ thống áp dụng kỹ thuật **biểu diễn đa vector theo section (Multi-Vector Section-Aware Representation)** kết hợp với **mô hình chấm điểm lai (Hybrid Fusion)** và **tái xếp hạng giải thích bằng LLM (LLM-based Explainable Reranking)**. Toàn bộ kiến trúc chạy hoàn toàn trên **CPU**, sử dụng mô hình embedding đóng băng (frozen BGE-M3) mà không cần huấn luyện lại mô hình từ đầu.

---

## Mục lục

- [1. Điểm Nổi Bật](#1-điểm-nổi-bật)
- [2. Kiến Trúc Hệ Thống](#2-kiến-trúc-hệ-thống)
- [3. Cơ Chế Hoạt Động Cốt Lõi](#3-cơ-chế-hoạt-động-cốt-lõi)
  - [3.1 Phân đoạn Section & Đa Vector](#31-phân-đoạn-section--đa-vector)
  - [3.2 Chiến lược Đối sánh Section (Section Alignment)](#32-chiến-lược-đối-sánh-section-section-alignment)
  - [3.3 Truy hồi Hai Giai đoạn (Two-Stage Retrieval)](#33-truy-hồi-hai-giai-đoạn-two-stage-retrieval)
  - [3.4 Công thức Chấm điểm Lai (Hybrid Scoring)](#34-công-thức-chấm-điểm-lai-hybrid-scoring)
  - [3.5 Matching Hai Chiều (Bidirectional Matching)](#35-matching-hai-chiều-bidirectional-matching)
  - [3.6 Tính Giải Thích & Tái Xếp Hạng bằng LLM](#36-tính-giải-thích--tái-xếp-hạng-bằng-llm)
- [4. Bộ Dữ Liệu & Thực Nghiệm (Benchmark)](#4-bộ-dữ-liệu--thực-nghiệm-benchmark)
  - [4.1 Bộ dữ liệu Frevia-Bench](#41-bộ-dữ-liệu-frevia-bench)
  - [4.2 Bộ dữ liệu Đối chứng Quốc tế (Hugging Face)](#42-bộ-dữ-liệu-đối-chứng-quốc-tế-hugging-face)
  - [4.3 Bảng Kết Quả Thực Nghiệm](#43-bảng-kết-quả-thực-nghiệm)
  - [4.4 Phân Tích Độ Nhạy Tham Số Alpha](#44-phân-tích-độ-nhạy-tham-số-alpha)
- [5. Cấu Trúc Thư Mục](#5-cấu-trúc-thư-mục)
- [6. Hướng Dẫn Cài Đặt & Sử Dụng](#6-hướng-dẫn-cài-đặt--sử-dụng)
- [7. Bản Thảo Bài Báo Khoa Học (Paper Draft)](#7-bản-thảo-bài-báo-khoa-học-paper-draft)

---

## 1. Điểm Nổi Bật

1. **Khắc phục hiện tượng suy giảm ngữ nghĩa (Semantic Dilution)**: Việc nhúng toàn bộ CV dài 2–3 trang vào một vector duy nhất làm loãng thông tin các kỹ năng then chốt. Hệ thống tách tài liệu thành từng phần độc lập (`skills`, `experience`, `projects`, `education`, `summary`) và nhúng đa vector.
2. **Chiến lược tổng hợp có trọng số (`weighted aggregation`)**: Cho phép gán trọng số linh hoạt cho từng phần (ví dụ: Kỹ năng chiếm 35%, Kinh nghiệm 35%, Dự án 15%, Học vấn 10%, Tóm tắt 5%).
3. **Mô hình Hybrid chặt chẽ**: Kết hợp độ tương đồng ngữ nghĩa (Semantic similarity) với các điều kiện nghiệp vụ thực tế (Bộ lọc ràng buộc cứng, Jaccard overlap kỹ năng, độ lệch kinh nghiệm, độ tương thích ngân sách).
4. **Tính minh bạch & Giải thích được (Explainability)**: Tích hợp mô hình ngôn ngữ lớn (LLM Qwen/DeepSeek qua Ollama) để chỉ rõ ứng viên mạnh ở điểm nào, thiếu kỹ năng gì và tại sao được đề xuất.
5. **Hiệu năng & Tối ưu chi phí**: Không cần GPU rời đắt đỏ, chạy ổn định trên CPU với BGE-M3 (1024-dim dense embedding) và ChromaDB vector store.

---

## 2. Kiến Trúc Hệ Thống

Hệ thống được thiết kế theo mô hình 5 tầng phân tách rõ ràng:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION                                                           │
│    PostgreSQL (Source of truth) ──db_export.py──▶ data/raw/*.jsonl          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. SECTIONING & MULTI-VECTOR INDEXING                                       │
│    CV/JD Text ──Regex/Heuristic Splitter──▶ Sections (Skills, Exp, Edu,...) │
│    Sections ──BAAI/BGE-M3 Embedder (1024d)──▶ ChromaDB (Collections)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. TWO-STAGE RETRIEVAL & SECTION MATCHING                                   │
│    Query ──Full-text coarse vector──▶ Candidate Pool (Top-K Coarse)         │
│    Pool ──Section-to-Section Matrix Scoring──▶ Semantic Score (S_sem)       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. HYBRID RANKING & CONSTRAINT FILTERING                                    │
│    Hard Filters (Privacy, Availability, Blocked status)                     │
│    S_hybrid = α · S_sem + (1 - α) · [w_sk·S_skills + w_exp·S_exp + ...]     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. SERVING, RE-RANKING & EXPLAINABILITY (apps/ai)                           │
│    FastAPI Gateway + Local LLM (Qwen/Ollama)                                │
│    JSON Response: Match Score, Strengths, Skill Gaps, Natural Explanation   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Cơ Chế Hoạt Động Cốt Lõi

### 3.1 Phân đoạn Section & Đa Vector
Mỗi tài liệu CV hoặc JD được bóc tách thành các section tiêu chuẩn:
- `summary`: Mục tiêu nghề nghiệp, giới thiệu tổng quan.
- `skills`: Danh sách kỹ năng chuyên môn, công nghệ, công cụ.
- `experience`: Lịch sử làm việc, vị trí đảm nhiệm, thành tựu.
- `projects`: Dự án thực tế, quy mô, công nghệ sử dụng.
- `education`: Trình độ học vấn, bằng cấp, chuyên ngành.
- `certifications`: Chứng chỉ chuyên môn quốc tế.
- `full_text`: Toàn văn tài liệu phục vụ truy hồi vòng thô (Coarse retrieval).

### 3.2 Chiến lược Đối sánh Section (Section Alignment)
Hệ thống hỗ trợ 3 chiến lược tính ma trận tương đồng giữa các section của CV ($C$) và JD ($J$):

1. **`fixed` (Cùng tên)**: Chỉ so sánh các section tương ứng cùng loại ($Skills_{CV} \leftrightarrow Skills_{JD}$, $Exp_{CV} \leftrightarrow Exp_{JD}$).
2. **`cross` (So khớp chéo cực đại)**: Với mỗi section của JD, tìm section của CV có độ tương đồng Cosine cao nhất:
   $$S_{cross}(J_i, C) = \max_{k} \cos(\mathbf{v}_{J_i}, \mathbf{u}_{C_k})$$
3. **`weighted` (Tổng hợp có trọng số - Khuyến nghị)**: Tổng hợp điểm số qua các section theo trọng số chuyên gia nhân sự:
   $$S_{sem}(J, C) = \sum_{k} w_k \cdot \cos(\mathbf{v}_{J_k}, \mathbf{u}_{C_k})$$
   *(Mặc định: $w_{skills} = 0.35, w_{exp} = 0.35, w_{proj} = 0.15, w_{edu} = 0.10, w_{summary} = 0.05$)*.

### 3.3 Truy hồi Hai Giai đoạn (Two-Stage Retrieval)
- **Giai đoạn 1 (Coarse Retrieval)**: Sử dụng vector toàn văn (`full_text`) để truy vấn nhanh qua ChromaDB (HNSW index), lọc từ hàng nghìn hồ sơ xuống ứng viên tiềm năng ($K = 50 \div 100$).
- **Giai đoạn 2 (Fine-grained Section Matching)**: Áp dụng phép tính ma trận đa vector trên tập ứng viên rút gọn, giúp tối ưu thời gian phản hồi (chỉ mất $\sim 10 - 15$ ms/query).

### 3.4 Công thức Chấm điểm Lai (Hybrid Scoring)
Điểm số cuối cùng kết hợp giữa tương đồng ngữ nghĩa và các tiêu chí nghiệp vụ có cấu trúc:

$$S_{hybrid} = \alpha \cdot S_{sem} + (1 - \alpha) \cdot S_{rules}$$

Trong đó $S_{rules}$ được tính từ:
$$S_{rules} = w_{sk} \cdot S_{skills} + w_{exp} \cdot S_{exp} + w_{budget} \cdot S_{budget} + w_{pref} \cdot S_{pref}$$
- **$S_{skills}$**: Chỉ số tương đồng tập hợp (Jaccard Overlap) giữa kỹ năng bắt buộc (Must-have) và điểm thưởng cho kỹ năng ưu tiên (Nice-to-have).
- **$S_{exp}$**: Hàm suy giảm phi tuyến dựa trên độ lệch giữa số năm kinh nghiệm của ứng viên và yêu cầu tối thiểu của công việc.
- **$S_{budget}$**: Mức độ phù hợp giữa mức lương/ngân sách kỳ vọng và khả năng chi trả của dự án.
- **Ràng buộc cứng (Hard Constraints)**: Tự động loại bỏ hồ sơ nếu tài khoản bị khóa, vi phạm trạng thái khả dụng, hoặc vi phạm chính sách bảo mật riêng tư.

### 3.5 Matching Hai Chiều (Bidirectional Matching)
Hệ thống thiết kế trừu tượng hóa cho phép chạy linh hoạt theo 2 chiều:
- **`jd2cv` (Client $\rightarrow$ Freelancer)**: Nhà tuyển dụng đăng bài toán, hệ thống tìm và xếp hạng các ứng viên tiềm năng nhất.
- **`cv2jd` (Freelancer $\rightarrow$ Job)**: Ứng viên tải CV lên, hệ thống tự động gợi ý các dự án phù hợp nhất với năng lực.

### 3.6 Tính Giải Thích & Tái Xếp Hạng bằng LLM
Sau khi có bảng xếp hạng top ứng viên từ tầng Hybrid, hệ thống có thể chuyển top-K sang mô hình ngôn ngữ lớn (Qwen-2.5 / DeepSeek chạy qua Ollama) để:
- Tinh chỉnh thứ hạng cuối cùng (Listwise Reranking).
- Trích xuất lý do gợi ý rõ ràng:
  - **Điểm mạnh (Strengths)**: Bằng chứng cụ thể từ dự án/kinh nghiệm thực tế.
  - **Điểm thiếu hụt (Skill Gaps)**: Kỹ năng hoặc công nghệ ứng viên còn thiếu so với JD.
  - **Khuyến nghị tuyển dụng**: Tóm tắt 2-3 câu đánh giá mức độ vừa vặn tổng thể.

---

## 4. Bộ Dữ Liệu & Thực Nghiệm (Benchmark)

### 4.1 Bộ dữ liệu Frevia-Bench
- **Nguồn gốc**: Trích xuất từ cơ sở dữ liệu thực tế của nền tảng Frevia tại thị trường Việt Nam (song ngữ Anh - Việt).
- **Quy mô**: **169 CVs** và **210 JDs**.
- **Tập đánh giá chuẩn (Qrels)**: **47 truy vấn** (25 truy vấn `jd2cv` và 22 truy vấn `cv2jd`), tổng cộng **2.784 cặp đánh giá**.
- **Thang nhãn 4 mức (Relevance Grade 0–3)**:
  - `Grade 3 (Highly Relevant)`: Khớp $\ge 80\%$ kỹ năng bắt buộc, chuẩn kinh nghiệm và cấp bậc.
  - `Grade 2 (Relevant)`: Khớp $60 - 80\%$ kỹ năng, kinh nghiệm lệch nhẹ.
  - `Grade 1 (Partially Relevant)`: Khớp ngành nhưng thiếu framework hoặc kinh nghiệm.
  - `Grade 0 (Irrelevant)`: Sai ngành hoặc không tương thích.
- **Độ tin cậy của nhãn**: Bộ quy tắc thẩm định đã được đối chứng qua mẫu kiểm định nhân sự (Human Evaluation) với độ đồng thuận **Cohen's Kappa $\kappa = 0.82$** (mức độ đồng thuận rất cao).

### 4.2 Bộ dữ liệu Đối chứng Quốc tế (Hugging Face)
Được lưu tại [`data/public_benchmark/resume-job-description-fit/`](data/public_benchmark/resume-job-description-fit/):
- **Quy mô**: 6.241 cặp huấn luyện (`train.csv`) và 1.759 cặp kiểm thử (`test.csv`).
- **Phân bố nhãn**: `No Fit` (Grade 0), `Potential Fit` (Grade 1), `Good Fit` (Grade 2).
- Giúp chứng minh tính tổng quát hóa của thuật toán trên dữ liệu quốc tế.

### 4.3 Bảng Kết Quả Thực Nghiệm

Đánh giá thực nghiệm so sánh 8 cấu hình trên bộ dữ liệu `Frevia-Bench` với các chỉ số Information Retrieval chuẩn:

#### Chiều Tuyển Dụng: Client $\rightarrow$ Freelancer (`jd2cv`)
| Phương pháp / Cấu hình | nDCG@5 | nDCG@10 | MRR | MAP | Recall@10 | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1: BM25 (Từ khóa)** | 0.5557 | 0.5920 | 0.7322 | 0.3187 | 0.3473 | 11.6 ms |
| **Baseline 2: Single-vector (Văn bản phẳng)** | 0.4265 | 0.4664 | 0.4441 | 0.2097 | 0.2110 | 0.8 ms |
| **Baseline 3: Chunk max-sim** | 0.4006 | 0.4547 | 0.3509 | 0.1802 | 0.2473 | 59.8 ms |
| **Section-aware (Fixed)** | 0.6056 | 0.6679 | 0.6111 | 0.6181 | 0.6691 | 0.8 ms |
| **Section-aware (Cross)** | 0.6876 | 0.7224 | 0.7598 | 0.5144 | 0.5779 | 10.6 ms |
| **Section-aware (Weighted)** | 0.6850 | 0.6953 | 0.7312 | 0.6017 | 0.5959 | 11.0 ms |
| **Hybrid (Section + BM25 + Rules)** | **0.9430** | **0.9542** | **1.0000** | **0.9212** | **0.8160** | 31.9 ms |
| **Full Pipeline (+ LLM Reranker)** | **1.0000** | **0.9690** | **1.0000** | **0.9621** | **0.8160** | 30.6 ms |

#### Chiều Tìm Việc: Freelancer $\rightarrow$ Job (`cv2jd`)
| Phương pháp / Cấu hình | nDCG@5 | nDCG@10 | MRR | MAP | Recall@10 | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1: BM25** | 0.4177 | 0.4533 | 0.3010 | 0.2043 | 0.0697 | 7.7 ms |
| **Baseline 2: Single-vector** | 0.3750 | 0.4373 | 0.3279 | 0.1994 | 0.0808 | 1.0 ms |
| **Section-aware (Weighted)** | **0.6522** | **0.6970** | **0.4530** | **0.3902** | **0.1244** | 14.1 ms |
| **Full Pipeline (+ LLM Reranker)** | **0.5606** | **0.5249** | **0.5175** | **0.3731** | **0.1353** | 34.8 ms |

### 4.4 Phân Tích Độ Nhạy Tham Số Alpha

Khảo sát hệ số dung hợp $\alpha \in [0.0, 1.0]$ giữa ngữ nghĩa và quy tắc nghiệp vụ:
- Khi $\alpha = 0.0$ (chỉ dùng Rules): nDCG@5 = 0.8705, MAP = 0.7877.
- Khi $\alpha = 1.0$ (chỉ dùng Semantic Sections): nDCG@5 = 0.9405, MAP = 0.9162.
- **Giá trị tối ưu $\alpha = 0.6$**: nDCG@5 đạt đỉnh **0.9578**, MAP = **0.9260**, chứng minh sự kết hợp bổ trợ hiệu quả giữa biểu diễn đa vector ngữ nghĩa và bộ lọc kỹ năng/kinh nghiệm thực tế.

---

## 5. Cấu Trúc Thư Mục

```
section-aware-cv-job-matching/
├── configs/                      # File cấu hình YAML (base.yaml, weights, thresholds)
├── data/                         # Dữ liệu thực nghiệm
│   ├── raw/                      # Dữ liệu xuất từ DB PostgreSQL (cvs.jsonl, jds.jsonl)
│   ├── benchmark_qrels.json      # Ground truth labels cho 47 queries
│   ├── public_benchmark/         # Dataset công khai từ Hugging Face
│   └── chroma/                   # Vector database lưu trữ embeddings
├── docs/                         # Tài liệu thiết kế chi tiết (WORKFLOW.md, kiến trúc)
├── paper/                        # Bản thảo bài báo khoa học (Springer LNCS LaTeX)
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
│   └── recommend.py              # CLI gợi ý matching nhanh
├── src/cvjd/                     # Mã nguồn lõi của module
│   ├── chunking/                 # Tách section CV & JD
│   ├── embedding/                # Trừu tượng hóa Embedding (BGE-M3, mock)
│   ├── models/                   # Section-aware matching, aggregation logic
│   ├── ranking/                  # Hybrid scoring, constraint rules, IR metrics
│   ├── retrieval/                # ChromaDB vector retriever
│   └── service.py                # Service tích hợp trung tâm
└── tests/                        # 146 unit test kiểm thử toàn diện module
```

---

## 6. Hướng Dẫn Cài Đặt & Sử Dụng

### 6.1 Yêu Cầu Môi Trường
- Python 3.10+
- Bộ thư viện phụ thuộc: `pip install -r requirements.txt` (gồm `chromadb`, `sentence-transformers`, `scikit-learn`, `pydantic`, `pyyaml`).

### 6.2 Các Lệnh Thực Thi Chính

#### 1. Xuất dữ liệu và tạo chỉ mục Vector
```powershell
# Xuất dữ liệu từ PostgreSQL
python scripts/db_export.py

# Bóc tách section và nạp vector vào ChromaDB
python scripts/index_chroma.py
```

#### 2. Chạy gợi ý ghép cặp qua CLI
```powershell
# Gợi ý Top 10 ứng viên cho JD có ID = 1 (Client -> Freelancer)
python scripts/recommend.py --direction jd2cv --query jd:1 --top-k 10

# Gợi ý Top 10 công việc phù hợp cho CV có ID = 1 (Freelancer -> Job)
python scripts/recommend.py --direction cv2jd --query cv:1 --top-k 10
```

#### 3. Chạy toàn bộ Unit Tests
```powershell
python -m pytest tests/
```

#### 4. Chạy Benchmark và Xuất Bảng Kết Quả
```powershell
python scripts/benchmark_paper.py
```
Kết quả sẽ được ghi tự động vào `results/benchmark_results.json`, `results/benchmark_table.md` và mã nguồn bảng LaTeX `results/benchmark_table.tex`.

---

## 7. Bản Thảo Bài Báo Khoa Học (Paper Draft)

Bản thảo bài báo hoàn chỉnh theo định dạng **Springer LNCS (Lecture Notes in Computer Science)** được đặt tại thư mục [`paper/`](paper/):
- Cấu trúc đầy đủ: Abstract, Introduction (RQ1–RQ4), Related Work, Methodology, Experimental Setup, Results & Analysis, Ethics/Fairness/Limitations, Conclusion.
- Biên dịch file PDF chỉ với 1 lệnh:
  ```powershell
  cd paper
  .\build.ps1
  ```
  File PDF được sinh ra tại: [`paper/main.pdf`](paper/main.pdf) (11 trang chuẩn proceedings).
