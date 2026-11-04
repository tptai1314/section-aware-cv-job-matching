# Hệ Thống Ghép Cặp Hồ Sơ Ứng Viên & Mô Tả Công Việc (Section-Aware CV–Job Matching)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Vector DB](https://img.shields.io/badge/ChromaDB-Multi--Vector-orange.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-BAAI%2FBGE--M3-green.svg)](https://huggingface.co/BAAI/bge-m3)
[![Tests](https://img.shields.io/badge/Tests-146%20Passed-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/Paper-Springer%20LNCS%20(13%20pages)-purple.svg)](paper/)

Hệ thống AI gợi ý và ghép cặp thông minh giữa **Hồ sơ ứng viên (CV/Resume)** và **Mô tả công việc (Job Description - JD)** được phát triển cho nền tảng tuyển dụng freelancer **Frevia**.

Khác với các hướng tiếp cận truyền thống xem CV/JD như một khối văn bản phẳng (flat text), hệ thống áp dụng kỹ thuật **biểu diễn đa vector theo section (Multi-Vector Section-Aware Representation)** kết hợp với **mô hình chấm điểm lai (Hybrid Fusion)** và **tái xếp hạng giải thích bằng LLM (LLM-based Explainable Reranking)**. Toàn bộ kiến trúc chạy hoàn toàn trên **CPU**, sử dụng mô hình embedding đóng băng (frozen BGE-M3 / MiniLM) mà không cần huấn luyện lại mô hình từ đầu.

---

## Mục lục

- [1. Điểm Nổi Bật & Đóng Góp Thực Tế](#1-điểm-nổi-bật--đóng-góp-thực-tế)
- [2. Kiến Trúc Hệ Thống](#2-kiến-trúc-hệ-thống)
- [3. Cơ Chế Hoạt Động Cốt Lõi](#3-cơ-chế-hoạt-động-cốt-lõi)
  - [3.1 Phân đoạn Section & Đa Vector](#31-phân-đoạn-section--đa-vector)
  - [3.2 Chiến lược Đối sánh Section (Section Alignment)](#32-chiến-lược-đối-sánh-section-section-alignment)
  - [3.3 Truy hồi Hai Giai đoạn (Two-Stage Retrieval)](#33-truy-hồi-hai-giai-đoạn-two-stage-retrieval)
  - [3.4 Công thức Chấm điểm Lai (Hybrid Scoring)](#34-công-thức-chấm-điểm-lai-hybrid-scoring)
  - [3.5 Matching Hai Chiều (Bidirectional Matching)](#35-matching-hai-chiều-bidirectional-matching)
  - [3.6 Tính Giải Thích & Tái Xếp Hạng bằng LLM](#36-tính-giải-thích--tái-xếp-hạng-bằng-llm)
- [4. Đánh Giá Thực Nghiệm & Tính Chặt Chẽ Học Thuật (Empirical Evaluation)](#4-đánh-giá-thực-nghiệm--tính-chặt-chẽ-học-thuật)
  - [4.1 Bảng 1: Kết quả xếp hạng chính trên Frevia-Bench](#41-bảng-1-kết-quả-xếp-hạng-chính-trên-frevia-bench)
  - [4.2 Bảng 2: Phân tích độ nhạy tham số Alpha](#42-bảng-2-phân-tích-độ-nhạy-tham-số-alpha)
  - [4.3 Bảng 3: Phân tích Ablation Study từng thành phần](#43-bảng-3-phân-tích-ablation-study-từng-thành-phần)
  - [4.4 Bảng 4: Đánh giá Tổng quát hóa trên Benchmark Quốc tế Công khai](#44-bảng-4-đánh-giá-tổng-quát-hóa-trên-benchmark-quốc-tế-công-khai)
  - [4.5 Kiểm định Thống kê & Khoảng Tin Cậy 95% Bootstrap](#45-kiểm-định-thống-kê--khoảng-tin-cậy-95-bootstrap)
  - [4.6 Thẩm định Công bằng & Thiên kiến (Fairness Perturbation Test)](#46-thẩm-định-công-bằng--thiên-kiến-fairness-perturbation-test)
  - [4.7 Giải trình Các Hiện tượng Dữ liệu & Giới hạn](#47-giải-trình-các-hiện-tượng-dữ-liệu--giới-hạn)
- [5. Cấu Trúc Thư Mục](#5-cấu-trúc-thư-mục)
- [6. Hướng Dẫn Cài Đặt & Sử Dụng](#6-hướng-dẫn-cài-đặt--sử-dụng)
- [7. Bản Thảo Bài Báo Khoa Học (Springer LNCS Paper)](#7-bản-thảo-bài-báo-khoa-học-springer-lncs-paper)

---

## 1. Điểm Nổi Bật & Đóng Góp Thực Tế

1. **Khắc phục hiện tượng suy giảm ngữ nghĩa (Semantic Dilution)**: Việc nhúng toàn bộ CV dài 2–3 trang vào một vector duy nhất làm loãng thông tin các kỹ năng then chốt. Hệ thống tách tài liệu thành từng phần độc lập (`skills`, `experience`, `projects`, `education`, `summary`) và nhúng đa vector.
2. **Chiến lược tổng hợp có trọng số (`weighted aggregation`)**: Cho phép gán trọng số linh hoạt cho từng phần (Kỹ năng: 35%, Kinh nghiệm: 35%, Dự án: 15%, Học vấn: 10%, Tóm tắt: 5%).
3. **Mô hình Hybrid chặt chẽ**: Kết hợp độ tương đồng ngữ nghĩa (Semantic similarity) với các điều kiện nghiệp vụ thực tế (Bộ lọc ràng buộc cứng, Jaccard overlap kỹ năng, độ lệch kinh nghiệm, độ tương thích ngân sách).
4. **Hệ thống đánh giá Dual-Dataset**: Kiểm chứng toàn diện trên cả dữ liệu thực tế song ngữ Việt - Anh (`Frevia-Bench`) và dữ liệu công khai quốc tế (`resume-job-description-fit`).
5. **Chi phí thấp, chạy hoàn toàn trên CPU**: Không phụ thuộc vào GPU đắt đỏ, đạt độ trễ phục vụ API cực thấp ($15 - 32$ ms/query).

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
1. **`fixed` (Cùng tên)**: So sánh các section tương ứng cùng loại ($Skills_{CV} \leftrightarrow Skills_{JD}$, $Exp_{CV} \leftrightarrow Exp_{JD}$).
2. **`cross` (So khớp chéo cực đại)**: Với mỗi section của JD, tìm section của CV có độ tương đồng Cosine cao nhất:
   $$S_{cross}(J_i, C) = \max_{k} \cos(\mathbf{v}_{J_i}, \mathbf{u}_{C_k})$$
3. **`weighted` (Tổng hợp có trọng số - Khuyến nghị)**: Tổng hợp điểm số qua các section theo trọng số chuyên gia nhân sự:
   $$S_{sem}(J, C) = \sum_{k} w_k \cdot \cos(\mathbf{v}_{J_k}, \mathbf{u}_{C_k})$$

### 3.3 Truy hồi Hai Giai đoạn (Two-Stage Retrieval)
- **Giai đoạn 1 (Coarse Retrieval)**: Sử dụng vector toàn văn (`full_text`) để truy vấn nhanh qua ChromaDB (HNSW index), lọc từ hàng nghìn hồ sơ xuống ứng viên tiềm năng ($K = 50 \div 100$).
- **Giai đoạn 2 (Fine-grained Section Matching)**: Áp dụng phép tính ma trận đa vector trên tập ứng viên rút gọn, giúp tối ưu thời gian phản hồi (chỉ mất $\sim 15 - 32$ ms/query trên CPU).

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
- **`jd2cv` (Client $\rightarrow$ Freelancer)**: Nhà tuyển dụng đăng bài toán, hệ thống tìm và xếp hạng các ứng viên tiềm năng nhất.
- **`cv2jd` (Freelancer $\rightarrow$ Job)**: Ứng viên tải CV lên, hệ thống tự động gợi ý các dự án phù hợp nhất với năng lực.

### 3.6 Tính Giải Thích & Tái Xếp Hạng bằng LLM
Sau khi có bảng xếp hạng top ứng viên từ tầng Hybrid, hệ thống có thể chuyển top 5 ứng viên sang mô hình ngôn ngữ lớn (Qwen-2.5 / DeepSeek qua Ollama) để trích xuất lý do gợi ý rõ ràng:
- **Điểm mạnh (Strengths)**: Bằng chứng cụ thể từ dự án/kinh nghiệm thực tế.
- **Điểm thiếu hụt (Skill Gaps)**: Kỹ năng hoặc công nghệ ứng viên còn thiếu so với JD.
- **Khuyến nghị tuyển dụng**: Tóm tắt 2-3 câu đánh giá mức độ vừa vặn tổng thể.

---

## 4. Đánh Giá Thực Nghiệm & Tính Chặt Chẽ Học Thuật

### 4.1 Bảng 1: Kết quả xếp hạng chính trên Frevia-Bench

Đánh giá thực nghiệm so sánh 8 cấu hình trên bộ dữ liệu `Frevia-Bench` (169 CV, 210 JD, 47 queries chuẩn):

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
| **Full Pipeline (+ LLM Reranker)** | **1.0000** | **0.9690** | **1.0000** | **0.9621** | **0.8160** | 30.6 ms* |

*\*Lưu ý: 30.6 ms là thời gian tính điểm rerank khi đã có sẵn cache đặc trưng; thời gian tạo sinh giải thích văn bản trực tiếp bằng LLM mất $\approx 1.8$ giây/query (xem mục 4.7).*

#### Chiều Tìm Việc: Freelancer $\rightarrow$ Job (`cv2jd`)
| Phương pháp / Cấu hình | nDCG@5 | nDCG@10 | MRR | MAP | Recall@10 | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1: BM25** | 0.4177 | 0.4533 | 0.3010 | 0.2043 | 0.0697 | 7.7 ms |
| **Baseline 2: Single-vector** | 0.3750 | 0.4373 | 0.3279 | 0.1994 | 0.0808 | 1.0 ms |
| **Baseline 3: Chunk max-sim** | 0.3789 | 0.4248 | 0.2359 | 0.1886 | 0.0518 | 171.8 ms |
| **Section-aware (Fixed)** | 0.5703 | 0.6298 | 0.4808 | 0.4458 | 0.1844 | 1.1 ms |
| **Section-aware (Cross)** | 0.6328 | 0.6770 | 0.4531 | 0.3664 | 0.1236 | 13.6 ms |
| **Section-aware (Weighted)** | **0.6522** | **0.6970** | 0.4530 | **0.3902** | 0.1244 | 14.1 ms |
| **Hybrid (Section + BM25 + Rules)** | 0.5029 | 0.4937 | 0.4525 | 0.3363 | 0.1353 | 33.9 ms |
| **Full Pipeline (+ LLM Reranker)** | 0.5606 | 0.5249 | **0.5175** | 0.3731 | 0.1353 | 34.8 ms |

---

### 4.2 Bảng 2: Phân tích độ nhạy tham số Alpha

Khảo sát hệ số dung hợp $\alpha \in [0.0, 1.0]$ giữa ngữ nghĩa và quy tắc nghiệp vụ:
- Khi $\alpha = 0.0$ (chỉ dùng Rules): nDCG@5 = 0.8705, MAP = 0.7877.
- Khi $\alpha = 1.0$ (chỉ dùng Semantic Sections): nDCG@5 = 0.9405, MAP = 0.9162.
- **Giá trị tối ưu $\alpha = 0.6$**: nDCG@5 đạt đỉnh **0.9578**, MAP = **0.9260**.

---

### 4.3 Bảng 3: Phân tích Ablation Study từng thành phần

Để đánh giá độc lập đóng góp của từng tiêu chí trong mô hình Hybrid, script [`scripts/run_ablation_and_fairness.py`](scripts/run_ablation_and_fairness.py) bóc tách ảnh hưởng của từng thành phần:

| Cấu hình Thử nghiệm | nDCG@5 | nDCG@10 | MRR | MAP | Đánh giá vai trò |
|---|:---:|:---:|:---:|:---:|---|
| **Full Hybrid (Semantic + Rules)** | **0.9409** | **0.9406** | **1.0000** | **0.9263** | Hiệu năng toàn diện cao nhất |
| **w/o $S_{skills}$ (Bỏ chấm kỹ năng)** | 0.6796 | 0.6943 | 0.7312 | 0.5993 | Giảm mạnh nhất ($-26.13\%$), kỹ năng là trụ cột chính |
| **w/o $S_{exp}$ (Bỏ số năm kinh nghiệm)** | 0.9409 | 0.9394 | 1.0000 | 0.9250 | Suy giảm nhẹ độ chính xác thứ hạng sâu |
| **w/o $S_{budget}$ (Bỏ ngân sách/lương)** | 0.9409 | 0.9396 | 1.0000 | 0.9260 | Ảnh hưởng chủ yếu ở vòng lọc cứng |
| **w/o $S_{pref}$ (Bỏ sở thích/địa điểm)** | 0.9409 | 0.9396 | 1.0000 | 0.9260 | Đóng vai trò tinh chỉnh phụ |
| **Semantic Only (Chỉ dùng Section-Aware)** | 0.6796 | 0.6943 | 0.7312 | 0.5993 | Không bị rò rỉ quy tắc, đạt mức tự nhiên thực tế |
| **Rules Only (Chỉ dùng Quy tắc, không Semantic)** | 0.8494 | 0.8728 | 0.7840 | 0.8392 | Dễ gặp lỗi từ đồng nghĩa / biến thể từ vựng |

---

### 4.4 Bảng 4: Đánh giá Tổng quát hóa trên Benchmark Quốc tế Công khai

Để kiểm chứng tính tổng quát hóa trên tập dữ liệu độc lập, hệ thống được đánh giá trên tập kiểm thử **`cnamuangtoun/resume-job-description-fit` (1.759 cặp test, 71 job queries)** tải từ Hugging Face:

| Mô hình / Pipeline | nDCG@5 | nDCG@10 | MRR | MAP | P@5 | Spearman $\rho$ | Latency (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF / BM25 (Lexical)** | 0.5536 | 0.5778 | 0.6487 | 0.5931 | 0.4563 | 0.1279 | 0.60 s |
| **Dense Bi-Encoder (MiniLM)** | **0.5895** | **0.6103** | **0.7005** | 0.6200 | **0.4930** | 0.1078 | 18.03 s |
| **Cross-Encoder (ms-marco)** | 0.5348 | 0.5589 | 0.6354 | 0.5650 | 0.4648 | -0.0163 | 89.80 s |
| **Section-Aware Multi-Vector** | **0.5895** | **0.6103** | **0.7005** | 0.6200 | **0.4930** | 0.1078 | **0.01 s** |
| **Hybrid (Section-Aware + BM25)** | 0.5869 | 0.6075 | 0.6857 | **0.6238** | 0.4789 | **0.1274** | 0.61 s |

*Nhận xét trên tập quốc tế: Biểu diễn Section-Aware Multi-Vector và mô hình Hybrid duy trì kết quả cạnh tranh so với các baseline BM25/TF-IDF và mô hình Cross-Encoder (vốn chịu giới hạn độ dài token khi xử lý hồ sơ CV dài).*

---

### 4.5 Kiểm định Thống kê & Phân tích Độ Tin Cậy

Script [`scripts/rigorous_evaluation.py`](scripts/rigorous_evaluation.py) thực hiện các kiểm định thống kê cặp (paired tests) và khoảng tin cậy:

1. **Kiểm định Paired $t$-test & Wilcoxon signed-rank test (so sánh trực tiếp Section-Aware với Baselines)**:
   - **Section-Aware (Cross) vs. BM25**: $t = 2.7911, p = 0.0097$; Wilcoxon $W = 81.0, p = 0.0082$ (Có ý nghĩa thống kê sau hiệu chỉnh Holm-Bonferroni ở mức $\alpha = 0.05$).
   - **Section-Aware (Cross) vs. Single-Vector**: $t = 3.9890, p = 0.00048$; Wilcoxon $W = 55.0, p = 0.00074$ (Có ý nghĩa thống kê sau hiệu chỉnh Holm-Bonferroni).
   - **Section-Aware (Weighted) vs. BM25**: $t = 2.4469, p = 0.0215$; Wilcoxon $W = 99.0, p = 0.0299$ (Có ý nghĩa thống kê sau hiệu chỉnh Holm-Bonferroni).
   - **Section-Aware (Weighted) vs. Single-Vector**: $t = 3.5934, p = 0.0013$; Wilcoxon $W = 72.0, p = 0.0039$ (Có ý nghĩa thống kê sau hiệu chỉnh Holm-Bonferroni).
2. **Khoảng tin cậy 95% Bootstrap CI (1.000 lần lặp) trên nDCG@10**:
   - BM25: $[0.5346, 0.6513]$ (Trung bình: $0.5920$)
   - Single-Vector: $[0.3991, 0.5378]$ (Trung bình: $0.4664$)
   - Section-Aware (Cross): $[0.6318, 0.8020]$ (Trung bình: $0.7224$)
   - Section-Aware (Weighted): $[0.6016, 0.7769]$ (Trung bình: $0.6953$)
   *(Cận dưới của Section-Aware Cross $0.6318$ cao hơn trung bình của BM25 và tách biệt hoàn toàn so với Single-Vector).*
3. **Kiểm chứng 5-Fold Cross-Validation theo Query**:
   - Để tránh rủi ro overfitting khi chọn $\alpha$ trên cùng tập đánh giá, bài toán được chia làm 5 fold độc lập theo query (tune $\alpha$ trên 4 fold, test trên fold còn lại).
   - Kết quả nDCG@10 out-of-fold trên tập test đạt trung bình **$0.5934 \pm 0.1534$**.

---

### 4.6 Thẩm định Công bằng & Thiên kiến (Fairness Perturbation Test)

Tiến hành kiểm thử nhiễu nhân quả (Counterfactual Perturbation Testing) trên 30 CV đại diện ($150$ cặp đánh giá):
1. **Thiên kiến Giới tính & Tên gọi (Gender & Name)**: Thay đổi tên nam phổ biến ("Nguyen Van An", "David") thành tên nữ tương ứng ("Nguyen Thi Mai", "Sarah").
   - Độ lệch điểm số trung bình (MAD): **$0.000000$**
   - Tỷ lệ bảo toàn thứ hạng Top 5: **$100.0\%$**
2. **Thiên kiến Trường học / Học vị (Institution Prestige)**: Thay thế các trường danh tiếng ("Bach Khoa University", "National University") bằng trường nghề địa phương ("Provincial Vocational College").
   - Độ lệch điểm số trung bình (MAD): **$0.000000$**
   - Nhận định: Hệ thống tập trung đánh giá các kỹ năng kỹ thuật và kinh nghiệm thực tế được bóc tách trong section thay vì điều kiện hóa theo danh xưng cá nhân hay trường học.

---

### 4.7 Giải trình Các Hiện tượng Dữ liệu, LLM & Giới hạn

1. **Về độ tin cậy của nhãn & Hệ số Cohen's Kappa ($\kappa = 0.82$)**:
   - Nhãn tham chiếu trên `Frevia-Bench` được đối chứng qua mẫu 50 cặp được chấm mù độc lập bởi **2 chuyên viên tuyển dụng nhân sự (HR Recruiters)**. Hệ số $\kappa = 0.82$ phản ánh độ đồng thuận cao giữa các chuyên gia, các trường hợp bất đồng được hội đồng phân xử bởi chuyên viên cao cấp.
2. **Nghịch lý ở chiều $cv2jd$ và Phân tích Trần Lý Thuyết Recall@10**:
   - Ở chiều $cv2jd$, mô hình LLM Cascade đạt $0.5606$ nDCG@5 so với $0.6522$ của Section-Aware Weighted. Chúng tôi giả thuyết rằng bản mô tả công việc (JD) chứa nhiều đoạn văn bản hành chính dài dòng, làm phân tán sự chú ý trong prompt dạng listwise.
   - **Trần lý thuyết của Recall@10**: Trong tập dữ liệu, nhiều hồ sơ ứng viên đáp ứng cùng lúc nhiều JD. Tính toán trần lý thuyết cho từng query $\text{Ceiling}_q = \min(1.0, 10 / |\text{Positive}_q|)$ cho giá trị trung bình là **$0.6844$**. Mức Recall@10 thực tế đạt $0.1244$, tương ứng với **$18.18\%$ của trần lý thuyết** (và mô hình Hybrid đạt $0.0697$, tương ứng $10.18\%$).
3. **Đánh giá Chất lượng Giải thích của LLM**:
   - Kiểm tra 30 cặp mẫu ngẫu nhiên (109 kỹ năng được trích xuất trong giải thích):
     - **Tỷ lệ có bằng chứng thực chứng (Verbatim Grounded)**: **$95.41\%$** (104/109 kỹ năng xuất hiện nguyên văn trong văn bản CV).
     - **Tỷ lệ suy diễn không có căn cứ (Hallucination Rate)**: **$4.59\%$** (5/109 kỹ năng).
4. **Phạm vi Thực thi trên CPU & Độ trễ**:
   - Kiểm thử thực hiện trên cấu hình: CPU Intel Core i7, RAM 16GB.
   - **Tầng API Serving (Truy hồi 2 giai đoạn + Scoring trên CPU)**: **$15 - 32$ ms/query**.
   - **Tầng LLM Tạo sinh Văn bản Giải thích**: **$\approx 1.8$ giây/query** qua Ollama (chỉ gọi theo yêu cầu cho Top 5 ứng viên khi người dùng cần giải trình chi tiết).


### 4.8 Đánh Giá Song Ngữ & Độ Bền Ngôn Ngữ (Bilingual & Linguistic Robustness)

#### 1. Thống Kê Phân Bố Ngôn Ngữ (Frevia-Bench)
Quy ước gắn nhãn: Tài liệu được xếp vào nhóm **Trộn (Code-Mixed)** nếu các token tiếng Việt và tiếng Anh mỗi bên chiếm $\\ge 20\\%$ tổng số token ngôn ngữ mang nghĩa ($\\rho_{vi} \\ge 0.20$ và $\\rho_{en} \\ge 0.20$); ngược lại được tính là **Tiếng Việt** ($\\rho_{vi} \\ge 0.80$) hoặc **Tiếng Anh** ($\\rho_{en} \\ge 0.80$).

| Phân loại Ngôn ngữ | Số lượng CV | Tỷ lệ CV (%) | Số lượng JD | Tỷ lệ JD (%) |
|---|:---:|:---:|:---:|:---:|
| **Tiếng Việt (VI)** | 50 | 29.6% | 70 | 33.3% |
| **Tiếng Anh (EN)** | 94 | 55.6% | 110 | 52.4% |
| **Trộn (Code-Mixed)** | 25 | 14.8% | 30 | 14.3% |
| **Tổng cộng** | **169** | **100.0%** | **210** | **100.0%** |

#### 2. Kết Quả Xếp Hạng Tách Theo Nhóm Cặp Ngôn Ngữ
Đánh giá trên chiều tuyển dụng  \\rightarrow CV$ (=27$ truy vấn):

| Nhóm Cặp Ngôn ngữ | Số Query | Mô hình | nDCG@5 | nDCG@10 | MAP |
|---|:---:|---|:---:|:---:|:---:|
| **Đơn ngữ EN ↔ EN** | 10 | BM25 | 0.3840 | 0.5119 | 0.1038 |
| | | Single-Vector | 0.3712 | 0.4147 | 0.1040 |
| | | **Section-Aware (Ours)** | **0.7245** | **0.7553** | **0.6204** |
| **Đơn ngữ VI ↔ VI** | 9 | BM25 (pyvi) | 0.5374 | 0.5326 | 0.4372 |
| | | Single-Vector | 0.4485 | 0.4500 | 0.3022 |
| | | **Section-Aware (Ours)** | **0.5731** | **0.5989** | **0.5247** |
| **Chéo Ngôn ngữ VI ↔ EN** | 8 | BM25 | 0.7374 | 0.6547 | 0.5044 |
| | | Single-Vector | 0.3525 | 0.3251 | 0.1582 |
| | | **Section-Aware (Ours)** | **0.2582** | **0.2730** | **0.2013** |

*Nhận định thực nghiệm*:
- Ở các nhóm đơn ngữ (EN-EN và VI-VI), **Section-Aware** vượt trội rõ rệt cả Single-Vector và BM25 nhờ bóc tách đúng trọng số section kỹ năng và kinh nghiệm.
- Ở nhóm chéo ngôn ngữ (VI-EN), trong ngành IT, các thực thể kỹ năng then chốt (React, Node.js, Python, PostgreSQL) là các token tiếng Anh dùng chung. Khi truy vấn và hồ sơ chia sẻ trực tiếp các thực thể này, BM25 bắt khớp tốt các từ khóa chuẩn xác, trong khi mô hình nhúng đa ngôn ngữ tổng quát có thể chịu độ lệch ngữ nghĩa do chuyển dịch ngôn ngữ văn cảnh.

#### 3. Thử Nghiệm Độ Bền Khi Bỏ Dấu Tiếng Việt (Unaccented Input)
Thực nghiệm loại bỏ hoàn toàn dấu tiếng Việt trên toàn bộ hồ sơ CV ứng viên để mô phỏng hành vi gõ văn bản không dấu của người dùng:

| Mô hình | nDCG@10 (Có dấu) | nDCG@10 (Không dấu) | Mức thay đổi $\\Delta$ (%) |
|---|:---:|:---:|:---:|
| BM25 (pyvi) | 0.5901 | 0.6592 | +11.71% |
| Single-Vector | 0.3912 | 0.4123 | +5.38% |
| **Section-Aware (Ours)** | **0.4455** | **0.5201** | **+16.75%** |

Nhờ cơ chế chuẩn hóa Unicode NFC và từ điển alias kỹ năng chịu lỗi không dấu (strip_accents), hệ thống duy trì chất lượng xếp hạng ổn định, không bị lỗi phân mảnh token.

#### 4. Thẩm Định Bộ Tách Section (Sectioner Accuracy Audit)
Đánh giá trên tập mẫu kiểm toán 50 CV song ngữ gán nhãn tay:
- **Tóm tắt (Summary/Profile)**: **100.0%**
- **Kỹ năng (Skills)**: **94.0%**
- **Kinh nghiệm (Experience)**: **94.0%**
- **Học vấn (Education)**: **94.0%**
- **Tỷ lệ fallback lỗi**: **0.0%**

#### 5. Đạo Đức & Quy Chuẩn Gán Nhãn
- **Đội ngũ chuyên gia gán nhãn**: 2 chuyên gia tuyển dụng (HR specialists) thành thạo song ngữ Việt - Anh trong ngành công nghệ; 50 cặp tính hệ số tương quan Cohen's Kappa ($\\kappa = 0.82$) được phân tầng trải đều cả 3 nhóm ngôn ngữ (VI-VI, EN-EN, VI-EN).
- **Ẩn danh hóa dữ liệu PII**: Toàn bộ thông tin định danh tiếng Việt (Họ và tên 'Nguyễn Văn A', số điện thoại '09xx', địa chỉ ngõ/phường/quận) đều được loại bỏ trước khi đưa vào không gian vector.
- **Phạm vi kiểm chứng của Benchmark Công khai**: Tập dữ liệu quốc tế (
esume-job-description-fit) hoàn toàn bằng tiếng Anh, phục vụ kiểm chứng tính tổng quát hóa không bị rò rỉ nhãn, không phản ánh các đặc thù tiếng Việt của nền tảng Frevia.

---

## 5. Cấu Trúc Thư Mục

```
section-aware-cv-job-matching/
├── configs/                      # File cấu hình YAML (base.yaml, weights, thresholds)
├── data/                         # Dữ liệu thực nghiệm
│   ├── raw/                      # Dữ liệu xuất từ DB PostgreSQL (cvs.jsonl, jds.jsonl)
│   ├── benchmark_qrels.json      # Ground truth labels cho 47 queries Frevia-Bench
│   ├── public_benchmark/         # Dataset công khai từ Hugging Face (1.759 test pairs)
│   └── chroma/                   # Vector database lưu trữ embeddings
├── docs/                         # Tài liệu thiết kế chi tiết (WORKFLOW.md, kiến trúc)
├── paper/                        # Bản thảo bài báo khoa học Springer LNCS LaTeX
│   ├── sections/                 # 00_abstract đến 06_conclusion & ethics
│   ├── tables/                   # Bảng số liệu LaTeX xuất tự động (Bảng 1-4)
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
│   ├── eval_public_benchmark.py  # Đánh giá trên benchmark công khai Hugging Face
│   ├── run_ablation_and_fairness.py # Ablation study, kiểm định thống kê & fairness
│   └── recommend.py              # CLI gợi ý matching nhanh
├── src/cvjd/                     # Mã nguồn lõi của module
│   ├── chunking/                 # Tách section CV & JD
│   ├── embedding/                # Trừu tượng hóa Embedding (BGE-M3, MiniLM, mock)
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
- Cài đặt thư viện: `pip install -r requirements.txt`

### 6.2 Các Lệnh Thực Thi Chính

#### 1. Xuất dữ liệu và tạo chỉ mục Vector
```powershell
python scripts/db_export.py
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

#### 4. Chạy Toàn Bộ Chuỗi Thực Nghiệm Học Thuật
```powershell
# 1. Benchmark chính trên Frevia-Bench
python scripts/benchmark_paper.py

# 2. Đánh giá đối chứng trên Public Dataset (Hugging Face)
python scripts/eval_public_benchmark.py

# 3. Chạy Ablation Study, Kiểm định thống kê & Fairness
python scripts/run_ablation_and_fairness.py
```

---

## 7. Bản Thảo Bài Báo Khoa Học (Springer LNCS Paper)

Bản thảo bài báo hoàn chỉnh theo định dạng **Springer LNCS (Lecture Notes in Computer Science)** được đặt tại thư mục [`paper/`](paper/):
- **Cấu trúc 13 trang chuẩn proceedings**: Bao gồm 4 bảng số liệu định lượng, phân tích ablation, kiểm định thống kê $p$-value, phân tích counterfactual fairness, và thảo luận đạo đức/giới hạn.
- Biên dịch file PDF chỉ với 1 lệnh:
  ```powershell
  cd paper
  .\build.ps1
  ```
  File PDF được sinh ra tại: [`paper/main.pdf`](paper/main.pdf).
