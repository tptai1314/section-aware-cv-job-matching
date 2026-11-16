# Hệ Thống Ghép Cặp Hồ Sơ Ứng Viên & Mô Tả Công Việc Theo Section (Section-Aware Multi-Vector CV–Job Matching)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Vector DB](https://img.shields.io/badge/ChromaDB-Multi--Vector-orange.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM%20%7C%20BGE--M3-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests](https://img.shields.io/badge/Tests-161%20Passed-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/Paper-Springer%20LNCS%20(12%20trang)-purple.svg)](paper/main.pdf)

Dự án nghiên cứu khoa học mở và hệ thống AI gợi ý, ghép cặp thông minh hai chiều giữa **Hồ sơ ứng viên (Curriculum Vitae / Resume - CV)** và **Yêu cầu công việc (Job Description - JD)** được phát triển cho nền tảng tuyển dụng **Frevia**.

Dự án tập trung giải quyết các hạn chế của mô hình truy hồi vector đơn (Single-Vector) và tìm kiếm từ khóa truyền thống thông qua:
1. **Biểu diễn Đa Vector theo Section (Section-Aware Multi-Vector Representation)**: Phân tách tài liệu thành các khối chức năng độc lập (`skills`, `experience`, `education`, `summary`) giúp giảm thiểu hiện tượng loãng thông tin (Semantic Dilution) khi nén toàn bộ CV dài vào một vector duy nhất.
2. **So khớp Chéo Section & Tái chuẩn hóa Trọng số Động (Dynamic Weight Renormalization)**: Tự động điều chỉnh mẫu số trọng số theo các section thực tế có mặt, loại bỏ việc phạt oan điểm số khi ứng viên thiếu các mục tùy chọn.
3. **Bộ Taxonomy Kỹ năng 3 Tầng & Độ phủ Bất đối xứng (3-Tier Skill Taxonomy with Asymmetric Coverage)**: Hỗ trợ so khớp chính xác (Layer 1), quan hệ phân cấp kỹ năng có trọng số (Layer 2: `is_superset_of`, `framework_of`, `ecosystem_companion`) và ngữ nghĩa liên tục (Layer 3). Đo lường độ phủ bất đối xứng phân biệt kỹ năng bắt buộc (*must-have*) và ưu tiên (*nice-to-have*) thay cho Jaccard đối xứng.
4. **Dung hợp Điểm Lai Cân Bằng Thang Đo & Baseline RRF Không Tham Số**: Chuẩn hóa Min-Max theo từng truy vấn để kết hợp từ khóa BM25, vector đa ngữ nghĩa và luật kỹ năng; đồng thời hỗ trợ thuật toán Reciprocal Rank Fusion (RRF, $k=60$) không cần tinh chỉnh tham số $\alpha$.
5. **Hiệu năng Thực thi trên CPU**: Vận hành hoàn toàn trên CPU phổ thông với độ trễ xếp hạng $T_{\text{rank}} = 3,48$\,ms/truy vấn cho pool ứng viên $N=25$ ($0,15$\,ms với Section-Aware Dense), nhanh hơn 337 lần so với Cross-Encoder ($1.174,67$\,ms) mà không bị suy giảm do trôi miền dữ liệu.

---

## Mục Lục

- [1. Câu Hỏi Nghiên Cứu (Research Questions)](#1-câu-hỏi-nghiên-cứu-research-questions)
- [2. Kiến Trúc Tổng Thể](#2-kiến-trúc-tổng-thể)
- [3. Phương Pháp Luận & Cải Tiến Cốt Lõi](#3-phương-pháp-luận--cải-tiến-cốt-lõi)
  - [3.1 Biểu diễn Đa Vector theo Section](#31-biểu-diễn-đa-vector-theo-section)
  - [3.2 Tái chuẩn hóa Trọng số Động (Dynamic Weight Renormalization)](#32-tái-chuẩn-hóa-trọng-số-động-dynamic-weight-renormalization)
  - [3.3 Taxonomy Kỹ năng 3 Tầng & Độ phủ Bất đối xứng](#33-taxonomy-kỹ-năng-3-tầng--độ-phủ-bất-đối-xứng)
  - [3.4 Phân tách Hai Tầng & Dung hợp Điểm Lai (Calibrated Hybrid)](#34-phân-tách-hai-tầng--dung-hợp-điểm-lai-calibrated-hybrid)
- [4. Đánh Giá Thực Nghiệm & Kết Quả Benchmark](#4-đánh-giá-thực-nghiệm--kết-quả-benchmark)
- [5. Cấu Trúc Thư Mục Dự Án](#5-cấu-trúc-thư-mục-dự-án)
- [6. Hướng Dẫn Cài Đặt & Chạy Tái Lập (Reproduction Guide)](#6-hướng-dẫn-cài-đặt--chạy-tái-lập-reproduction-guide)
  - [6.1 Thiết lập Môi trường](#61-thiết-lập-môi-trường)
  - [6.2 Chạy Bộ Kiểm Thử (161 Unit Tests)](#62-chạy-bộ-kiểm-thử-161-unit-tests)
  - [6.3 Chạy Benchmark Độc lập](#63-chạy-benchmark-độc-lập)
  - [6.4 Biên dịch Bài báo Khoa học Springer LNCS](#64-biên-dịch-bài-báo-khoa-học-springer-lncs)
- [7. Ứng Dụng Microservice FastAPI (`apps/ai`)](#7-ứng-dụng-microservice-fastapi-appsai)

---

## 1. Câu Hỏi Nghiên Cứu (Research Questions)

Đề tài tập trung giải quyết và kiểm chứng 4 câu hỏi nghiên cứu cốt lõi trên tập benchmark công khai quốc tế:

* **RQ1 (Độ mịn Section so với Vector phẳng toàn văn)**: *Biểu diễn đa vector phân tách theo section có cải thiện độ chính xác xếp hạng so với vector đơn phẳng (Single-Vector) và kỹ thuật phân đoạn trượt (Chunk Max-Sim) không?*
* **RQ2 (So sánh Từ khóa BM25, Bi-Encoder và Dung hợp Lai)**: *Mô hình từ khóa (BM25) so với Dense Bi-Encoder có ưu/nhược điểm gì, và việc kết hợp lai (Calibrated Hybrid) hoặc RRF không tham số giúp cải thiện độ ổn định thứ hạng ra sao?*
* **RQ3 (Đóng góp của Taxonomy Kỹ năng 3 Tầng & Độ phủ Bất đối xứng)**: *Việc tích hợp quan hệ phân cấp kỹ năng và tính độ phủ bất đối xứng (Must-have vs. Nice-to-have) mang lại giá trị thực tế như thế nào so với phép đo tương đồng tập hợp Jaccard truyền thống?*
* **RQ4 (Đánh đổi Hiệu năng Tính toán & Reranking 2 Giai đoạn)**: *Việc áp dụng Cross-Encoder xếp hạng lại ở giai đoạn 2 có bù đắp được chi phí độ trễ suy luận lớn trên CPU so với mô hình Hybrid 1 giai đoạn không?*

---

## 2. Kiến Trúc Tổng Thể

Hệ thống được thiết kế theo mô hình 5 tầng phân tách rõ ràng:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. TIỀN XỬ LÝ & BỘ BÓC TÁCH SECTION THEO LUẬT                               │
│    Văn bản thô CV / JD ──preprocessing.py──▶ Phân đoạn thành các Section   │
│    (Các khối: Kỹ năng kỹ thuật, Kinh nghiệm làm việc, Học vấn, Tóm tắt)     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. CHỈ MỤC ĐA VECTOR ĐỘC LẬP (SECTION-AWARE INDEXING)                       │
│    Bi-Encoder đóng băng (MiniLM / BGE-M3) ──▶ Nhúng vector từng section     │
│    ChromaDB (HNSW, Độ đo Cosine, Đồng bộ bất đồng bộ qua Outbox)            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. SO KHỚP CHÉO SECTION & TÁI CHUẨN HÓA TRỌNG SỐ ĐỘNG                       │
│    Ma trận so khớp chéo: c(s_m) = max sim(v_q^{s_m}, v_c^{s_k})             │
│    Tái chuẩn hóa theo các section thực tế (Không phạt điểm section rỗng)   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. TAXONOMY KỸ NĂNG 3 TẦNG & ĐỘ PHỦ BẤT ĐỐI XỨNG                            │
│    Lớp 1: Khớp chính xác / từ đồng nghĩa (Trọng số = 1.0)                   │
│    Lớp 2: Quan hệ phân cấp (is_superset_of: 0.8, framework_of: 0.6)        │
│    Lớp 3: So khớp ngữ nghĩa vector                                          │
│    Coverage: Tính độ phủ bất đối xứng trên Must-Have + Thưởng Nice-To-Have   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. DUNG HỢP ĐIỂM LAI CÂN BẰNG THANG ĐO & BASELINE RRF                       │
│    Chuẩn hóa Min-Max theo query: S_sem_norm, S_bm25_norm                    │
│    Cách 1: Dung hợp tuyến tính có tham số (α = 0.60 dense, 0.40 lexical)    │
│    Cách 2: Dung hợp thứ hạng tương hỗ không tham số RRF (k = 60)            │
│    S_final = 0.60 * S_retrieval + 0.40 * S_taxonomy                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phương Pháp Luận & Cải Tiến Cốt Lõi

### 3.1 Biểu diễn Đa Vector theo Section
Khi nén toàn bộ một bản CV dài 2–3 trang vào một vector duy nhất, các đoạn tóm tắt kinh nghiệm dài dòng sẽ lấn át các từ khóa kỹ năng then chốt nhưng ngắn gọn. Hệ thống chia tài liệu thành các section chức năng và mã hóa thành các vector không gian riêng biệt:
$$\mathbf{v}_m = \mathcal{E}(T_m) \in \mathbb{R}^d, \quad m \in \{\text{skills}, \text{experience}, \text{education}, \text{summary}\}$$

Nhằm xử lý các trường hợp ứng viên trình bày kỹ năng lồng ghép bên trong mục kinh nghiệm làm việc (thay vì viết ở mục kỹ năng riêng), hệ thống sử dụng thuật toán **So khớp chéo (Cross-Section Pairing)**:
$$c(s_m) = \max_{s_k \in D_c} \text{sim}\big(\mathbf{v}_q^{s_m}, \mathbf{v}_c^{s_k}\big)$$

### 3.2 Tái chuẩn hóa Trọng số Động (Dynamic Weight Renormalization)
Nhiều ứng viên lập trình viên trẻ chưa có dự án ngoài, hoặc các bài đăng tuyển việc làm tự do không yêu cầu bằng cấp. Nếu gán điểm $0$ cố định cho các section bị thiếu sẽ dẫn đến việc phạt oan ứng viên tiềm năng. Hệ thống tự động tái cân bằng trọng số theo các section thực tế xuất hiện:
$$S_{sem} = \frac{\sum_{m \in \mathcal{S}_{\text{matched}}} w_m \cdot c(s_m)}{\sum_{m \in \mathcal{S}_{\text{matched}}} w_m}$$

### 3.3 Taxonomy Kỹ năng 3 Tầng & Độ phủ Bất đối xứng
Hệ thống chấm điểm tự động truyền thống thường áp dụng độ tương đồng Jaccard đối xứng ($|S_q \cap S_c| / |S_q \cup S_c|$). Cách làm này mắc lỗi nghiêm trọng: phạt điểm ứng viên kỳ cựu sở hữu nhiều kỹ năng phụ vượt ngoài yêu cầu tuyển dụng ("phạt ứng viên quá tiêu chuẩn"). 

Hệ thống thay thế bằng công thức **Độ phủ Bất đối xứng (Asymmetric Coverage)**:
$$\text{Cov}(\mathcal{T}) = \frac{\sum_{t \in \mathcal{T}} \mathbb{I}(t \in \mathcal{C}_{skills}) \cdot w_t}{\sum_{t \in \mathcal{T}} w_t}, \quad S_{skills} = 0.85 \cdot \text{Cov}(\text{MustHave}) + 0.15 \cdot \text{Cov}(\text{NiceToHave})$$

Bộ Taxonomy đóng băng (`data/taxonomy/skills_taxonomy.json`) với hơn 600 kỹ năng kỹ thuật chuẩn hóa quốc tế hoạt động theo 3 lớp:
* **Lớp 1 (Exact / Alias)**: Khớp trực tiếp hoặc qua từ đồng nghĩa (ví dụ `NodeJS` $\rightarrow$ `Node.js`).
* **Lớp 2 (Taxonomy Relational)**: Cho điểm một phần có định danh dựa trên quan hệ hệ sinh thái:
  * `is_superset_of` (trọng số 0.80, ví dụ: ứng viên biết TypeScript, JD yêu cầu JavaScript).
  * `framework_of` (trọng số 0.60, ví dụ: ứng viên biết Next.js, JD yêu cầu React).
  * `ecosystem_companion` (trọng số 0.25 - 0.50, ví dụ: Redux $\rightarrow$ React).
  * Tuyệt đối không đánh đồng hai kỹ năng cùng hệ sinh thái nếu không có quan hệ (ví dụ: Next.js $\neq$ Redux nhận điểm 0.0).
* **Lớp 3 (Semantic)**: So khớp ngữ nghĩa thông qua vector embedding.

### 3.4 Phân tách Hai Tầng & Dung hợp Điểm Lai (Calibrated Hybrid)
Hệ thống tách biệt tuyệt đối giữa **Bộ lọc Cứng (Eligibility Hard Filters)** (trạng thái hoạt động, giới hạn ngân sách tối đa, quyền riêng tư) và **Xếp hạng Năng lực (Competency Ranking)**.

Điểm truy hồi ngữ nghĩa và từ khóa được cân bằng phương sai qua phép chuẩn hóa Min-Max theo từng query trước khi tổ hợp:
$$S_{retrieval}(q, d) = \alpha \cdot \text{MinMax}(S_{dense}) + (1 - \alpha) \cdot \text{MinMax}(S_{bm25})$$
$$S_{final}(q, d) = 0.60 \cdot S_{retrieval}(q, d) + 0.40 \cdot S_{taxonomy}(q, d)$$

Bên cạnh đó, hệ thống hỗ trợ phương pháp không cần dò tìm tham số **Reciprocal Rank Fusion (RRF)**:
$$\text{RRF}(d) = \frac{1}{60 + r_{dense}(d)} + \frac{1}{60 + r_{bm25}(d)}$$

---

## 4. Đánh Giá Thực Nghiệm & Kết Quả Benchmark

Được đánh giá độc lập trên tập dữ liệu chuẩn quốc tế **`cnamuangtoun/resume-job-description-fit`** ($1.759$ cặp kiểm thử, $71$ truy vấn JD, $477$ CV ứng viên) với nhãn mức độ phù hợp 3 cấp (`0: No Fit`, `1: Potential Fit`, `2: Good Fit`).

| Mô hình / Pipeline | nDCG@5 | nDCG@10 | MRR | MAP | P@5 | Spearman $\rho$ | $T_{\text{rank}}$ ($N=25$)* | $T_{\text{online}}$* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Ranking (Floor)** | 0.4532 | 0.4963 | 0.5721 | 0.4941 | 0.4173 | -0.026 | -- | -- |
| **BM25 / TF-IDF (Lexical)** | 0.5532 | 0.5773 | 0.6393 | 0.5930 | 0.4592 | 0.209 | 0.16 ms | 0.88 ms |
| **Single-Vector Dense (MiniLM)** | 0.5889 | 0.6146 | 0.7055 | 0.6210 | 0.4845 | 0.256 | $<0.10$ ms | 29.79 ms |
| **Chunk-Mean (MiniLM, 128w)** | 0.6123 | 0.6193 | 0.6982 | 0.6356 | 0.5099 | 0.306 | $<0.10$ ms | 29.79 ms |
| **Chunk-MaxSim (MiniLM, 128w)** | 0.5758 | 0.5992 | 0.6785 | 0.6119 | 0.4817 | 0.267 | 0.39 ms | 30.17 ms |
| **Section-Aware Dense (Ours)** | 0.6053 | 0.6170 | 0.7091 | 0.6328 | 0.4986 | 0.283 | 0.12 ms | 29.90 ms |
| **BGE-M3 Flat (Long-Context)** | 0.5688 | 0.5937 | 0.6555 | 0.6167 | 0.4789 | 0.248 | $<0.10$ ms | 1,646.13 ms |
| **BGE-M3 Section-Aware** | 0.5934 | 0.6086 | 0.7126 | 0.6169 | 0.4901 | 0.257 | 0.13 ms | 1,646.25 ms |
| **Dense + BM25 (No Taxonomy)** | 0.6009 | 0.6123 | 0.6801 | 0.6337 | 0.4873 | 0.276 | 0.28 ms | 30.06 ms |
| **Calibrated Hybrid (Ours)** | 0.5998 | 0.6162 | 0.6764 | **0.6366** | 0.4817 | **0.295** | 2.69 ms | 33.44 ms |
| **Parameter-Free RRF + Taxonomy** | 0.5980 | 0.6107 | 0.6870 | 0.6299 | 0.4873 | 0.293 | 2.83 ms | 33.57 ms |
| **Cascaded 2-Stage: Hybrid $\rightarrow$ CE (Top-10)** | 0.5513 | 0.5766 | 0.6408 | 0.5767 | 0.4761 | 0.212 | 426.9 ms | 457.63 ms |
| **Full Cross-Encoder (ms-marco)** | 0.5292 | 0.5531 | 0.6420 | 0.5650 | 0.4592 | 0.167 | 1,178.0 ms | 1,178.05 ms |
| **Oracle Ranking (Ceiling)** | 0.8592 | 0.8592 | 0.8592 | 0.8592 | 0.6958 | 0.791 | -- | -- |

*(Số liệu trên là Setting A: Toàn bộ 71 truy vấn, bao gồm 10 truy vấn không có positive. Trên Setting B lọc theo chuẩn TREC có positive: ở ngưỡng $\text{rel}\ge 1$ ($N=61$), Calibrated Hybrid đạt MAP 0.7410, Section-Aware Dense đạt nDCG@10 0.7182 và MRR 0.8253; ở ngưỡng $\text{rel}=2$ ($N=28$), Section-Aware Dense đạt nDCG@10 0.7770, MRR 0.8869, MAP 0.7411).*

*\*Ghi chú về phân tách độ trễ (Intel Core i7 CPU, PyTorch CPU, đo lặp 15-20 lần kèm 10 warm-up):*
- $T_{\text{index}}$: Mã hóa ngoại tuyến (offline indexing: MiniLM 19.25 ms/doc, Section-Aware 84.85 ms/doc, BGE-M3 3,012.9 ms/doc).
- $T_{\text{query}}$: Mã hóa trực tuyến câu truy vấn (online query encoding: MiniLM 29.78 ms/query, BGE-M3 1,646.12 ms/query).
- $T_{\text{rank}}$: Thời gian tính điểm và xếp hạng danh sách ứng viên (candidate scoring & ranking) đo thực tế với quy mô pool chuẩn $N=25$ ứng viên. Các phép nhân ma trận có độ trễ $<0.10$\,ms được ghi nhận giới hạn độ phân giải đồng hồ đo.
- $T_{\text{online}} = T_{\text{query}} + T_{\text{rank}}$: Tổng độ trễ phục vụ truy vấn trực tuyến (end-to-end online query latency).

### Phát hiện Thực nghiệm Chính:
1. **Bản chất của con số Fallback 0.00% (Commit `41e1d27`)**: Phân tích toàn bộ 477 CV test cho thấy phân bố số section chức năng (`skills`, `experience`, `education`, `summary`): 0 section: **0 CV** (0.00%); 1 section: **2 CV** (0.42%, đều có `experience`); 2 section: **28 CV** (5.87%); 3 section: **107 CV** (22.43%); 4 section: **340 CV** (71.28%). 72 CV có `preamble` (thông tin liên hệ trước heading đầu tiên) nhưng đều có kèm các section chức năng. Với 2 CV chỉ có `experience`, thuật toán Dynamic Weight Renormalization tự động tái chuẩn hóa mẫu số về $0.35$ (tính $0.35 \cdot S_{\text{exp}} / 0.35 = S_{\text{exp}}$) nên không kích hoạt fallback single-vector.
2. **Section-Aware trên tập dữ liệu sạch (Held-out Clean Split)**: Khi loại bỏ 68 CV bị trùng giữa train và test trong dataset gốc, Section-Aware Dense cải thiện điểm số so với Single-Vector: nDCG@10 từ 0.6199 lên **0.6354** (+0.0155) và MAP từ 0.6290 lên **0.6458** (+0.0168). Tuy nhiên, trên toàn bộ tập test $N=71$, kiểm định cặp bootstrap 95% CI cho hiệu số nDCG@10 là $[-0.0296, +0.0333]$ ($p_{\text{adj}}=1.0000$), cho thấy sự tương đương thống kê về thứ hạng top đầu.
3. **Hiệu ứng của Section-Aware trên mô hình ngữ cảnh dài (BGE-M3)**: Trên metric chính nDCG@10, Section-Aware đạt 0.6086 so với 0.5937 của Flat BGE-M3 ($\bar{\Delta} = +0.0149$, 95% bootstrap CI $[-0.0202, +0.0496]$, $p_{\text{adj}} = 1.0000$), chưa đạt ý nghĩa thống kê. Trong phân tích thăm dò phụ, MRR ghi nhận mức tăng từ 0.6555 lên **0.7126** (+0.0571); tuy nhiên giả thuyết "ngăn loãng thông tin kỹ năng" cần được kiểm chứng thêm bằng thí nghiệm cơ chế chuyên biệt.
4. **Kiểm định Thống kê Holm-Bonferroni cho Taxonomy**: Fusing Taxonomy vào Calibrated Hybrid đạt MAP 0.6366 trên Setting A (0.7410 trên Setting B) và Spearman $\rho_{\text{mac}} = 0.295$. So với Dense+BM25, hiệu số là $\bar{\Delta} = +0.0029$ MAP (95% CI $[-0.0055, +0.0127]$, $p_{\text{adj}} = 1.0000$), không đạt ý nghĩa thống kê; giá trị thực tế của Taxonomy là khả năng áp đặt ràng buộc cứng (hard constraints) và hỗ trợ giải trình cho nhà tuyển dụng.
5. **Đánh đổi Hiệu năng & Khảo sát Pareto**: Calibrated Hybrid có xu hướng tốt hơn Cross-Encoder MS-MARCO ($\bar{\Delta} = +0.0632$ nDCG@10, 95% CI $[+0.0218, +0.1049]$, raw $p=0.0041$), nhưng sau hiệu chỉnh Holm-Bonferroni, $p_{\text{adj}} = 0.0586$ chưa vượt qua ngưỡng $\alpha = 0.05$. Reranking Top-10 qua Cross-Encoder cũng có $p_{\text{adj}} = 0.4440$ (không đạt ý nghĩa). Xét về chi phí vận hành: vì không có mô hình nào vượt trội có ý nghĩa thống kê so với Single-Vector Dense sau hiệu chỉnh đa kiểm định, **Single-Vector Dense là lựa chọn kinh tế nhất về tài nguyên tính toán** ($T_{\text{online}} \approx 29.8$\,ms, $T_{\text{rank}} < 0.10$\,ms cho $N=25$) trong nhóm có cùng độ chính xác thực nghiệm.
6. **Phân tích Độ nhạy Tham số trên Tập Huấn luyện (Train Sensitivity Analysis)**: Đánh giá độ nhạy độc lập trên tập train (`train.csv`, 50 queries, 1.627 pairs) cho thấy các hằng số được chọn là tối ưu và ổn định:
   - *Section Weights*: Điểm nDCG@10 ổn định trong khoảng 0.6578 đến 0.6636 qua các cấu hình khác nhau (Default 0.40/0.35/0.15/0.10 đạt 0.6601; Equal đạt 0.6613; Experience-heavy đạt 0.6636).
   - *Alpha (Dense vs. Lexical)*: Đạt đỉnh tại $\alpha = 0.60$ (nDCG@10 = **0.6994** so với 0.6601 của pure dense và 0.6626 của pure lexical).
   - *Beta (Retrieval vs. Taxonomy)*: Đạt đỉnh tại $\beta = 0.60$ (nDCG@10 = **0.7078** so với 0.6994 khi không có taxonomy).

---

## 5. Cấu Trúc Thư Mục Dự Án

```
section-aware-cv-job-matching/
├── apps/
│   └── ai/                             # Microservice FastAPI cho ứng dụng web thực tế
│       ├── app/                        # REST endpoints, cấu hình, tích hợp Ollama LLM, OCR
│       └── tests/                      # Kiểm thử cho microservice
├── data/
│   ├── public_benchmark/
│   │   └── resume-job-description-fit/ # Dataset chuẩn quốc tế (train.csv, test.csv)
│   └── taxonomy/
│       └── skills_taxonomy.json        # 600+ kỹ năng chuẩn hóa và quan hệ phân cấp
├── paper/                              # Mã nguồn LaTeX bài báo khoa học Springer LNCS
│   ├── sections/                       # Abstract, Intro, Methodology, Experiments, Results
│   ├── tables/                         # Các bảng LaTeX sinh tự động từ thực nghiệm
│   ├── build.ps1                       # Kịch bản biên dịch PDF qua MiKTeX / TeX Live
│   └── main.pdf                        # File PDF bài báo 12 trang hoàn chỉnh
├── results/                            # Kết quả JSON và nhật ký thực nghiệm
├── scripts/
│   └── benchmark_english.py            # Script chạy thực nghiệm chuẩn hóa độc lập
├── src/
│   └── cvjd/                           # Core engine nghiên cứu
│       ├── data/                       # Bộ bóc tách section & taxonomy loader
│       ├── models/                     # BM25, Bi-Encoder embeddings, chunking
│       ├── ranking/                    # Section-aware matching, hybrid fusion, metrics
│       ├── retrieval/                  # ChromaDB multi-vector retriever
│       └── schema.py                   # Định nghĩa cấu trúc dữ liệu (Document, Section, Pair)
├── tests/                              # Bộ kiểm thử toàn diện (161 unit tests)
├── pyproject.toml                      # Cấu hình dự án và dependencies
└── README.md                           # Tài liệu hướng dẫn dự án
```

---

## 6. Hướng Dẫn Cài Đặt & Chạy Tái Lập (Reproduction Guide)

### 6.1 Thiết lập Môi trường

```bash
# Clone kho mã nguồn
git clone https://github.com/tptai1314/section-aware-cv-job-matching.git
cd section-aware-cv-job-matching

# Khởi tạo và kích hoạt môi trường ảo (Python 3.11+)
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

# Cài đặt gói ở chế độ phát triển
pip install -e .
pip install sentence-transformers rank-bm25 scikit-learn pandas
```

### 6.2 Chạy Bộ Kiểm Thử (161 Unit Tests)

```bash
python -m pytest tests/
```
*Toàn bộ 161 tests đều PASS trong khoảng 6,3 giây.*

### 6.3 Chạy Benchmark Độc lập

```bash
# Chạy đo đạc thực nghiệm toàn diện trên 1.759 cặp kiểm thử
python scripts/benchmark_english.py
```
*Kết quả sẽ được ghi vào `results/english_benchmark_results.json` và tự động cập nhật bảng LaTeX `paper/tables/tab_main_benchmark.tex`.*

### 6.4 Biên dịch Bài báo Khoa học Springer LNCS

```powershell
# Trên Windows (PowerShell):
powershell -ExecutionPolicy Bypass -File paper/build.ps1
```
*Biên dịch ra file `paper/main.pdf` (12 trang chuẩn định dạng Springer LNCS, 0 lỗi).*

---

## 7. Ứng Dụng Microservice FastAPI (`apps/ai`)

Bên cạnh engine thực nghiệm nghiên cứu, kho mã nguồn cung cấp sẵn một microservice FastAPI hoàn chỉnh tại thư mục `apps/ai`:
* **Bóc tách PDF / DOCX & Nhận dạng Ký tự Quang học (OCR)**: Tự động phát hiện PDF dạng ảnh scan và kích hoạt `pytesseract` (300 DPI) để trích xuất văn bản.
* **Tích hợp LLM Cục bộ qua Ollama**: Sinh đánh giá năng lực theo rubric có căn cứ thực chứng và xuất giải thích quyết định ghép cặp.
* **Đồng bộ hóa Vector**: Quản lý lưu trữ và truy hồi trên ChromaDB đồng bộ với cơ sở dữ liệu qua mẫu thiết kế transactional outbox.

Khởi chạy server FastAPI phát triển:
```bash
cd apps/ai
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Giấy Phép (License)

Dự án được phân phối dưới giấy phép **Apache 2.0 License**. Các bộ dữ liệu và mô hình nền tảng tuân theo các giấy phép mã nguồn mở tương ứng.
