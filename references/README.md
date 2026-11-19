# Thư Mục Tài Liệu Tham Khảo Mới Nhất (Related Work 2026)
## Các Nghiên Cứu Đột Phá Về Ghép Cặp Hồ Sơ Ứng Viên & Mô Tả Công Việc (Job–Resume Matching)

Thư mục này lưu trữ tài liệu toàn văn (PDF), trích dẫn BibTeX chuẩn và bảng so sánh đối sánh kỹ thuật của **3 bài báo khoa học xuất bản năm 2026** liên quan trực tiếp đến kiến trúc hệ thống truy xuất và tái xếp hạng hai giai đoạn (Retrieve-then-Rerank), biểu diễn đa góc nhìn (Multiview / Section-aware), và ứng dụng mô hình ngôn ngữ lớn (LLM):

---

### Danh Sách Tài Liệu Trong Thư Mục

| STT | Tên Bài Báo | Tác Giả & Nơi Xuất Bản | Trạng Thái Tệp PDF | Đường Dẫn Trực Tiếp |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **ConFit v3: Improving Resume-Job Matching with LLM-based Re-Ranking** | jasonyux et al.<br>*arXiv:2605.09760* (2026) | `[ĐÃ TẢI VỀ]`<br>(1.07 MB) | [2026_ConFit_v3_Improving_Resume_Job_Matching.pdf](2026_ConFit_v3_Improving_Resume_Job_Matching.pdf)<br>arXiv: [abs/2605.09760](https://arxiv.org/abs/2605.09760) |
| **2** | **Multiview Job–Resume Matching With Contrastive Learning and LLM-Based Paraphrase Augmentation** | Helena Ćeović, Klemo Vladimir, Goran Delač, Marin Šilić<br>*IEEE Access*, Vol. 14, 2026 | `[OPEN ACCESS]`<br>(Truy cập trực tuyến) | [IEEE Xplore DOI: 10.1109/ACCESS.2026.3709651](https://doi.org/10.1109/ACCESS.2026.3709651)<br>ResearchGate: [Ấn phẩm 393282297](https://www.researchgate.net/publication/393282297_Multiview_Job-Resume_Matching_With_Contrastive_Learning_and_LLM-Based_Paraphrase_Augmentation) |
| **3** | **Semantic Candidate-Job Matching: A Comparative Evaluation of Dense Embedding Models in Hybrid Retrieval** | Sai Yashwant, Siddhartha Jain, Anurag Dubey et al.<br>*arXiv:2609.23307* (2026) | `[ĐÃ TẢI VỀ]`<br>(594 KB) | [2026_Semantic_Candidate_Job_Matching.pdf](2026_Semantic_Candidate_Job_Matching.pdf)<br>arXiv: [abs/2609.23307](https://arxiv.org/abs/2609.23307) |

---

## 1. Tóm Tắt Chuyên Sâu Từng Bài Báo

### 1.1. ConFit v3 (arXiv:2605.09760)
* **Trọng tâm nghiên cứu**: Kiến trúc tái xếp hạng (Re-ranking) bằng LLM sau khi đã truy xuất ứng viên bằng mô hình embedding (ConFit v1/v2).
* **Đóng góp kỹ thuật**:
  - Khắc phục giới hạn độ dài CV bằng cơ chế **Multi-pass Sliding-Window Re-ranking** (xếp hạng trượt nhiều lượt).
  - Tối ưu hóa mô hình qua **Listwise Reinforcement Learning (RL)** kết hợp chưng cất tri thức (**Distillation**) từ các mô hình mạnh hơn trước khi tinh chỉnh.
  - Tinh chỉnh trên nền tảng **Qwen3-8B** và **Qwen3-32B**, vượt qua cả GPT-5 và Claude Opus-4.5 về độ chính xác và khả năng giải trình trên tập dữ liệu tuyển dụng thực tế.
* **Mã nguồn chính thức**: [https://github.com/jasonyux/ConFit-v3](https://github.com/jasonyux/ConFit-v3)

### 1.2. Multiview Job–Resume Matching (IEEE Access 2026)
* **Trọng tâm nghiên cứu**: Biểu diễn đa trường/đa góc nhìn (Multiview Structured Representation) kết hợp học tương phản (Contrastive Learning) và tăng cường dữ liệu bằng LLM.
* **Đóng góp kỹ thuật**:
  - Mô hình hóa CV thành tài liệu đa vùng cấu trúc (Multi-field documents) tổng hợp qua Transformer siêu nhẹ, tích hợp precomputed embeddings về cấu trúc section và lộ trình thăng tiến sự nghiệp.
  - Sử dụng LLM tạo các bản diễn giải tương đương (Paraphrased Resumes) làm các góc nhìn phụ (Auxiliary views) trong quá trình huấn luyện, ép mô hình học được tính bất biến ngữ nghĩa (Semantic Invariance).
  - **Tối ưu hóa thời gian suy luận (Inference Efficiency)**: Giữ nguyên tốc độ siêu nhanh của Dual-Encoder trong lúc phục vụ (Inference time) vì các khâu sinh paraphrase và tính toán phức tạp chỉ diễn ra trong pha huấn luyện offline.
  - Đạt MAP 0.852 và nDCG@10 0.922 trên benchmark tuyển dụng thực tế.

### 1.3. Semantic Candidate-Job Matching (arXiv:2609.23307)
* **Trọng tâm nghiên cứu**: Đánh giá thực nghiệm các mô hình Dense Embedding trong hệ thống truy xuất lai (Hybrid Retrieval) quy mô doanh nghiệp.
* **Đóng góp kỹ thuật**:
  - Kết hợp vector similarity với truy xuất từ khóa toàn văn thông qua giải thuật **Reciprocal Rank Fusion (RRF)**.
  - Tinh chỉnh mô hình **EmbeddingGemma (base)** với hàm mất mát **Cached Multiple Negatives Ranking Loss (MNRL)** và các biến thể AnglE/CoSENT.
  - Đánh giá có đối chứng giữa điểm số tự động AI-Match và khuôn khổ chấm điểm độc lập qua **LLM-as-a-Judge**.

---

## 2. Ma Trận Đối Sánh Với Hệ Thống Của Chúng Ta (Section-Aware Matching)

| Tiêu Chí So Sánh | ConFit v3 (2026) | Multiview (IEEE Access 2026) | Semantic Candidate (2026) | Hệ Thống Của Chúng Ta (Đề Xuất) |
| :--- | :--- | :--- | :--- | :--- |
| **Cơ chế biểu diễn Section** | Cửa sổ trượt (Sliding-window) | Multi-field + Precomputed Trajectory | Structured Search Text qua LLM | **Section Parser Regex tất định** (Skills, Exp, Edu, Sum) |
| **Mô hình mã hóa** | Qwen3-8B / Qwen3-32B (LLM) | Lightweight Dual-Encoder | EmbeddingGemma + MNRL | **MiniLM-L6-v2 / BGE-M3** (Bi-Encoder đóng băng) |
| **Xử lý thiếu Section** | Ghép chuỗi văn bản dài | Transformer Aggregation | LLM tái cấu trúc | **Tái chuẩn hóa trọng số động (Dynamic Renormalization)** |
| **Ràng buộc cứng kỹ năng** | Không (dựa vào Prompting/RL) | Không (Contrastive Latent Space) | RRF (BM25 + Dense) | **Ontology Cây 3 Tầng + Độ phủ Bất đối xứng (Must/Nice)** |
| **Độ trễ phục vụ ($T_{\text{rank}}$)** | Hàng trăm mili-giây đến vài giây (LLM) | Dual-Encoder nhanh | Dual-Encoder + RRF | **Cực thấp: 0.12 - 2.69 ms trên CPU thông thường** |
| **Khả năng giải trình** | Tự sinh văn bản qua LLM | Điểm tương đồng đa trường | Điểm LLM-as-a-Judge | **Minh bạch cơ học (Traceable to specific section & skill rule)** |

---

## 3. Giá Trị Học Thuật & Đề Xuất Cập Nhật Bài Báo (Paper Updates)

1. **Bổ sung vào mục *Related Work (Phần 2)***:
   - Trích dẫn *ConFit v3* để minh chứng cho xu hướng dùng LLM reranking hai giai đoạn, đồng thời nêu rõ lý do hệ thống của chúng ta chọn giải pháp Hybrid + Taxonomy để đạt độ trễ CPU dưới $3$\,ms và không phụ thuộc GPU đắt đỏ.
   - Trích dẫn *Multiview Matching (IEEE Access)* để củng cố luận điểm khoa học: việc phân rã tài liệu thành đa góc nhìn (Multiview / Section-aware) mang lại biểu diễn vượt trội so với nén mù toàn văn (Single-Vector Flat).
   - Trích dẫn *Semantic Candidate-Job Matching (arXiv:2609.23307)* để hỗ trợ cho việc sử dụng thuật toán Reciprocal Rank Fusion (RRF) kết hợp Dense và BM25.
2. **File BibTeX sẵn sàng tích hợp**: Xem tệp [related_work_2026.bib](related_work_2026.bib).
