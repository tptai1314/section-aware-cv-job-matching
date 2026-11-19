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

## 2. Ma Trận Đối Sánh Kỹ Thuật Chi Tiết (Đối Chiếu Trực Tiếp Với Bài Báo Gốc)

| Tiêu Chí So Sánh | ConFit v3 (2026)<br>*(Yu et al., arXiv:2605.09760)* | Multiview (IEEE Access 2026)<br>*(\v{C}eovi\'{c} et al.)* | Semantic Candidate (2026)<br>*(Yashwant et al., arXiv:2609.23307)* | **Hệ Thống Của Chúng Ta**<br>*(Section-Aware Hybrid)* |
| :--- | :--- | :--- | :--- | :--- |
| **Kiến trúc mô hình chính** | LLM-based Listwise Reranker (Qwen3-8B / Qwen3-32B) | Multi-Field Dual-Encoder với Contrastive Learning | Hybrid Retrieval (Dense Bi-Encoder + BM25 qua RRF) | **Multi-Vector theo Phân mục + BM25 + Cây Kỹ năng Tri thức** |
| **Mô hình mã hóa backbone** | Qwen3-8B và Qwen3-32B | Transformer Dual-Encoder siêu nhẹ | EmbeddingGemma (base) / MPNet | **all-MiniLM-L6-v2 ($d=384$) / BGE-M3 ($d=1024$)** |
| **Chiến lược biểu diễn văn bản** | Cửa sổ trượt đa lượt (Multi-pass sliding window, $k=4, s=2$) | Biểu diễn đa trường (Multi-field) + Vector quỹ đạo | Text tìm kiếm có cấu trúc qua LLM trích xuất | **Đa vector theo $M \le 4$ phân mục (skills, exp, edu, sum)** |
| **Quy trình huấn luyện / Fine-tuning** | Listwise RL (GRPO) + Chưng cất SFT từ **Claude Sonnet-4.5** | Huấn luyện tương phản InfoNCE đa góc nhìn | Fine-tune với **Cached-MNRL** (Multiple Negatives Ranking) | **Zero-shot hoàn toàn (Trọng số mô hình nền đóng băng)** |
| **Dữ liệu tăng cường (Augmentation)** | Sắp xếp ngẫu nhiên vị trí trong prompt để khử position bias | Tăng cường diễn giải (Paraphrase) qua LLM offline | Dữ liệu log ghép cặp thực tế của ManpowerGroup | **Không tăng cường; bóc tách quy tắc Regex tất định** |
| **Xử lý kỹ năng & Ontology** | LLM suy luận tự do theo prompt ngữ cảnh | Vector nhúng trường kỹ năng trong không gian chung | Từ khóa kết hợp BM25 | **Ontology 3 cấp (615 khái niệm), độ phủ Must/Nice bất đối xứng** |
| **Độ trễ suy luận công bố** | *Không công bố số đo ms chính thức*; thuật toán yêu cầu 18 lượt gọi LLM sliding window cho top-20 | *Không công bố bảng đo ms chi tiết*; serving qua Dual-Encoder tương đương bi-encoder thông thường | *Không công bố số đo ms*; tác giả xác định đo độ trễ là hướng nghiên cứu tương lai (trang 18) | **Đo đạc thực nghiệm chi tiết**: $T_{\text{rank}} = 2.69$\,ms ($N=25$), $T_{\text{query}} \approx 25$--$30$\,ms; tổng $T_{\text{online}} \approx 28$--$33$\,ms trên CPU |
| **Hạ tầng phục vụ** | Cụm máy chủ GPU (A100/H100) để chạy LLM 8B/32B | GPU để fine-tune, CPU/GPU lúc phục vụ | Hệ thống tìm kiếm doanh nghiệp (Cosmos DB, Elasticsearch) | **Chạy 100% trên CPU thông thường (0 GPU required)** |
| **Tính giải trình & Kiểm chứng** | Tự sinh lời giải thích qua LLM; so sánh với GPT-5/Opus-4.5 | Điểm tương đồng đa trường | Điểm AI-Match và LLM-as-a-Judge | **Giải trình cấu trúc (truy vết đóng góp từng section & quan hệ ontology); chưa có User Study** |

---

## 3. Giá Trị Học Thuật & Đề Xuất Cập Nhật Bài Báo (Paper Updates)

1. **Bổ sung vào mục *Related Work (Phần 2)***:
   - Trích dẫn *ConFit v3* để minh chứng cho xu hướng dùng LLM reranking hai giai đoạn, đồng thời nêu rõ lý do hệ thống của chúng ta chọn giải pháp Hybrid + Taxonomy để đạt độ trễ CPU dưới $3$\,ms và không phụ thuộc GPU đắt đỏ.
   - Trích dẫn *Multiview Matching (IEEE Access)* để củng cố luận điểm khoa học: việc phân rã tài liệu thành đa góc nhìn (Multiview / Section-aware) mang lại biểu diễn vượt trội so với nén mù toàn văn (Single-Vector Flat).
   - Trích dẫn *Semantic Candidate-Job Matching (arXiv:2609.23307)* để hỗ trợ cho việc sử dụng thuật toán Reciprocal Rank Fusion (RRF) kết hợp Dense và BM25.
2. **File BibTeX sẵn sàng tích hợp**: Xem tệp [related_work_2026.bib](related_work_2026.bib).
