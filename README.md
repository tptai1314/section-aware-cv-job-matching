# Section-aware Multi-vector CV–JD Matching

Đề tài nghiên cứu sự đánh đổi giữa chất lượng xếp hạng và hiệu năng của so khớp CV–JD bằng biểu diễn đa vector theo section trên dữ liệu nội bộ của hệ thống tuyển dụng: mỗi hồ sơ (CV) và mỗi mô tả công việc (JD) được tách thành các section (kinh nghiệm, kỹ năng, học vấn, v.v.), mỗi section được nhúng thành một vector riêng, điểm tương ứng giữa CV và JD được tổng hợp qua cách ghép cặp section để tạo bảng xếp hạng ứng viên; đề tài là nghiên cứu ứng dụng – thực nghiệm, không đề xuất thuật toán mới, mà đo lường đồng thời chất lượng xếp hạng (NDCG, MRR), chi phí tính toán (latency, RAM) và khả năng giải thích của điểm thành phần, so với các baseline BM25, single-vector, chunk max-sim và upper bound là cross-encoder.

## Câu hỏi nghiên cứu và giả thuyết

| RQ | Câu hỏi | Giả thuyết |
|---|---|---|
| RQ1 | Section-aware có xếp hạng tốt hơn single-vector và chunk max-sim không? | **H1:** NDCG@10 của section-aware không thấp hơn hai baseline, đồng thời sinh ra điểm thành phần theo section giải thích được vì sao một CV được xếp hạng cao. |
| RQ2 | Section-aware đạt gần cross-encoder/LLM đến mức nào và rẻ hơn bao nhiêu? | **H2:** Section-aware nhanh hơn và tốn ít RAM hơn rõ rệt, trong khi NDCG@10 chỉ chênh nhẹ so với cross-encoder. |
| RQ3 | Kỹ thuật tăng tốc (hai tầng, lượng tử hoá/PCA) làm giảm chất lượng bao nhiêu? | **H3:** Pipeline hai tầng giảm latency đáng kể nhưng mất rất ít chất lượng xếp hạng. |
| RQ4 | Ghép cặp chéo giữa section CV và section JD có tốt hơn ghép cặp cố định không? | **H4:** Ghép cặp chéo cho kết quả tốt hơn, đặc biệt khi các section CV–JD không tương ứng một-một. |

## Phạm vi

- **Không train model mới.** Mọi mô hình đều là embedding có sẵn, frozen (không fine-tune).
- **384 chiều**, chạy hoàn toàn trên **CPU**, không cần GPU.
- Dữ liệu nội bộ đồ án, không dùng dataset bên ngoài.
- Chỉ đánh giá thực nghiệm: so sánh, ablation, phân tích đánh đổi chất lượng – hiệu năng.

## Đơn vị ranking

- **Query = JD** (mô tả công việc cần tìm người phù hợp).
- **Candidate = CV** (hồ sơ ứng viên).
- Với mỗi JD, hệ thống xếp hạng toàn bộ CV trong tập ứng viên.

## Thang nhãn

Nhãn đánh giá mức độ phù hợp giữa một cặp (JD, CV) theo thang **0–3**:

| Điểm | Ý nghĩa |
|---|---|
| 0 | Không phù hợp |
| 1 | Hơi phù hợp |
| 2 | Phù hợp |
| 3 | Rất phù hợp |
