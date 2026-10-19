# Hệ thống AI Matching CV–JD (Section-aware)

Hệ thống gợi ý/ghép cặp hồ sơ ứng viên (CV) với mô tả công việc (JD) cho nền tảng tuyển dụng freelancer Frevia.
Điểm cốt lõi: mỗi tài liệu được tách thành các **section** (kỹ năng, kinh nghiệm, học vấn…), **mỗi section được nhúng thành một vector riêng**,
sau đó điểm phù hợp giữa CV và JD được tính bằng cách so khớp các section và tổng hợp lại thành một bảng xếp hạng.

- **Query = JD**, **Candidate = CV**. Với mỗi JD, hệ thống xếp hạng các CV ứng viên phù hợp nhất.
- Chạy hoàn toàn trên **CPU**, dùng embedding có sẵn (BGE-M3) — không cần huấn luyện model.
- Hỗ trợ matching **hai chiều**: tìm CV cho JD (client → freelancer) và tìm JD cho CV (freelancer → client).

## Cách hoạt động

1. **Tách section** — mỗi CV/JD được chia thành các section theo quy tắc, mỗi section + toàn văn được nhúng thành vector.
2. **Indexing** — các vector được lưu vào ChromaDB để truy hồi nhanh.
3. **Retrieval** — dùng vector toàn văn để lấy ra pool ứng viên tiềm năng (top-K).
4. **Matching** — tính điểm tương đồng theo từng section, ghép cặp và tổng hợp thành điểm semantic.
5. **Hybrid ranking** — kết hợp điểm semantic với các tiêu chí có cấu trúc (kỹ năng, kinh nghiệm, ngân sách, ưu tiên)
   và lọc ràng buộc cứng (trạng thái, quyền) để ra kết quả cuối cùng.
6. **Giải thích & phục vụ** — AI service (FastAPI + LLM Qwen) trả về kết quả kèm giải thích dựa trên bằng chứng.

## Thành phần chính

| Thành phần | Mô tả |
|---|---|
| Embedding | BGE-M3 (dense 1024 chiều, đa ngôn ngữ), frozen |
| Vector store | ChromaDB (`frevia_cvs`, `frevia_jobs`) |
| Nguồn dữ liệu | PostgreSQL (read-only) |
| Ghép cặp section | `fixed` (cùng tên) hoặc `cross` (so mọi section, lấy max) |
| Tổng hợp điểm | `mean` hoặc `max` qua các section |
| Ranking | Hybrid: semantic + skills + experience + budget + preferences |
| AI service | FastAPI + Ollama (Qwen) cho giải thích/rerank |
| Hàng đợi xử lý file | AI service xử lý nền CV/JD upload (`PENDING → PROCESSING → READY/FAILED`, có `step`/`progress` để poll) — [thiết kế](docs/DESIGN_processing_queue.md) |

## Thang điểm phù hợp

Nhãn đánh giá mức độ phù hợp giữa một cặp (JD, CV) theo thang **0–3**:

| Điểm | Ý nghĩa |
|---|---|
| 0 | Không phù hợp |
| 1 | Hơi phù hợp |
| 2 | Phù hợp |
| 3 | Rất phù hợp |

## Chạy nhanh

```powershell
python scripts/db_export.py           # export dữ liệu từ Postgres
python scripts/index_chroma.py        # nhúng + index vào ChromaDB
python scripts/recommend.py --direction jd2cv --query jd:1 --top-k 10
```

Xem chi tiết luồng hoạt động đầy đủ tại [docs/WORKFLOW.md](docs/WORKFLOW.md).
