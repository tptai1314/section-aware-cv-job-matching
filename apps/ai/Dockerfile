# frevia-ai — Dịch vụ AI nội bộ (FastAPI)
#
# Đọc file CV (PDF/DOCX), trích xuất thông tin có cấu trúc bằng LLM local (Ollama),
# sinh embedding đa ngôn ngữ bằng BGE-M3. Service không xác thực, không truy cập DB.
#
# Cách chạy local:
#     pip install -r requirements.txt
#     copy .env.example .env -> sửa OLLAMA_URL
#     uvicorn app.main:app --port 8000
#
# Endpoint:
#     GET  /healthz
#     GET  /readyz
#     POST /api/v1/ai/analyze-cv      (multipart, field `file`)  — contract chuẩn của NestJS
#     POST /internal/extract-cv      (alias của analyze-cv, theo plan.cv.md §10.2)
#     POST /internal/extract-job     (JSON {title?, text})
#     POST /internal/embeddings      (JSON {texts})
#     POST /internal/index           (upsert tài liệu đã trích xuất vào Chroma)
#     POST /internal/recommend       (hybrid §7: stateless nếu có candidates,
#                                     ngược lại retrieval Chroma + payload store)
#     POST /internal/jobs            (multipart: file CV | text JD) -> 202 {job_id}
#     GET  /internal/jobs/{job_id}   (status/step/progress)
#     DELETE /internal/jobs/{job_id} (hủy job PENDING)
#     POST /internal/jobs/{job_id}/retry (retry job FAILED)
#
# Lưu ý dữ liệu: mount volume cho /app/data để giữ `processing.db`, file upload tạm
# và index Chroma qua các lần restart (ví dụ: -v frevia_data:/app/data).
#
# Build context = repo root (vì copy thêm `src/` package cvjd + `configs/`).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY apps/ai/requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY apps/ai/app app
COPY src src
COPY configs configs

EXPOSE 8000

# Không chạy healthcheck bằng uvicorn --reload; entrypoint đơn giản.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]