# Luồng hoạt động của hệ thống — AI Matching CV–JD (Section-aware)

> Tài liệu này mô tả **toàn bộ luồng hoạt động (workflow)** của hệ thống AI matching CV–JD:
> từ nguồn dữ liệu PostgreSQL, qua xử lý/tách section, nhúng vector, lưu vào ChromaDB, cho tới
> truy vấn – ghép cặp – xếp hạng, phục vụ qua AI service (FastAPI) kèm giải thích bằng LLM.

---

## 1. Tổng quan

### 1.1 Mục tiêu
Hệ thống gợi ý/ghép cặp hồ sơ ứng viên (CV) với mô tả công việc (JD) cho nền tảng tuyển dụng freelancer Frevia
bằng **biểu diễn đa vector theo section**. Mỗi tài liệu (CV/JD) được tách thành các section (kỹ năng, kinh nghiệm,
học vấn, tóm tắt…). **Mỗi section sinh ra một vector riêng**, điểm tương ứng CV–JD được tổng hợp thông qua
**cách ghép cặp section** để tạo bảng xếp hạng ứng viên.

Hệ thống **không huấn luyện model mới** — dùng embedding có sẵn (frozen), chạy hoàn toàn trên **CPU**.

### 1.2 Đơn vị xếp hạng
- **Query = JD**, **Candidate = CV**.
- Với mỗi JD, hệ thống xếp hạng toàn bộ CV ứng viên.
- Hỗ trợ **matching hai chiều**: JD→CV (client tìm freelancer) và CV→JD (freelancer tìm việc).

### 1.3 Các phương pháp matching
Hệ thống hỗ trợ nhiều phương pháp để lựa chọn khi cấu hình:

| Phương pháp | Mô tả |
|---|---|
| `section_aware` | **Phương pháp chính** — so khớp đa vector theo section |
| `single_vector` | Chỉ so 1 vector toàn văn |
| `chunk_max_sim` | Cắt nhỏ thành chunk rồi lấy độ tương đồng lớn nhất |
| `bm25` | Tìm theo từ khoá |

---

## 2. Kiến trúc tổng thể (5 tầng)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION        PostgreSQL ──export──▶ JSONL (CV/JD)            │
│ 2. INDEXING              JSONL ──chunk──▶ sections ──embed──▶ ChromaDB     │
│ 3. RETRIEVAL/MATCHING    query vector ──▶ pool ──▶ section-aware scoring   │
│ 4. HYBRID RANKING        S_hybrid = semantic + skills + exp + budget + pref│
│ 5. SERVING (apps/ai)     FastAPI + Ollama(Qwen) + BGE-M3 + Chroma          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Luồng dữ liệu cốt lõi:**

```
PostgreSQL (source of truth, read-only)
      │  db_export.py
      ▼
data/raw/{cvs,jds}.jsonl + manifest.json
      │  chunking.py (tách section)
      ▼
Section texts  ──embed (BGE-M3)──▶  vectors
      │  index_chroma.py / sync_worker.py
      ▼
ChromaDB (data/chroma)  ──query──▶  pool ứng viên
      │  retriever.py + matching.py + hybrid.py
      ▼
Bảng xếp hạng (Recommendation)  ──▶  API / kết quả
```

---

## 3. Cấu trúc thư mục

```
section-aware-cv-job-matching/
├── configs/base.yaml         # cấu hình trung tâm (embedding, sections, weights…)
├── src/cvjd/                 # thư viện lõi (package chính)
│   ├── data/                 # loader, preprocessing, payload, schema
│   ├── models/               # chunking, embedding, matching, bm25
│   ├── retrieval/            # retriever (ChromaRetriever)
│   ├── ranking/              # hybrid, structured, engine, constraints
│   ├── storage/              # chroma, postgres, outbox
│   ├── utils/                # config
│   ├── io.py                 # đọc/ghi JSONL
│   └── service.py            # MatchingService (điểm vào matching)
├── apps/ai/                  # AI service (FastAPI)
│   ├── app/                  # routes, recommend, explain, embeddings, index_store, llm…
│   └── tests/
├── scripts/                  # các script CLI (export, index, recommend)
├── data/                     # raw, chroma, outbox
├── tests/                    # unit/integration tests của lõi
├── docs/                     # WORKFLOW.md
├── plan.cv.md                # kế hoạch triển khai chi tiết
└── README.md                 # mô tả hệ thống
```

---

## 4. TẦNG 1 — Data Ingestion (PostgreSQL → JSONL)

### 4.1 Nguồn dữ liệu
- **PostgreSQL** là *source of truth* duy nhất (quyền, trạng thái, nội dung gốc).
- Truy cập bằng `psycopg` v3 (SQL thô, **không ORM**), **chỉ đọc (SELECT-only)**.
- Bảng chính: `Job`, `FreelancerProfile`; kèm `CvAnalysis` (dữ liệu có cấu trúc).
- File: `src/cvjd/storage/postgres.py` — có introspection `information_schema` để "degrade gracefully"
  nếu thiếu cột/bảng.

### 4.2 Cấu hình kết nối
- `src/cvjd/utils/config.py` → `load_env()` đọc `.env` ở gốc, `get_database_url()` trả `DATABASE_URL`.
- Ví dụ: `DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME`.

### 4.3 Export dữ liệu
Script `scripts/db_export.py`:
- Đọc Postgres, ưu tiên nguồn có cấu trúc: `CvAnalysis` → structured fields → legacy.
- **Loại bỏ PII (privacy-safe)** trước khi export.
- Ghi ra:
  - `data/raw/cvs.jsonl`
  - `data/raw/jds.jsonl`
  - `data/raw/manifest.json` — ánh xạ `doc_id ↔ DB id`.

### 4.4 Kiểm tra
- `scripts/db_inspect.py --schema public --no-count` — audit schema/số bản ghi.

### 4.5 Định dạng dữ liệu nội bộ
- `src/cvjd/data/schema.py`: `Section`, `Document`, `Pair`, `Query`, `SCHEMA_VERSION = "1.0.0"`.
- `src/cvjd/io.py`: round-trip JSONL (`save_jsonl`, `save_documents/load_documents`, `save_pairs`, `save_queries`).
- `src/cvjd/data/loader.py`: `load_cvs()`/`load_jds()` đọc JSONL hoặc thư mục `data/raw/{cvs,jds}/*.txt|md`.
  **Chưa hỗ trợ PDF/DOCX/OCR.**
- `src/cvjd/data/payload.py`: `document_from_payload()` — cầu nối payload API ngoài → `Document` (AI service dùng).
- `src/cvjd/data/preprocessing.py`: chuẩn hoá text (hiện ở mức stub).

---

## 5. TẦNG 2 — Indexing (tách section → embed → ChromaDB)

### 5.1 Tách section (`src/cvjd/models/chunking.py`)
- Tách **rule-based**, danh sách section **cấu hình được** (không hard-code).
- Section CV chuẩn: `skills`, `experience`, `education`, `summary` (+ `_full`).
- Section JD chuẩn: `responsibilities`, `required_skills`, `experience_level`… (+ `_full`).
- **Mỗi tài liệu → N vector section + 1 vector full-text** (`FULL_KEY = "_full"`).
- Có 3 dạng biểu diễn: `raw_file`, `structured_data`, `embedding_text`.

### 5.2 Văn bản để nhúng (`embedding_text`)
Ví dụ định dạng:
```
Document type: Freelancer CV. Section: skills.
Node.js, TypeScript, NestJS, PostgreSQL.
```

### 5.3 Mô hình nhúng (`src/cvjd/models/embedding.py`)
- `MockEmbedder` — 384 chiều, tất định (cho test).
- `st` — `all-MiniLM-L6-v2` (384 chiều).
- `bge-m3` — `BAAI/bge-m3` (**1024 chiều, đa ngôn ngữ**) — mặc định production.
- `EmbedderFactory` / `models/factory.py`: `METHODS`, `prepare_ranker()`, `build_rankings()`.
- **Ràng buộc:** vector query và collection phải cùng checkpoint/không gian — không so sánh chéo model.

### 5.4 ChromaDB (`src/cvjd/storage/chroma.py`)
- `build_client(path)` → `PersistentClient` (có path) hoặc `EphemeralClient` (không path).
- Collections: **`frevia_cvs`**, **`frevia_jobs`**, `hnsw:space = cosine`.
- **Scheme vector ID ổn định:** `{type}:{doc_id}:v{version}:{section}`
  ví dụ `cv:123:v1:skills`, `job:45:v1:_full`.
- Metadata mỗi vector: `document_id`, `document_type`, `owner_id`, `document_version`, `section`,
  `status`, `visibility`, `language`, `embedding_model`, `content_hash`.
- Dữ liệu trên đĩa: `data/chroma/chroma.sqlite3` + các segment HNSW.
- **Chroma không bao giờ là nguồn có thẩm quyền về quyền/trạng thái** — Postgres mới là.

### 5.5 Index hàng loạt
```
python scripts/index_chroma.py --raw-dir data/raw --chroma-dir data/chroma --embedder bge-m3 --device cpu
```
→ nhúng toàn bộ dữ liệu, upsert vào Chroma, chạy smoke query kiểm tra.

### 5.6 Outbox sync (PostgreSQL → Chroma)
`src/cvjd/storage/outbox.py` — `OutboxEvent`, `Outbox`, `process_events()`, `reconcile()`:
- Outbox JSONL append-only: `data/outbox.jsonl` + con trỏ sidecar `*.jsonl.cursor`.
- **Idempotent** nhờ vector ID ổn định + `content_hash`, có retry với giới hạn số lần.
- `reconcile()` phát hiện lệch (drift) giữa tài liệu PG và vector trong Chroma.

Worker:
```
python scripts/sync_worker.py --outbox data/outbox.jsonl --raw-dir data/raw \
       --chroma data/chroma --embedder bge-m3 --once      # hoặc --loop
```
Luồng sự kiện:
```
Thay đổi nghiệp vụ → ghi PG + phát sự kiện outbox
   → worker nhúng LẠI toàn bộ vector section của tài liệu
   → upsert vào Chroma
   → xoá vector phiên bản cũ (stale)
   → ghi nhận kết quả
```
Lưu ý: revoke quyền / xoá phải được tôn trọng **ngay cả khi vector chưa bị purge** — nhờ status-gate lúc query.

---

## 6. TẦNG 3 — Retrieval & Matching

### 6.1 Truy hồi (`src/cvjd/retrieval/retriever.py`)
`ChromaRetriever`, `RetrievalResult`:
1. Query bằng **vector full-text** của tài liệu đối ứng → lấy **pool top-K** (mặc định 100).
2. Lấy **section vector** của từng ứng viên trong pool (`section_vectors`).
3. Tính các **thành phần semantic theo section**.
4. **Lọc ràng buộc cứng** (kiểm tra qua Postgres) → sau đó mới xếp hạng hybrid.

Chi tiết:
- `pool_by_text()`: encode text → query Chroma → **dedup theo `doc_id`** (giữ distance nhỏ nhất),
  oversample để bù trùng lặp.
- `retrieve()`: lấy pool → `section_scoring()` → sắp xếp giảm dần theo điểm semantic → cắt `top_k`.

### 6.2 Ghép cặp section (`src/cvjd/models/matching.py`)
`embed_corpus()`, `section_components()`, `aggregate()`, `cosine()`:
- **Ghép cặp cố định (fixed):** `component_s = sim(U_s, V_s)` (0 nếu thiếu).
- **Ghép cặp chéo (cross):** `component_s = max_{t∈T} sim(U_s, V_t)`.
- **Tổng hợp:** `mean` hoặc `max` qua các section JD → ra `S_semantic`.

### 6.3 Các phương pháp khác & tăng tốc
- `src/cvjd/models/bm25.py` — BM25 (keyword, build index in-memory mỗi lần chạy).
- Chunk max-sim, single-vector.
- `src/cvjd/models/acceleration.py` — **hai tầng (two-stage)**, **PCA**, **lượng tử hoá int8**.
- `src/cvjd/models/rerank.py` — cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

## 7. TẦNG 4 — Hybrid Ranking

### 7.1 Công thức tổng hợp (`src/cvjd/ranking/hybrid.py`)
```
S_hybrid = 0.40*S_semantic + 0.30*S_skills + 0.15*S_experience
         + 0.10*S_budget   + 0.05*S_preferences
```
Trọng số nằm trong `configs/base.yaml` (`ranking.weights`), tất cả thành phần được chuẩn hoá về `[0,1]`.

Thành phần: `skills_score()`, `experience_score()`, `budget_score()`, `preferences_score()`,
cùng `JobRequirement`, `CandidateProfile`, `DEFAULT_WEIGHTS`.

### 7.2 Các module liên quan
- `ranking/structured.py` — trích xuất đặc trưng có cấu trúc.
- `ranking/engine.py` — `HybridRanker`, `RankedCandidate`, `min_max()`, `clamp01()`.
- `ranking/constraints.py` — `is_allowed()`, `apply_hard_constraints()`, `public_candidates()`
  (job đang mở, CV chia sẻ được, quyền owner/visibility, không bị xoá).

### 7.3 Điểm vào matching (`src/cvjd/service.py`)
`MatchingService.recommend_cvs_for_job()` và `recommend_jobs_for_cv()` → trả về `Recommendation`.
Đây là **matching hai chiều** (freelancer→job và client→freelancer).

---

## 8. TẦNG 5 — AI Service (`apps/ai/`, FastAPI + Ollama + BGE-M3)

### 8.1 Khởi chạy
- Entrypoint: `uvicorn app.main:app`, port `8000`, `Dockerfile` đặt `PYTHONPATH=/app/src`.
- Config `app/config.py`: `OLLAMA_URL`, `OLLAMA_MODEL` (ví dụ `qwen3:4b`), embedding model, `CHROMA_PATH`.
  - Nếu `CHROMA_PATH` trống → chạy **ephemeral** (mất dữ liệu khi restart).

### 8.2 Endpoints (`app/routes.py`)
| Endpoint | Chức năng |
|---|---|
| `GET /healthz`, `GET /readyz` | Health / readiness |
| `POST /api/v1/ai/analyze-cv` | Phân tích CV đầy đủ: extract + embed + match |
| `POST /internal/extract-cv`, `/internal/extract-job` | Trích xuất bằng LLM với JSON-schema ràng buộc |
| `POST /internal/embeddings` | Nhúng batch BGE-M3 |
| `POST /internal/index` | Index vào Chroma (`app/index_store.py`) |
| `POST /internal/recommend` | Gợi ý (`app/recommend.py`): stateless hoặc retrieval. Thêm `explain: true` để gộp luôn giải thích LLM |
| `POST /internal/explain` | Giải thích LLM riêng cho một cặp (job, cv) |
| `POST /internal/rerank` | Chấm lại danh sách candidate bằng LLM theo rubric 0–100 |

### 8.2.1 Endpoint gộp: xếp hạng + giải thích trong 1 call
`POST /internal/recommend` với `explain: true` sẽ: truy hồi → xếp hạng hybrid → gọi LLM giải thích cho **top-N** candidate
rồi trả về tất cả trong một response. Không cần gọi `/internal/explain` tuần tự như trước.

- `explain` (bool, mặc định `false`): bật gắn giải thích.
- `explain_top_n` (int 0–50, mặc định `3`): số candidate đầu bảng được giải thích (mỗi candidate tốn 1 lượt LLM).
- Mỗi item trong `results` có thêm `explanation` (`ready`, `summary`, `claims`...); item ngoài top-N có `explanation = null`.
- JD luôn đóng vai `job`, CV đóng vai `cv` bất kể chiều `jd2cv`/`cv2jd`.
- LLM lỗi → `explanation.ready = false` + `fallback_reason`, **không làm hỏng** phần xếp hạng, không trả 5xx.

### 8.3 Giải thích & rerank bằng LLM (`app/explain.py`)
- `explain_with_llm()` — Qwen tạo giải thích **dựa trên bằng chứng (evidence-grounded)**;
  phán quyết `hit/missing/unknown`; mọi khẳng định phải trích dẫn bằng chứng.
- `rerank_with_llm()` — rerank pointwise theo rubric (0–100): kỹ năng bắt buộc 40 /
  kỹ năng ưu tiên 20 / kinh nghiệm 20 / trách nhiệm 10 / điều kiện 10.
- **Fallback:** nếu LLM không sẵn sàng → `ready=false` + `fallback_reason`, **không bao giờ trả 5xx**.

### 8.4 Lưu trữ phía service
- `app/index_store.py` — `DocumentStore` giữ payload kỹ năng/kinh nghiệm/ngân sách có cấu trúc tại
  `<chroma_path>.documents.jsonl` (ghi full khi upsert/delete); **chỉ trong RAM** nếu `chroma_path = None`.
- `app/text.py` — dùng `tempfile` cho file upload (`*.upload`), xoá sau khi extract; LibreOffice
  chuyển đổi ghi file `.docx` cạnh đó.

### 8.5 Bảo mật (`app/security.py`)
- Các route `/internal/*` bảo vệ bằng header `X-Internal-Token` so với `INTERNAL_API_KEY`
  (dùng `hmac.compare_digest`).

---

## 9. Hệ thống cấu hình

### 9.1 `configs/base.yaml` — nguồn cấu hình trung tâm
Được nạp bởi `utils/config.py` (`load_config()`), có thể override qua `.env`.
| Nhóm | Nội dung |
|---|---|
| `embedding` | model name, dim, device, batch size, frozen |
| `sections` | danh sách section theo loại tài liệu + trọng số |
| `matching` | `pairing` (cross/fixed), `aggregation` (mean/max), `method` |
| `retrieval` | `pool_size`, `top_k`, `status`, `keyword_alpha` |
| `ranking` | `DEFAULT_WEIGHTS` (0.40/0.30/0.15/0.10/0.05), chuẩn hoá |

### 9.2 Biến môi trường (`.env` / `.env.example`)
`DATABASE_URL`, `INTERNAL_API_KEY`, `OLLAMA_URL`, `OLLAMA_MODEL`, `CHROMA_PATH`.

### 9.3 Dependencies
- `requirements.txt` (lõi) và `apps/ai/requirements.txt` (service) **tách riêng**.
- `pyproject.toml` — metadata package + cấu hình pytest.

---

## 10. Entry points & thứ tự chạy End-to-End

> Không có Makefile / package.json — tất cả là CLI Python + một app FastAPI.

### Giai đoạn A — Dữ liệu
```powershell
python scripts/db_inspect.py
python scripts/db_export.py
```

### Giai đoạn B — Index
```powershell
python scripts/index_chroma.py --embedder bge-m3
python scripts/sync_worker.py --once        # hoặc --loop
```

### Giai đoạn C — Match & Recommend
```powershell
python scripts/recommend.py --direction jd2cv --query jd:1 --top-k 10
python scripts/recommend.py --direction cv2jd --query cv:1
```

### Giai đoạn D — AI service
```powershell
cd apps/ai
docker build -t frevia-ai .
docker run -p 8000:8000 --env-file .env frevia-ai
# sau đó gọi: POST /internal/embeddings, /internal/index, /internal/recommend,
#             POST /api/v1/ai/analyze-cv
```

### Kiểm thử
```powershell
pytest                       # tests lõi
pytest apps/ai/tests         # tests AI service
```

---

## 11. Vị trí lưu dữ liệu (Data Storage)

| Loại | Vị trí | Ghi chú |
|---|---|---|
| **PostgreSQL** | `DATABASE_URL` (external) | Source of truth, chỉ đọc |
| **ChromaDB** | `data/chroma/chroma.sqlite3` + segment | Vector store, HNSW cosine |
| **JSONL thô** | `data/raw/{cvs,jds}.jsonl`, `manifest.json` | Export từ Postgres |
| **Outbox** | `data/outbox.jsonl` + `*.jsonl.cursor` | Hàng đợi sync PG→Chroma |
| **Payload AI service** | `<chroma_path>.documents.jsonl` | Structured data cạnh Chroma |

---

## 12. Hạn chế hiện tại

- **Hàng đợi/trạng thái xử lý file (queue/status)** — chưa triển khai. AI service đã trích xuất được
  **PDF/DOCX/OCR** khi upload (`apps/ai/app/text.py`), nhưng chưa có pipeline xử lý nền.
- Endpoints **NestJS** và **frontend UI** — không nằm trong repo này.
