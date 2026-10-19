# THIẾT KẾ — Hàng đợi & Trạng thái xử lý file (Document Processing Queue)

> **Trạng thái tài liệu:** thiết kế đã **triển khai P1–P3** trong repo này (xem §9);
> P4 (NestJS) nằm ngoài repo, P5–P6 còn lại.
> **Phủ sóng:** `plan.cv.md` §9 (quy trình CV/Job), §9.3–9.4 (trạng thái, retry/idempotency),
> §11.2 (`DocumentProcessingJob`), §10.1 (`GET /api/cvs/:id/processing-status`),
> mục 16 Definition of Done: *"CV được upload và xử lý tự động"*, *"Job được tạo và lập chỉ mục tự động"*.

**Code tương ứng:** `apps/ai/app/processing/{store,pipeline,worker}.py`,
endpoint `/internal/jobs*` trong `apps/ai/app/routes.py`, wiring ở `apps/ai/app/main.py`,
test ở `apps/ai/tests/test_processing_{store,worker,routes}.py`.

---

## 0. Tóm tắt quyết định thiết kế

| # | Quyết định | Đề xuất |
|---|---|---|
| D1 | Trạng thái xử lý lưu ở đâu | Interface `ProcessingStore` + **2 backend**: `PostgresStore` (đúng plan, khi có quyền ghi PG) và `SqliteStore` (cho repo hiện tại, PG đang read-only) → **đã dùng SqliteStore** (OQ1) |
| D2 | Hàng đợi ở đâu | **In-process queue trong AI service** (worker thread riêng), không Redis/BullMQ; NestJS chỉ đẩy job qua HTTP |
| D3 | Luồng upload | Tách 2 phase: HTTP trả `202 + job_id + PENDING` ngay, xử lý nền, NestJS poll trạng thái |
| D4 | Ai ghi index | Upload mới → worker gọi thẳng `recommend_mod.index()` (single writer). Sự kiện PG thay đổi → giữ nguyên outbox `sync_worker` hiện có |
| D5 | Retry | Phân loại lỗi **tạm thời vs vĩnh viễn**; tối đa 3 lần, backoff; hết lượt → `FAILED` nhưng cho phép retry thủ công |

---

## 1. Hiện trạng & điểm tích hợp (code đang có)

### 1.1 Đã sẵn sàng — tái sử dụng nguyên vẹn

| Khối | Module | Vai trò trong pipeline mới |
|---|---|---|
| Đọc + trích xuất file | `apps/ai/app/text.py` — `read_upload()`, `extract_text()`, `merge_broken_lines()` | Bước **EXTRACT**: PDF/DOCX/OCR, có giới hạn `max_upload_mb`, `max_pages` |
| Parse CV/JD bằng LLM | `apps/ai/app/llm.py` + `normalize.py` — `chat()`, `normalize_cv()`, `normalize_job()` | Bước **PARSE**: trả payload có cấu trúc + warnings |
| Embedding + upsert Chroma | `apps/ai/app/recommend.py:index()` → `MatchingService.index()` → `src/cvjd/storage/chroma.py:upsert_document()` | Bước **INDEX**: `vector_id = {doc_type}:{doc_id}:v{version}:{section}` (`chroma.py:83`), metadata đã có `content_hash`, `status`, `owner_id`, `visibility` (`chroma.py:112-138`); `upsert_document` tự xoá vector version cũ (`_stale_filter`, `chroma.py:162`) |
| Payload có cấu trúc | `apps/ai/app/index_store.py:DocumentStore` → `<chroma_path>.documents.jsonl` | Lưu skills/experience/budget phục vụ hybrid ranking |
| Đồng bộ PG → Chroma | `src/cvjd/storage/outbox.py` + `scripts/sync_worker.py` | Giữ nguyên cho **những thay đổi xuất phát từ PostgreSQL** |
| Hard constraints | `src/cvjd/ranking/constraints.py` | Không đổi: `status=READY` + `visibility=public` + chưa xoá |

### 1.2 Khoảng trống đã đóng (P1–P3) và phần còn lại

| # | Trước đây | Nay |
|---|---|---|
| 1 | Không có state machine (chỉ có `READY` làm filter Chroma) | `processing/store.py`: `PENDING → PROCESSING → READY/FAILED/CANCELLED` + `step`, có test transitions/claim/retry |
| 2 | `POST /api/v1/ai/analyze-cv` chặn tới 180s (Ollama timeout) | `POST /internal/jobs` trả `202` ngay; worker nền xử lý; poll `GET /internal/jobs/{job_id}` |
| 3 | Không có bảng job & endpoint trạng thái | `SqliteProcessingStore` (`data/processing.db`) + `GET /internal/jobs/{job_id}` (§4.1) |
| 4 | `content_hash` chưa dùng để quyết định embed lại | Đã *tính & lưu* vào job (`set_content_hash`); chống trùng theo `(document_type, document_id, version)`. **Skip theo hash vẫn là tối ưu để sau** |
| 5 | Batch là CLI thủ công (`db_export.py` → `index_chroma.py`) | Upload/`/internal/jobs` tự trigger worker, không cần chạy tay |
| 6 | NestJS/frontend chưa có đường async | Hợp đồng NestJS đã chốt (§4.2) — triển khai ở repo khác (P4) |

---

## 2. State machine

### 2.1 Trạng thái

```
                    ┌──────────────────────────────────────────┐
                    │                                          ▼
 PENDING ──▶ PROCESSING ──▶ READY (terminal)            FAILED (terminal)
   │             │  ▲            ▲                           ▲
   │             │  └──retry─────┼───────────────────────────┘
   │             │               │
   └──cancel────▶▼               │
             CANCELLED (terminal) │
                                 │
                    (retry thủ công: FAILED → PENDING)
```

| Trạng thái | Ý nghĩa | Terminal |
|---|---|---|
| `PENDING` | Job đã ghi nhận, chờ worker nhận | không |
| `PROCESSING` | Worker đang xử lý (kèm trường `step` xem §2.3) | không |
| `READY` | Đã extract + parse + embed + index xong; vector đã ở Chroma, retrieval thấy được | có |
| `FAILED` | Lỗi sau khi hết retry (hoặc lỗi vĩnh viễn) — lưu `last_error` | có |
| `CANCELLED` | User/NestJS hủy trước khi xử lý xong (tuỳ chọn, giai đoạn 2) | có |

**Nguyên tắc:** `status` trong record xử lý và `status` metadata trong Chroma là **hai thứ khác nhau**.
- Record xử lý: tiến trình của *job*.
- Metadata Chroma `status="READY"`: cờ *đủ điều kiện truy xuất* — chỉ được ghi khi job chuyển `READY` (điều này sẵn có: `chroma.py` ghi `status` khi upsert).

### 2.2 Bảng chuyển trạng thái

| Từ | Đến | Ai gây ra | Điều kiện |
|---|---|---|---|
| *(mới)* | `PENDING` | API `POST /internal/jobs` (NestJS gọi) | file hợp lệ, `size ≤ max_upload_mb`, không có job đang `PENDING/PROCESSING` cho cùng `doc_id + version` |
| `PENDING` | `PROCESSING` | Worker (claim job) | atomic claim (xem §5.3) |
| `PROCESSING` | `READY` | Worker | mọi bước xong, index thành công |
| `PROCESSING` | `PROCESSING` | Worker | retry có lỗi tạm thời: `attempt_count += 1`, quay về chờ, backoff |
| `PROCESSING` | `FAILED` | Worker | lỗi vĩnh viễn **hoặc** `attempt_count ≥ max_attempts` |
| `FAILED` | `PENDING` | API `POST /internal/jobs/:id/retry` (thủ công) | reset `attempt_count` |
| `PENDING` | `CANCELLED` | API `DELETE /internal/jobs/:id` (tuỳ chọn) | chưa có worker claim |

Không có đường `FAILED → READY` trực tiếp: mọi xử lý lại đều qua `PENDING` để giữ thống nhất log.

### 2.3 `step` (chi tiết bên trong `PROCESSING`)

| `step` | Công việc | Hàm hiện có |
|---|---|---|
| `RECEIVED` | Đã ghi job, chưa chạm file | — |
| `EXTRACTING` | Đọc file, trích text (+OCR nếu bật) | `text.read_upload` + `text.extract_text` |
| `PARSING` | Gọi LLM parse CV/JD → payload chuẩn hóa | `llm.chat` + `normalize_cv/normalize_job` |
| `EMBEDDING` | Tách section + nhúng từng section + full-text, upsert Chroma + DocumentStore, ghi `status=READY` | `recommend_mod.index()` |
| `FINALIZING` | Ghi `READY` + `warnings`, dọn file tạm, log | `store.mark_ready()`, `store.remove_source_file()` |

> Bản triển khai gộp `EMBEDDING` + `INDEXING` thành **một** bước `EMBEDDING`:
> `recommend_mod.index()` vừa nhúng vừa upsert Chroma + DocumentStore trong cùng một
> lời gọi, nên không có ranh giới thật giữa hai bước để checkpoint.
> `warnings` (section rỗng, `TRUNCATED`, OCR...) được ghi khi job `READY`.

`step` chỉ để hiển thị tiến độ cho UI (`"đang nhúng..."`), không phải state machine độc lập — nếu job thất bại ở bước nào thì `last_error` kèm `step`.

---

## 3. Nguồn trạng thái (D1)

### 3.1 Interface chung

```
ProcessingStore (apps/ai/app/processing/store.py)
  create(*, document_type, document_id, version, owner_id, visibility,
         source_kind, source_ref, text, title) -> (JobRecord, deduplicated)
  get(job_id) -> JobRecord | None
  get_by_doc(document_type, document_id, version) -> JobRecord | None      # chống trùng
  list_active() -> list[JobRecord]                                        # monitoring
  claim_next(now=None) -> JobRecord | None    # atomic: PENDING(+tới hạn) -> PROCESSING, attempt+1
  set_step(job_id, step) -> JobRecord         # chỉ khi PROCESSING
  set_content_hash(job_id, hash) -> JobRecord
  mark_ready(job_id, warnings) -> JobRecord   # PROCESSING -> READY
  mark_failed(job_id, last_error) -> JobRecord
  requeue(job_id, delay_s, last_error) -> JobRecord   # PROCESSING -> PENDING (backoff)
  retry(job_id) -> JobRecord                  # FAILED -> PENDING (reset attempt)
  cancel(job_id) -> JobRecord                 # PENDING -> CANCELLED
  requeue_stale(max_age_s, max_attempts) -> int   # worker chết -> PENDING/FAILED
```

### 3.2 Backend A — PostgreSQL (đúng plan §11.2)

Bảng **`DocumentProcessingJob`**:

| Kiểu | Trường | Ghi chú |
|---|---|---|
| uuid/text | `id` | PK |
| text | `documentType` | `cv` \| `jd` |
| text | `documentId` | ID nghiệp vụ bên NestJS |
| int | `documentVersion` | bump khi CV/job đổi |
| text | `status` | `PENDING/PROCESSING/READY/FAILED/CANCELLED` |
| text | `step` | §2.3 |
| int | `attemptCount` | mặc định 0 |
| text | `lastError` | cắt ngắn ~2000 ký tự |
| text | `contentHash` | sha256 của text đã chuẩn hoá — dùng để bỏ qua embed lại |
| text | `sourceRef` | đường dẫn file tạm / object key |
| uuid | `ownerId`, text `visibility` | để ghi metadata Chroma khớp `access_where()` |
| int | `payloadVersion` | version vector sẽ ghi (`v{n}` trong `vector_id`) |
| timestamp | `createdAt/updatedAt/startedAt/finishedAt` | |
| int | `durationMs` | đo p50/p95 thời gian xử lý |

**Index/uniqueness:**
- `UNIQUE (documentType, documentId, documentVersion)` → 2 job trùng cho cùng bản ghi bị loại ngay khi `create`.
- index `(status, createdAt)` cho `claim_next()`.
- index `(documentType, documentId)` cho endpoint poll theo tài liệu.

### 3.3 Backend B — SQLite (cho repo hiện tại)

Repo này dùng PostgreSQL **read-only** (README: *"Nguồn dữ liệu PostgreSQL — read-only"*), chưa có quyền ghi.

→ `SqliteStore`: 1 file `data/processing.db`, schema **giống hệt** §3.2 (đổi tên cột về snake_case), cùng interface. Lý do chọn SQLite thay JSONL: cần `UPDATE` có điều kiện (atomic claim) và truy vấn theo `status` — JSONL không làm được an toàn.

**Khi nào chuyển sang Backend A:** khi NestJS cần đọc trạng thái trực tiếp từ PG (không qua AI service) hoặc cần xử lý lại sau khi service restart với data share giữa nhiều instance. Việc chuyển chỉ là thay 1 dòng construct store — không đổi logic pipeline.

> ⚠️ **Cần chốt với người dùng:** quyền ghi PostgreSQL của AI service (plan gốc muốn bảng này ở PG). Đây là open question OQ1 (§9).

### 3.4 Bảng `EmbeddingIndexRecord` (plan §11.2) — quyết định

**Không tạo riêng.** Thông tin này đã có đủ trong metadata Chroma (`content_hash`, `embedding_model`, `document_version`, `status` — `chroma.py:112-138`) + record xử lý ở trên. Tạo thêm sẽ phải đồng bộ 2 nơi, tăng lỗi. Nếu sau này cần audit độc lập thì thêm sau, không chặn MVP.

---

## 4. Hợp đồng API

### 4.1 NestJS → AI service (internal,-auth bằng `INTERNAL_API_KEY`)

| Method | Endpoint | Body / Query | Trả về |
|---|---|---|---|
| `POST` | `/internal/jobs` | multipart/form: `document_type` (`cv`\|`jd`), `document_id`, `version`=1, `owner_id?`, `visibility?`, `title?`, `file?` **hoặc** `text?` | `202 {job_id, status, document_type, document_id, version, deduplicated, created_at}` |
| `GET` | `/internal/jobs/{job_id}` | — | `{job_id, status, step, progress, attempt_count, max_attempts, last_error, warnings[], created_at, updated_at, finished_at, duration_ms}` |
| `DELETE` | `/internal/jobs/{job_id}` | — | trạng thái sau hủy (`CANCELLED`); `409` nếu job không còn `PENDING` |
| `POST` | `/internal/jobs/{job_id}/retry` | — | `PENDING` (attempt reset); `409` nếu job không ở `FAILED` |

Quy tắc nguồn nội dung (đã hiện thực trong `routes.create_processing_job`):

- `document_type=cv`: cần **`file`** (PDF/DOCX/DOC — magic bytes + `max_upload_mb`) **hoặc** `text` (CV đã trích xuất → bỏ qua `EXTRACTING`).
- `document_type=jd`: cần **`text`** ≥ 10 ký tự; gửi kèm file → `400 JD_FILE_NOT_ALLOWED`.
- **Idempotency:** cùng `(document_type, document_id, version)` → trả job cũ với `deduplicated=true`; nếu job cũ `FAILED`/`CANCELLED` thì *requeue* với nguồn mới (attempt reset).

> Bản hiện tại nhận `file` trực tiếp qua multipart thay vì `fileRef` — đơn giản cho MVP, không cần
> storage dùng chung giữa NestJS và AI service. Khi NestJS có storage riêng (OQ2), thêm nhánh `fileRef`
> không đổi pipeline: worker chỉ cần `source_ref` là đường dẫn đọc được.

### 4.2 NestJS → người dùng (theo plan §10.1)

| Method | Endpoint | Ghi chú |
|---|---|---|
| `POST` | `/api/cvs/upload` | Lưu file + tạo record PG → gọi `/internal/jobs` → trả **`202 {cvId, jobId, status:"PENDING"}`** |
| `GET` | `/api/cvs/:id/processing-status` | Proxy `GET /internal/jobs/{jobId}` → `{status, step, progress, error?}` |
| `POST` | `/api/jobs` | Tương tự cho JD: tạo job → `202 {jobId}` |
| `POST` | `/api/admin/processing/:jobId/retry` | Gọi retry nội bộ |

`progress` suy ra từ `step` (`store.progress_for`): `RECEIVED 5% | EXTRACTING 25% | PARSING 55% | EMBEDDING 85% | FINALIZING 95% | READY 100%`.

### 4.3 Tương thích ngược

Giữ nguyên `POST /api/v1/ai/analyze-cv` (đồng bộ) cho: dev, test fixture (`apps/ai/scripts/make_fixture.py`), và tài liệu nhỏ. Endpoint async mới là `POST /internal/jobs`. Hai đường dùng chung exact pipeline (§5.2) — khác nhau chỉ ở việc chạy đồng bộ hay qua queue.

---

## 5. Worker & pipeline

### 5.1 Vị trí chạy (D2)

**Worker thread trong process FastAPI** (`apps/ai/app/main.py:create_app` + `lifespan`):

```
create_app():
  store   = make_store(settings)                     # SQLite: PROCESSING_DB_PATH
  worker  = ProcessingWorker(store, settings, llm)   # chưa chạy thread
lifespan:
  if settings.processing_worker_enabled: worker.start()
      -> store.requeue_stale(JOB_STALE_SECONDS, JOB_MAX_ATTEMPTS)   # rà job kẹt
  ...
  finally: worker.stop(); llm.close()
```

- `POST /internal/jobs` → `store.create()` → `worker.submit(job_id)` (chỉ đánh thức; DB là nguồn sự thật).
- Worker loop: `claim_next()` → pipeline → `mark_ready/mark_failed/requeue`; không có job thì
  `queue.get(timeout=JOB_POLL_INTERVAL_S)` rồi thử `claim_next()` lại.
- Retry có backoff **không** `sleep` chặn: `store.requeue(delay_s)` ghi `next_attempt_at`
  trong DB, `claim_next()` chỉ lấy job đã tới hạn ⇒ worker vẫn nhận job khác.
- Job gửi khi worker chưa chạy (hoặc service restart) không mất: vẫn `PENDING` trong DB.


**Vì sao không Redis/BullMQ:** repo thuần Python + CLI, 1 instance, job nhẹ (tần suất upload thấp); infra queue mới sẽ thêm điểm hỏng và khó test. Nút thắt thật sự là **Ollama + BGE-M3 trên CPU** — đã được giới hạn bằng concurrency (§5.4). Nếu sau này chạy đa instance → nâng cấp lên backend A (PG) + `SELECT ... FOR UPDATE SKIP LOCKED` là đủ, không đổi pipeline.

### 5.2 Pipeline từng bước (checkpoint = `step`)

```
1. claim job                        -> PROCESSING / step=RECEIVED, attempt_count += 1
2. content-hash: sha256(text đã chuẩn hoá) -> job.content_hash (audit/observability)
   Chống trùng ở mức job (§5.2): cùng (document_type, document_id, version) không tạo
   bản mới; NestJS tăng `version` khi nội dung đổi. (Bỏ qua embed lại theo hash khi
   version không đổi là tối ưu để sau — plan §9.4.)
3. EXTRACTING   text.read_upload + extract_text (worker chạy trong thread riêng, không cần threadpool)
                - chars < 50        -> lỗi vĩnh viễn "CV_TEXT_EMPTY"
                - chars > max_llm_chars -> cắt + warning TRUNCATED (đã có)
                - file nguồn biến mất -> lỗi vĩnh viễn "SOURCE_MISSING"
4. PARSING      llm.chat(SYSTEM_PROMPT, user_prompt) -> normalize_cv/normalize_job
                - mọi ApiError của Ollama (timeout / kết nối / JSON sai) -> TẠM THỜI, retry
5. EMBEDDING    recommend_mod.index([document])  # nhúng section + upsert Chroma + DocumentStore
                - metadata: status=READY, ownerId, visibility, content_hash, version=job.version
                - lỗi hạ tầng (embed/Chroma/ổ đĩa) -> TẠM THỜI, retry
6. FINALIZING   store.mark_ready(warnings) -> READY, finished_at, duration_ms; xoá file tạm
                log: job_id, doc, step cuối, attempts, duration_ms
```


Mỗi bước `update(step=...)` **trước khi** làm việc đó ⇒ nếu worker chết, `requeue_stale()` thấy job `PROCESSING` quá hạn và đưa về `PENDING` (mọi bước đều idempotency-safe nhờ `vector_id` theo version + upsert).

### 5.3 Claim atomic

- Backend PG: `UPDATE ... SET status='PROCESSING', startedAt=now()
  WHERE id = (SELECT id FROM DocumentProcessingJob
              WHERE status='PENDING' ORDER BY createdAt LIMIT 1
              FOR UPDATE SKIP LOCKED) RETURNING *`.
- Backend SQLite: `BEGIN IMMEDIATE` + `UPDATE ... WHERE id=? AND status='PENDING'`,
  kiểm tra `rowcount==1` (đủ cho 1 worker thread).

### 5.4 Giới hạn tài nguyên

| Hạn | Nguồn | Hành động vượt |
|---|---|---|
| File ≤ 10 MB | `MAX_UPLOAD_MB` (`config.py:55`) | `422` khi create job |
| ≤ 20 trang | `MAX_PAGES` | warning + cắt |
| ≤ 24 000 ký tự cho LLM | `MAX_LLM_CHARS` | cắt + `TRUNCATED` |
| OCR mặc định tắt | `OCR_ENABLED=false` | bật theo env |
| **Concurrency = 1 job/lần** | thiết kế | BGE-M3 + OCR chạy CPU, song song sẽ cán RAM/VRAM (plan §12.3 đo RAM) |
| Job kẹt > 300s | `JOB_STALE_SECONDS` → `requeue_stale` | về `PENDING` (hoặc `FAILED` nếu đã hết lượt) |
| Retry tối đa 3 | `JOB_MAX_ATTEMPTS` (plan §9.4) | quá → `FAILED` + cho retry thủ công |
| Backoff 2^n | `JOB_RETRY_BASE_DELAY_S` (1s → 1s, 2s, 4s) | lưu ở `next_attempt_at` |

---

## 6. Phân loại lỗi & retry

| Lỗi | Loại | Ví dụ | Xử lý |
|---|---|---|---|
| File hỏng / text rỗng / mã hoá | **Vĩnh viễn** | `CV_FILE_INVALID`, `CV_TEXT_EMPTY`, `CV_ENCRYPTED`, `CV_FILE_TOO_LARGE`, `SOURCE_MISSING` | `FAILED` ngay, không tốn lượt retry |
| Text/nguồn thiếu khi tạo job | **Vĩnh viễn (400/422)** | `CV_SOURCE_REQUIRED`, `JD_TEXT_REQUIRED` | chặn ngay ở endpoint, không tạo job |
| Ollama timeout / connection refused / JSON sai | **Tạm thời** | `LLM_UNAVAILABLE`, `LLM_TIMEOUT`, `LLM_INVALID_OUTPUT` | requeue + backoff, ≤ `JOB_MAX_ATTEMPTS` |
| Chroma/embed/ổ đĩa/bug nội bộ | **Tạm thời** | `INTERNAL: <ExceptionType>: <msg>` | requeue + backoff, ≤ `JOB_MAX_ATTEMPTS` |
| Job trùng `(document_type, document_id, version)` | **Không phải lỗi** | 2 request đua | trả job cũ, `deduplicated=true` (§5.2) |
| Worker chết giữa chừng | **Tạm thời** | job kẹt `PROCESSING` > `JOB_STALE_SECONDS` | `requeue_stale` → `PENDING` (giữ `attempt_count`); hết lượt → `FAILED` |

`last_error` = `{step}: {code}: {message}` (cắt 2000 ký tự, **không** chứa nội dung CV — plan §15).
Mã vĩnh viễn nằm ở `apps/ai/app/processing/pipeline.py:PERMANENT_CODES`; mọi `ApiError` khác và mọi
exception không lường trước được xếp loại *tạm thời*.

**Ollama-down (chốt OQ5 khi triển khai):** bản này **không** dùng fallback kiểu plan §8.5.
Ollama lỗi → coi là tạm thời → retry 3 lượt → `FAILED` kèm `last_error`; người dùng/admin có thể
gọi `POST /internal/jobs/{id}/retry`. Lý do: payload LLM là đầu vào bắt buộc của embedding theo
section — "READY mà không có payload" sẽ tạo vector kém chất lượng và match sai. Muốn fallback
structured thì thêm nhánh riêng + `warning=LLM_UNAVAILABLE`, không đổi state machine.

---

## 7. Tương tác với outbox (D4)

```
Upload mới (từ user) ──► worker pipeline ──► recommend_mod.index()   [trực tiếp]
Thay đổi từ PostgreSQL ──► outbox.jsonl ──► sync_worker ──► upsert    [giữ nguyên]
Xoá/hạn quyền ──────────► /internal/index/delete                     [giữ nguyên]
```

- **Không** cho upload mới đi qua outbox: sẽ phải serialize payload ra file rồi worker khác đọc lại — thêm trung gian vô ích, trong khi `recommend_mod.index()` đã là single-writer an toàn (upsert theo `vector_id` có version, tự xoá version cũ).
- Outbox giữ nguyên vai trò **reconciliation** plan §6.6: định kỳ `sync_worker --reconcile` so Chroma vs source → job nào lệch thì đưa về `PENDING` để xử lý lại.
- Nếu về sau dùng Backend A (PG), *khi đó* cân nhắc chuyển cả 2 nhánh sang transactional outbox cho đồng nhất — không cần từ đầu.

---

## 8. Bảo mật & quan sát

**Bảo mật (plan §15):**
- Mọi endpoint `/internal/*` (gồm `/internal/jobs*`) đi qua `require_internal_token`; `security.py` hiện **chỉ cảnh báo** khi `INTERNAL_API_KEY` chưa cấu hình ⇒ dev/test vẫn chạy được, nhưng **production bắt buộc đặt khoá mạnh** (đã có test `test_internal_key_required_when_configured`).
- Nội dung file/CV: không ghi vào log (chỉ log `job_id`, `doc`, `step`, `duration`); file tạm xóa sau khi xử lý (`text.temp_file` đã dùng context manager).
- Xem nội dung CV/JD là dữ liệu không đáng tin: đã có sẵn `clean_text`/prompt chống injection — `text.py` + `llm.py` làm rồi, pipeline chỉ không làm nặng thêm.
- Quyền: job chỉ được tạo cho `ownerId` người dùng; `visibility` ghi vào metadata Chroma như hiện tại (`access_where()`).

**Quan sát:**
- Log 1 dòng JSON/job: `{ts, job_id, doc_type, doc_id, version, step, status, attempts, duration_ms, error?}` → `logs/processing_YYYYMMDD.log`.
- Metric tối thiểu (để plan §12.3 báo cáo): `queue_depth`, `jobs_failed_total`, `processing_duration_p50/p95` — lấy từ record, không cần hệ metric riêng giai đoạn này.
- `/readyz` hiện chỉ ping Ollama (`main.py:43`) → thêm check store + queue depth.

---

## 9. Lộ trình triển khai — trạng thái

| Giai đoạn | Việc | Trạng thái |
|---|---|---|
| **P1** | `ProcessingStore` (SQLite) + state machine + unit test transitions/claim/retry | ✅ `app/processing/store.py`, `tests/test_processing_store.py` |
| **P2** | Pipeline worker tái dùng `text/llm/recommend_mod` + checkpoint `step` | ✅ `app/processing/pipeline.py`, `worker.py`, `tests/test_processing_worker.py` |
| **P3** | Endpoint `/internal/jobs*` + test API | ✅ `routes.py` (§4.1), `tests/test_processing_routes.py` — chặn bằng `INTERNAL_API_KEY` như toàn bộ `/internal/*` |
| **P4** | NestJS: `POST /api/cvs/upload` async + `GET .../processing-status` | ⛔ **ngoài repo này** (hợp đồng ở §4.2 đã sẵn) |
| **P5** | Backend PG (`DocumentProcessingJob`) + `FOR UPDATE SKIP LOCKED` | ⏳ chờ OQ1 (quyền ghi PG) |
| **P6** | Reconcile tự động outbox → `PENDING`, metric | ⏳ tuỳ chọn (outbox hiện có vẫn chạy) |

**Definition of Done ứng với phần này**

- [x] Job upload → xử lý nền, không block request (P2–P3; P4 phía NestJS)
- [x] Endpoint trạng thái trả `PENDING/PROCESSING/READY/FAILED` + `step` + `progress` (P3)
- [x] Job tạo → tự index (P2–P3; nhánh `text` cho JD/CV đã trích)
- [x] Retry có giới hạn + `last_error` + chống trùng theo bản ghi (§5.2)
- [x] Worker chết giữa chừng → requeue, không kẹt `PROCESSING` (P1, có test)
- [x] `/internal/jobs*` yêu cầu API key khi `INTERNAL_API_KEY` được cấu hình (P3)
- [ ] `content_hash` dùng để **bỏ qua embed lại** (hiện chỉ ghi nhận) — tối ưu sau

---

## 10. Open questions — trạng thái chốt

| # | Câu hỏi | Chốt khi triển khai | Ảnh hưởng còn lại |
|---|---|---|---|
| **OQ1** | AI service có **quyền ghi PostgreSQL** không? | **Chưa có** → dùng SQLite (`PROCESSING_DB_PATH`, mặc định `data/processing.db`). Interface `ProcessingStore` giữ nguyên để thay backend | P5: khi có quyền ghi, thêm `PostgresProcessingStore` + `SELECT … FOR UPDATE SKIP LOCKED` |
| **OQ2** | NestJS lưu file rồi gửi `fileRef`, hay đẩy bytes? | **Đẩy bytes qua multipart** (`file`) — AI service lưu tạm ở `PROCESSING_FILES_DIR`, xoá sau khi `READY`/`CANCELLED` | Khi NestJS có storage: thêm nhánh `fileRef`, không đổi pipeline |
| **OQ3** | JD gửi `text` thẳng (bỏ `EXTRACTING`)? | **Có** — `document_type=jd` bắt buộc `text`; CV cũng nhận `text` đã trích | — |
| **OQ4** | Giữ `analyze-cv` đồng bộ hay async-only? | **Giữ song song** (`POST /api/v1/ai/analyze-cv` không đổi) cho dev/fixture; async là `/internal/jobs` | Có thể deprecate khi NestJS chuyển hẳn |
| **OQ5** | Ollama sập → `FAILED` hay `READY` fallback? | **Retry tạm thời 3 lượt → `FAILED`** (không fallback structured), có retry thủ công | Nếu cần fallback §8.5: thêm nhánh riêng + warning |
| **OQ6** | `content_hash` dùng để skip re-embed? | **Chỉ ghi nhận** ở bản này; chống trùng theo `(document_type, document_id, version)` | Tối ưu sau khi biết tần suất re-upload cùng version |
