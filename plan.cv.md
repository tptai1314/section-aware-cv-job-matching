# KẾ HOẠCH XÂY DỰNG HỆ THỐNG AI JOB–CV RECOMMENDATION CHO FREVIA

**Dự án:** Frevia — Nền tảng kết nối Freelancer và Client  
**Tính năng:** AI Job–CV Recommendation hai chiều  
**LLM:** Qwen 4B  
**Embedding Model:** BGE-M3 (mô hình baseline đề xuất)  
**Vector Database:** ChromaDB  
**Database nghiệp vụ:** PostgreSQL  
**ORM:** Prisma  
**Backend:** NestJS  
**AI Service:** Python + FastAPI  
**Thời gian dự kiến:** 8 tuần  
**Phương pháp matching chính:** Section-aware Multi-vector + Hybrid Ranking  
**Định hướng nghiên cứu:** Hybrid Retrieval, Ranking và Explainable Recommendation

---

# 1. Mục tiêu dự án

## 1.1. Mục tiêu chính

Xây dựng hệ thống AI Recommendation cho Frevia, sử dụng dữ liệu CV của freelancer và dữ liệu job của client để tìm kiếm, đánh giá và đề xuất những cặp CV–job phù hợp nhất.

Hệ thống phải hỗ trợ hai chiều:

### Chiều 1: Freelancer → Jobs

Khi freelancer tải CV lên hoặc mở trang Recommended Jobs, hệ thống:

1. Đọc và xử lý CV.
2. Tách CV thành các section (kỹ năng, kinh nghiệm, học vấn, tóm tắt, ...) và tạo một embedding vector cho mỗi section cùng một vector full-text.
3. Tìm các job có ngữ nghĩa liên quan trong ChromaDB.
4. Kiểm tra trạng thái và điều kiện của từng job.
5. Tính điểm phù hợp bằng thuật toán Hybrid Ranking.
6. Sử dụng Qwen 4B để giải thích hoặc hỗ trợ xếp hạng lại các job tiềm năng.
7. Trả về danh sách job được sắp xếp theo mức độ phù hợp.

### Chiều 2: Client → Freelancers/CVs

Khi client tạo job hoặc mở trang Recommended Freelancers, hệ thống:

1. Đọc và phân tích job description.
2. Trích xuất kỹ năng bắt buộc, kỹ năng ưu tiên và điều kiện công việc.
3. Tách job thành các section và tạo một embedding vector cho mỗi section cùng một vector full-text.
4. Tìm các CV có ngữ nghĩa liên quan trong ChromaDB.
5. Kiểm tra quyền chia sẻ CV và trạng thái freelancer.
6. Tính điểm phù hợp dựa trên kỹ năng, kinh nghiệm, ngân sách và các yếu tố liên quan.
7. Sử dụng Qwen 4B để giải thích hoặc hỗ trợ reranking.
8. Trả về danh sách freelancer phù hợp nhất.

## 1.2. Kết quả cần đạt được

- Tự động xử lý CV/job khi người dùng upload hoặc cập nhật.
- Chuyển CV/job thành embedding vector.
- Lưu vector vào ChromaDB.
- Sử dụng vector similarity search để tìm các candidate tiềm năng.
- Kết hợp semantic search với keyword search và dữ liệu có cấu trúc.
- Xây dựng thuật toán Hybrid Ranking.
- Tích hợp Qwen 4B vào pipeline recommendation.
- Hiển thị điểm phù hợp và giải thích có bằng chứng.
- Ghi nhận phản hồi của người dùng.
- Có benchmark để đo chất lượng retrieval và ranking.
- Có khả năng phát triển thành paper khoa học.

---

# 2. Kiến trúc hệ thống

## 2.1. Sơ đồ tổng thể

```text
                FRE VIA WEB APPLICATION
                          |
             +------------+------------+
             |                         |
       Freelancer                   Client
        Upload CV                  Create Job
             |                         |
             +------------+------------+
                          |
                     NestJS API
                          |
                   Ingestion Pipeline
                          |
              +-----------+-----------+
              |                       |
         CV Processor            Job Processor
              |                       |
              +-----------+-----------+
                          |
Text Extraction / Cleaning
                           |
                Section Splitting / Normalization
                           |
                Structured Information
                           |
             +------------+-------------+
             |                          |
        PostgreSQL                Python AI Service
        Business Data             FastAPI / Worker
                                         |
                           +-------------+-------------+
                           |                           |
                    Embedding Model                 Qwen 4B
                      BGE-M3                 Extraction / Reranking /
                          |                      Explanation
              CV/Job Section Vectors
                          |
                       ChromaDB
                          |
                    Vector Search
                          |
                   Candidate Retrieval
                          |
                 PostgreSQL Validation
                          |
                    Hybrid Ranking
                          |
                  Optional Qwen Reranking
                          |
                    Top-K Results
                          |
                 Qwen Explanation
                          |
                    NestJS API
                          |
                  Frevia Recommendation UI
                          |
                  User Feedback / Logs
                          |
                     PostgreSQL
```

## 2.2. Vai trò của từng thành phần

| Thành phần | Vai trò |
|---|---|
| Next.js | Giao diện upload CV, tạo job và hiển thị recommendation |
| NestJS | API, authentication, authorization, nghiệp vụ và điều phối recommendation |
| PostgreSQL | Lưu dữ liệu user, CV, job, skills, application và lịch sử recommendation |
| Prisma | Quản lý schema và truy vấn PostgreSQL |
| Python FastAPI | Cung cấp dịch vụ embedding và inference AI |
| BGE-M3 | Chuyển từng section của CV/job (kỹ năng, kinh nghiệm, ...) thành vector |
| ChromaDB | Lưu vector, metadata và thực hiện vector similarity search |
| Qwen 4B | Phân tích CV/job, hỗ trợ reranking và tạo lời giải thích |
| Background Worker | Xử lý upload, embedding, retry và đồng bộ vector |
| Redis/BullMQ | Hàng đợi background job nếu cần |
| Hybrid Ranking Engine | Kết hợp section-aware semantic score và các yếu tố nghiệp vụ |

## 2.3. Nguyên tắc thiết kế

1. PostgreSQL là nguồn dữ liệu nghiệp vụ chính.
2. ChromaDB là hệ thống lưu trữ và tìm kiếm vector.
3. Embedding model chịu trách nhiệm tạo vector.
4. ChromaDB không tự hiểu CV/job; nó tìm kiếm dựa trên vector được cung cấp.
5. Qwen 4B có thể tham gia recommendation thông qua phân tích và reranking, nhưng không bắt buộc phải xử lý mọi tài liệu trong database.
6. Hard constraints và quyền truy cập được kiểm tra bằng logic backend.
7. Mọi kết quả do LLM tạo ra phải được kiểm tra và không được xem là dữ kiện chắc chắn nếu chưa có bằng chứng.
8. Các thành phần phải có thể đánh giá độc lập thông qua benchmark.
9. Semantic matching ưu tiên biểu diễn đa vector theo section thay vì một vector cho toàn tài liệu.

---

# 3. Tại sao hệ thống cần cả Qwen 4B và Embedding Model?

## 3.1. Vấn đề khi chỉ sử dụng Qwen 4B

Qwen 4B là mô hình ngôn ngữ sinh văn bản. Nó có thể đọc CV và job rồi đánh giá mức độ phù hợp, nhưng nếu phải so sánh trực tiếp với hàng nghìn tài liệu trong mỗi request thì sẽ có những vấn đề:

- Phải thực hiện nhiều lần inference.
- Tăng thời gian phản hồi.
- Tăng chi phí tính toán.
- Khó bảo đảm thứ hạng ổn định.
- Điểm số do LLM tạo ra có thể không được hiệu chuẩn.
- Không có sẵn chỉ mục vector để truy xuất nhanh trên tập tài liệu lớn.

Vì vậy, không nên dùng Qwen làm công cụ duy nhất để quét toàn bộ database.

Tuy nhiên, Qwen vẫn có thể tham gia vào giai đoạn đánh giá một tập candidate nhỏ sau khi hệ thống đã truy xuất được các CV/job tiềm năng.

## 3.2. Vấn đề khi chỉ sử dụng Vector Similarity

Embedding và vector search tìm kiếm văn bản có ngữ nghĩa gần nhau, nhưng không trực tiếp chứng minh rằng freelancer đáp ứng mọi yêu cầu của job.

Ví dụ:

- CV A có ngữ nghĩa gần job nhưng thiếu một kỹ năng bắt buộc.
- CV B có similarity thấp hơn một chút nhưng đáp ứng yêu cầu kỹ thuật và ngân sách tốt hơn.

Nếu chỉ sắp xếp theo similarity, CV A có thể đứng trên CV B dù không phải lựa chọn phù hợp nhất.

Do đó, cần có tầng Hybrid Ranking và có thể thêm Qwen reranking.

## 3.3. Phân chia trách nhiệm

| Thành phần | Trả lời câu hỏi |
|---|---|
| Embedding model | Mỗi section của CV/job được biểu diễn như thế nào trong không gian vector? |
| ChromaDB | Những vector nào gần query vector nhất? |
| Hard filters | Những đối tượng nào đủ điều kiện được xem xét? |
| Hybrid Ranking | Những candidate nào phù hợp nhất theo các yếu tố nghiệp vụ? |
| Qwen reranker | Với một cặp CV–job cụ thể, mức độ phù hợp được phân tích ra sao? |
| Qwen explanation | Vì sao kết quả được đề xuất và bằng chứng nào hỗ trợ kết luận đó? |

---

# 4. Embedding Model

## 4.1. Model đề xuất

Sử dụng BGE-M3 làm embedding baseline ban đầu.

Lý do:

- Hỗ trợ văn bản đa ngôn ngữ.
- Có thể sử dụng cho dữ liệu tiếng Việt và tiếng Anh.
- Phù hợp với bài toán semantic retrieval.
- Có thể dùng để tạo vector cho cả CV và job.
- Có thể benchmark với các embedding model khác.

Đây là lựa chọn ban đầu, không phải kết luận rằng BGE-M3 chắc chắn tốt nhất cho Frevia.

## 4.2. Biểu diễn đa vector theo section

Mỗi CV/job không được nhúng thành một vector duy nhất mà được tách thành các section rồi nhúng từng phần:

1. Tách tài liệu thành các section bằng quy tắc (rule-based): `skills`, `experience`, `education`, `summary`. Danh sách section có thể cấu hình, không hardcode.
2. Mỗi section được nhúng thành một vector riêng.
3. Giữ thêm một vector full-text của toàn tài liệu dùng cho candidate retrieval.

Một tài liệu ban đầu → N vector section + 1 vector full-text.

Lý do dùng đa vector theo section:

- Đối sánh từng section cho ra điểm thành phần theo section, tự nhiên giải thích được vì sao một CV/job được xếp hạng cao hay thấp.
- Ghép cặp chéo (cross-pairing) xử lý được trường hợp CV và JD không sắp xếp section trùng khớp 1–1.
- Tập trung tín hiệu: kỹ năng nằm trong section `skills`, không bị "loãng" bởi toàn bộ văn bản.
- Có thể so sánh được với baseline một vector và chunk max-sim để đo lợi ích thực sự của việc tách section.

## 4.3. Quy trình embedding

Với mỗi CV hoặc job:

1. Lấy nội dung đã được trích xuất.
2. Chuẩn hóa văn bản.
3. Tách section và tạo các `embedding_text` riêng cho từng section cùng một bản cho full-text.
4. Gửi từng văn bản đến embedding service.
5. Nhận các vector tương ứng.
6. Kiểm tra dimension và dữ liệu đầu ra.
7. Upsert các vector vào ChromaDB với ID ổn định theo section.
8. Lưu mapping section ↔ vector và thông tin phiên bản vào PostgreSQL.

Embedding của query và collection phải được tạo bằng các checkpoint tương thích và cùng không gian vector.

Không so sánh trực tiếp vector từ hai embedding model khác nhau nếu chưa có một phương pháp ánh xạ được xác định và đánh giá.

## 4.4. Embedding representation

Mỗi tài liệu có ba dạng dữ liệu:

- `raw_file`: CV gốc hoặc nội dung job gốc.
- `structured_data`: các trường có cấu trúc trong PostgreSQL.
- `embedding_text`: nội dung được thiết kế để tạo vector — **một bản cho mỗi section** và một bản cho full-text.

Ví dụ embedding text theo section cho CV:

```text
Document type: Freelancer CV.
Section: skills.

Node.js, TypeScript, NestJS, PostgreSQL.
```

```text
Document type: Freelancer CV.
Section: experience.

Developed REST APIs using NestJS and PostgreSQL.
Built marketplace features using Next.js.
```

Ví dụ embedding text theo section cho job:

```text
Document type: Freelance Job.
Section: responsibilities.

Build REST APIs, maintain backend services,
design database schemas and write automated tests.
```

```text
Document type: Freelance Job.
Section: required_skills.

Node.js, TypeScript, PostgreSQL.
```

```text
Document type: Freelance Job.
Section: experience_level.

Junior. Remote work.
```

Định dạng section và nội dung nhúng sẽ được chốt trong benchmark spec.

Các ví dụ trên chỉ minh họa cách biểu diễn dữ liệu.

Không đưa thông tin cá nhân nhạy cảm vào embedding text nếu không cần thiết.

---

# 5. Thuật toán Vector Similarity

## 5.1. Biểu diễn toán học

Gọi:

- \(u\): vector embedding của CV.
- \(v\): vector embedding của job.
- \(d\): số chiều của vector.

Mức độ tương đồng được tính bằng cosine similarity:

**Cosine Similarity:**

<div align="center">
  <b>sim(u, v) = (u · v) / (‖u‖ × ‖v‖)</b>
</div>

Trong đó:

**Tích vô hướng:**

<div align="center">
  <b>u · v = ∑<sub>i=1</sub><sup>d</sup> u<sub>i</sub>v<sub>i</sub></b>
</div>

Và:

**Độ dài vector:**

<div align="center">
  <b>‖u‖ = √(∑<sub>i=1</sub><sup>d</sup> u<sub>i</sub><sup>2</sup>)</b>
</div>


Cosine similarity đo mức độ tương đồng về hướng giữa hai vector.

Giá trị lý thuyết nằm trong khoảng từ -1 đến 1. Điểm cao hơn thường thể hiện ngữ nghĩa tương đồng hơn khi dùng một embedding model phù hợp.

Không được xem cosine similarity là xác suất một freelancer sẽ được tuyển dụng.

### Công thức tính Cosine Similarity

Cosine Similarity đo độ tương đồng giữa hai vector embedding, được tính bằng công thức:

**Cosine Similarity = Tích vô hướng của hai vector / (Độ dài vector thứ nhất × Độ dài vector thứ hai)**

**Công thức dạng ký hiệu:**

`sim(u, v) = (u · v) / (||u|| × ||v||)`

Trong đó:

- `u`, `v`: Hai vector embedding, ví dụ vector CV của freelancer và vector mô tả công việc.
- `u · v`: Tích vô hướng của hai vector.
- `||u||`: Độ dài của vector `u`.
- `||v||`: Độ dài của vector `v`.

**Cách tính tích vô hướng:**

`u · v = u₁v₁ + u₂v₂ + ... + u_dv_d`

**Cách tính độ dài vector:**

`||u|| = √(u₁² + u₂² + ... + u_d²)`

`||v|| = √(v₁² + v₂² + ... + v_d²)`

Trong đó `d` là số chiều của vector embedding.

**Ý nghĩa kết quả:**

- Gần `1`: Hai vector tương đồng cao.
- Gần `0`: Hai vector ít tương đồng theo hướng.
- Gần `-1`: Hai vector ngược hướng.

Trong hệ thống Frevia, Cosine Similarity được dùng để so sánh embedding của CV freelancer với embedding của mô tả công việc. Điểm càng cao thường cho thấy mức độ tương đồng ngữ nghĩa càng lớn, nhưng không đồng nghĩa với việc freelancer chắc chắn phù hợp với công việc.


## 5.2. Quy trình tính similarity

Ví dụ, freelancer cần tìm job:

1. Lấy CV đang hoạt động.
2. Lấy các embedding vector của CV (vector full-text + từng section).
3. Truy vấn collection `frevia_jobs` bằng vector full-text để lấy candidate pool.
4. Tính hoặc tìm các vector gần nhất theo distance metric đã cấu hình.
5. Trả về top 100 job candidate.
6. Kiểm tra trạng thái và quyền truy cập.
7. Tính điểm semantic theo section (xem mục 5.6) và đưa candidate vào Hybrid Ranking.

Đối với client tìm freelancer, thực hiện ngược lại bằng cách truy vấn collection `frevia_cvs` với full-text vector của job.

## 5.3. Thuật toán tìm kiếm vector

### Exact nearest-neighbor search

Tính khoảng cách từ query vector đến từng vector trong tập dữ liệu.

Ưu điểm:

- Có thể tìm kết quả chính xác theo distance metric đã chọn.
- Phù hợp để làm baseline hoặc kiểm chứng kết quả ANN.

Nhược điểm:

- Chi phí tìm kiếm tăng khi số lượng vector lớn.

### Approximate nearest-neighbor search

Sử dụng chỉ mục tìm kiếm lân cận gần đúng để giảm thời gian truy vấn.

Với ChromaDB, HNSW là một lựa chọn quan trọng cần benchmark.

HNSW xây dựng đồ thị nhiều tầng, trong đó các vector được liên kết để hỗ trợ điều hướng nhanh đến vùng lân cận của query vector.

Ưu điểm:

- Tăng tốc truy vấn trên tập vector lớn.
- Phù hợp cho semantic retrieval có độ trễ thấp.

Nhược điểm:

- Có thể bỏ sót một số nearest neighbors so với exact search.
- Cần đánh giá recall, latency và chi phí bộ nhớ.
- Hiệu năng phụ thuộc kích thước dữ liệu và cấu hình index.

Không cần tự viết thuật toán HNSW. Sử dụng khả năng lập chỉ mục của ChromaDB và benchmark cấu hình phù hợp.

## 5.4. Distance và similarity

ChromaDB có thể trả về distance thay vì cosine similarity trực tiếp.

Cần:

1. Xác nhận distance metric của collection.
2. Kiểm tra ý nghĩa của giá trị trả về theo phiên bản ChromaDB đang dùng.
3. Chuẩn hóa score trong recommendation engine.
4. Không giả định mọi loại distance đều có thể chuyển thành similarity bằng cùng một công thức.

Nếu cần so sánh score của nhiều embedding model, hãy đánh giá cách chuẩn hóa phù hợp trên validation set.

## 5.5. Candidate Retrieval

Giá trị ban đầu để benchmark:

- Vector retrieval: top 100.
- Hybrid ranking: top 50–100.
- Qwen reranking nếu bật: top 20–50.
- Kết quả cuối cùng: top 5–10.

Đây là cấu hình ban đầu. Các giá trị cần được điều chỉnh theo Recall@K và latency.

## 5.6. Điểm semantic theo section (Section-aware Semantic Score)

Điểm semantic của một cặp (job, CV) không phải một cosine duy nhất mà được phân rã theo section.

Gọi:

- \(U_s\): vector embedding của section \(s\) trong job (JD).
- \(V_t\): vector embedding của section \(t\) trong CV.
- \(S\): tập các section của JD; \(T\): tập các section của CV.

Cosine similarity giữa hai section:

<div align="center">
  <b>sim(U<sub>s</sub>, V<sub>t</sub>) = (U<sub>s</sub> · V<sub>t</sub>) / (‖U<sub>s</sub>‖ × ‖V<sub>t</sub>‖)</b>
</div>

**Ghép cặp cố định (fixed):** mỗi section của JD chỉ so với section cùng tên trong CV:

<div align="center">
  <b>component<sub>s</sub> = sim(U<sub>s</sub>, V<sub>s</sub>) nếu V<sub>s</sub> tồn tại, ngược lại 0</b>
</div>

**Ghép cặp chéo (cross):** mỗi section của JD lấy giá trị cao nhất so với mọi section của CV:

<div align="center">
  <b>component<sub>s</sub> = max<sub>t ∈ T</sub> sim(U<sub>s</sub>, V<sub>t</sub>)</b>
</div>

**Tổng hợp:**

<div align="center">
  <b>S<sub>semantic</sub> = mean<sub>s ∈ S</sub> component<sub>s</sub>  (hoặc max<sub>s ∈ S</sub> component<sub>s</sub>)</b>
</div>

Các thành phần `component_s` đồng thời là điểm giải thích: biết CV/job mạnh ở section nào, yếu ở section nào. Ghép cặp chéo nên được ưu tiên so với ghép cặp cố định khi CV và JD không sắp xếp section khớp 1–1, nhưng cần kiểm tra trong ablation để tránh việc một section "thắng" nhiều section JD làm lệch điểm.

---

# 6. Thiết kế ChromaDB

## 6.1. Collections

Tạo hai collection:

- `frevia_cvs`
- `frevia_jobs`

Collection CV lưu vector và metadata phục vụ tìm freelancer.

Collection job lưu vector và metadata phục vụ tìm công việc.

## 6.2. Metadata

Các trường có thể lưu:

- `document_id`
- `document_type`
- `owner_id`
- `document_version`
- `section` — tên section mà vector này đại diện, `_full` cho full-text
- `status`
- `visibility`
- `language`
- `embedding_model`
- `content_hash`

Một tài liệu tương ứng với nhiều vector (một vector mỗi section + một vector full-text), phân biệt bằng trường `section`. ID vector ổn định theo dạng `cv:123:v1:skills`, `cv:123:v1:_full`, ...

Chỉ lưu các metadata cần thiết cho truy vấn và quản lý index.

Không dùng metadata trong ChromaDB thay cho việc kiểm tra quyền ở PostgreSQL.

## 6.3. Ví dụ ChromaDB

```python
import chromadb

client = chromadb.PersistentClient(
    path="./data/chroma"
)

cv_collection = client.get_or_create_collection(
    name="frevia_cvs",
    metadata={"hnsw:space": "cosine"}
)

job_collection = client.get_or_create_collection(
    name="frevia_jobs",
    metadata={"hnsw:space": "cosine"}
)
```

Đây là ví dụ khởi đầu. Cần kiểm tra khả năng hỗ trợ và cấu hình theo phiên bản ChromaDB được cài đặt.

## 6.4. Upsert CV

Với mỗi section và full-text, một vector và một metadata riêng:

```python
cv_collection.upsert(
    ids=["cv:123:v1:skills", "cv:123:v1:_full"],
    embeddings=[cv_skills_embedding, cv_full_embedding],
    documents=[
        "Document type: Freelancer CV. Section: skills. Node.js, TypeScript, ...",
        cv_full_embedding_text,
    ],
    metadatas=[
        {
            "document_id": "123",
            "document_type": "cv",
            "owner_id": "freelancer_001",
            "document_version": 1,
            "section": "skills",
            "status": "READY",
            "embedding_model": "BAAI/bge-m3",
            "content_hash": "example_hash"
        },
        {
            "document_id": "123",
            "document_type": "cv",
            "owner_id": "freelancer_001",
            "document_version": 1,
            "section": "_full",
            "status": "READY",
            "embedding_model": "BAAI/bge-m3",
            "content_hash": "example_hash"
        }
    ]
)
```

`cv_skills_embedding`, `cv_full_embedding` phải là các vector thực tế được tạo từ embedding service.

## 6.5. Query jobs

Candidate retrieval dùng vector full-text của CV:

```python
results = job_collection.query(
    query_embeddings=[cv_full_embedding],
    n_results=100,
    where={"status": "READY"},
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)
```

Sau khi có candidate pool, lấy các vector theo section của từng job candidate để tính điểm semantic theo section (mục 5.6):

```python
section_vectors = job_collection.get(
    ids=[f"job:{jid}:v{ver}:{section}" for jid in candidate_job_ids for section in sections],
    include=["embeddings"],
)
```

Sau khi truy vấn:

1. Lấy các ID candidate.
2. Truy vấn PostgreSQL để kiểm tra job còn mở hay không.
3. Kiểm tra điều kiện và quyền.
4. Loại candidate không hợp lệ.
5. Tính Hybrid Ranking.
6. Trả về kết quả cuối cùng.

## 6.6. Đồng bộ PostgreSQL và ChromaDB

Hai hệ thống không chia sẻ một transaction nguyên tử.

Để tránh tình trạng PostgreSQL đã cập nhật nhưng ChromaDB vẫn chứa vector cũ:

1. Ghi thay đổi nghiệp vụ và sự kiện đồng bộ vào PostgreSQL.
2. Background worker nhận sự kiện.
3. Tách section và tạo lại toàn bộ embedding (section + full-text) của tài liệu.
4. Upsert đồng bộ tất cả vector của tài liệu vào ChromaDB (thay thế các vector cũ theo `document_id` + `document_version`).
5. Xóa các vector thuộc version cũ không còn dùng.
6. Ghi nhận kết quả đồng bộ.
7. Retry khi lỗi.
8. Chạy reconciliation định kỳ để phát hiện dữ liệu lệch.

Nên dùng transactional outbox nếu cần đảm bảo khả năng phục hồi.

Khi tài liệu bị xóa hoặc mất quyền chia sẻ, backend phải ngăn nó xuất hiện trong recommendation ngay cả khi vector cũ chưa được xóa khỏi ChromaDB.

---

# 7. Hybrid Ranking Engine

## 7.1. Vì sao cần Hybrid Ranking?

Vector similarity chỉ cho biết độ tương đồng ngữ nghĩa.

Recommendation cần đánh giá thêm:

- Kỹ năng bắt buộc.
- Kỹ năng ưu tiên.
- Kinh nghiệm liên quan.
- Mức ngân sách.
- Hình thức làm việc.
- Ngôn ngữ.
- Thời gian sẵn sàng làm việc.

Do đó, hệ thống sẽ dùng vector search để tạo candidate pool và Hybrid Ranking để sắp xếp kết quả.

## 7.2. Công thức ban đầu

### Công thức Hybrid Ranking

**Điểm xếp hạng tổng hợp:**

<div align="center">
  <b>S<sub>hybrid</sub> = 0.40 × S<sub>semantic</sub> + 0.30 × S<sub>skills</sub> + 0.15 × S<sub>experience</sub> + 0.10 × S<sub>budget</sub> + 0.05 × S<sub>preferences</sub></b>
</div>

Trong đó:

- **S_semantic (40%):** Điểm tương đồng ngữ nghĩa từ phương pháp section-aware — đối sánh từng section bằng ghép cặp fixed/cross rồi tổng hợp mean/max (xem mục 5.6).
- **S_skills (30%):** Mức độ bao phủ các kỹ năng yêu cầu của công việc.
- **S_experience (15%):** Mức độ phù hợp về kinh nghiệm làm việc.
- **S_budget (10%):** Mức độ tương thích giữa ngân sách của khách hàng và mức giá freelancer đề xuất.
- **S_preferences (5%):** Mức độ phù hợp về sở thích và điều kiện làm việc.

Tất cả các thành phần điểm nên được chuẩn hóa về cùng thang điểm từ 0 đến 1 trước khi tính toán.

**Ví dụ:**

Giả sử một freelancer có các điểm thành phần:

- S_semantic = 0.90
- S_skills = 0.80
- S_experience = 0.70
- S_budget = 1.00
- S_preferences = 0.80

Khi đó:

<div align="center">
  <b>S<sub>hybrid</sub> = 0.40 × 0.90 + 0.30 × 0.80 + 0.15 × 0.70 + 0.10 × 1.00 + 0.05 × 0.80</b>
</div>

<div align="center">
  <b>S<sub>hybrid</sub> = 0.835 = 83.5%</b>
</div>

Freelancer này đạt điểm phù hợp tổng hợp là **83.5%** theo bộ trọng số đang sử dụng.

Tất cả thành phần cần có quy tắc chuẩn hóa rõ ràng.

Trọng số chỉ là baseline ban đầu. Không được khẳng định đây là trọng số tối ưu trước khi thực nghiệm.

## 7.3. Skill Score

Một baseline:

### Công thức tính Skill Matching Score

**Điểm phù hợp về kỹ năng:**

<div align="center">
  <b>S<sub>skills</sub> = (Tổng trọng số kỹ năng phù hợp) / (Tổng trọng số kỹ năng bắt buộc)</b>
</div>

Viết dưới dạng ký hiệu toán học:

<div align="center">
  <b>S<sub>skills</sub> = [∑<sub>k ∈ K<sub>matched</sub></sub> w<sub>k</sub>] / [∑<sub>k ∈ K<sub>required</sub></sub> w<sub>k</sub>]</b>
</div>

Trong đó:

- **K_required:** Tập hợp các kỹ năng bắt buộc của công việc.
- **K_matched:** Tập hợp các kỹ năng bắt buộc được xác định là phù hợp với kỹ năng của freelancer.
- **w_k:** Trọng số của kỹ năng k, thể hiện mức độ quan trọng của kỹ năng đó.

**Ví dụ:**

Một công việc yêu cầu ba kỹ năng:

| Kỹ năng | Trọng số | Freelancer đáp ứng? |
|---|---:|---|
| TypeScript | 3 | Có |
| NestJS | 2 | Có |
| Docker | 1 | Không |

Tổng trọng số kỹ năng bắt buộc = 3 + 2 + 1 = 6.

Tổng trọng số kỹ năng phù hợp = 3 + 2 = 5.

Vậy:

<div align="center">
  <b>S<sub>skills</sub> = 5 / 6 ≈ 0.8333 = 83.33%</b>
</div>

Freelancer đạt **83.33% điểm phù hợp về kỹ năng**.

**Lưu ý:** Nếu công việc không có kỹ năng bắt buộc, cần quy định cách xử lý riêng để tránh phép chia cho 0.


Nếu không có kỹ năng bắt buộc, cần định nghĩa quy tắc xử lý riêng để tránh chia cho 0.

Phải phân biệt kỹ năng bắt buộc và kỹ năng ưu tiên.

## 7.4. Hard Constraints

Không đưa tất cả điều kiện vào điểm số.

Các điều kiện cần kiểm tra riêng:

- Job còn mở.
- CV được phép chia sẻ.
- User có quyền truy cập.
- Tài liệu không bị xóa.
- Các điều kiện bắt buộc được đáp ứng theo chính sách sản phẩm.

Nếu một điều kiện bắt buộc không được đáp ứng, candidate có thể bị loại hoặc đánh dấu rõ ràng tùy quy tắc nghiệp vụ.

Không để similarity cao bù trừ cho vi phạm quyền truy cập hoặc điều kiện bắt buộc.

## 7.5. Hai chiều ranking

### Freelancer → Jobs

Đánh giá:

- Semantic similarity theo section giữa CV và job.
- Kỹ năng CV đáp ứng yêu cầu job.
- Kinh nghiệm phù hợp.
- Ngân sách job so với expected rate.
- Work type và sở thích.

### Client → Freelancers

Đánh giá:

- Semantic similarity theo section giữa job và CV.
- Kỹ năng freelancer đáp ứng yêu cầu.
- Kinh nghiệm liên quan.
- Rate của freelancer so với ngân sách.
- Availability và điều kiện làm việc.

Có thể dùng chung phần lớn scoring components nhưng trọng số cần được đánh giá riêng cho từng hướng.

---

# 8. Tích hợp Qwen 4B vào Recommendation

## 8.1. Chiến lược A — Qwen giải thích kết quả

Pipeline:

```text
CV/Job
   |
ChromaDB Retrieval (full-text vector)
   |
Section-aware Scoring
   |
Hybrid Ranking
   |
Top 10
   |
Qwen 4B Explanation
   |
Final Recommendation UI
```

Qwen tạo:

- Tóm tắt mức phù hợp.
- Kỹ năng phù hợp.
- Yêu cầu còn thiếu.
- Bằng chứng trong CV/job.
- Những thông tin chưa xác định.

Đây là chiến lược nên triển khai đầu tiên.

## 8.2. Chiến lược B — Qwen tham gia reranking

Pipeline:

```text
CV/Job
   |
ChromaDB Retrieval: Top 100
   |
PostgreSQL Validation
   |
Section-aware Scoring
   |
Hybrid Ranking: Top 20
   |
Qwen 4B Pairwise/Pointwise Evaluation
   |
Final Top 10
   |
Evidence-Grounded Explanation
```

Qwen nhận một cặp CV–job và đánh giá dựa trên rubric cố định:

- Kỹ năng bắt buộc.
- Kỹ năng ưu tiên.
- Kinh nghiệm.
- Trách nhiệm công việc.
- Thiếu hụt thông tin.
- Mức độ phù hợp tổng thể.

Qwen có thể trả về nhãn relevance hoặc điểm theo thang quy định.

Điểm do LLM sinh ra không được coi là xác suất đã hiệu chuẩn.

Nếu muốn reranking ổn định hơn, cần so sánh pointwise scoring, pairwise ranking và một cross-encoder chuyên dụng khi có đủ nguồn lực.

## 8.3. Rubric reranking

Ví dụ rubric:

| Thành phần | Điểm tối đa |
|---|---:|
| Kỹ năng bắt buộc | 40 |
| Kỹ năng ưu tiên | 20 |
| Kinh nghiệm liên quan | 20 |
| Phù hợp trách nhiệm công việc | 10 |
| Điều kiện bổ sung | 10 |
| Tổng | 100 |

Rubric này chỉ là cấu hình thử nghiệm. Hard constraints vẫn phải do backend xử lý.

Yêu cầu Qwen trả về JSON có cấu trúc, bao gồm điểm từng thành phần, bằng chứng và thông tin còn thiếu.

## 8.4. Đánh giá Qwen reranking

So sánh:

1. Dense retrieval.
2. Hybrid Ranking.
3. Hybrid Ranking + Qwen reranking.
4. Hybrid Ranking + cross-encoder reranking nếu có thể.

Metrics:

- NDCG@10.
- MRR@10.
- Precision@10.
- p50/p95 latency.
- Tỷ lệ JSON hợp lệ.
- Tỷ lệ kết luận không có bằng chứng.

Chỉ giữ Qwen reranking trong production nếu chất lượng cải thiện đủ để biện minh cho độ trễ và tài nguyên bổ sung.

## 8.5. Fallback

Nếu Qwen không khả dụng:

- Vẫn trả về kết quả Hybrid Ranking.
- Đánh dấu explanation chưa sẵn sàng.
- Retry việc tạo explanation ở background nếu cần.
- Không để lỗi LLM làm hỏng toàn bộ recommendation API.

---

# 9. Quy trình xử lý CV và Job

## 9.1. CV Upload

1. Xác thực freelancer.
2. Kiểm tra định dạng và kích thước file.
3. Lưu file gốc.
4. Tạo CV record trong PostgreSQL.
5. Đưa tác vụ vào background queue.
6. Trích xuất văn bản từ PDF/DOCX.
7. OCR nếu cần.
8. Chuẩn hóa văn bản.
9. Trích xuất cấu trúc CV.
10. Tách section (rule-based).
11. Validate dữ liệu.
12. Tạo embedding cho từng section + full-text.
13. Upsert các vector vào ChromaDB.
14. Cập nhật trạng thái `READY`.
15. Ghi log.

## 9.2. Job Creation

1. Xác thực client.
2. Kiểm tra dữ liệu job.
3. Lưu job trong PostgreSQL.
4. Chuẩn hóa job description.
5. Trích xuất required/optional skills.
6. Tách section (rule-based).
7. Tạo embedding cho từng section + full-text.
8. Upsert các vector vào ChromaDB.
9. Cập nhật trạng thái tìm kiếm.
10. Ghi log.

## 9.3. Trạng thái xử lý

Sử dụng:

- `PENDING`
- `PROCESSING`
- `READY`
- `FAILED`

Trạng thái phải được lưu trong PostgreSQL.

## 9.4. Retry và idempotency

- Sử dụng ID ổn định cho vector (bao gồm cả định danh section).
- Dùng content hash để tránh embedding lại khi không cần thiết.
- Retry lỗi tạm thời.
- Giới hạn số lần retry.
- Ghi lỗi cuối cùng.
- Cho phép reindex.
- Không tạo vector trùng khi worker chạy lại.
- Khi tài liệu đổi, thay thế toàn bộ vector cũ của tài liệu, không chỉ một vector.

---

# 10. API Design

## 10.1. NestJS endpoints

| Method | Endpoint | Chức năng |
|---|---|---|
| POST | `/api/cvs/upload` | Upload CV |
| GET | `/api/cvs/:id/processing-status` | Kiểm tra trạng thái |
| POST | `/api/jobs` | Tạo job |
| PATCH | `/api/jobs/:id` | Cập nhật job |
| GET | `/api/recommendations/jobs` | Freelancer tìm job |
| GET | `/api/recommendations/freelancers?jobId=...` | Client tìm freelancer |
| GET | `/api/recommendations/:id/explanation` | Lấy giải thích |
| POST | `/api/recommendations/:id/feedback` | Ghi nhận phản hồi |
| POST | `/api/admin/recommendations/reindex` | Reindex có kiểm soát |

Các endpoint phải tuân theo authentication, role và ownership hiện có của Frevia.

## 10.2. Python AI service

Các endpoint nội bộ có thể bao gồm:

- `POST /internal/embeddings`
- `POST /internal/extract-cv`
- `POST /internal/extract-job`
- `POST /internal/rerank`
- `POST /internal/explanations`

Chỉ triển khai endpoint reranking khi đã xác định rõ rubric và quy trình đánh giá.

Không để client công khai truy cập trực tiếp ChromaDB hoặc model inference.

---

# 11. Database Design

## 11.1. PostgreSQL

Tiếp tục sử dụng các bảng nghiệp vụ hiện có:

- `User`
- `FreelancerProfile`
- `ClientProfile`
- `CVDocument`
- `Job`
- `CVSkill`
- `JobSkill`
- `Application`

Không tạo bảng trùng với schema hiện tại.

## 11.2. Các bảng bổ sung

### DocumentProcessingJob

Lưu trạng thái xử lý CV/job.

Trường đề xuất:

- `id`
- `documentType`
- `documentId`
- `documentVersion`
- `status`
- `attemptCount`
- `errorMessage`
- `createdAt`
- `updatedAt`

### EmbeddingIndexRecord

Theo dõi đồng bộ ChromaDB.

Trường đề xuất:

- `id`
- `documentType`
- `documentId`
- `documentVersion`
- `embeddingModel`
- `collectionName`
- `vectorRecordId`
- `contentHash`
- `indexStatus`
- `indexedAt`

### Recommendation

Lưu lịch sử kết quả.

Trường đề xuất:

- `id`
- `requesterId`
- `sourceType`
- `sourceDocumentId`
- `targetType`
- `targetDocumentId`
- `semanticScore`
- `skillScore`
- `experienceScore`
- `budgetScore`
- `preferenceScore`
- `finalScore`
- `rankingVersion`
- `embeddingModel`
- `createdAt`

### RecommendationFeedback

Lưu sự kiện phản hồi:

- `VIEW`
- `SAVE`
- `APPLY`
- `DISMISS`
- `SHORTLIST`
- `HIRE`

Không xem mọi lượt click hoặc apply là nhãn positive tuyệt đối.

### EmbeddingOutboxEvent

Lưu sự kiện cần đồng bộ:

- `id`
- `eventType`
- `documentType`
- `documentId`
- `documentVersion`
- `status`
- `attemptCount`
- `lastError`
- `createdAt`
- `processedAt`

---

# 12. Đánh giá thực nghiệm

## 12.1. Dataset (LÀM ƠN NHỚ SỬ DỤNG DATABASE CỦA DỰ ÁN, XIN CẢM ƠN!!!)

Tạo benchmark gồm các cặp CV–job đã ẩn danh và được gán nhãn.

Nhãn relevance:

- `0`: Không phù hợp.
- `1`: Phù hợp một phần.
- `2`: Phù hợp tốt.

Ghi lại các lý do:

- Matching section (section nào khớp, section nào lệch).
- Matching skills.
- Missing required skills.
- Experience fit.
- Budget compatibility.
- Các điều kiện liên quan khác.

Cần có hướng dẫn gán nhãn rõ ràng và đo độ đồng thuận giữa annotators khi có nhiều người gán nhãn.

## 12.2. Baselines

| ID | Phương pháp |
|---|---|
| B0 | Keyword/BM25 |
| B1 | Single-vector BGE-M3 + ChromaDB |
| B2 | Chunk max-sim |
| B3 | Dense + keyword retrieval |
| B4 | Section-aware multi-vector |
| B5 | Section-aware + keyword retrieval |
| B6 | Hybrid Ranking (S_semantic = section-aware) |
| B7 | Hybrid Ranking + Qwen reranking |
| B8 | Hybrid Ranking + cross-encoder |
| B9 | Fine-tuned embedding, nếu đủ dữ liệu |

Section-aware multi-vector (B4) là phương pháp chính; các baseline còn lại để đo lợi ích của từng thành phần.

## 12.3. Metrics

### Retrieval

- Recall@10.
- Recall@50.
- Precision@K.

### Ranking

- NDCG@5.
- NDCG@10.
- MRR@10.

### Hiệu năng

- p50/p95 latency.
- Throughput.
- Thời gian embedding.
- Thời gian indexing.
- RAM/VRAM.
- Thời gian xử lý end-to-end.

### Explanation

- Evidence precision.
- Factual consistency.
- Missing requirement accuracy.
- Unsupported claim rate.
- JSON validity rate.
- Human-rated usefulness.

## 12.4. Ablation Study

Chạy lần lượt:

1. Dense retrieval (single-vector).
2. Section-aware multi-vector.
3. Section-aware + ghép cặp cố định vs ghép cặp chéo.
4. Section-aware + tổng hợp mean vs max.
5. Section-aware + bỏ một số section / dùng toàn bộ full-text (bỏ section).
6. Section-aware + keyword retrieval.
7. Section-aware + skill score.
8. Section-aware + skill + experience/budget.
9. Hybrid Ranking + Qwen reranking.
10. Hybrid Ranking + cross-encoder reranking.
11. Hybrid Ranking + Qwen explanation.

Mục tiêu là xác định thành phần nào thực sự cải thiện chất lượng và chi phí tương ứng.

Không gộp chất lượng ranking với chất lượng explanation thành cùng một metric.

---

# 13. Kế hoạch triển khai

## 1 — Phân tích hệ thống

- Kiểm tra Prisma schema hiện tại.
- Xác định luồng upload CV và tạo job.
- Kiểm tra storage và authentication.
- Chốt cấu trúc dữ liệu.
- Chọn embedding baseline.
- Xác định checkpoint Qwen 4B.
- Thiết kế benchmark sơ bộ.

**Deliverables:** Architecture, data mapping và benchmark specification.

## 2 — Document Ingestion

- Xây dựng PDF/DOCX parser.
- OCR fallback nếu cần.
- Chuẩn hóa văn bản.
- Trích xuất cấu trúc CV/job.
- Tách section bằng rule (skills, experience, education, summary).
- Validate dữ liệu.
- Thêm processing status.
- Thêm logging và retry.

**Deliverables:** CV/job ingestion pipeline.

## 3 — Embedding và ChromaDB

- Xây dựng FastAPI embedding service.
- Triển khai BGE-M3.
- Tạo collections.
- Nhúng từng section + full-text.
- Implement upsert/delete theo section.
- Lưu metadata (bao gồm `section`) và content hash.
- Xây dựng synchronization worker.
- Kiểm thử persistence và reindexing.

**Deliverables:** Embedding service và ChromaDB integration.

## 4 — Vector Retrieval và Section Scoring

- Implement CV-to-job search.
- Implement job-to-CV search.
- Implement điểm semantic theo section (ghép cặp fixed/cross, tổng hợp mean/max).
- Thêm metadata filtering.
- Validate candidate bằng PostgreSQL.
- Xây dựng keyword baseline.
- Đo Recall@K và latency.

**Deliverables:** Hai luồng retrieval và section scoring hoạt động.

## 5 — Hybrid Ranking

- Implement semantic score (section-aware).
- Implement skill score.
- Implement experience score.
- Implement budget score.
- Implement preference score.
- Thêm hard constraints.
- Cấu hình ranking weights.
- Chạy ablation sơ bộ (single-vector vs section-aware, fixed vs cross).

**Deliverables:** Hybrid Ranking Engine v1.

## 6 — Qwen và Frontend

- Tích hợp Qwen 4B.
- Xây dựng explanation prompt.
- Validate output schema.
- Kiểm tra bằng chứng.
- Tích hợp Recommended Jobs UI.
- Tích hợp Recommended Freelancers UI.
- Thêm feedback.
- Thử nghiệm Qwen reranking trên candidate pool nhỏ.

**Deliverables:** End-to-end recommendation.

## 7 — Benchmark

- Hoàn thiện benchmark.
- Đánh giá retrieval và ranking.
- So sánh single-vector, section-aware và hybrid.
- Đánh giá Qwen reranking.
- Đánh giá explanation quality.
- Đo latency và memory.
- Phân tích lỗi.
- Tính confidence intervals khi phù hợp.

**Deliverables:** Evaluation report và kết quả thực nghiệm.

## 8 — Hoàn thiện và Paper

- Fix bug.
- Kiểm tra đồng bộ và xóa dữ liệu.
- Kiểm tra authorization.
- Kiểm tra prompt injection.
- Tối ưu latency.
- Chuẩn bị bảng và biểu đồ.
- Viết paper draft.
- Hoàn thiện README và hướng dẫn tái lập.

**Deliverables:** MVP hoàn chỉnh và bản thảo nghiên cứu.

---

# 14. Định hướng Paper

## 14.1. Tên đề tài đề xuất

**Beyond Semantic Similarity: Section-aware Multi-vector Retrieval and LLM-Assisted Ranking for Bidirectional CV–Job Recommendation**

Tên tiếng Việt:

**Vượt qua tương đồng ngữ nghĩa: Truy xuất đa vector theo section và xếp hạng hỗ trợ bởi LLM cho hệ thống đề xuất CV–Job hai chiều**

## 14.2. Vấn đề nghiên cứu

Dense retrieval một vector có thể tìm các CV/job tương đồng về ngữ nghĩa nhưng dễ làm mờ tín hiệu của từng phần tài liệu và không nhất thiết đáp ứng các yêu cầu kỹ thuật, điều kiện công việc và tiêu chí tuyển dụng cụ thể.

Nghiên cứu sẽ đánh giá liệu biểu diễn đa vector theo section, kết hợp keyword search, structured skill scoring và LLM-assisted reranking có cải thiện chất lượng recommendation hay không.

## 14.3. Research Questions

**RQ1:** Section-aware multi-vector có cải thiện Recall@K/NDCG@10 so với single-vector và chunk max-sim không?

**RQ2:** Structured ranking có cải thiện NDCG@10 so với xếp hạng bằng similarity không?

**RQ3:** Qwen 4B reranking có cải thiện chất lượng top-ranked recommendations không?

**RQ4:** Qwen 4B có tạo explanation chính xác và bám sát bằng chứng không?

**RQ5:** Chất lượng cải thiện có tương xứng với latency và tài nguyên bổ sung không?

## 14.4. Giả thuyết

- H1: Section-aware multi-vector cải thiện khả năng truy xuất và xếp hạng candidate phù hợp so với single-vector.
- H2: Structured ranking cải thiện chất lượng xếp hạng.
- H3: Qwen reranking có thể cải thiện chất lượng top-K, nhưng cần kiểm chứng bằng thực nghiệm.
- H4: Evidence-grounded prompting giảm các kết luận không có bằng chứng.
- H5: Qwen reranking làm tăng latency so với Hybrid Ranking thuần túy.

Đây là giả thuyết cần kiểm nghiệm, không phải kết quả đã được chứng minh.

## 14.5. Các đóng góp tiềm năng

1. Benchmark CV–job matching hai chiều.
2. Biểu diễn đa vector theo section và so sánh ghép cặp fixed vs cross, tổng hợp mean vs max.
3. Hybrid retrieval/ranking framework.
4. So sánh embedding model cho dữ liệu tiếng Việt và tiếng Anh.
5. Đánh giá Qwen 4B trong reranking.
6. Đánh giá evidence-grounded explanation.
7. Ablation study và phân tích hiệu năng.

Chỉ tuyên bố đóng góp đã được hỗ trợ bởi kết quả thực nghiệm.

## 14.6. Cấu trúc paper

1. Introduction.
2. Related Work.
3. Problem Formulation.
4. Dataset and Annotation.
5. Proposed Method.
6. Experimental Setup.
7. Results.
8. Ablation Study.
9. Error Analysis.
10. Efficiency Analysis.
11. Ethics and Limitations.
12. Conclusion.

## 14.7. Yêu cầu thực nghiệm

- Cố định train/validation/test split.
- Tránh leakage giữa các tập dữ liệu.
- Dùng cùng benchmark khi so sánh các phương pháp.
- Công bố checkpoint và cấu hình model.
- Báo cáo số lượng query và relevant documents.
- Dùng confidence intervals và kiểm định phù hợp khi có thể.
- Tách kết quả dữ liệu tổng hợp và dữ liệu thật.
- Báo cáo cả kết quả không cải thiện.
- Không tuyên bố novelty chỉ vì sử dụng Qwen, embedding và ChromaDB.

---

# 15. Bảo mật và quyền riêng tư

- Chỉ client được phép mới có thể truy cập CV.
- Chỉ lưu metadata cần thiết trong ChromaDB.
- Không đưa thông tin cá nhân không liên quan vào ranking.
- Xử lý yêu cầu xóa CV/job trên cả PostgreSQL và ChromaDB.
- Bảo vệ endpoint nội bộ.
- Không cho LLM quyết định quyền truy cập.
- Xem nội dung CV/job là dữ liệu không đáng tin cậy.
- Ngăn prompt injection từ tài liệu.
- Kiểm tra mọi claim do Qwen tạo ra.
- Ghi log lỗi nhưng hạn chế lưu nội dung cá nhân.
- Có quy trình backup và khôi phục index.

---

# 16. Definition of Done

- [ ] CV được upload và xử lý tự động.
- [ ] Job được tạo và lập chỉ mục tự động.
- [ ] Embedding được tạo bằng model chuyên dụng.
- [ ] Tài liệu được nhúng đa vector theo section + full-text.
- [ ] Có điểm semantic theo section (ghép cặp fixed/cross).
- [ ] ChromaDB lưu vector bền vững.
- [ ] PostgreSQL vẫn là nguồn dữ liệu nghiệp vụ chính.
- [ ] Có cơ chế đồng bộ và reindex.
- [ ] Vector search hoạt động hai chiều.
- [ ] Có kiểm tra quyền và trạng thái.
- [ ] Có Hybrid Ranking.
- [ ] Có Qwen explanation.
- [ ] Có thể thử nghiệm Qwen reranking độc lập.
- [ ] Giao diện hiển thị recommendation.
- [ ] Có feedback và log.
- [ ] Có benchmark retrieval/ranking.
- [ ] Có ablation study.
- [ ] Có tài liệu triển khai.
- [ ] Có kết quả thực nghiệm cho paper.

---

# 17. Quyết định kỹ thuật cuối cùng

| Hạng mục | Phương án |
|---|---|
| LLM | Qwen 4B |
| Embedding | BGE-M3 baseline |
| Matching | Section-aware multi-vector (1 vector mỗi section + full-text) |
| Vector database | ChromaDB |
| Business database | PostgreSQL |
| ORM | Prisma |
| Backend | NestJS |
| AI service | Python + FastAPI |
| Section pairing | Cross (so với fixed) |
| Section aggregation | Mean/max, chọn theo ablation |
| Vector similarity | Cosine distance/similarity theo cấu hình collection |
| Vector index | HNSW nếu phù hợp phiên bản và workload |
| Candidate retrieval | Full-text vector top-K, rồi chấm điểm theo section |
| Ranking | Hybrid Ranking (S_semantic = section-aware) |
| LLM reranking | Thử nghiệm trên candidate pool nhỏ |
| Explanation | Qwen 4B với bằng chứng |
| Research | Retrieval, ranking, reranking và explainability |

## Kết luận

Hệ thống không phải là Qwen hoặc vector search, mà là sự kết hợp các thành phần với vai trò khác nhau.

- **BGE-M3** chuyển từng section (và full-text) của CV/job thành vector.
- **ChromaDB** tìm những vector gần nhất bằng distance metric và index phù hợp.
- **Section-aware scoring** đối chiếu từng section (ghép cặp fixed/cross) và tạo điểm thành phần giải thích được.
- **Hybrid Ranking** đánh giá thêm kỹ năng, kinh nghiệm, ngân sách và điều kiện nghiệp vụ.
- **Qwen 4B** có thể hỗ trợ trích xuất, reranking và giải thích recommendation.
- **PostgreSQL** đảm bảo dữ liệu nghiệp vụ và quyền truy cập chính xác.

Hướng triển khai an toàn nhất là xây dựng section-aware retrieval và Hybrid Ranking làm baseline trước, sau đó thử nghiệm Qwen reranking. Điều này giúp Frevia có một hệ thống thực tế, đồng thời cho phép đánh giá định lượng xem Qwen có thực sự cải thiện chất lượng recommendation hay chỉ làm tăng thời gian xử lý.

Trước khi triển khai, cần kiểm tra schema, module upload và authentication hiện tại của Frevia để điều chỉnh kế hoạch cho đúng repository.