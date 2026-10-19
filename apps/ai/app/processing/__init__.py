"""Hàng đợi & trạng thái xử lý file (plan §9.3–9.4, §11.2).

Thiết kế chi tiết: `docs/DESIGN_processing_queue.md`.

- `store.py`   — state machine + `SqliteProcessingStore` (backend thay thế PG, xem §3.3)
- `pipeline.py`— các bước EXTRACTING → PARSING → EMBEDDING → FINALIZING
- `worker.py`  — thread nền claim job, retry/backoff, requeue job kẹt

Lưu ý: `__init__.py` cố tình KHÔNG import submodule để tránh vòng phụ thuộc
`routes → processing.store → (pipeline) → routes`.
"""

__all__: list[str] = []
