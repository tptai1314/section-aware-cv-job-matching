"""PostgreSQL validation cho candidate trước khi trả kết quả (plan §7.4).

Chroma KHÔNG lưu hard constraints; thay vào đó engine phải kiểm tra lại với
PostgreSQL: job còn mở (OPEN/IN_PROGRESS), tài liệu chưa xoá, và quyền truy
cập (CV được chia sẻ) — qua policy thuần ở `cvjd.ranking.constraints`.

Module này chỉ *đọc* (SELECT). Nếu bảng/cột không tồn tại (môi trường nghiên
cứu), nó suy giảm an toàn: bỏ qua cột thiếu thay vì crash.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..ranking.constraints import CandidateRecord, apply_hard_constraints

OPEN_JOB_STATUSES = ("OPEN", "IN_PROGRESS")

_JOB_COLUMNS = ("id", "status", "deletedAt", "clientId")
_CV_COLUMNS = ("id", "userId", "deletedAt", "visibility")


def parse_doc_id(value: Any) -> tuple[str, int] | None:
    """'jd_12' -> ('jd', 12); None nếu không đúng dạng '{type}_{id}'."""
    match = re.fullmatch(r"(cv|jd)_(\d+)", str(value))
    if not match:
        return None
    return match.group(1), int(match.group(2))


def parse_doc_ids(doc_ids: Iterable[Any], doc_type: str) -> list[int]:
    """Lọc ra các id số hợp lệ của `doc_type` từ danh sách doc_id ('jd_1')."""
    out: list[int] = []
    for value in doc_ids:
        match = re.fullmatch(rf"{re.escape(doc_type)}_(\d+)", str(value))
        if match:
            out.append(int(match.group(1)))
    return out


def _as_iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def job_record_from_row(row: Mapping[str, Any]):
    """Row của bảng Job -> CandidateRecord (job mở -> status READY)."""
    from ..ranking.constraints import CandidateRecord

    job_id = row.get("id")
    if job_id is None:
        return None
    raw_status = str(row.get("status") or "").upper()
    status = "READY" if raw_status in OPEN_JOB_STATUSES else (raw_status.lower() or "unknown")
    owner = row.get("clientId")
    return CandidateRecord(
        doc_id=f"jd_{job_id}",
        doc_type="jd",
        status=status,
        visibility="public",
        owner_id=str(owner) if owner is not None else None,
        deleted_at=_as_iso(row.get("deletedAt")),
    )


def cv_record_from_row(row: Mapping[str, Any]):
    """Row của FreelancerProfile -> CandidateRecord."""
    profile_id = row.get("id")
    if profile_id is None:
        return None
    owner = row.get("userId")
    return CandidateRecord(
        doc_id=f"cv_{profile_id}",
        doc_type="cv",
        status="READY",
        visibility=str(row.get("visibility") or "public"),
        owner_id=str(owner) if owner is not None else None,
        deleted_at=_as_iso(row.get("deletedAt")),
    )


def _catalog(conn) -> dict[str, set[str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns"
            " WHERE table_schema = 'public'"
        )
        rows = cur.fetchall()
    cols: dict[str, set[str]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            table, column = row["table_name"], row["column_name"]
        else:
            table, column = row[0], row[1]
        cols.setdefault(str(table).lower(), set()).add(str(column).lower())
    return cols


def _existing(catalog: Mapping[str, set[str]], table: str, wanted: Sequence[str]) -> list[str]:
    have = catalog.get(table.lower(), set())
    return [c for c in wanted if c in have]


def _select(conn, table: str, columns: Sequence[str], ids: Sequence[int]):
    if not columns or not ids:
        return []
    from psycopg.rows import dict_row

    cols = ", ".join(f'"{c}"' for c in columns)
    sql = f'SELECT {cols} FROM "{table}" WHERE "id" = ANY(%s)'
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (list(ids),))
        return cur.fetchall()


def _catalog_columns(conn) -> dict[str, set[str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns"
            " WHERE table_schema = 'public'"
        )
        rows = cur.fetchall()
    cols: dict[str, set[str]] = {}
    for row in rows:
        table = row["table_name"] if isinstance(row, Mapping) else row[0]
        column = row["column_name"] if isinstance(row, Mapping) else row[1]
        cols.setdefault(str(table).lower(), set()).add(str(column).lower())
    return cols


def _catalog_or_load(conn, catalog) -> dict[str, set[str]]:
    return catalog if catalog is not None else _catalog(conn)


def fetch_candidate_records(
    conn, doc_type: str, doc_ids: Iterable[Any], catalog: Mapping[str, set[str]] | None = None,
):
    """Đọc record từ PostgreSQL cho `doc_ids` (bỏ qua id không hợp lệ)."""
    if doc_type == "jd":
        table, wanted, builder = "Job", _JOB_COLUMNS, job_record_from_row
    elif doc_type == "cv":
        table, wanted, builder = "FreelancerProfile", _CV_COLUMNS, cv_record_from_row
    else:
        raise ValueError(f"doc_type must be 'cv' or 'jd', got {doc_type!r}")

    ids = parse_doc_ids(doc_ids, doc_type)
    if not ids:
        return []
    if catalog is None:
        catalog = _catalog(conn)

    rows = _select(conn, table, _existing(catalog, table, wanted), ids)
    records = []
    for row in rows:
        record = builder(row)
        if record is not None:
            records.append(record)
    return records


def validate_ids(
    records: Sequence[Any],
    ctx,
    require_ready: bool = True,
    allowed_roles: Sequence[str] = (),
) -> list[str]:
    """Giữ lại doc_id hợp lệ theo policy hard-constraints."""
    allowed = apply_hard_constraints(records, ctx, require_ready, allowed_roles)
    return [record.doc_id for record in allowed]


def validate_candidates(
    conn,
    doc_type: str,
    doc_ids: Iterable[Any],
    ctx,
    require_ready: bool = True,
    allowed_roles: Sequence[str] = (),
) -> list[str]:
    """Fetch + lọc candidate theo hard constraints; trả doc_id được phép."""
    records = fetch_candidate_records(conn, doc_type, doc_ids)
    return validate_ids(records, ctx, require_ready, allowed_roles)


__all__ = [
    "OPEN_JOB_STATUSES",
    "cv_record_from_row",
    "fetch_candidate_records",
    "job_record_from_row",
    "parse_doc_id",
    "parse_doc_ids",
    "validate_candidates",
    "validate_ids",
]
