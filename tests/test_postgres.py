"""Tests PostgreSQL candidate validation (plan §7.4) — policy + fetch (fake conn)."""

from cvjd.ranking.constraints import AccessContext, CandidateRecord
from cvjd.storage import postgres


def test_parse_doc_id_and_ids():
    assert postgres.parse_doc_id("jd_12") == ("jd", 12)
    assert postgres.parse_doc_id("cv_3") == ("cv", 3)
    assert postgres.parse_doc_id("bogus") is None
    assert postgres.parse_doc_ids(["jd_1", "jd_2", "x"], "jd") == [1, 2]
    assert postgres.parse_doc_ids(["cv_5", "jd_9"], "cv") == [5]


def test_job_record_from_row_open_and_closed():
    record = postgres.job_record_from_row(
        {"id": 1, "status": "OPEN", "deletedAt": None, "clientId": 7}
    )
    assert record.doc_id == "jd_1"
    assert record.doc_type == "jd"
    assert record.status == "READY"
    assert record.owner_id == "7"
    assert record.deleted_at is None

    closed = postgres.job_record_from_row({"id": 2, "status": "CLOSED"})
    assert closed.status != "READY"


def test_job_record_maps_deleted_timestamp():
    from datetime import datetime, timezone

    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    record = postgres.job_record_from_row(
        {"id": 9, "status": "OPEN", "deletedAt": ts, "clientId": None}
    )
    assert record.deleted_at == ts.isoformat()


def test_record_from_row_missing_id_returns_none():
    assert postgres.job_record_from_row({}) is None
    assert postgres.cv_record_from_row({}) is None


def test_cv_record_from_row_defaults():
    record = postgres.cv_record_from_row({"id": 7})
    assert record.doc_id == "cv_7"
    assert record.doc_type == "cv"
    assert record.status == "READY"
    assert record.visibility == "public"
    assert record.deleted_at is None


def test_validate_ids_applies_access_policy():
    records = [
        CandidateRecord(doc_id="cv_1", doc_type="cv", visibility="public"),
        CandidateRecord(doc_id="cv_2", doc_type="cv", visibility="private", owner_id="u9"),
        CandidateRecord(
            doc_id="cv_3", doc_type="cv", visibility="public", deleted_at="2026-01-01"
        ),
    ]
    anonymous = AccessContext()
    assert postgres.validate_ids(records, anonymous) == ["cv_1"]
    owner = AccessContext(user_id="u9")
    assert postgres.validate_ids(records, owner) == ["cv_1", "cv_2"]


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.rows = self.conn.resolve(sql, params)

    def fetchall(self):
        return self.rows


class _FakeConn:
    def __init__(self, catalog, tables):
        self._catalog = catalog
        self._tables = tables

    def cursor(self, row_factory=None):
        return _FakeCursor(self)

    def resolve(self, sql, params=None):
        if "information_schema" in sql:
            return self._catalog
        for table, rows in self._tables.items():
            if f'"{table}"' in sql:
                if params:
                    wanted = set(params[0])
                    return [row for row in rows if row["id"] in wanted]
                return rows
        return []


def _fake_conn():
    catalog = [
        ("Job", "id"),
        ("Job", "status"),
        ("Job", "deletedAt"),
        ("Job", "clientId"),
    ]
    tables = {
        "Job": [
            {"id": 1, "status": "OPEN", "deletedAt": None, "clientId": "c1"},
            {"id": 2, "status": "CLOSED", "deletedAt": None, "clientId": "c2"},
        ]
    }
    return _FakeConn(catalog, tables)


def test_validate_candidates_fetches_and_filters():
    conn = _fake_conn()
    ctx = AccessContext()
    allowed = postgres.validate_candidates(conn, "jd", ["jd_1", "jd_2", "nope"], ctx)
    assert allowed == ["jd_1"]


def test_fetch_candidate_records_ignores_unparseable():
    conn = _fake_conn()
    records = postgres.fetch_candidate_records(conn, "jd", ["jd_1", "oops"])
    assert len(records) == 1
    assert records[0].doc_id == "jd_1"


def test_fetch_rejects_unknown_doc_type():
    conn = _fake_conn()
    try:
        postgres.fetch_candidate_records(conn, "exe", ["jd_1"])
    except ValueError as exc:
        assert "doc_type" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
