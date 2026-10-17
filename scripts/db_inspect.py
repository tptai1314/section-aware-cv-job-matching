"""Inspect a PostgreSQL database: tables, text columns and row counts.

Connects with DATABASE_URL (from .env or the environment). Use this to find
which tables/columns hold CV and JD text before running scripts/db_export.py.

Usage:
    python scripts/db_inspect.py
    python scripts/db_inspect.py --schema public
    python scripts/db_inspect.py --no-count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.utils.config import get_database_url  # noqa: E402

_TEXT_TYPES = ("text", "character varying", "character")

_TABLES_SQL = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = %s AND table_type = 'BASE TABLE'
ORDER BY table_name
"""

_COLUMNS_SQL = """
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = %s
ORDER BY table_name, ordinal_position
"""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="public")
    parser.add_argument("--no-count", action="store_true",
                        help="skip SELECT count(*) per table")
    args = parser.parse_args(argv)

    import psycopg

    with psycopg.connect(get_database_url()) as conn, conn.cursor() as cur:
        cur.execute(_TABLES_SQL, (args.schema,))
        tables = [row[0] for row in cur.fetchall()]

        cur.execute(_COLUMNS_SQL, (args.schema,))
        columns: dict[str, list[tuple[str, str]]] = {}
        for table, column, dtype in cur.fetchall():
            columns.setdefault(table, []).append((column, dtype))

    if not tables:
        print(f"no tables in schema {args.schema!r}")
        return

    print(f"schema {args.schema!r}: {len(tables)} tables")
    for table in tables:
        count = ""
        if not args.no_count:
            with psycopg.connect(get_database_url()) as conn, conn.cursor() as cur:
                cur.execute(f'SELECT count(*) FROM "{args.schema}"."{table}"')
                count = f"  rows={cur.fetchone()[0]}"
        print(f"\n== {table}{count}")
        for column, dtype in columns.get(table, []):
            marker = " <text>" if dtype in _TEXT_TYPES else ""
            print(f"   {column:<32} {dtype}{marker}")


if __name__ == "__main__":
    main()
