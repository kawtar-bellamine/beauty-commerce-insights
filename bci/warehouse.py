"""Builds the DuckDB warehouse: raw tables, staging, marts, then data-quality checks."""
from __future__ import annotations

from pathlib import Path

import duckdb

from .data import ROOT, load_raw, source_dir

SQL = ROOT / "sql"


def build(db_path: str | Path = ":memory:", sample: bool = False, folder: Path | None = None):
    con = duckdb.connect(str(db_path))
    load_raw(con, folder or source_dir(sample))
    for f in sorted(SQL.glob("*.sql")):
        con.execute(f.read_text())
    return con
