"""Utility functions for collecting real Postgres/DuckDB latency measurements.

These helpers run inside the OpenHands workspace and emit structured JSONL/CSV
artifacts so failures are actionable (error class/code/traceback, etc.).
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import psycopg


@dataclass
class QueryResult:
    engine: str
    query_id: str
    sql_sha256: str
    ok: bool
    latency_ms: Optional[float]
    rows: Optional[int]
    error_class: Optional[str]
    error_code: Optional[str]
    error_message: Optional[str]
    traceback: Optional[str]


def _artifacts_dir() -> Path:
    target = Path(os.environ.get("GROUNDTRUTH_ART_DIR", "/workspace/artifacts/groundtruth"))
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_csv(path: Path, fields: Tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(
                "engine,query_id,sql_sha256,ok,latency_ms,rows,error_class,error_code,error_message\n"
            )
        row = ",".join(
            [
                "" if value is None else str(value).replace("\n", " ").replace(",", ";")
                for value in fields
            ]
        )
        handle.write(row + "\n")


def _hash_sql(sql: str) -> str:
    import hashlib

    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:12]


def time_postgres(dsn: str, sql: str, timeout_ms: int = 60_000) -> QueryResult:
    """Run the query on PostgreSQL and capture latency/errors."""

    query_id = _hash_sql(sql)
    started = time.perf_counter()
    try:
        with psycopg.connect(
            dsn,
            autocommit=True,
            options=f"-c statement_timeout={timeout_ms}"
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                try:
                    rows = len(cur.fetchall())
                except psycopg.ProgrammingError:
                    rows = 0
        latency = (time.perf_counter() - started) * 1000
        result = QueryResult(
            engine="postgres",
            query_id=query_id,
            sql_sha256=query_id,
            ok=True,
            latency_ms=latency,
            rows=rows,
            error_class=None,
            error_code=None,
            error_message=None,
            traceback=None,
        )
    except Exception as exc:
        result = QueryResult(
            engine="postgres",
            query_id=query_id,
            sql_sha256=query_id,
            ok=False,
            latency_ms=None,
            rows=None,
            error_class=exc.__class__.__name__,
            error_code=getattr(exc, "sqlstate", None),
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )

    artifacts = _artifacts_dir()
    _write_jsonl(artifacts / "logs.jsonl", asdict(result))
    _append_csv(
        artifacts / "results.csv",
        (
            result.engine,
            result.query_id,
            result.sql_sha256,
            result.ok,
            result.latency_ms,
            result.rows,
            result.error_class,
            result.error_code,
            result.error_message,
        ),
    )
    return result


def time_duckdb(database: Optional[str], sql: str, timeout_s: int = 60) -> QueryResult:
    """Run the query on DuckDB with a hard timeout."""

    query_id = _hash_sql(sql)
    container: dict[str, QueryResult] = {}

    def _worker() -> None:
        try:
            connection = duckdb.connect(database) if database else duckdb.connect()
            started = time.perf_counter()
            cursor = connection.execute(sql)
            try:
                rows = len(cursor.fetchall())
            except Exception:
                rows = 0
            latency = (time.perf_counter() - started) * 1000
            container["result"] = QueryResult(
                engine="duckdb",
                query_id=query_id,
                sql_sha256=query_id,
                ok=True,
                latency_ms=latency,
                rows=rows,
                error_class=None,
                error_code=None,
                error_message=None,
                traceback=None,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            container["result"] = QueryResult(
                engine="duckdb",
                query_id=query_id,
                sql_sha256=query_id,
                ok=False,
                latency_ms=None,
                rows=None,
                error_class=exc.__class__.__name__,
                error_code=None,
                error_message=str(exc),
                traceback=traceback.format_exc(),
            )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        try:
            duckdb.interrupt()
        except Exception:
            pass
        result = QueryResult(
            engine="duckdb",
            query_id=query_id,
            sql_sha256=query_id,
            ok=False,
            latency_ms=None,
            rows=None,
            error_class="TimeoutError",
            error_code=None,
            error_message=f"DuckDB query exceeded {timeout_s}s",
            traceback=None,
        )
    else:
        result = container["result"]

    artifacts = _artifacts_dir()
    _write_jsonl(artifacts / "logs.jsonl", asdict(result))
    _append_csv(
        artifacts / "results.csv",
        (
            result.engine,
            result.query_id,
            result.sql_sha256,
            result.ok,
            result.latency_ms,
            result.rows,
            result.error_class,
            result.error_code,
            result.error_message,
        ),
    )
    return result


def compute_speedup_label(pg_ms: Optional[float], duck_ms: Optional[float], minimum_gain: float = 0.20) -> Tuple[Optional[float], Optional[int]]:
    """Return (speedup, label) where label=1 when DuckDB is faster by >= minimum_gain."""

    if pg_ms is None or duck_ms is None:
        return None, None
    baseline = max(pg_ms, 1e-6)
    speedup = (pg_ms - duck_ms) / baseline
    return speedup, 1 if speedup >= minimum_gain else 0


__all__ = ["QueryResult", "time_postgres", "time_duckdb", "compute_speedup_label"]
