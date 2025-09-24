"""Standalone utilities for scientific research experiments."""

from .groundtruth_runner import QueryResult, time_postgres, time_duckdb, compute_speedup_label

__all__ = [
    "QueryResult",
    "time_postgres",
    "time_duckdb",
    "compute_speedup_label",
]
