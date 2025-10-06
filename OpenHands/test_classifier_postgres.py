#!/usr/bin/env python3
"""Test the classifier with the postgres/pg_duckdb query"""

import sys
sys.path.insert(0, './OpenHands')

from extensions.uagent_research.classifier.task_classifier import task_classifier

# Your specific query
postgres_query = """please modify postgres and pg_duckdb source code （ to download source code you can utilize the proxy on port localhost:7890, do not use the system-wide postgresql）, first extract pre-opt features from postgres kernel and log to files, then collect dual-execution data (pre-optimization query features that can be found in kernel structures and execution times on dual engine) and train a machine learning model to predict whether postgres engine or duckdb engine executes a query fast and embed the machine learning model into database source code (using the language of the database for example c language) to online route each query to the faster engine, and execute end-to-end experiments to test the ml-based system's performance. A baseline method called threshold-based method should also be implemented, which routes query based on threshold, for example threshold can be 10000 or 50000 or any other value, if postgres estimates the cost of a query is above threshold, then send to duckdb, otherwise send to postgres, and compare the postgres-only, duckdb-only, different threshold-based methods and lightgbm-based method"""

should_trigger, task_type, confidence, reasoning = task_classifier.should_trigger_research(postgres_query)

print("=" * 80)
print("POSTGRES/PG_DUCKDB QUERY CLASSIFICATION")
print("=" * 80)
print(f"\nQuery: {postgres_query[:200]}...")
print(f"\nShould Trigger Research: {should_trigger}")
print(f"Task Type: {task_type.value}")
print(f"Confidence: {confidence:.2f}")
print(f"\nReasoning:")
for key, value in reasoning.items():
    print(f"  {key}: {value}")
print("=" * 80)
