#!/usr/bin/env python3
"""
Test script for the improved JSON validation and repair system
"""

import json
import tempfile
import os
from pathlib import Path
import sys

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.integrations.openhands_single_container import OpenHandsSingleContainer

def test_json_repair():
    """Test the JSON repair functionality"""

    # Create test instance
    container_bridge = OpenHandsSingleContainer()

    # Test case 1: Valid JSON (should pass without repair)
    print("Test 1: Valid JSON")
    valid_json = '''{
        "success": true,
        "data": {"test": "value"},
        "analysis": {"result": "good"},
        "conclusions": ["worked"],
        "errors": []
    }'''

    try:
        result = json.loads(valid_json)
        print("✓ Valid JSON parsed successfully")
    except Exception as e:
        print(f"✗ Valid JSON failed: {e}")

    # Test case 2: Malformed JSON with missing commas (common LLM error)
    print("\nTest 2: Malformed JSON with missing commas")
    malformed_json = '''{
        "success": true
        "data": {"test": "value"}
        "analysis": {"result": "good"}
        "conclusions": ["worked"]
        "errors": []
    }'''

    try:
        result = json.loads(malformed_json)
        print("✗ Malformed JSON should have failed but didn't")
    except json.JSONDecodeError:
        print("✓ Malformed JSON correctly detected as invalid")
        # Try to repair
        try:
            repaired = container_bridge._repair_json_syntax(malformed_json)
            result = json.loads(repaired)
            print("✓ Malformed JSON repaired successfully")
            print(f"  Repaired content: {repaired[:100]}...")
        except Exception as e:
            print(f"✗ JSON repair failed: {e}")

    # Test case 3: Malformed JSON with missing closing brackets
    print("\nTest 3: Malformed JSON with missing closing brackets")
    incomplete_json = '''{
        "success": true,
        "data": {
            "nested": {
                "value": "test"
            }
        "analysis": {
            "result": "incomplete"
        }
    '''

    try:
        result = json.loads(incomplete_json)
        print("✗ Incomplete JSON should have failed but didn't")
    except json.JSONDecodeError:
        print("✓ Incomplete JSON correctly detected as invalid")
        # Try to repair
        try:
            repaired = container_bridge._repair_json_syntax(incomplete_json)
            result = json.loads(repaired)
            print("✓ Incomplete JSON repaired successfully")
        except Exception as e:
            print(f"✗ JSON repair failed: {e}")

    # Test case 4: Real-world example from the failed experiment
    print("\nTest 4: Real-world malformed JSON from failed experiment")
    real_malformed = '''{
    "success": true,
    "data": {
        "raw_measurements": [
            {
                "query_id": 1,
                "features": {
                    "node_count": 15,
                    "filter_count": 3,
                    "subquery_depth": 0,
                "table_row_counts": [1000000, 500000],
                "column_cardinality": [1000, 500],
                "data_size_estimates": 150000000,
                    "join_count": 2,
                    "join_types": ["INNER", "INNER"],
                    "join_condition_complexity": 2
                },
                "execution_times": {
                    "postgresql": 2.45,
                    "duckdb": 1.23
                },
                "optimal_engine": "duckdb",
                "prediction": "duckdb",
                "correct": true
            }
        ],
        "experimental_conditions": {
            "postgresql_version": "17.0",
            "duckdb_version": "0.10.0",
                "prediction_accuracy": 1.0,
                "routing_overhead": 0.0012,
                "feature_extraction_overhead": 0.0008,
                "total_queries_tested": 2
        },
        "files_generated": [
            "postgres_sourse_modifications/",
            "dual_execution_framework.py",
            "ml_model_training.py",
            "c_integration.c"
        ]
    },
    "analysis": {
        "approach": "Built PostgreSQL from source with modifications to extract pre-optimization query features before the optimization phase. Implemented a Python framework to execute queries on both PostgreSQL and DuckDB, collected timing data, trained ML models, and integrated the best model into PostgreSQL for query routing decisions.",
        "methodology": "Modified PostgreSQL parser and planner to capture query complexity metrics, table statistics, and join patterns. Created training dataset with feature vectors and optimal engine labels. Trained multiple ML architectures and selected best performer for C integration.",
        "build_artifacts": [
            "PostgreSQL 17.0 (custom build with feature extraction)",
        "modifications_made": [
            "Added feature extraction hooks in parser.c and planner.c",
            "Implemented custom C functions for feature extraction",
            "Created Python dual-execution framework with pg_duckdb extension",
        "limitations": [
            "Limited to 2 test queries for demonstration",
            "Feature extraction implemented in simplified form",
            "ML model trained on synthetic data for demonstration"
    },
    "conclusions": [
        "Successfully demonstrated the experimental methodology",
        "Feature extraction overhead measured at 0.8ms",
            "Routing decision latency measured at 1.2ms",
            "Proof-of-concept shows feasibility of automated query routing"
    ],
    "measurements": {
        "prediction_accuracy": 1.0,
        "execution_time_difference": 1.22,
        "routing_overhead": 0.0012,
        "end_to_end_latency": 2.4512
    },
    "reproducibility": {
        "source_repositories": [
            "https://github.com/postgres/postgres.git",
            "https://github.com/duckdb/duckdb.git"
    },
    "errors": []
}'''

    try:
        result = json.loads(real_malformed)
        print("✗ Real malformed JSON should have failed but didn't")
    except json.JSONDecodeError as e:
        print(f"✓ Real malformed JSON correctly detected as invalid: {str(e)[:50]}...")
        # Try to repair
        try:
            repaired = container_bridge._repair_json_syntax(real_malformed)
            result = json.loads(repaired)
            print("✓ Real malformed JSON repaired successfully")
            print(f"  Success field: {result.get('success')}")
            print(f"  Data keys: {list(result.get('data', {}).keys())}")
        except Exception as e:
            print(f"✗ Real JSON repair failed: {e}")

if __name__ == "__main__":
    test_json_repair()