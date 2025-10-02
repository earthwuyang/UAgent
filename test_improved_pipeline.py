#!/usr/bin/env python3
"""
Test improved research pipeline with better requirements extraction
"""

import asyncio
import json
import logging
from typing import Dict, Any
import aiohttp
import os

logger = logging.getLogger(__name__)

TEST_RESEARCH_QUERY = """please modify postgres and pg_duckdb source code （ to download source code you can utilize the proxy on port localhost:7890, do not use the system-wide postgresql), first extract pre-opt features from postgres kernel and log to files, then collect dual-execution data (pre-optimization query features that can be found in kernel structures and execution times on dual engine) and train a machine learning model to predict whether postgres engine or duckdb engine executes a query fast and embed the machine learning model into database source code (using the language of the database for example c language) to online route each query to the faster engine, and execute end-to-end experiments to test the ml-based system's performance."""

async def test_improved_pipeline():
    """Test the improved requirement extraction pipeline"""

    base_url = "http://localhost:8000"  # Adjust based on your uagent deployment

    print("Testing Improved Research Pipeline")
    print("=" * 80)

    # Test 1: Basic scientific research request
    print("\n1. Testing basic scientific research request...")

    payload = {
        "query": TEST_RESEARCH_QUERY,
        "include_literature_review": True,
        "include_code_analysis": True,
        "enable_iteration": True,
        "max_iterations": 3
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/research/scientific", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"Research ID: {result.get('research_id', 'N/A')}")
                    print(f"Status: {result.get('status', 'N/A')}")
                    print(f"Hypotheses Generated: {result.get('hypotheses_generated', 0)}")
                    print(f"Experiments Conducted: {result.get('experiments_conducted', 0)}")
                    print("✓ Basic research request successful")
                else:
                    print(f"HTTP Error: {response.status}")
                    error_text = await response.text()
                    print(f"Error details: {error_text[:200]}...")
    except Exception as e:
        print(f"Connection Error: {e}")

    # Test 2: Request with explicit technical requirements
    print("\n2. Testing request with explicit technical requirements...")

    custom_requirements = {
        "source_code_modifications": ["PostgreSQL source code", "pg_duckdb extension"],
        "execution_engines": ["PostgreSQL", "pg_duckdb", "DuckDB"],
        "programming_languages": ["C", "Python"],
        "integration_requirements": [
            "ML model integration",
            "dual-engine query routing",
            "pre-optimization feature extraction"
        ],
        "data_collection_requirements": [
            "query execution times",
            "pre-optimization features",
            "comparison metrics"
        ],
        "prohibited_shortcuts": ["system-wide PostgreSQL", "simulation-based approaches"],
        "technical_guidance_needed": ["PostgreSQL kernel modification", "C-based ML model embedding"]
    }

    enhanced_payload = {
        "query": TEST_RESEARCH_QUERY,"include_literature_review": True,
        "include_code_analysis": True,
        "enable_iteration": True,
        "max_iterations": 5,
        "technical_requirements": custom_requirements
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/research/scientific", json=enhanced_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"Research ID: {result.get('research_id', 'N/A')}")
                    print(f"Status: {result.get('status', 'N/A')}")
                    print(f"Hypotheses Generated: {result.get('hypotheses_generated', 0)}")
                    print(f"Experiments Conducted: {result.get('experiments_conducted', 0)}")
                    print("✓ Enhanced research request with technical requirements successful")
                else:
                    print(f"HTTP Error: {response.status}")
                    error_text = await response.text()
                    print(f"Error details: {error_text[:200]}...")
    except Exception as e:
        print(f"Connection Error: {e}")

    # Test 3: Check research sessions
    print("\n3. Checking research sessions...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/research/sessions") as response:
                if response.status == 200:
                    sessions = await response.json()
                    print(f"Total sessions: {sessions.get('total', 0)}")
                    print(f"Active sessions: {len(sessions.get('active', []))}")
                    print(f"Completed sessions: {len(sessions.get('completed', []))}")
                    for session_data in sessions.get('sessions', [])[:2]:  # Show first 2 sessions
                        print(f"  - Session {session_data['session_id']}: {session_data['type']} ({session_data['status']})")
                    print("✓ Sessions retrieval successful")
                else:
                    print(f"Error retrieving sessions: {response.status}")
    except Exception as e:
        print(f"Connection Error: {e}")

    print("\n" + "=" * 80)
    print("Test completed. Check experiment details by querying specific session endpoints.")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(test_improved_pipeline())