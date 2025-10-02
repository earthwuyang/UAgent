#!/usr/bin/env python3
"""
Test script to validate the improved technical requirements extraction
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.core.research_engines.scientific_research import RequirementExtractor, TechnicalRequirements
from app.core.llm_client import LLMClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_requirements_extraction():
    """Test the improved requirements extraction"""

    # Create LLM client (using environment variables)
    llm_client = LLMClient("openai")

    # Create requirement extractor
    requirements_extractor = RequirementExtractor(llm_client)

    # Test queries to validate
    test_queries = [
        # Original problematic query
        """please modify postgres and pg_duckdb source code （ to download source code you can utilize the proxy on port localhost:7890, do not use the system-wide postgresql), first extract pre-opt features from postgres kernel and log to files, then collect dual-execution data (pre-optimization query features that can be found in kernel structures and execution times on dual engine) and train a machine learning model to predict whether postgres engine or duckdb engine executes a query fast and embed the machine learning model into database source code (using the language of the database for example c language) to online route each query to the faster engine, and execute end-to-end experiments to test the ml-based system's performance.""",

        # Simpler version for testing
        "PostgreSQL and pg_duckdb ML model integration for query routing",

        # Different domain test
        "Compare Redis vs Memcached performance with benchmark suite"
    ]

    print("Testing Improved Requirement Extraction")
    print("=" * 80)

    for query in test_queries:
        print(f"\nQuery: {query[:100]}...")
        print("-" * 40)

        try:
            # Extract requirements
            technical_requirements = await requirements_extractor.extract_requirements(query)

            # Print results
            print("Extracted Technical Requirements:")
            print(f"Source Code Modifications: {technical_requirements.source_code_modifications}")
            print(f"Programming Languages: {technical_requirements.programming_languages}")
            print(f"Execution Engines: {technical_requirements.execution_engines}")
            print(f"Integration Requirements: {technical_requirements.integration_requirements}")
            print(f"Data Collection Requirements: {technical_requirements.data_collection_requirements}")
            print(f"Prohibited Shortcuts: {technical_requirements.prohibited_shortcuts}")
            print(f"Technical Guidance Needed: {technical_requirements.technical_guidance_needed}")

            # Validation
            print("\nValidation:")
            if "PostgreSQL" in query and "PostgreSQL" in technical_requirements.source_code_modifications:
                print("✓ PostgreSQL correctly identified for source code modification")
            if "pg_duckdb" in query and "pg_duckdb" in technical_requirements.execution_engines:
                print("✓ pg_duckdb correctly identified for execution engines")
            if "machine learning" in query.lower() and "ML" in str(technical_requirements.integration_requirements):
                print("✓ Machine learning integration correctly identified")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_requirements_extraction())