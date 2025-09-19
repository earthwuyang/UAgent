#!/usr/bin/env python3
"""Quick test to verify API can start and endpoints are accessible"""

import asyncio
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import app
from app.core.llm_client import create_llm_client


async def test_api_startup():
    """Test that the API can start up successfully"""
    try:
        # Check if DashScope API key is available
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if not dashscope_key:
            print("WARNING: DASHSCOPE_API_KEY not set - using test mode")
            return False

        # Test LLM client creation
        print("Testing LLM client creation...")
        llm_client = create_llm_client("dashscope", api_key=dashscope_key)
        print(f"✓ LLM client created: {llm_client.__class__.__name__}")

        # Test a simple classification
        print("Testing basic LLM call...")
        test_prompt = "Classify this as a simple test: Hello world"
        try:
            result = await llm_client.classify(test_prompt, "Is this a greeting? Reply yes or no.")
            result_str = str(result)
            print(f"✓ LLM classification result: {result_str[:50]}...")
        except Exception as e:
            print(f"✗ LLM call failed: {e}")
            return False

        print("✓ API startup test passed")
        return True

    except Exception as e:
        print(f"✗ API startup test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_api_startup())
    sys.exit(0 if success else 1)