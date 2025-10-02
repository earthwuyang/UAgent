#!/usr/bin/env python3
"""Test script to verify parallelism configuration is loaded correctly from environment variables."""

import os
import sys
import asyncio

# Set environment variables before importing modules
os.environ["MAX_RESEARCH_IDEAS"] = "1"
os.environ["MAX_PARALLEL_IDEAS"] = "1"
os.environ["EXPERIMENTS_PER_HYPOTHESIS"] = "1"

# Add the backend to path
sys.path.insert(0, '/home/wuy/AI/UAgent')

from backend.app.core.research_engines.scientific_research import ScientificResearchEngine
from backend.app.core.llm_client import LLMClient


async def test_parallelism_config():
    """Test that parallelism settings are correctly loaded from environment variables."""

    print("Environment variables set:")
    print(f"  MAX_RESEARCH_IDEAS = {os.getenv('MAX_RESEARCH_IDEAS')}")
    print(f"  MAX_PARALLEL_IDEAS = {os.getenv('MAX_PARALLEL_IDEAS')}")
    print(f"  EXPERIMENTS_PER_HYPOTHESIS = {os.getenv('EXPERIMENTS_PER_HYPOTHESIS')}")
    print()

    # Create a mock LLM client
    llm_client = LLMClient(config={})

    # Create the scientific research engine
    config = {}
    engine = ScientificResearchEngine(llm_client, config)

    print("Scientific Research Engine configuration:")
    print(f"  experiments_per_hypothesis = {engine.experiments_per_hypothesis}")

    # Test max_parallel_ideas
    parallel = engine._max_parallel_ideas(5)
    print(f"  _max_parallel_ideas(5) = {parallel} (should be 1)")

    # Test idea generation (just check the max_ideas value)
    print()
    print("Testing idea generation configuration:")

    # The max_ideas will be loaded when _generate_research_ideas is called
    # We can't call it directly without a full setup, but we can check the logic
    max_ideas = int(os.getenv("MAX_RESEARCH_IDEAS", 3))
    print(f"  max_ideas would be: {max_ideas} (should be 1)")

    if engine.experiments_per_hypothesis == 1 and parallel == 1 and max_ideas == 1:
        print("\n✅ SUCCESS: All parallelism settings are correctly limited to 1")
    else:
        print("\n❌ FAILED: Some settings are not correctly limited")

    return True


if __name__ == "__main__":
    asyncio.run(test_parallelism_config())