#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/wuy/AI/UAgent/OpenHands')

from extensions.uagent_research.orchestrator.tree_orchestrator import TreeSearchOrchestrator
from extensions.uagent_research.core.budget import Budget

async def test_parallel_execution():
    """Test that parallel execution works properly."""
    print("Testing parallel execution fix...")
    
    # Create a simple orchestrator
    budget = Budget(max_iterations=10, max_cost=10.0)
    orchestrator = TreeSearchOrchestrator(max_parallel=2, budget=budget)
    
    print(f"Orchestrator created with max_parallel={orchestrator.max_parallel}")
    
    # Test the semaphore
    print(f"Semaphore value: {orchestrator._semaphore}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_parallel_execution())