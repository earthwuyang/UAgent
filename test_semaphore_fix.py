#!/usr/bin/env python3
"""
Test script to verify the parallel execution fix.
This script tests that the semaphore properly limits concurrent execution.
"""

import asyncio
import time

# Mock classes to simulate the orchestrator behavior
class MockSemaphore:
    def __init__(self, value):
        self._value = value
        self._lock = asyncio.Lock()
        
    async def __aenter__(self):
        async with self._lock:
            while self._value <= 0:
                await asyncio.sleep(0.01)
            self._value -= 1
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            self._value += 1

class MockNode:
    def __init__(self, id, title):
        self.id = id
        self.title = title
        self.type = "test"

class TestOrchestrator:
    def __init__(self, max_parallel=3):
        self.max_parallel = max_parallel
        self._semaphore = MockSemaphore(max_parallel)
        self._running_tasks = {}
        self.execution_log = []
        
    async def _execute_node(self, node):
        """Simulate node execution with timing."""
        start_time = time.time()
        self.execution_log.append(f"Started {node.id} at {start_time}")
        
        # Simulate some work
        await asyncio.sleep(0.5)
        
        end_time = time.time()
        self.execution_log.append(f"Finished {node.id} at {end_time}")
        return f"Result for {node.id}"
        
    # FIXED VERSION
    async def _execute_children_parallel_fixed(self, children):
        """
        Execute children nodes in parallel with bounded concurrency (FIXED VERSION).
        """
        print(f"Starting parallel execution of {len(children)} children")
        print(f"Concurrency limit: {self.max_parallel}")
        
        # Use semaphore to limit concurrent execution properly
        semaphore = self._semaphore
        
        async def limited_execute_node(node):
            """Execute node with semaphore limiting."""
            async with semaphore:
                return await self._execute_node(node)
        
        tasks = []

        for child in children:
            task = asyncio.create_task(limited_execute_node(child))
            self._running_tasks[child.id] = task
            task.add_done_callback(
                lambda t, node_id=child.id: self._running_tasks.pop(node_id, None)
            )
            tasks.append(task)

        print(f"Created {len(tasks)} asyncio tasks for parallel execution")
        
        try:
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print(f"All {len(tasks)} tasks completed")
            return results
        finally:
            for child in children:
                self._running_tasks.pop(child.id, None)

    # ORIGINAL BUGGY VERSION
    async def _execute_children_parallel_buggy(self, children):
        """
        Execute children nodes in parallel with bounded concurrency (BUGGY VERSION).
        """
        print(f"Starting parallel execution of {len(children)} children")
        print(f"Concurrency limit: {self.max_parallel}")
        
        tasks = []

        for child in children:
            # BUG: All tasks are created at once without semaphore limiting
            task = asyncio.create_task(self._execute_node(child))
            self._running_tasks[child.id] = task
            task.add_done_callback(
                lambda t, node_id=child.id: self._running_tasks.pop(node_id, None)
            )
            tasks.append(task)

        print(f"Created {len(tasks)} asyncio tasks for parallel execution")
        
        try:
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print(f"All {len(tasks)} tasks completed")
            return results
        finally:
            for child in children:
                self._running_tasks.pop(child.id, None)

async def test_buggy_version():
    """Test the buggy version - all tasks start at once."""
    print("\n=== Testing BUGGY version ===")
    orchestrator = TestOrchestrator(max_parallel=2)
    
    # Create 5 children
    children = [MockNode(f"node_{i}", f"Node {i}") for i in range(5)]
    
    start_time = time.time()
    await orchestrator._execute_children_parallel_buggy(children)
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print("Execution log:")
    for log in orchestrator.execution_log:
        print(f"  {log}")

async def test_fixed_version():
    """Test the fixed version - tasks are limited by semaphore."""
    print("\n=== Testing FIXED version ===")
    orchestrator = TestOrchestrator(max_parallel=2)
    
    # Create 5 children
    children = [MockNode(f"node_{i}", f"Node {i}") for i in range(5)]
    
    start_time = time.time()
    await orchestrator._execute_children_parallel_fixed(children)
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print("Execution log:")
    for log in orchestrator.execution_log:
        print(f"  {log}")

async def main():
    """Run both tests to compare behavior."""
    print("Testing parallel execution fix...")
    await test_buggy_version()
    await test_fixed_version()
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(main())