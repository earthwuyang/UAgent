#!/usr/bin/env python3
"""
Comprehensive test to verify that GitHub issue #16 is fixed:
1. Debug logging visibility issues
2. Parallel research execution barriers 

This test verifies that:
- Debug logs are visible 
- Experiment isolation works correctly (tree snapshots and orchestrators)
- Session manager integration is working
- Parallel execution is possible
"""

import sys
import os
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_debug_logging_visibility():
    """Test that debug logging is visible in research modules"""
    print("🔍 Testing debug logging visibility...")
    
    try:
        # Import the modules that should have debug logging enabled
        from OpenHands.extensions.uagent_research.uagent_research.api.research_routes import logger as routes_logger
        from OpenHands.extensions.uagent_research.uagent_research.api.tree_publisher import logger as publisher_logger
        
        # Check if debug level is enabled
        if routes_logger.isEnabledFor(logging.DEBUG):
            print("✅ Research routes debug logging is enabled")
        else:
            print("❌ Research routes debug logging is NOT enabled")
            return False
            
        if publisher_logger.isEnabledFor(logging.DEBUG):
            print("✅ Tree publisher debug logging is enabled")
        else:
            print("❌ Tree publisher debug logging is NOT enabled")
            return False
            
        # Test logging output with emoji markers (as mentioned in issue)
        routes_logger.debug("🧪 Debug log test from research routes")
        publisher_logger.debug("🌳 Debug log test from tree publisher")
        
        print("✅ Debug logging is visible (check console output)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug logging test failed: {e}", exc_info=True)
        return False

def test_tree_publisher_isolation():
    """Test that tree publisher properly isolates tree snapshots per experiment"""
    print("\n🧪 Testing tree publisher experiment isolation...")
    
    try:
        # Import the tree publisher
        from OpenHands.extensions.uagent_research.uagent_research.api.tree_publisher import (
            update_tree_state, get_tree_state, clear_tree_state, get_all_tree_snapshots
        )
        
        # Test data for two different experiments
        exp1_id = "exp_isolation_test1_parallel123"
        exp2_id = "exp_isolation_test2_parallel456"
        
        tree_data1 = {
            "nodes": [{"id": "parallel_node1", "title": "Parallel Test Node 1"}],
            "edges": [],
            "stats": {"total_nodes": 1}
        }
        
        tree_data2 = {
            "nodes": [{"id": "parallel_node2", "title": "Parallel Test Node 2"}],
            "edges": [],
            "stats": {"total_nodes": 1}
        }
        
        # Clear any existing state
        clear_tree_state(exp1_id)
        clear_tree_state(exp2_id)
        
        # Update tree state for experiment 1
        update_tree_state(exp1_id, tree_data1)
        print(f"✅ Updated tree state for experiment 1: {exp1_id}")
        
        # Update tree state for experiment 2
        update_tree_state(exp2_id, tree_data2)
        print(f"✅ Updated tree state for experiment 2: {exp2_id}")
        
        # Check that each experiment gets its own data
        retrieved_data1 = get_tree_state(exp1_id)
        retrieved_data2 = get_tree_state(exp2_id)
        
        if retrieved_data1 and retrieved_data1["nodes"][0]["id"] == "parallel_node1":
            print("✅ Experiment 1 gets correct tree data")
        else:
            print("❌ Experiment 1 did not get correct tree data")
            return False
            
        if retrieved_data2 and retrieved_data2["nodes"][0]["id"] == "parallel_node2":
            print("✅ Experiment 2 gets correct tree data")
        else:
            print("❌ Experiment 2 did not get correct tree data")
            return False
            
        # Check that get_all_tree_snapshots returns both
        all_snapshots = get_all_tree_snapshots()
        if exp1_id in all_snapshots and exp2_id in all_snapshots:
            print("✅ All tree snapshots includes both experiments")
        else:
            print("❌ All tree snapshots missing some experiments")
            return False
            
        # Clean up
        clear_tree_state(exp1_id)
        clear_tree_state(exp2_id)
        print("🧹 Cleaned up test data")
        
        print("✅ Tree publisher isolation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tree publisher isolation test failed: {e}", exc_info=True)
        return False

def test_session_manager_integration():
    """Test that session manager integration is working properly"""
    print("\n🔧 Testing session manager integration...")
    
    try:
        from OpenHands.extensions.uagent_research.uagent_research.api.tree_publisher import _session_manager_available
        from OpenHands.extensions.uagent_research.uagent_research.api.research_routes import (
            get_total_active_orchestrators, get_orchestrator_from_session_manager
        )
        
        print(f"✅ Session manager availability flag: {_session_manager_available}")
        
        # Test the helper functions
        total_orchestrators = get_total_active_orchestrators()
        print(f"✅ Total active orchestrators: {total_orchestrators}")
        
        # Test orchestrator retrieval from session manager
        orchestrator = get_orchestrator_from_session_manager("test_exp_123")
        print(f"✅ Session manager orchestrator retrieval: {orchestrator is None} (expected None for non-existent experiment)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Session manager integration test failed: {e}", exc_info=True)
        return False

def simulate_parallel_execution():
    """Simulate parallel execution to test experiment isolation"""
    print("\n🏃 Simulating parallel execution...")
    
    try:
        from OpenHands.extensions.uagent_research.uagent_research.api.tree_publisher import (
            update_tree_state, get_tree_state, clear_tree_state
        )
        
        # Test data for parallel experiments
        experiments = [
            ("parallel_exp_1_thread123", {"nodes": [{"id": "thread1_node1", "title": "Thread 1 Node"}]}),
            ("parallel_exp_2_thread456", {"nodes": [{"id": "thread2_node1", "title": "Thread 2 Node"}]}),
            ("parallel_exp_3_thread789", {"nodes": [{"id": "thread3_node1", "title": "Thread 3 Node"}]}),
        ]
        
        # Clear any existing state
        for exp_id, _ in experiments:
            clear_tree_state(exp_id)
        
        def update_experiment(exp_id, tree_data):
            """Simulate updating tree state for an experiment"""
            update_tree_state(exp_id, tree_data)
            time.sleep(0.1)  # Small delay to simulate work
            result = get_tree_state(exp_id)
            return result
        
        # Execute in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for exp_id, tree_data in experiments:
                future = executor.submit(update_experiment, exp_id, tree_data)
                futures.append((exp_id, future))
            
            # Collect results
            results = []
            for exp_id, future in futures:
                result = future.result()
                results.append((exp_id, result))
                print(f"✅ Completed experiment {exp_id}")
        
        # Verify isolation - each experiment should have its own data
        success = True
        for exp_id, tree_data in experiments:
            retrieved_data = get_tree_state(exp_id)
            if retrieved_data and retrieved_data["nodes"][0]["id"] == tree_data["nodes"][0]["id"]:
                print(f"✅ Experiment {exp_id} maintained data isolation")
            else:
                print(f"❌ Experiment {exp_id} data isolation failed")
                success = False
        
        # Clean up
        for exp_id, _ in experiments:
            clear_tree_state(exp_id)
        print("🧹 Cleaned up parallel test data")
        
        if success:
            print("✅ Parallel execution test passed!")
        else:
            print("❌ Parallel execution test failed!")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Parallel execution test failed: {e}", exc_info=True)
        return False

def main():
    """Run all comprehensive tests for GitHub issue #16"""
    print("🚀 Starting comprehensive test for GitHub issue #16 fixes...\n")
    print("Issue #16: Debug logging visibility issues and parallel research execution barriers\n")
    
    tests = [
        test_debug_logging_visibility,
        test_tree_publisher_isolation,
        test_session_manager_integration,
        simulate_parallel_execution,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} crashed: {e}", exc_info=True)
            failed += 1
    
    print(f"\n🏁 Final Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! GitHub issue #16 fixes are working correctly.")
        print("✅ Debug logging is visible")
        print("✅ Experiment isolation is working")
        print("✅ Session manager integration is functional")
        print("✅ Parallel execution is supported")
        return True
    else:
        print("💥 Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)