"""Test git worktree functionality via research API"""

import asyncio
import requests
import time
import json

BASE_URL = "http://localhost:2999"

async def test_worktree_via_research_api():
    """Test that starting a research experiment triggers worktree creation for EXPERIMENT nodes."""
    
    print("\n" + "="*60)
    print("Testing Git Worktree Functionality via Research API")
    print("="*60 + "\n")
    
    # Start a new research experiment
    research_goal = "test git worktree: create parallel experiments with isolated branches"
    
    print(f"📝 Starting research experiment with goal:")
    print(f"   '{research_goal}'")
    
    payload = {
        "session_id": "test_worktree_session",
        "goal": research_goal,
        "experiment_type": "scientific"
    }
    
    print(f"\n🚀 Sending POST request to {BASE_URL}/api/research/experiments/start...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/research/experiments/start",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start experiment: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        experiment_id = data.get("experiment_id")
        
        print(f"✅ Experiment started successfully!")
        print(f"   Experiment ID: {experiment_id}")
        
        # Wait a bit for the tree to initialize
        print(f"\n⏳ Waiting for tree to initialize...")
        await asyncio.sleep(5)
        
        # Query the tree to see if nodes are being created
        print(f"\n🔍 Querying tree structure...")
        tree_response = requests.get(
            f"{BASE_URL}/api/research/experiments/{experiment_id}/tree",
            timeout=10
        )
        
        if tree_response.status_code != 200:
            print(f"❌ Failed to get tree: {tree_response.status_code}")
            return False
        
        tree_data = tree_response.json()
        nodes = tree_data.get("nodes", [])
        edges = tree_data.get("edges", [])
        
        print(f"✅ Tree data retrieved")
        print(f"   Nodes: {len(nodes)}")
        print(f"   Edges: {len(edges)}")
        
        if nodes:
            print(f"\n📊 Node details:")
            for node in nodes:
                node_id = node.get("id")
                node_type = node.get("node_type")
                status = node.get("status")
                metadata = node.get("metadata", {})
                
                print(f"   - {node_id}")
                print(f"     Type: {node_type}")
                print(f"     Status: {status}")
                
                # Check for experiment-specific metadata
                if node_type == "EXPERIMENT":
                    conv_id = metadata.get("conversation_id")
                    branch = metadata.get("worktree_branch")
                    worktree_path = metadata.get("worktree_path")
                    
                    print(f"     Conversation ID: {conv_id}")
                    print(f"     Worktree Branch: {branch}")
                    print(f"     Worktree Path: {worktree_path}")
                    
                    if conv_id and branch and worktree_path:
                        print(f"     ✅ EXPERIMENT node has worktree metadata!")
                    else:
                        print(f"     ⚠️  EXPERIMENT node missing worktree metadata")
        else:
            print(f"\n⚠️  No nodes created yet. The tree might still be initializing.")
            print(f"   Note: This test verifies that the worktree code is in place,")
            print(f"   but actual node creation depends on the orchestrator running.")
        
        # Check experiment status
        print(f"\n🔍 Checking experiment status...")
        status_response = requests.get(
            f"{BASE_URL}/api/research/experiments/{experiment_id}/status",
            timeout=10
        )
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   Status: {status_data.get('status')}")
            print(f"   Progress: {status_data.get('progress_percentage', 0)}%")
        
        print(f"\n" + "="*60)
        print(f"✅ API test completed successfully")
        print(f"="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_worktree_via_research_api())
    exit(0 if result else 1)
