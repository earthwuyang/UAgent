#!/usr/bin/env python3
"""
Test Complete Workflow: Infrastructure Deployment + Report Generation
"""

import requests
import json
import time

API_BASE = "http://localhost:8012/api/research-tree"

def test_complete_workflow():
    print("🧪 Testing Complete UAgent Workflow")
    print("=" * 60)

    # Step 1: Create a PostgreSQL infrastructure deployment goal
    print("\n📋 Step 1: Creating PostgreSQL Infrastructure Goal")
    goal_data = {
        "title": "Setup PostgreSQL database for ML research",
        "description": "Deploy a PostgreSQL instance with research schema for AI/ML experiments",
        "success_criteria": [
            "PostgreSQL instance is running",
            "Database accepts connections",
            "Research schema is configured",
            "Deployment scripts are ready"
        ],
        "constraints": {
            "database_name": "ml_research_db",
            "username": "researcher",
            "password": "secure_research_123"
        }
    }

    try:
        response = requests.post(f"{API_BASE}/goals/start", json=goal_data)
        response.raise_for_status()
        result = response.json()
        goal_id = result["goal_id"]
        print(f"✅ Goal created: {goal_id}")
        print(f"   Title: {result['title']}")
        print(f"   Status: {result['status']}")
    except Exception as e:
        print(f"❌ Failed to create goal: {e}")
        return None

    # Step 2: Check goal status
    print(f"\n📊 Step 2: Checking Goal Status")
    try:
        response = requests.get(f"{API_BASE}/goals/{goal_id}/status")
        response.raise_for_status()
        status = response.json()
        print(f"✅ Goal status retrieved")
        print(f"   Goal: {status['goal']['title']}")
        print(f"   Status: {status['goal']['status']}")
        print(f"   Nodes: {status['tree_stats']['total_nodes']}")
        print(f"   Best confidence: {status['tree_stats']['best_confidence']}")
    except Exception as e:
        print(f"❌ Failed to get status: {e}")

    # Step 3: Generate report
    print(f"\n📄 Step 3: Generating Report")
    try:
        response = requests.post(f"{API_BASE}/goals/{goal_id}/generate-report")
        response.raise_for_status()
        report_result = response.json()
        print(f"✅ Report generated")
        print(f"   Message: {report_result['message']}")
        print(f"   Report path: {report_result['report_path']}")
        print(f"   View URL: {report_result['view_url']}")
        print(f"   Download URL: {report_result['download_url']}")
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")

    # Step 4: Get raw report content
    print(f"\n📖 Step 4: Getting Report Content")
    try:
        response = requests.get(f"{API_BASE}/goals/{goal_id}/report/raw")
        response.raise_for_status()
        raw_report = response.json()
        print(f"✅ Raw report retrieved")
        print(f"   Filename: {raw_report['filename']}")
        print(f"   Generated at: {raw_report['generated_at']}")
        print(f"\n📄 Report Preview (first 500 chars):")
        print("-" * 50)
        print(raw_report['markdown_content'][:500] + "...")
        print("-" * 50)
    except Exception as e:
        print(f"❌ Failed to get raw report: {e}")

    # Step 5: Test report viewer
    print(f"\n🌐 Step 5: Testing Report Viewer")
    try:
        response = requests.get(f"{API_BASE}/goals/{goal_id}/report/view")
        if response.status_code == 200:
            print(f"✅ Report viewer accessible")
            print(f"   Content type: {response.headers.get('content-type')}")
            print(f"   Content length: {len(response.text)} characters")
            print(f"   URL: http://localhost:8012{API_BASE}/goals/{goal_id}/report/view")
        else:
            print(f"❌ Report viewer failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to access report viewer: {e}")

    # Step 6: Check workspace files
    print(f"\n📁 Step 6: Checking Workspace Files")
    try:
        # Get goal details to find workspace path
        response = requests.get(f"{API_BASE}/goals/{goal_id}/status")
        response.raise_for_status()
        status = response.json()

        # Look for infrastructure deployment information
        if 'constraints' in status['goal']:
            workspace_path = status['goal']['constraints'].get('workspace_path')
            if workspace_path:
                print(f"✅ Infrastructure workspace found: {workspace_path}")
                import os
                if os.path.exists(workspace_path):
                    files = os.listdir(workspace_path)
                    print(f"   Generated files: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")
                else:
                    print(f"⚠️  Workspace path doesn't exist: {workspace_path}")
            else:
                print("ℹ️  No infrastructure workspace (might be research-only goal)")
        else:
            print("ℹ️  No constraint information available")
    except Exception as e:
        print(f"❌ Failed to check workspace: {e}")

    # Step 7: Test history persistence
    print(f"\n💾 Step 7: Testing History Persistence")
    try:
        response = requests.get(f"{API_BASE}/history")
        response.raise_for_status()
        history = response.json()
        print(f"✅ History retrieved")
        print(f"   Total goals in history: {history['total_count']}")
        if history['history']:
            latest = history['history'][0]
            print(f"   Latest goal: {latest['title']} (Status: {latest['status']})")
    except Exception as e:
        print(f"❌ Failed to get history: {e}")

    print(f"\n🎉 Workflow Test Complete!")
    print(f"Goal ID: {goal_id}")
    return goal_id

if __name__ == "__main__":
    test_complete_workflow()