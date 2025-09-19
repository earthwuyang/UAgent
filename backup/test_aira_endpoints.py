#!/usr/bin/env python3
"""
Test script for AIRA integration endpoints
Tests the API structure without requiring full module imports
"""

import requests
import json

def test_aira_endpoints():
    """Test AIRA endpoints with curl-like requests"""
    base_url = "http://localhost:8000/api/research-tree"

    endpoints_to_test = [
        ("GET", "/aira/config", "Get AIRA configuration"),
        ("POST", "/aira/enable", "Enable AIRA mode"),
        ("GET", "/aira/status", "Get AIRA status"),
        ("POST", "/aira/disable", "Disable AIRA mode"),
    ]

    print("🧪 AIRA Integration Test Plan")
    print("=" * 50)

    for method, endpoint, description in endpoints_to_test:
        print(f"{method:6} {endpoint:20} - {description}")

    print("\n📋 Test Goals:")
    print("1. Verify AIRA configuration endpoints work")
    print("2. Test enable/disable AIRA mode")
    print("3. Confirm AIRA-enhanced goal creation")
    print("4. Validate MCTS policy configuration")

    print("\n🔬 Manual Testing Steps:")
    print("1. Start the uvicorn server: uvicorn app.main:app --reload")
    print("2. Test endpoints with curl or Postman:")

    for method, endpoint, description in endpoints_to_test:
        if method == "GET":
            print(f"   curl -X {method} {base_url}{endpoint}")
        else:
            print(f"   curl -X {method} {base_url}{endpoint} -H 'Content-Type: application/json'")

    print("\n3. Test AIRA-enhanced goal creation:")
    print(f"   curl -X POST {base_url}/goals/start \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{")
    print('            "title": "Test AIRA ML Task",')
    print('            "description": "Test MCTS-enhanced ML research",')
    print('            "success_criteria": ["Successfully run AIRA operators"],')
    print('            "use_aira": true')
    print("        }'")

    print("\n✨ Expected AIRA Enhancements:")
    print("- MCTS tree search with UCT (c=0.25)")
    print("- Draft/Improve/Debug operators")
    print("- 5-fold cross validation enforcement")
    print("- Top-k final selection")
    print("- Enhanced evaluation metrics")

if __name__ == "__main__":
    test_aira_endpoints()