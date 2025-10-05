#!/usr/bin/env python3
"""
Test UAgent Research Extension Integration

This script tests the UAgent research extension endpoints.
"""

import requests
import json
import time
from datetime import datetime

# Base URL
BASE_URL = "http://120.46.207.248:3000"
API_URL = f"{BASE_URL}/api/research"

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")

def test_health_check():
    """Test 1: Health Check"""
    print_header("Test 1: Health Check")

    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()

        print_success("Health check passed")
        print(f"   Extension: {data['extension']}")
        print(f"   Version: {data['version']}")
        print(f"   Status: {data['status']}")
        return True
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_start_experiment():
    """Test 2: Start Scientific Experiment"""
    print_header("Test 2: Start Scientific Experiment")

    try:
        payload = {
            "goal": "Compare the performance of quicksort vs mergesort on random data",
            "session_id": f"test_session_{int(time.time())}",
            "research_type": "scientific",
            "config": {
                "max_iterations": 10,
                "timeout": 300
            }
        }

        print_info(f"Starting experiment with goal: {payload['goal']}")

        response = requests.post(f"{API_URL}/experiments/start", json=payload)
        response.raise_for_status()
        data = response.json()

        print_success("Experiment created")
        print(f"   ID: {data['id']}")
        print(f"   Status: {data['status']}")
        print(f"   Type: {data['experiment_type']}")
        print(f"   Session: {data['session_id']}")

        return data['id']
    except Exception as e:
        print_error(f"Start experiment failed: {e}")
        if hasattr(e, 'response'):
            print(f"   Response: {e.response.text}")
        return None

def test_get_experiment(experiment_id):
    """Test 3: Get Experiment Status"""
    print_header("Test 3: Get Experiment Status")

    if not experiment_id:
        print_error("No experiment ID provided")
        return False

    try:
        response = requests.get(f"{API_URL}/experiments/{experiment_id}")
        response.raise_for_status()
        data = response.json()

        print_success("Retrieved experiment")
        print(f"   ID: {data['id']}")
        print(f"   Status: {data['status']}")
        print(f"   Progress: {data['progress_percentage']}%")
        print(f"   Goal: {data['goal']}")
        if data['current_step']:
            print(f"   Current Step: {data['current_step']}")
        if data['results']:
            print(f"   Results: {json.dumps(data['results'], indent=2)}")

        return True
    except Exception as e:
        print_error(f"Get experiment failed: {e}")
        return False

def test_list_experiments():
    """Test 4: List All Experiments"""
    print_header("Test 4: List All Experiments")

    try:
        response = requests.get(f"{API_URL}/experiments?limit=10")
        response.raise_for_status()
        data = response.json()

        print_success(f"Retrieved {len(data)} experiments")
        for i, exp in enumerate(data[:5], 1):  # Show first 5
            print(f"   {i}. {exp['id']}")
            print(f"      Status: {exp['status']}, Progress: {exp['progress_percentage']}%")
            print(f"      Goal: {exp['goal'][:60]}...")

        if len(data) > 5:
            print(f"   ... and {len(data) - 5} more")

        return True
    except Exception as e:
        print_error(f"List experiments failed: {e}")
        return False

def test_generate_ideas():
    """Test 5: Generate Research Ideas"""
    print_header("Test 5: Generate Research Ideas")

    try:
        payload = {
            "topic": "Machine Learning Optimization",
            "context": "Exploring novel approaches to improve ML model training",
            "num_ideas": 3,
            "creativity": 0.8
        }

        print_info(f"Generating ideas for topic: {payload['topic']}")

        response = requests.post(f"{API_URL}/ideas/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        print_success(f"Generated {len(data['ideas'])} ideas")
        for i, idea in enumerate(data['ideas'], 1):
            print(f"   {i}. {idea['title']}")
            print(f"      Novelty: {idea['novelty_score']}, "
                  f"Feasibility: {idea['feasibility_score']}, "
                  f"Impact: {idea['impact_score']}")

        return True
    except Exception as e:
        print_error(f"Generate ideas failed: {e}")
        return False

def test_generate_hypotheses():
    """Test 6: Generate Hypotheses"""
    print_header("Test 6: Generate Hypotheses")

    try:
        payload = {
            "idea": "Use neural architecture search to find optimal CNN structures",
            "background": "Recent advances in AutoML",
            "num_hypotheses": 2
        }

        print_info(f"Generating hypotheses for idea: {payload['idea']}")

        response = requests.post(f"{API_URL}/hypotheses/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        print_success(f"Generated {len(data['hypotheses'])} hypotheses")
        for i, hyp in enumerate(data['hypotheses'], 1):
            print(f"   {i}. {hyp['statement']}")
            print(f"      Testability: {hyp['testability_score']}")

        return True
    except Exception as e:
        print_error(f"Generate hypotheses failed: {e}")
        return False

def test_code_research_experiment():
    """Test 7: Start Code Research Experiment"""
    print_header("Test 7: Start Code Research Experiment (RepoMaster)")

    try:
        payload = {
            "goal": "Analyze the OpenHands codebase and identify key architectural patterns",
            "session_id": f"code_session_{int(time.time())}",
            "research_type": "code",
            "config": {
                "repository_path": "/workspace"
            }
        }

        print_info(f"Starting code research: {payload['goal']}")

        response = requests.post(f"{API_URL}/experiments/start", json=payload)
        response.raise_for_status()
        data = response.json()

        print_success("Code research experiment created")
        print(f"   ID: {data['id']}")
        print(f"   Type: {data['experiment_type']}")
        print(f"   Status: {data['status']}")

        return data['id']
    except Exception as e:
        print_error(f"Code research experiment failed: {e}")
        return None

def run_all_tests():
    """Run all tests"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║  UAgent Research Extension Integration Tests              ║")
    print("╚════════════════════════════════════════════════════════════╝")

    results = []

    # Test 1: Health check
    results.append(("Health Check", test_health_check()))

    # Test 2: Start scientific experiment
    experiment_id = test_start_experiment()
    results.append(("Start Experiment", experiment_id is not None))

    # Test 3: Get experiment
    results.append(("Get Experiment", test_get_experiment(experiment_id)))

    # Test 4: List experiments
    results.append(("List Experiments", test_list_experiments()))

    # Test 5: Generate ideas
    results.append(("Generate Ideas", test_generate_ideas()))

    # Test 6: Generate hypotheses
    results.append(("Generate Hypotheses", test_generate_hypotheses()))

    # Test 7: Code research
    code_exp_id = test_code_research_experiment()
    results.append(("Code Research", code_exp_id is not None))

    # Summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")

    print(f"\n{'='*60}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*60}\n")

    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        exit(1)
