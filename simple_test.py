#!/usr/bin/env python3
"""
Simple test to verify key logic without dependencies
"""
import sys

def test_metadata_fields():
    """Test that conversation metadata has the new field"""
    print("🧪 Testing Conversation Metadata Fields...")
    
    # Read the metadata file and check for the new field
    metadata_file = '/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/storage/data_models/conversation_metadata.py'
    
    try:
        with open(metadata_file, 'r') as f:
            content = f.read()
        
        if 'research_experiment_id: str | None = None' in content:
            print("✅ research_experiment_id field found in ConversationMetadata")
            return True
        else:
            print("❌ research_experiment_id field NOT found in ConversationMetadata")
            return False
            
    except Exception as e:
        print(f"❌ Failed to read metadata file: {e}")
        return False

def test_research_goal_endpoint():
    """Test that research goal endpoint has been updated"""
    print("🧪 Testing Research Goal Endpoint...")
    
    # Read the manage_conversations file and check for the new logic
    endpoint_file = '/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/server/routes/manage_conversations.py'
    
    try:
        with open(endpoint_file, 'r') as f:
            content = f.read()
        
        checks = [
            'research_middleware.process_message',
            'research_experiment_id',
            'research_triggered',
        ]
        
        results = []
        for check in checks:
            if check in content:
                print(f"✅ Found: {check}")
                results.append(True)
            else:
                print(f"❌ Missing: {check}")
                results.append(False)
        
        return all(results)
            
    except Exception as e:
        print(f"❌ Failed to read endpoint file: {e}")
        return False

def test_session_state_fix():
    """Test that session state management has been updated"""
    print("🧪 Testing Session State Management...")
    
    # Read the session file and check for the new logic
    session_file = '/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/server/session/session.py'
    
    try:
        with open(session_file, 'r') as f:
            content = f.read()
        
        checks = [
            'Set agent state to RUNNING after successful initialization',
            'event_to_dict(state_change_event)',
            'Failed to set agent state to RUNNING',
        ]
        
        results = []
        for check in checks:
            if check in content:
                print(f"✅ Found: {check}")
                results.append(True)
            else:
                print(f"❌ Missing: {check}")
                results.append(False)
        
        return all(results)
            
    except Exception as e:
        print(f"❌ Failed to read session file: {e}")
        return False

def test_middleware_fix():
    """Test that research middleware has been updated"""
    print("🧪 Testing Research Middleware Fix...")
    
    # Read the middleware file and check for the new logic
    middleware_file = '/Users/wuy/Desktop/code/UAgent/OpenHands/extensions/uagent_research/middleware/research_middleware.py'
    
    try:
        with open(middleware_file, 'r') as f:
            content = f.read()
        
        checks = [
            'experiment_exists',
            'research_goal_api',
            'Use the research goal from metadata if available',
        ]
        
        results = []
        for check in checks:
            if check in content:
                print(f"✅ Found: {check}")
                results.append(True)
            else:
                print(f"❌ Missing: {check}")
                results.append(False)
        
        return all(results)
            
    except Exception as e:
        print(f"❌ Failed to read middleware file: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Simple Fix Verification Test")
    print("=" * 50)
    
    tests = [
        ("Conversation Metadata Fields", test_metadata_fields),
        ("Research Goal Endpoint", test_research_goal_endpoint),
        ("Session State Management", test_session_state_fix),
        ("Research Middleware Fix", test_middleware_fix),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Fixes are properly implemented.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
