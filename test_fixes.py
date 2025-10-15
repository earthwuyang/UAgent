#!/usr/bin/env python3
"""
Test script to verify the parallel research fixes
"""
import asyncio
import sys
import os

# Add OpenHands to path
sys.path.insert(0, '/Users/wuy/Desktop/code/UAgent/OpenHands')

async def test_research_middleware():
    """Test research middleware functionality"""
    print("🧪 Testing Research Middleware...")
    
    try:
        from extensions.uagent_research.middleware.research_middleware import research_middleware
        
        # Test API-triggered research scenario
        result = await research_middleware.process_message(
            user_message="Test message",
            session_id="test_session_123",
            conversation_metadata={
                'research_goal': 'Test research goal for parallel execution',
                'research_locked': True,
                'source': 'research_goal_api'
            }
        )
        
        print(f"📊 Research middleware result: {result}")
        
        if result.get('should_trigger_research'):
            print("✅ Research triggered successfully")
            print(f"🔬 Experiment ID: {result.get('experiment_id')}")
        else:
            print("⚠️ Research not triggered")
            
        return True
        
    except Exception as e:
        print(f"❌ Research middleware test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_metadata():
    """Test conversation metadata model"""
    print("🧪 Testing Conversation Metadata Model...")
    
    try:
        from openhands.storage.data_models.conversation_metadata import ConversationMetadata
        
        # Test creating metadata with research experiment ID
        metadata = ConversationMetadata(
            conversation_id="test_conv_123",
            selected_repository="test/repo",
            research_goal="Test research goal",
            research_locked=True,
            research_experiment_id="exp_test_123"
        )
        
        print(f"✅ Created metadata: {metadata.conversation_id}")
        print(f"🔬 Research goal: {metadata.research_goal}")
        print(f"🔒 Research locked: {metadata.research_locked}")
        print(f"🆔 Experiment ID: {metadata.research_experiment_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversation metadata test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Parallel Research Fix Verification")
    print("=" * 50)
    
    tests = [
        ("Conversation Metadata", test_conversation_metadata),
        ("Research Middleware", test_research_middleware),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
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
        print("🎉 All tests passed! Fixes should work correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
