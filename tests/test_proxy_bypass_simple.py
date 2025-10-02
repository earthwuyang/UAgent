#!/usr/bin/env python3
"""
Simple test script to verify proxy bypass for localhost connections
"""

import os
import sys
import httpx

print("🧪 Testing Proxy Bypass for Localhost Connections")
print("=" * 50)

# Check initial environment
print(f"Initial NO_PROXY: {os.environ.get('NO_PROXY', 'not set')}")
print(f"Initial no_proxy: {os.environ.get('no_proxy', 'not set')}")

# Test the HTTP session bypass
print("\n🌐 Testing HTTP Session Proxy Bypass")
print("-" * 40)

# Add OpenHands to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OpenHands'))

# Import the HTTP session module that should set proxy bypass
try:
    from OpenHands.openhands.utils.http_session import HttpSession
    print("✅ HTTP session module imported successfully")

    print(f"After import NO_PROXY: {os.environ.get('NO_PROXY', 'not set')}")
    print(f"After import no_proxy: {os.environ.get('no_proxy', 'not set')}")

    # Create a session
    session = HttpSession()
    print("✅ HttpSession created successfully")

    # Test localhost connection
    print("\n🔌 Testing localhost connection bypass...")
    try:
        # This should bypass proxy for localhost
        response = session.get("http://localhost:35100/test", timeout=2)
        print("✅ Localhost connection successful")
    except Exception as e:
        if "Connection refused" in str(e) or "refused" in str(e):
            print("✅ Connection refused (expected - no server), but bypassed proxy")
        elif "502 Bad Gateway" in str(e):
            print("❌ 502 Bad Gateway - proxy is still being used!")
        else:
            print(f"⚠️ Other connection error: {e}")

except ImportError as e:
    print(f"❌ Failed to import HTTP session: {e}")

# Test MCP utils
print("\n🔧 Testing MCP Utils Proxy Bypass")
print("-" * 40)

try:
    from OpenHands.openhands.mcp.utils import create_mcp_clients
    print("✅ MCP utils module imported successfully")

    print(f"After MCP import NO_PROXY: {os.environ.get('NO_PROXY', 'not set')}")
    print(f"After MCP import no_proxy: {os.environ.get('no_proxy', 'not set')}")

except ImportError as e:
    print(f"❌ Failed to import MCP utils: {e}")

# Final check
print(f"\n🎯 Final Environment:")
print(f"NO_PROXY: {os.environ.get('NO_PROXY', 'not set')}")
print(f"no_proxy: {os.environ.get('no_proxy', 'not set')}")

# Check if localhost is in the proxy bypass list
no_proxy_value = os.environ.get('NO_PROXY', '') or os.environ.get('no_proxy', '')
if 'localhost' in no_proxy_value.lower():
    print("✅ localhost is in proxy bypass list")
else:
    print("❌ localhost is NOT in proxy bypass list")

print("\n✅ Proxy bypass test completed")