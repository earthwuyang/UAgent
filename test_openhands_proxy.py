#!/usr/bin/env python3
"""
Test script to verify that OpenHands Docker containers receive proxy configuration.
This script creates a test conversation and checks the runtime environment.
"""

import requests
import time
import json
import subprocess

def test_proxy_configuration():
    base_url = "http://localhost:2999"
    
    # Create a new conversation
    print("Creating new conversation...")
    response = requests.post(f"{base_url}/api/conversations")
    if response.status_code != 200:
        print(f"Failed to create conversation: {response.text}")
        return
    
    conversation = response.json()
    conversation_id = conversation['conversation_id']
    print(f"Created conversation: {conversation_id}")
    
    # Wait for runtime to start
    print("Waiting for runtime to start...")
    time.sleep(5)
    
    # Check Docker container
    container_name = f"openhands-runtime-{conversation_id}"
    print(f"Checking container: {container_name}")
    
    # Check if container exists and is running
    result = subprocess.run(
        ["docker", "ps", "-f", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    
    if container_name not in result.stdout:
        print(f"Container {container_name} not found or not running")
        # Try to find any openhands-runtime containers
        result = subprocess.run(
            ["docker", "ps", "-f", "name=openhands-runtime", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )
        if result.stdout:
            print(f"Found containers: {result.stdout}")
            container_name = result.stdout.strip().split('\n')[0]
        else:
            print("No OpenHands runtime containers found")
            return
    
    # Check environment variables in container
    print(f"\nChecking proxy environment in container {container_name}:")
    result = subprocess.run(
        ["docker", "exec", container_name, "sh", "-c", "env | grep -i proxy"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0 and result.stdout:
        print("✅ Proxy configuration found in container:")
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
    else:
        print("❌ No proxy configuration found in container")
        print(f"Error: {result.stderr}")
    
    # Test proxy connectivity from inside container
    print("\nTesting proxy connectivity from container:")
    result = subprocess.run(
        ["docker", "exec", container_name, "sh", "-c",
         "wget -q -O /dev/null --timeout=5 https://www.google.com && echo '✅ Successfully connected through proxy' || echo '❌ Failed to connect through proxy'"],
        capture_output=True, text=True
    )
    print(result.stdout)

if __name__ == "__main__":
    test_proxy_configuration()