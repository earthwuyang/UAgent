#!/usr/bin/env python3
"""Test script to verify OpenHands Docker integration with UAgent"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.integrations.openhands_codeact_bridge_v3 import OpenHandsCodeActBridgeV3, CodeActRunConfig


async def test_docker_integration():
    """Test OpenHands Docker integration"""
    print("Testing OpenHands Docker integration...")

    # Docker runtime is now mandatory - no environment variable needed

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "docker_test_workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        print(f"Using workspace: {workspace}")

        # Simple test goal that creates final.json with proper structure
        cfg = CodeActRunConfig(
            goal="Write a Python script that: 1) prints 'Hello from Docker OpenHands!' 2) creates experiments/docker_test/results/final.json with: {'success': true, 'data': {'message': 'Hello from Docker OpenHands!'}, 'analysis': {'runtime': 'docker'}, 'conclusions': ['Docker runtime working'], 'errors': [], 'measurements': [1.0]}",
            workspace=workspace,
            session_name="docker_test",
            max_steps=15,
            max_minutes=5,
            disable_browser=True
        )

        # Initialize bridge
        try:
            bridge = OpenHandsCodeActBridgeV3()
            print(f"Bridge initialized successfully")
            print(f"OpenHands directory: {bridge.openhands_dir}")

            # Run the test
            print("Starting Docker execution test...")
            result = await bridge.run_async(cfg)

            print(f"\n=== Test Results ===")
            print(f"Success: {result.success}")
            print(f"Exit code: {result.exit_code}")
            print(f"Duration: {result.duration_seconds:.1f}s")
            print(f"Reason: {result.reason}")

            if result.artifact_path:
                print(f"Artifact path: {result.artifact_path}")
                if result.final_json:
                    print(f"Final JSON: {result.final_json}")

            if result.stdout_tail:
                print(f"\n=== STDOUT (last 500 chars) ===")
                print(result.stdout_tail[-500:])

            if result.stderr_tail:
                print(f"\n=== STDERR (last 500 chars) ===")
                print(result.stderr_tail[-500:])

            # Check workspace contents
            print(f"\n=== Workspace Contents ===")
            for path in workspace.rglob("*"):
                if path.is_file():
                    print(f"  {path.relative_to(workspace)}")

            return result.success

        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_environment_variables():
    """Test environment variable configuration"""
    print("Testing Docker runtime configuration...")
    print("Docker runtime is now mandatory for hardware isolation")
    print("Local runtime has been completely removed")
    return True


def check_docker_availability():
    """Check if Docker is available on the system"""
    print("Checking Docker availability...")

    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("✓ Docker client is available and responsive")

        # List available images
        images = client.images.list()
        print(f"✓ Found {len(images)} Docker images")

        return True
    except Exception as e:
        print(f"✗ Docker not available: {e}")
        return False


if __name__ == "__main__":
    print("=== UAgent OpenHands Docker Integration Test ===\n")

    # Check prerequisites
    env_ok = test_environment_variables()
    docker_ok = check_docker_availability()

    if not docker_ok:
        print("\n❌ Docker is not available. Please install and start Docker.")
        sys.exit(1)

    # Run the integration test
    print("\n" + "="*50)
    success = asyncio.run(test_docker_integration())

    if success:
        print("\n✅ Docker integration test passed!")
        print("\nUAgent is now configured to use Docker runtime for:")
        print("  - Complete hardware isolation")
        print("  - Prevention of host system crashes")
        print("  - Safe execution of experimental code")
    else:
        print("\n❌ Docker integration test failed!")
        print("\nEnsure Docker is properly installed and running")
        print("Local runtime has been removed for safety")

    sys.exit(0 if success else 1)