#!/usr/bin/env python3
"""
Test script to verify the enhanced OpenHands instructions include JSON validation
"""

import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_enhanced_instructions():
    """Test that the enhanced instructions include JSON validation requirements"""

    # Test single container instructions
    from app.integrations.openhands_single_container import SingleContainerConfig

    # Create a mock config
    class MockConfig:
        def __init__(self):
            self.goal = "Test scientific experiment"
            self.session_name = "test_session"
            self.workspace = Path("/tmp/test")
            self.max_steps = 100

    cfg = MockConfig()

    # Import the class to trigger the enhanced goal creation
    from app.integrations.openhands_single_container import OpenHandsSingleContainer

    # Create instance and check the enhanced goal
    bridge = OpenHandsSingleContainer()

    # We can't easily test the internal enhanced goal creation without running the full method,
    # but we can verify the source file contains our enhancements

    single_container_path = Path(__file__).parent / "backend" / "app" / "integrations" / "openhands_single_container.py"

    if single_container_path.exists():
        content = single_container_path.read_text()
        if "IMPORTANT SELF-VALIDATION REQUIREMENT" in content:
            print("✓ Single container enhanced instructions include JSON validation requirement")
        else:
            print("✗ Single container enhanced instructions missing JSON validation requirement")

        if "python -m json.tool" in content:
            print("✓ Single container instructions include json.tool validation command")
        else:
            print("✗ Single container instructions missing json.tool validation command")
    else:
        print(f"✗ Could not find single container file: {single_container_path}")

    # Test codeact bridge v3 instructions
    codeact_path = Path(__file__).parent / "backend" / "app" / "integrations" / "openhands_codeact_bridge_v3.py"

    if codeact_path.exists():
        content = codeact_path.read_text()
        if "IMPORTANT SELF-VALIDATION REQUIREMENT" in content:
            print("✓ CodeAct bridge v3 instructions include JSON validation requirement")
        else:
            print("✗ CodeAct bridge v3 instructions missing JSON validation requirement")
    else:
        print(f"✗ Could not find codeact bridge v3 file: {codeact_path}")

    # Test backend service instructions
    backend_path = Path(__file__).parent / "backend" / "app" / "integrations" / "openhands_backend_service.py"

    if backend_path.exists():
        content = backend_path.read_text()
        if "IMPORTANT SELF-VALIDATION REQUIREMENT" in content:
            print("✓ Backend service instructions include JSON validation requirement")
        else:
            print("✗ Backend service instructions missing JSON validation requirement")
    else:
        print(f"✗ Could not find backend service file: {backend_path}")

if __name__ == "__main__":
    test_enhanced_instructions()