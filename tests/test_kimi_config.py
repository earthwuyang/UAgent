#!/usr/bin/env python3
"""Test that OpenHands V3 correctly uses Kimi model configuration from .env"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Load .env file
load_dotenv()

from backend.app.integrations.openhands_codeact_bridge_v3 import (
    OpenHandsCodeActBridgeV3,
    CodeActRunConfig
)

def test_kimi_config():
    """Test that Kimi model configuration is properly passed to OpenHands"""

    # Create a test workspace
    workspace = Path("/tmp/test_kimi_config")
    workspace.mkdir(exist_ok=True, parents=True)

    # Create test config
    cfg = CodeActRunConfig(
        goal="Print the model name you are using",
        workspace=workspace,
        session_name="test_kimi",
        max_steps=5,
        max_minutes=1,
        disable_browser=True
    )

    # Create bridge and build environment
    bridge = OpenHandsCodeActBridgeV3()
    env = bridge._build_environment(cfg)

    # Check that LITELLM variables are mapped to LLM variables
    print("Environment variables that will be passed to OpenHands:")
    print(f"  LLM_MODEL: {env.get('LLM_MODEL', 'NOT SET')}")
    print(f"  LLM_API_KEY: {env.get('LLM_API_KEY', 'NOT SET')[:20] if env.get('LLM_API_KEY') else 'NOT SET'}...")
    print(f"  LLM_BASE_URL: {env.get('LLM_BASE_URL', 'NOT SET')}")

    # Verify the model is Kimi
    assert env.get('LLM_MODEL') == 'openai/kimi-k2-turbo-preview', f"Expected Kimi model, got {env.get('LLM_MODEL')}"
    assert env.get('LLM_API_BASE') or env.get('LLM_BASE_URL'), "No API base URL set"
    assert env.get('LLM_API_KEY'), "No API key set"

    print("\n✅ Kimi model configuration is correctly set for OpenHands V3!")
    print(f"   Model: {env.get('LLM_MODEL')}")
    print(f"   Base URL: {env.get('LLM_BASE_URL')}")

    return True

if __name__ == "__main__":
    test_kimi_config()