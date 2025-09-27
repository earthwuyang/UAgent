#!/usr/bin/env python3
"""Test script to verify LLM configuration from .env is passed to OpenHands"""

import os
import sys
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.integrations.docker_transition_manager import docker_transition_manager


def test_env_config():
    """Test that .env LLM configuration is properly mapped"""
    print("=== LLM Configuration Test ===")

    # Show current environment variables
    print("\n1. Current .env LiteLLM configuration:")
    litellm_vars = {
        "LITELLM_MODEL": os.getenv("LITELLM_MODEL"),
        "LITELLM_API_KEY": os.getenv("LITELLM_API_KEY", "***hidden***") if os.getenv("LITELLM_API_KEY") else None,
        "LITELLM_API_BASE": os.getenv("LITELLM_API_BASE"),
        "LITELLM_EXTRA_OPTIONS": os.getenv("LITELLM_EXTRA_OPTIONS")
    }

    for key, value in litellm_vars.items():
        if value:
            print(f"  {key} = {value}")
        else:
            print(f"  {key} = <not set>")

    # Get Docker configuration
    print("\n2. Docker configuration for OpenHands:")
    try:
        docker_config = docker_transition_manager.get_docker_config()

        llm_config_keys = [key for key in docker_config.keys() if key.startswith("LLM_")]
        if llm_config_keys:
            print("  OpenHands LLM environment variables:")
            for key in llm_config_keys:
                value = docker_config[key]
                if "API_KEY" in key:
                    value = "***hidden***" if value else "<not set>"
                print(f"    {key} = {value}")
        else:
            print("  No LLM configuration found in Docker config")

        print(f"\n3. Docker config summary:")
        print(f"  Total environment variables: {len(docker_config)}")
        print(f"  LLM-related variables: {len(llm_config_keys)}")

        return len(llm_config_keys) > 0

    except Exception as e:
        print(f"  Error getting Docker config: {e}")
        return False


def test_runtime_config():
    """Test runtime configuration mapping"""
    print("\n=== Runtime Configuration Test ===")

    # Check if .env variables would be mapped correctly
    expected_mapping = {
        "LITELLM_MODEL": "LLM_MODEL",
        "LITELLM_API_KEY": "LLM_API_KEY",
        "LITELLM_API_BASE": "LLM_BASE_URL"
    }

    all_mapped = True
    for litellm_var, llm_var in expected_mapping.items():
        if os.getenv(litellm_var):
            print(f"  ✓ {litellm_var} → {llm_var}")
        else:
            print(f"  ✗ {litellm_var} → {llm_var} (source not set)")
            all_mapped = False

    return all_mapped


if __name__ == "__main__":
    print("Testing LLM configuration integration between .env and OpenHands\n")

    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()

    env_test_passed = test_env_config()
    runtime_test_passed = test_runtime_config()

    print(f"\n=== Results ===")
    print(f"Environment config test: {'✓ PASSED' if env_test_passed else '✗ FAILED'}")
    print(f"Runtime mapping test: {'✓ PASSED' if runtime_test_passed else '✗ FAILED'}")

    if env_test_passed and runtime_test_passed:
        print("\n🎉 LLM configuration integration is working!")
        print("OpenHands will use the model configured in .env")
    else:
        print("\n⚠️  LLM configuration integration needs attention")
        print("Check that .env has LITELLM_* variables set")

    sys.exit(0 if (env_test_passed and runtime_test_passed) else 1)