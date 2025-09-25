#!/usr/bin/env python3
"""Simple script to verify environment variables are set correctly."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load the .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")
else:
    print(f"❌ No .env file found at {env_path}")

print("\nParallelism configuration from environment:")
print(f"  MAX_RESEARCH_IDEAS = {os.getenv('MAX_RESEARCH_IDEAS', 'not set')}")
print(f"  MAX_PARALLEL_IDEAS = {os.getenv('MAX_PARALLEL_IDEAS', 'not set')}")
print(f"  EXPERIMENTS_PER_HYPOTHESIS = {os.getenv('EXPERIMENTS_PER_HYPOTHESIS', 'not set')}")

print("\nOpenHands timeout configuration:")
print(f"  OPENHANDS_ACTION_TIMEOUT = {os.getenv('OPENHANDS_ACTION_TIMEOUT', 'not set')}")
print(f"  OPENHANDS_MAX_ACTION_TIMEOUT = {os.getenv('OPENHANDS_MAX_ACTION_TIMEOUT', 'not set')}")
print(f"  OPENHANDS_RUN_ADAPTIVE_MULTIPLIER = {os.getenv('OPENHANDS_RUN_ADAPTIVE_MULTIPLIER', 'not set')}")
print(f"  OPENHANDS_RUN_MAX_ATTEMPTS = {os.getenv('OPENHANDS_RUN_MAX_ATTEMPTS', 'not set')}")
print(f"  OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT = {os.getenv('OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT', 'not set')}")

# Check values
expected = {
    "MAX_RESEARCH_IDEAS": "1",
    "MAX_PARALLEL_IDEAS": "1",
    "EXPERIMENTS_PER_HYPOTHESIS": "1"
}

all_correct = True
for key, expected_val in expected.items():
    actual = os.getenv(key)
    if actual == expected_val:
        print(f"\n✅ {key} = {actual} (correct)")
    else:
        print(f"\n❌ {key} = {actual} (expected {expected_val})")
        all_correct = False

if all_correct:
    print("\n🎉 All parallelism settings are correctly set to 1 for debugging!")
else:
    print("\n⚠️ Some settings need adjustment in .env file")