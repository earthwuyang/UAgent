#!/usr/bin/env python3
"""Test concise LLM logging only"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Add backend and OpenHands to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "OpenHands"))

# Set environment for testing
os.environ['WORKSPACE_BASE'] = '/tmp/test_llm_log'
os.environ['DEBUG_LLM'] = 'true'
os.environ['DEBUG_LLM_AUTO_CONFIRM'] = 'true'
os.environ['LOG_TO_FILE'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG'

# Create test workspace
workspace = Path('/tmp/test_llm_log')
workspace.mkdir(exist_ok=True)
(workspace / 'logs').mkdir(exist_ok=True)

# Test the logger
try:
    from openhands.core.logger import _setup_concise_llm_logger

    logger = _setup_concise_llm_logger('/tmp/test_llm_log')
    print(f"Created logger: {logger}")

    # Test logging
    logger.info("USER: Test prompt\nASSISTANT: Test response")
    print("✅ Concise LLM logging test successful")

    # Check log file
    log_file = workspace / 'logs' / 'llm_interactions.log'
    if log_file.exists():
        print(f"✅ Log file created: {log_file}")
        with open(log_file) as f:
            content = f.read()
        print(f"Log content:\n{content}")
    else:
        print("❌ Log file not created")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()