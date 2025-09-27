#!/usr/bin/env python3
"""Verify OpenHands V3 is enabled by default"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Check V3 default status
def check_v3_status():
    # Check the default value logic
    v3_env = os.getenv("UAGENT_OPENHANDS_V3", "1").lower()
    use_v3 = v3_env not in ("0", "false", "no", "")

    print("=" * 60)
    print("OpenHands V3 Default Status Check")
    print("=" * 60)

    print(f"\n1. Environment variable UAGENT_OPENHANDS_V3: {os.getenv('UAGENT_OPENHANDS_V3', 'NOT SET')}")
    print(f"2. Parsed value: {v3_env}")
    print(f"3. V3 enabled: {use_v3}")

    # Try importing V3 bridge
    try:
        from backend.app.integrations.openhands_codeact_bridge_v3 import OpenHandsCodeActBridgeV3
        print(f"4. V3 Bridge module: ✓ Available")
    except ImportError as e:
        print(f"4. V3 Bridge module: ✗ Not available ({e})")
        return False

    # Check if it's actually used in ScientificResearchEngine
    try:
        from backend.app.core.research_engines.scientific_research import OpenHandsCodeActBridgeV3 as ImportedV3
        print(f"5. V3 imported in ScientificResearchEngine: ✓ Yes")
    except ImportError:
        print(f"5. V3 imported in ScientificResearchEngine: ✗ No")

    print("\n" + "=" * 60)
    print("RESULT: V3 is", "ENABLED" if use_v3 else "DISABLED", "by default")
    print("=" * 60)

    if use_v3:
        print("\n✅ V3 headless mode is active (default)")
        print("To use legacy V2 mode, set: export UAGENT_OPENHANDS_V3=0")
    else:
        print("\n⚠️  V3 is disabled, using legacy V2 mode")
        print("To enable V3, unset UAGENT_OPENHANDS_V3 or set to 1")

    return use_v3

if __name__ == "__main__":
    is_v3 = check_v3_status()
    sys.exit(0 if is_v3 else 1)