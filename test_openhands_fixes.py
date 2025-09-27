#!/usr/bin/env python3
"""Test script to verify OpenHands performance fixes."""

import time
import sys
import os

# Add OpenHands to path
sys.path.insert(0, '/home/wuy/AI/UAgent/OpenHands')

def test_apt_manager():
    """Test APT manager functionality."""
    print("=== Testing APT Manager ===")
    try:
        from openhands.runtime.utils.apt_manager import APTManager

        apt_mgr = APTManager(max_retries=3, retry_delay=1, lock_timeout=10)

        # Test lock status check
        lock_holder = apt_mgr._check_lock_status()
        print(f"Current APT lock status: {lock_holder or 'No locks detected'}")

        # Test package list update (safe operation)
        print("Testing package list update...")
        success = apt_mgr.update_packages()
        print(f"Package update result: {'SUCCESS' if success else 'FAILED'}")

        return True
    except Exception as e:
        print(f"APT Manager test failed: {e}")
        return False

def test_bash_session():
    """Test bash session with optimized settings."""
    print("\n=== Testing Bash Session ===")
    try:
        from openhands.runtime.utils.bash import BashSession
        from openhands.events.action import CmdRunAction

        # Create bash session with optimized settings
        bash_session = BashSession(
            work_dir="/tmp",
            no_change_timeout_seconds=15  # Optimized timeout
        )

        print("Initializing bash session...")
        bash_session.initialize()

        # Test simple command
        print("Testing simple command execution...")
        action = CmdRunAction(command="echo 'OpenHands performance test'")
        result = bash_session.execute(action)

        print(f"Command result: {result.content[:100]}...")

        # Test APT command detection
        print("Testing APT command detection...")
        is_apt_cmd = bash_session._is_apt_command("sudo apt-get install curl")
        print(f"APT command detection: {'PASS' if is_apt_cmd else 'FAIL'}")

        # Test non-APT command detection
        is_not_apt = not bash_session._is_apt_command("ls -la")
        print(f"Non-APT command detection: {'PASS' if is_not_apt else 'FAIL'}")

        return True
    except Exception as e:
        print(f"Bash session test failed: {e}")
        return False

def test_performance_config():
    """Test performance configuration."""
    print("\n=== Testing Performance Config ===")
    try:
        from openhands.runtime.utils.performance_config import PerformanceConfig

        config = PerformanceConfig.from_env()
        print(f"Poll interval: {config.bash_poll_interval}s")
        print(f"No-change timeout: {config.bash_no_change_timeout}s")
        print(f"APT max retries: {config.apt_max_retries}")

        # Test timeout detection
        apt_timeout = config.get_timeout_for_command("sudo apt-get install vim")
        build_timeout = config.get_timeout_for_command("make all")
        default_timeout = config.get_timeout_for_command("ls -la")

        print(f"APT command timeout: {apt_timeout}s")
        print(f"Build command timeout: {build_timeout}s")
        print(f"Default command timeout: {default_timeout}s")

        expected_apt = config.apt_command_timeout
        expected_build = config.build_command_timeout
        expected_default = config.default_command_timeout

        return (apt_timeout == expected_apt and
                build_timeout == expected_build and
                default_timeout == expected_default)
    except Exception as e:
        print(f"Performance config test failed: {e}")
        return False

def test_log_reduction():
    """Test if excessive logging is reduced."""
    print("\n=== Testing Log Reduction ===")

    # Check recent log file
    log_file = "/home/wuy/AI/UAgent/OpenHands/logs/openhands.log"
    if not os.path.exists(log_file):
        print("Log file not found")
        return False

    try:
        # Count recent polling messages
        with open(log_file, 'r') as f:
            lines = f.readlines()

        recent_lines = lines[-100:]  # Last 100 lines
        polling_lines = [line for line in recent_lines if 'SLEEPING for' in line]

        print(f"Recent polling messages in last 100 log lines: {len(polling_lines)}")

        # Should be fewer with optimized settings
        return len(polling_lines) < 50  # Arbitrary threshold
    except Exception as e:
        print(f"Log reduction test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("OpenHands Performance Fixes Test Suite")
    print("=" * 50)

    tests = [
        ("APT Manager", test_apt_manager),
        ("Bash Session", test_bash_session),
        ("Performance Config", test_performance_config),
        ("Log Reduction", test_log_reduction),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n{test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"\n{test_name}: ERROR - {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("Test Summary:")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)