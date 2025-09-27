#!/usr/bin/env python3
"""Test script to verify OpenHands logging improvements."""

import os
import sys
import tempfile
import time
import logging
from pathlib import Path

# Add OpenHands to path
sys.path.insert(0, '/home/wuy/AI/UAgent/OpenHands')

def test_smart_logging_filter():
    """Test the smart logging filter functionality."""
    print("=== Testing Smart Logging Filter ===")

    try:
        from openhands.core.smart_logger_filter import SmartLoggingFilter, ImportantOnlyFilter, setup_smart_logging

        # Test filter creation
        smart_filter = SmartLoggingFilter()
        important_filter = ImportantOnlyFilter()

        # Create test logger
        test_logger = logging.getLogger('test_smart_filter')
        test_logger.setLevel(logging.DEBUG)

        # Test message filtering
        verbose_record = logging.LogRecord(
            name='test',
            level=logging.DEBUG,
            pathname='test.py',
            lineno=1,
            msg='PANE CONTENT GOT after 0.01 seconds',
            args=(),
            exc_info=None
        )

        important_record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='test.py',
            lineno=1,
            msg='ERROR: Command failed with critical error',
            args=(),
            exc_info=None
        )

        # Test smart filter with repeated verbose messages
        # First few should be allowed, then suppressed
        verbose_results = []
        for i in range(5):
            verbose_results.append(smart_filter.filter(verbose_record))

        important_allowed = smart_filter.filter(important_record)

        print(f"Smart filter - Verbose messages (first 5): {verbose_results}")
        print(f"Smart filter - Important message allowed: {important_allowed}")

        # Test important-only filter
        verbose_allowed_strict = important_filter.filter(verbose_record)
        important_allowed_strict = important_filter.filter(important_record)

        print(f"Important-only filter - Verbose message allowed: {verbose_allowed_strict}")
        print(f"Important-only filter - Important message allowed: {important_allowed_strict}")

        # Expected: smart filter should start suppressing after max_repetitions, important should always pass
        # Important-only filter should block verbose but allow important
        smart_suppressing = not all(verbose_results)  # Some should be False (suppressed)
        expected_smart = smart_suppressing and important_allowed
        expected_strict = not verbose_allowed_strict and important_allowed_strict

        return expected_smart and expected_strict

    except Exception as e:
        print(f"Smart logging filter test failed: {e}")
        return False

def test_component_log_levels():
    """Test that component log levels are properly set."""
    print("\n=== Testing Component Log Levels ===")

    try:
        from openhands.core.smart_logger_filter import ComponentLogLevelManager

        # Apply component levels
        ComponentLogLevelManager.apply_component_levels()

        # Check that verbose components have higher thresholds
        bash_logger = logging.getLogger('openhands.runtime.utils.bash')
        controller_logger = logging.getLogger('openhands.controller.agent_controller')

        print(f"Bash logger level: {bash_logger.level} (should be WARNING={logging.WARNING})")
        print(f"Controller logger level: {controller_logger.level} (should be INFO={logging.INFO})")

        # Verify levels are correctly set
        bash_correct = bash_logger.level >= logging.WARNING
        controller_correct = controller_logger.level >= logging.INFO

        return bash_correct and controller_correct

    except Exception as e:
        print(f"Component log levels test failed: {e}")
        return False

def test_bash_logging_optimization():
    """Test bash session logging optimizations."""
    print("\n=== Testing Bash Logging Optimization ===")

    try:
        # Set environment variables for testing
        os.environ['OPENHANDS_BASH_VERBOSE_DEBUG'] = 'False'

        from openhands.runtime.utils.bash import BashSession
        from openhands.events.action import CmdRunAction

        # Create a temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            bash_session = BashSession(work_dir=tmpdir, no_change_timeout_seconds=5)
            bash_session.initialize()

            # Test trivial content change detection
            trivial = bash_session._is_trivial_content_change(
                "10:30:15 - session_abc123: some content",
                "10:30:16 - session_def456: some content"
            )

            non_trivial = bash_session._is_trivial_content_change(
                "old content here",
                "completely different content"
            )

            print(f"Trivial change detection: {trivial} (should be True)")
            print(f"Non-trivial change detection: {non_trivial} (should be False)")

            # Test APT command detection
            apt_detected = bash_session._is_apt_command("sudo apt-get install vim")
            non_apt_detected = bash_session._is_apt_command("ls -la")

            print(f"APT command detection: {apt_detected} (should be True)")
            print(f"Non-APT command detection: {non_apt_detected} (should be False)")

            return trivial and not non_trivial and apt_detected and not non_apt_detected

    except Exception as e:
        print(f"Bash logging optimization test failed: {e}")
        return False

def test_environment_configuration():
    """Test environment variable configuration for logging."""
    print("\n=== Testing Environment Configuration ===")

    try:
        # Test environment variable reading
        os.environ['OPENHANDS_SMART_LOGGING'] = 'True'
        os.environ['OPENHANDS_SMART_LOGGING_MODE'] = 'important'

        # Reload the logger module to pick up new environment variables
        import importlib
        import openhands.core.logger
        importlib.reload(openhands.core.logger)

        from openhands.core.logger import SMART_LOGGING, SMART_LOGGING_MODE

        print(f"Smart logging enabled: {SMART_LOGGING}")
        print(f"Smart logging mode: {SMART_LOGGING_MODE}")

        # Verify configuration is read correctly
        config_correct = SMART_LOGGING and SMART_LOGGING_MODE == 'important'

        # Test disabling smart logging
        os.environ['OPENHANDS_SMART_LOGGING'] = 'False'
        importlib.reload(openhands.core.logger)

        from openhands.core.logger import SMART_LOGGING as DISABLED_SMART_LOGGING
        disabled_correct = not DISABLED_SMART_LOGGING

        print(f"Smart logging disabled: {disabled_correct}")

        return config_correct and disabled_correct

    except Exception as e:
        print(f"Environment configuration test failed: {e}")
        return False

def test_log_volume_reduction():
    """Test that log volume is actually reduced."""
    print("\n=== Testing Log Volume Reduction ===")

    try:
        # Count recent log entries of different types
        log_file = Path("/home/wuy/AI/UAgent/OpenHands/logs/openhands.log")

        if not log_file.exists():
            print("Log file not found - cannot test volume reduction")
            return False

        # Read recent log entries
        with open(log_file, 'r') as f:
            lines = f.readlines()

        recent_lines = lines[-500:]  # Last 500 lines

        # Count verbose message types
        verbose_patterns = [
            'PANE CONTENT GOT',
            'BEGIN OF PANE CONTENT',
            'END OF PANE CONTENT',
            'SLEEPING for',
            'GETTING PANE CONTENT',
            'CHECKING NO CHANGE TIMEOUT',
            'CHECKING HARD TIMEOUT',
        ]

        verbose_count = 0
        for line in recent_lines:
            for pattern in verbose_patterns:
                if pattern in line:
                    verbose_count += 1
                    break

        total_lines = len(recent_lines)
        verbose_percentage = (verbose_count / total_lines * 100) if total_lines > 0 else 0

        print(f"Recent log lines: {total_lines}")
        print(f"Verbose messages: {verbose_count} ({verbose_percentage:.1f}%)")

        # With optimizations, verbose messages should be less than 20% of total
        return verbose_percentage < 20

    except Exception as e:
        print(f"Log volume reduction test failed: {e}")
        return False

def main():
    """Run all logging improvement tests."""
    print("OpenHands Logging Improvements Test Suite")
    print("=" * 60)

    tests = [
        ("Smart Logging Filter", test_smart_logging_filter),
        ("Component Log Levels", test_component_log_levels),
        ("Bash Logging Optimization", test_bash_logging_optimization),
        ("Environment Configuration", test_environment_configuration),
        ("Log Volume Reduction", test_log_volume_reduction),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ERROR - {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Test Summary:")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All logging improvements are working correctly!")
        print("\nTo use the optimized logging:")
        print("  export OPENHANDS_SMART_LOGGING=true")
        print("  export OPENHANDS_SMART_LOGGING_MODE=smart  # or 'important' for minimal logs")
        print("  export OPENHANDS_BASH_VERBOSE_DEBUG=false  # disable verbose bash debug")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)