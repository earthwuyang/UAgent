#!/usr/bin/env python3
"""
Comprehensive Demo: Python Performance Issues & Bugs
This file contains various performance anti-patterns and bugs that Semgrep can detect.
"""

import os
import time
import asyncio
import subprocess
import requests
from pathlib import Path

# =============================================================================
# PERFORMANCE ISSUES
# =============================================================================

def performance_demo():
    """Demonstrates various performance anti-patterns"""
    
    # 1. String concatenation in loop (O(n²) complexity)
    result = ""
    for i in range(1000):
        result += f"item_{i} "  # Performance issue: creates new string each iteration
    
    # 2. Inefficient list building
    data = []
    for i in range(1000):
        data.append(str(i))  # Could use list comprehension
    
    # 3. Multiple file operations on same file
    with open("test.txt", "r") as f:
        content = f.read()
    
    # Later in same function...
    with open("test.txt", "w") as f:
        f.write(content.upper())
    
    # 4. Inefficient dictionary access
    config = {"timeout": 30, "retries": 3}
    if "timeout" in config:
        timeout = config["timeout"]
    else:
        timeout = 10
    
    return result, data, timeout

# =============================================================================
# ASYNC/CONCURRENCY BUGS  
# =============================================================================

async def async_bugs_demo():
    """Demonstrates async-related bugs"""
    
    # 1. Using time.sleep in async function (blocks event loop)
    time.sleep(1)  # BUG: Should be await asyncio.sleep(1)
    
    # 2. Creating tasks in loop without concurrency control
    tasks = []
    items = ['a', 'b', 'c', 'd', 'e']
    
    for item in items:
        # BUG: All tasks start immediately, ignoring max_parallel limits
        task = asyncio.create_task(process_item(item))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    # 3. Potential infinite loop without timeout
    while True:
        await asyncio.sleep(0.1)
        result = check_condition()
        if result:
            break  # What if check_condition() never returns True?

async def process_item(item):
    """Simulate processing"""
    await asyncio.sleep(0.5)
    return f"processed_{item}"

def check_condition():
    """Simulate condition check"""
    return False  # Always False = infinite loop!

# =============================================================================
# RESOURCE MANAGEMENT BUGS
# =============================================================================

def resource_bugs_demo():
    """Demonstrates resource management issues"""
    
    # 1. File not properly closed (resource leak)
    file = open("output.txt", "w")
    file.write("some data")
    # BUG: Missing file.close() or context manager
    
    # 2. Subprocess without proper error handling
    result = subprocess.run(["ls", "-la"], capture_output=True, text=True)
    print(result.stdout)  # What if command fails?
    
    # 3. Network request without timeout
    response = requests.get("https://api.example.com/data")  # BUG: No timeout
    return response.json()

# =============================================================================
# ERROR HANDLING BUGS
# =============================================================================

def error_handling_bugs():
    """Demonstrates poor error handling patterns"""
    
    # 1. Bare except clause (swallows all exceptions)
    try:
        risky_operation()
    except:
        pass  # BUG: Hides all errors, makes debugging impossible
    
    # 2. Catching but not logging the actual error
    try:
        another_risky_operation()
    except Exception as e:
        print("Something went wrong")  # BUG: Doesn't log the actual error
        return None
    
    # 3. Not handling specific exceptions
    try:
        value = int(user_input)  # Could raise ValueError
        result = 10 / value      # Could raise ZeroDivisionError
    except Exception:  # Too broad
        return "Error occurred"

def risky_operation():
    """Simulates a risky operation"""
    raise ValueError("Something bad happened")

def another_risky_operation():
    """Another risky operation"""
    raise ConnectionError("Network issue")

user_input = "not_a_number"

# =============================================================================
# CODE QUALITY ISSUES
# =============================================================================

class QualityIssues:
    """Demonstrates code quality problems"""
    
    def __init__(self):
        # 1. Complex nested data structure that's hard to maintain
        self.config = {
            'database': {
                'primary': {
                    'host': 'localhost',
                    'port': 5432,
                    'settings': {
                        'pool_size': 10,
                        'timeout': 30,
                        'retry_config': {
                            'max_retries': 3,
                            'backoff_factor': 2,
                            'retry_codes': [500, 502, 503, 504]
                        }
                    }
                }
            }
        }
    
    def process_data(self, data):
        """Method with too many nested conditions"""
        if data:
            if isinstance(data, list):
                if len(data) > 0:
                    if data[0]:
                        if hasattr(data[0], 'value'):
                            if data[0].value is not None:
                                return data[0].value
        return None

# =============================================================================
# SUBPROCESS SECURITY ISSUES
# =============================================================================

def _ensure_safe_path(raw_path: str) -> Path:
    """Normalize user input to a path within the current working directory."""

    if not raw_path or raw_path.startswith("-"):
        raise ValueError("Path must not be empty or start with '-' (option injection)")

    base_dir = Path.cwd().resolve()
    candidate = (base_dir / raw_path).expanduser()
    resolved = candidate.resolve(strict=False)

    if base_dir == resolved:
        return resolved

    if base_dir not in resolved.parents:
        raise ValueError("Path escapes working directory")

    return resolved


def subprocess_security_demo(user_input):
    """Demonstrates secure subprocess usage with untrusted input."""

    try:
        safe_path = _ensure_safe_path(user_input)
    except ValueError as exc:
        print(f"Rejected unsafe path: {exc}")
        return None

    # 1. Execute command without shell=True and with sanitized arguments
    result = subprocess.run([
        "ls",
        "-l",
        "-a",
        str(safe_path),
    ], check=False)

    # 2. Safely handle file content without invoking shell utilities
    if safe_path.is_file():
        try:
            safe_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"Failed to read file {safe_path}: {exc}")

    return result

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Performance issues
    performance_demo()
    
    # Async issues
    asyncio.run(async_bugs_demo())
    
    # Resource management
    resource_bugs_demo()
    
    # Error handling
    error_handling_bugs()
    
    # Subprocess security
    subprocess_security_demo("../../etc/passwd")
    
    print("Demo completed - Semgrep should find many issues!")
