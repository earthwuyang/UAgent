"""
Unit tests for OpenHands CodeExecutor
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import pytest_asyncio

from app.core.openhands.workspace_manager import WorkspaceManager
from app.core.openhands.code_executor import (
    CodeExecutor,
    ExecutionResult,
    ExecutionCommand
)


class TestCodeExecutor:
    """Test CodeExecutor functionality"""

    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def workspace_manager(self, temp_base_dir):
        """Create WorkspaceManager instance for testing"""
        return WorkspaceManager(temp_base_dir)

    @pytest.fixture
    def code_executor(self, workspace_manager):
        """Create CodeExecutor instance for testing"""
        return CodeExecutor(workspace_manager)

    @pytest_asyncio.fixture
    async def test_workspace(self, workspace_manager):
        """Create a test workspace"""
        config = await workspace_manager.create_workspace("test_workspace")
        return config.workspace_id

    @pytest.mark.asyncio
    async def test_python_code_execution_success(self, code_executor, test_workspace):
        """Test successful Python code execution"""
        python_code = """
print("Hello, World!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""

        result = await code_executor.execute_python_code(
            test_workspace, python_code, "test_script.py"
        )

        assert result.success is True
        assert result.exit_code == 0
        assert "Hello, World!" in result.stdout
        assert "2 + 2 = 4" in result.stdout
        assert result.stderr == ""
        assert result.execution_time > 0
        assert any("test_script.py" in f for f in result.files_created + result.files_modified)

    @pytest.mark.asyncio
    async def test_python_code_execution_with_error(self, code_executor, test_workspace):
        """Test Python code execution with syntax error"""
        python_code = """
print("This has a syntax error"
# Missing closing parenthesis
"""

        result = await code_executor.execute_python_code(
            test_workspace, python_code, "error_script.py"
        )

        assert result.success is False
        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr or "invalid syntax" in result.stderr
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_python_code_execution_with_runtime_error(self, code_executor, test_workspace):
        """Test Python code execution with runtime error"""
        python_code = """
print("Before error")
x = 1 / 0  # Division by zero
print("After error")
"""

        result = await code_executor.execute_python_code(
            test_workspace, python_code, "runtime_error.py"
        )

        assert result.success is False
        assert result.exit_code != 0
        assert "Before error" in result.stdout
        assert "ZeroDivisionError" in result.stderr
        assert "After error" not in result.stdout

    @pytest.mark.asyncio
    async def test_python_code_file_operations(self, code_executor, test_workspace):
        """Test Python code that creates and modifies files"""
        python_code = """
import json

# Create a data file
data = {"name": "test", "value": 42}
with open("data/test_data.json", "w") as f:
    json.dump(data, f)

# Create another file
with open("output/results.txt", "w") as f:
    f.write("Test results\\n")
    f.write("All tests passed\\n")

print("Files created successfully")
"""

        result = await code_executor.execute_python_code(
            test_workspace, python_code, "file_ops.py"
        )

        assert result.success is True
        assert "Files created successfully" in result.stdout
        assert len(result.files_created) >= 2  # At least the script and created files

    @pytest.mark.asyncio
    async def test_bash_command_execution(self, code_executor, test_workspace):
        """Test bash command execution"""
        command = "echo 'Hello from bash'"

        result = await code_executor.execute_bash_command(test_workspace, command)

        assert result.success is True
        assert result.exit_code == 0
        assert "Hello from bash" in result.stdout

    @pytest.mark.asyncio
    async def test_bash_command_with_forbidden_command(self, code_executor, test_workspace):
        """Test bash command with forbidden command"""
        # Try to use a forbidden command
        command = "sudo rm -rf /"

        result = await code_executor.execute_bash_command(test_workspace, command)

        assert result.success is False
        assert "Command not allowed" in result.stderr

    @pytest.mark.asyncio
    async def test_command_safety_checking(self, code_executor):
        """Test command safety checking"""
        executor = code_executor

        # Test safe commands
        assert executor._is_command_safe("python script.py") is True
        assert executor._is_command_safe("ls -la") is True
        assert executor._is_command_safe("cat file.txt") is True

        # Test forbidden commands
        assert executor._is_command_safe("rm -rf /") is False
        assert executor._is_command_safe("sudo something") is False
        assert executor._is_command_safe("format c:") is False
        assert executor._is_command_safe("../../etc/passwd") is False

    @pytest.mark.asyncio
    async def test_execution_timeout(self, code_executor, test_workspace):
        """Test execution timeout functionality"""
        # Code that sleeps longer than timeout
        python_code = """
import time
print("Starting long operation")
time.sleep(10)  # Sleep for 10 seconds
print("This should not be printed")
"""

        result = await code_executor.execute_python_code(
            test_workspace, python_code, "timeout_test.py", timeout=2  # 2 second timeout
        )

        assert result.success is False
        assert result.exit_code == -9  # Killed by timeout
        assert "Starting long operation" in result.stdout
        assert "This should not be printed" not in result.stdout
        assert "timed out" in result.stderr

    @pytest.mark.asyncio
    async def test_execution_command_object(self, code_executor, test_workspace):
        """Test execution using ExecutionCommand object"""
        command = ExecutionCommand(
            command="echo 'Test with command object'",
            working_directory=".",
            timeout=30,
            capture_output=True,
            shell=True
        )

        result = await code_executor.execute_command(test_workspace, command)

        assert result.success is True
        assert "Test with command object" in result.stdout

    @pytest.mark.asyncio
    async def test_execution_with_environment_variables(self, code_executor, test_workspace):
        """Test execution with custom environment variables"""
        python_code = """
import os
print(f"TEST_VAR: {os.environ.get('TEST_VAR', 'Not found')}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
"""

        command = ExecutionCommand(
            command=f"python3 code/env_test.py",
            env_vars={"TEST_VAR": "test_value"}
        )

        # First write the Python code to a file
        await code_executor.workspace_manager.write_file(
            test_workspace, "code/env_test.py", python_code
        )

        result = await code_executor.execute_command(test_workspace, command)

        assert result.success is True
        assert "TEST_VAR: test_value" in result.stdout
        # PYTHONPATH should be set by the executor
        assert "PYTHONPATH:" in result.stdout

    @pytest.mark.asyncio
    async def test_nonexistent_workspace(self, code_executor):
        """Test execution on non-existent workspace"""
        python_code = "print('This should fail')"

        result = await code_executor.execute_python_code(
            "nonexistent_workspace", python_code
        )

        assert result.success is False
        assert "Workspace nonexistent_workspace not found" in result.stderr

    @pytest.mark.asyncio
    async def test_jupyter_notebook_execution(self, code_executor, test_workspace):
        """Test Jupyter notebook execution"""
        notebook_content = """
{
  "cells": [
    {
      "cell_type": "code",
      "source": ["import numpy as np\\nprint('NumPy version:', np.__version__)"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": ["x = 5\\ny = 10\\nprint(f'x + y = {x + y}')"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": ["# This is a markdown cell\\nIt should be skipped."]
    },
    {
      "cell_type": "code",
      "source": ["# Empty cell"],
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {},
  "nbformat": 4,
  "nbformat_minor": 4
}
"""

        results = await code_executor.execute_jupyter_notebook(
            test_workspace, notebook_content
        )

        # Should have results for executable code cells (cells 0, 1, 3)
        assert len(results) >= 2  # At least 2 non-empty code cells

        # Check first cell result
        first_result = results[0]
        assert "NumPy version:" in first_result.stdout or first_result.success is False  # NumPy might not be installed

        # Check second cell result
        if len(results) > 1:
            second_result = results[1]
            assert second_result.success is True
            assert "x + y = 15" in second_result.stdout

    @pytest.mark.asyncio
    async def test_jupyter_notebook_invalid_json(self, code_executor, test_workspace):
        """Test Jupyter notebook with invalid JSON"""
        invalid_notebook = "{ invalid json content"

        results = await code_executor.execute_jupyter_notebook(
            test_workspace, invalid_notebook
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "Invalid notebook JSON" in results[0].stderr

    @pytest.mark.asyncio
    async def test_concurrent_executions(self, code_executor, test_workspace):
        """Test concurrent code executions"""
        python_codes = [
            "print(f'Task {i}'); import time; time.sleep(0.1); print(f'Task {i} completed')"
            for i in range(5)
        ]

        # Execute all codes concurrently
        tasks = [
            code_executor.execute_python_code(test_workspace, code, f"task_{i}.py")
            for i, code in enumerate(python_codes)
        ]

        results = await asyncio.gather(*tasks)

        # All executions should succeed
        assert all(result.success for result in results)

        # Each should have the correct output
        for i, result in enumerate(results):
            assert f"Task {i}" in result.stdout
            assert f"Task {i} completed" in result.stdout

    @pytest.mark.asyncio
    async def test_workspace_files_tracking(self, code_executor, test_workspace):
        """Test tracking of created and modified files"""
        # Get initial file list
        initial_files = await code_executor._get_workspace_files(test_workspace)

        python_code = """
# Create a new file
with open("new_file.txt", "w") as f:
    f.write("This is a new file")

# Modify existing file (requirements.txt)
with open("requirements.txt", "a") as f:
    f.write("\\n# Added by test")
"""

        result = await code_executor.execute_python_code(
            test_workspace, python_code, "file_tracking.py"
        )

        assert result.success is True
        assert len(result.files_created) > len(initial_files)  # New files created
        assert "new_file.txt" in str(result.files_created)

    @pytest.mark.asyncio
    async def test_error_handling_in_execution(self, code_executor, test_workspace):
        """Test error handling in various execution scenarios"""
        # Test with invalid Python syntax
        invalid_code = "print('unclosed string"
        result = await code_executor.execute_python_code(test_workspace, invalid_code)
        assert result.success is False

        # Test with runtime error
        runtime_error_code = "raise ValueError('Test error')"
        result = await code_executor.execute_python_code(test_workspace, runtime_error_code)
        assert result.success is False
        assert "ValueError: Test error" in result.stderr

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_shell')
    async def test_stream_execution(self, mock_subprocess, code_executor, test_workspace):
        """Test streaming execution functionality"""
        # Mock the subprocess
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = [
            b"Line 1\n",
            b"Line 2\n",
            b"Line 3\n",
            b""  # End of stream
        ]
        mock_process.stderr.readline.side_effect = [b""]
        mock_process.wait.return_value = asyncio.Future()
        mock_process.wait.return_value.set_result(0)
        mock_subprocess.return_value = mock_process

        command = ExecutionCommand(command="echo 'test'")

        events = []
        async for event in code_executor.stream_execution(test_workspace, command):
            events.append(event)

        # Should have start, stdout events, and complete event
        assert len(events) >= 2
        assert events[0]["type"] == "start"
        assert any(event["type"] == "stdout" for event in events)

    def test_execution_result_creation(self):
        """Test ExecutionResult object creation"""
        result = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="test output",
            stderr="",
            execution_time=1.5,
            files_created=["test.py"],
            files_modified=[]
        )

        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "test output"
        assert result.execution_time == 1.5
        assert result.files_created == ["test.py"]

    def test_execution_command_creation(self):
        """Test ExecutionCommand object creation"""
        command = ExecutionCommand(
            command="python test.py",
            working_directory="/tmp",
            timeout=120,
            capture_output=True,
            shell=True,
            env_vars={"TEST": "value"}
        )

        assert command.command == "python test.py"
        assert command.working_directory == "/tmp"
        assert command.timeout == 120
        assert command.capture_output is True
        assert command.shell is True
        assert command.env_vars == {"TEST": "value"}

    def test_execution_command_defaults(self):
        """Test ExecutionCommand default values"""
        command = ExecutionCommand(command="test command")

        assert command.working_directory == "."
        assert command.timeout == 300
        assert command.capture_output is True
        assert command.shell is True
        assert command.env_vars is None