"""
Unit tests for OpenHands WorkspaceManager
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from app.core.openhands.workspace_manager import (
    WorkspaceManager,
    WorkspaceConfig,
    WorkspaceStatus
)


class TestWorkspaceManager:
    """Test WorkspaceManager functionality"""

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

    @pytest.mark.asyncio
    async def test_workspace_creation(self, workspace_manager):
        """Test basic workspace creation"""
        research_id = "test_research_123"

        config = await workspace_manager.create_workspace(research_id)

        assert config.workspace_id == research_id
        assert Path(config.base_path).exists()
        assert Path(config.base_path, "code").exists()
        assert Path(config.base_path, "data").exists()
        assert Path(config.base_path, "output").exists()
        assert Path(config.base_path, "logs").exists()

    @pytest.mark.asyncio
    async def test_workspace_creation_with_config(self, workspace_manager):
        """Test workspace creation with custom configuration"""
        research_id = "test_research_456"
        custom_config = {
            "max_file_size": 5 * 1024 * 1024,  # 5MB
            "timeout": 600
        }

        config = await workspace_manager.create_workspace(research_id, custom_config)

        assert config.workspace_id == research_id
        assert config.max_file_size == 5 * 1024 * 1024
        assert config.timeout == 600

    @pytest.mark.asyncio
    async def test_workspace_initialization_files(self, workspace_manager):
        """Test that workspace is initialized with proper files"""
        research_id = "test_research_789"

        config = await workspace_manager.create_workspace(research_id)
        workspace_path = Path(config.base_path)

        # Check initialization files
        assert (workspace_path / "requirements.txt").exists()
        assert (workspace_path / ".gitignore").exists()
        assert (workspace_path / "README.md").exists()

        # Verify content of requirements.txt
        requirements_content = (workspace_path / "requirements.txt").read_text()
        assert "numpy" in requirements_content
        assert "pandas" in requirements_content
        assert "matplotlib" in requirements_content

    @pytest.mark.asyncio
    async def test_workspace_status(self, workspace_manager):
        """Test workspace status retrieval"""
        research_id = "test_research_status"

        # Create workspace
        config = await workspace_manager.create_workspace(research_id)

        # Get status
        status = await workspace_manager.get_workspace_status(research_id)

        assert status is not None
        assert status.workspace_id == research_id
        assert status.status in ["active", "stopped"]
        assert status.file_count >= 3  # At least requirements.txt, .gitignore, README.md
        assert status.total_size > 0

    @pytest.mark.asyncio
    async def test_workspace_status_nonexistent(self, workspace_manager):
        """Test status for non-existent workspace"""
        status = await workspace_manager.get_workspace_status("nonexistent_workspace")
        assert status is None

    @pytest.mark.asyncio
    async def test_file_write_and_read(self, workspace_manager):
        """Test file write and read operations"""
        research_id = "test_file_ops"
        config = await workspace_manager.create_workspace(research_id)

        # Test file write
        file_content = "print('Hello, World!')"
        success = await workspace_manager.write_file(
            research_id, "code/hello.py", file_content
        )
        assert success is True

        # Test file read
        read_content = await workspace_manager.read_file(research_id, "code/hello.py")
        assert read_content == file_content

    @pytest.mark.asyncio
    async def test_file_write_security_boundary(self, workspace_manager):
        """Test that file operations respect security boundaries"""
        research_id = "test_security"
        config = await workspace_manager.create_workspace(research_id)

        # Try to write outside workspace (should fail)
        success = await workspace_manager.write_file(
            research_id, "../../etc/passwd", "malicious content"
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_file_size_limits(self, workspace_manager):
        """Test file size limit enforcement"""
        research_id = "test_file_limits"
        config = await workspace_manager.create_workspace(research_id)

        # Create content that exceeds default limit (10MB)
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        success = await workspace_manager.write_file(
            research_id, "large_file.txt", large_content
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_file_overwrite_behavior(self, workspace_manager):
        """Test file overwrite behavior"""
        research_id = "test_overwrite"
        config = await workspace_manager.create_workspace(research_id)

        # Write initial file
        initial_content = "original content"
        success = await workspace_manager.write_file(
            research_id, "test_file.txt", initial_content, overwrite=True
        )
        assert success is True

        # Overwrite with new content
        new_content = "updated content"
        success = await workspace_manager.write_file(
            research_id, "test_file.txt", new_content, overwrite=True
        )
        assert success is True

        # Verify new content
        read_content = await workspace_manager.read_file(research_id, "test_file.txt")
        assert read_content == new_content

        # Test overwrite=False
        success = await workspace_manager.write_file(
            research_id, "test_file.txt", "should not overwrite", overwrite=False
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_list_files(self, workspace_manager):
        """Test file listing functionality"""
        research_id = "test_list_files"
        config = await workspace_manager.create_workspace(research_id)

        # Create some test files
        await workspace_manager.write_file(research_id, "code/script1.py", "# Script 1")
        await workspace_manager.write_file(research_id, "code/script2.py", "# Script 2")
        await workspace_manager.write_file(research_id, "data/dataset.csv", "col1,col2\n1,2")

        # List root directory
        files = await workspace_manager.list_files(research_id, ".")
        file_names = [f["name"] for f in files]
        assert "code" in file_names
        assert "data" in file_names
        assert "requirements.txt" in file_names

        # List code directory
        code_files = await workspace_manager.list_files(research_id, "code")
        code_file_names = [f["name"] for f in code_files]
        assert "script1.py" in code_file_names
        assert "script2.py" in code_file_names

    @pytest.mark.asyncio
    async def test_list_files_security(self, workspace_manager):
        """Test that file listing respects security boundaries"""
        research_id = "test_list_security"
        config = await workspace_manager.create_workspace(research_id)

        # Try to list outside workspace
        files = await workspace_manager.list_files(research_id, "../../etc")
        assert files == []  # Should return empty list for security violation

    @pytest.mark.asyncio
    async def test_workspace_cleanup(self, workspace_manager):
        """Test workspace cleanup functionality"""
        research_id = "test_cleanup"
        config = await workspace_manager.create_workspace(research_id)
        workspace_path = Path(config.base_path)

        # Verify workspace exists
        assert workspace_path.exists()
        assert research_id in workspace_manager.workspaces

        # Mock active processes for testing
        mock_process = MagicMock()
        mock_process.returncode = None  # Still running
        workspace_manager.active_processes[research_id] = [mock_process]

        # Cleanup workspace
        success = await workspace_manager.cleanup_workspace(research_id)

        # Verify cleanup
        assert success is True
        assert not workspace_path.exists()
        assert research_id not in workspace_manager.workspaces
        assert research_id not in workspace_manager.active_processes

        # Verify process was terminated
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_all_workspaces(self, workspace_manager):
        """Test cleanup of all workspaces"""
        # Create multiple workspaces
        research_ids = ["test_cleanup_1", "test_cleanup_2", "test_cleanup_3"]
        for research_id in research_ids:
            await workspace_manager.create_workspace(research_id)

        # Verify all workspaces exist
        assert len(workspace_manager.workspaces) == 3

        # Cleanup all
        await workspace_manager.cleanup_all_workspaces()

        # Verify all cleaned up
        assert len(workspace_manager.workspaces) == 0
        assert len(workspace_manager.active_processes) == 0

    @pytest.mark.asyncio
    async def test_get_workspace_path(self, workspace_manager):
        """Test workspace path retrieval"""
        research_id = "test_get_path"
        config = await workspace_manager.create_workspace(research_id)

        # Test valid workspace
        path = workspace_manager.get_workspace_path(research_id)
        assert path is not None
        assert str(path) == config.base_path

        # Test invalid workspace
        invalid_path = workspace_manager.get_workspace_path("nonexistent")
        assert invalid_path is None

    @pytest.mark.asyncio
    async def test_workspace_duplicate_creation(self, workspace_manager):
        """Test behavior when creating workspace with existing ID"""
        research_id = "test_duplicate"

        # Create first workspace
        config1 = await workspace_manager.create_workspace(research_id)
        original_path = Path(config1.base_path)

        # Create workspace with same ID (should replace)
        config2 = await workspace_manager.create_workspace(research_id)
        new_path = Path(config2.base_path)

        # New workspace should exist, old one should be cleaned up
        assert new_path.exists()
        assert config2.workspace_id == research_id

    @pytest.mark.asyncio
    async def test_concurrent_workspace_operations(self, workspace_manager):
        """Test concurrent workspace operations"""
        research_ids = [f"test_concurrent_{i}" for i in range(5)]

        # Create workspaces concurrently
        tasks = [
            workspace_manager.create_workspace(research_id)
            for research_id in research_ids
        ]
        configs = await asyncio.gather(*tasks)

        # Verify all workspaces created successfully
        assert len(configs) == 5
        for i, config in enumerate(configs):
            assert config.workspace_id == research_ids[i]
            assert Path(config.base_path).exists()

    @pytest.mark.asyncio
    async def test_error_handling_in_file_operations(self, workspace_manager):
        """Test error handling in file operations"""
        research_id = "test_error_handling"

        # Test operations on non-existent workspace
        success = await workspace_manager.write_file(
            "nonexistent_workspace", "test.txt", "content"
        )
        assert success is False

        content = await workspace_manager.read_file(
            "nonexistent_workspace", "test.txt"
        )
        assert content is None

        files = await workspace_manager.list_files("nonexistent_workspace")
        assert files == []

    def test_workspace_config_defaults(self):
        """Test WorkspaceConfig default values"""
        config = WorkspaceConfig(
            workspace_id="test_defaults",
            base_path="/tmp/test"
        )

        assert config.max_file_size == 10 * 1024 * 1024  # 10MB
        assert config.max_total_size == 100 * 1024 * 1024  # 100MB
        assert config.timeout == 300  # 5 minutes
        assert config.python_path == "/usr/bin/python3"

    def test_workspace_status_creation(self):
        """Test WorkspaceStatus object creation"""
        status = WorkspaceStatus(
            workspace_id="test_status",
            status="active",
            created_at="2024-01-01T00:00:00Z",
            last_activity="2024-01-01T01:00:00Z",
            file_count=10,
            total_size=1024,
            processes=[]
        )

        assert status.workspace_id == "test_status"
        assert status.status == "active"
        assert status.file_count == 10
        assert status.total_size == 1024
        assert status.processes == []

    @pytest.mark.asyncio
    async def test_workspace_manager_initialization(self, temp_base_dir):
        """Test WorkspaceManager initialization"""
        # Test with custom base directory
        manager1 = WorkspaceManager(temp_base_dir)
        assert str(manager1.base_dir) == str(Path(temp_base_dir) / "uagent_workspaces")

        # Test with default directory (None)
        manager2 = WorkspaceManager(None)
        assert "uagent_workspaces" in str(manager2.base_dir)

        # Verify base directory is created
        assert manager1.base_dir.exists()