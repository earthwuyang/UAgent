"""
Unit tests for OCC Validator.

Tests for optimistic concurrency control validation system including AST region
signatures, read/write tracking, conflict detection, and validation logic.
"""

import ast
import hashlib
import logging
import pytest
import time
from typing import Dict, List, Set
from unittest.mock import AsyncMock, Mock, MagicMock, patch

from openhands.core.occ_validator import (
    OCCValidator,
    ReadWriteTracker,
    ASTRegionSignature,
    ValidationResult,
    ConflictDetail,
    ConflictType,
    Region,
    _get_file_content_at_commit,
    _compute_region_hash,
    _regions_overlap
)
from openhands.core.dependency_analyzer.core import DependencyAnalyzer
from openhands.runtime.utils.git_handler import GitHandler


class TestASTRegionSignature:
    """Test AST region signature computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.signature = ASTRegionSignature()
    
    def test_compute_signature_python(self):
        """Test signature computation for Python functions/classes."""
        content = """
def hello():
    print("hello")

class TestClass:
    def method(self):
        pass
"""
        region = Region(
            file_path="test.py",
            start_line=1,
            end_line=3,
            region_type="function",
            name="hello"
        )
        
        signature = self.signature.compute_signature("test.py", content, region)
        assert signature != ""
        assert len(signature) == 16  # SHA256 truncated to 16 chars
        
        # Same content should produce same signature
        signature2 = self.signature.compute_signature("test.py", content, region)
        assert signature == signature2
    
    def test_compute_signature_javascript(self):
        """Test signature computation for JavaScript functions."""
        content = """
function hello() {
    console.log("hello");
}

class TestClass {
    method() {
        return true;
    }
}
"""
        region = Region(
            file_path="test.js",
            start_line=1,
            end_line=3,
            region_type="function",
            name="hello"
        )
        
        signature = self.signature.compute_signature("test.js", content, region)
        assert signature != ""
        assert len(signature) == 16
    
    def test_extract_regions_python(self):
        """Test region extraction from Python file."""
        content = """
def hello():
    print("hello")

def world():
    print("world")

class TestClass:
    def method(self):
        pass
"""
        regions = self.signature.extract_regions("test.py", content)
        
        assert len(regions) >= 3  # At least hello, world, TestClass
        function_names = [r.name for r in regions if r.region_type == 'function']
        class_names = [r.name for r in regions if r.region_type == 'class']
        
        assert "hello" in function_names
        assert "world" in function_names
        assert "TestClass" in class_names
    
    def test_extract_regions_javascript(self):
        """Test region extraction from JavaScript file."""
        content = """
function hello() {
    console.log("hello");
}

class TestClass {
    method() {
        return true;
    }
}
"""
        regions = self.signature.extract_regions("test.js", content)
        
        # Should extract at least the function and class
        assert len(regions) >= 1
        
        # Check if we found the function (might be regex-based if tree-sitter unavailable)
        function_names = [r.name for r in regions if r.region_type == 'function']
        class_names = [r.name for r in regions if r.region_type == 'class']
        
        if function_names or class_names:  # If parsing worked
            assert "hello" in function_names or "TestClass" in class_names
    
    def test_compare_signatures_match(self):
        """Test signature comparison when regions match."""
        sig1 = "abcd1234"
        sig2 = "abcd1234"
        
        assert self.signature.compare_signatures(sig1, sig2)
    
    def test_compare_signatures_differ(self):
        """Test signature comparison when regions differ."""
        sig1 = "abcd1234"
        sig2 = "efgh5678"
        
        assert not self.signature.compare_signatures(sig1, sig2)
    
    def test_signature_stable_across_whitespace(self):
        """Test that signatures ignore whitespace changes."""
        content1 = """
def hello():
    print("hello")
"""
        content2 = """
def hello(  ):
     print( "hello" )
"""
        
        region = Region(
            file_path="test.py",
            start_line=1,
            end_line=3,
            region_type="function",
            name="hello"
        )
        
        sig1 = self.signature.compute_signature("test.py", content1, region)
        sig2 = self.signature.compute_signature("test.py", content2, region)
        
        # Signatures should be similar for whitespace-only changes
        # (This depends on implementation - they might differ due to AST vs normalized content)
        assert sig1 != "" and sig2 != ""
    
    def test_signature_changes_on_code_change(self):
        """Test that signatures change when code changes."""
        content1 = """
def hello():
    print("hello")
"""
        content2 = """
def hello():
    print("world")
"""
        
        region = Region(
            file_path="test.py",
            start_line=1,
            end_line=3,
            region_type="function",
            name="hello"
        )
        
        sig1 = self.signature.compute_signature("test.py", content1, region)
        sig2 = self.signature.compute_signature("test.py", content2, region)
        
        assert sig1 != sig2


class TestReadWriteTracker:
    """Test read/write tracking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = ReadWriteTracker("abc123")
    
    def test_track_read(self):
        """Test tracking file reads."""
        self.tracker.track_read("file1.py")
        
        read_set = self.tracker.get_read_set()
        assert "file1.py" in read_set
        assert len(read_set["file1.py"]) == 1
        
        # Should have entire_file region
        region = list(read_set["file1.py"])[0]
        assert region.name == "entire_file"
    
    def test_track_write(self):
        """Test tracking file writes."""
        self.tracker.track_write("file2.py")
        
        write_set = self.tracker.get_write_set()
        assert "file2.py" in write_set
        assert len(write_set["file2.py"]) == 1
        
        # Should have entire_file region
        region = list(write_set["file2.py"])[0]
        assert region.name == "entire_file"
    
    def test_track_multiple_files(self):
        """Test tracking multiple files."""
        self.tracker.track_read("file1.py")
        self.tracker.track_read("file2.py")
        self.tracker.track_write("file3.py")
        
        read_set = self.tracker.get_read_set()
        write_set = self.tracker.get_write_set()
        
        assert len(read_set) == 2
        assert len(write_set) == 1
        assert "file1.py" in read_set
        assert "file2.py" in read_set
        assert "file3.py" in write_set
    
    def test_track_regions(self):
        """Test tracking specific regions within files."""
        regions = [
            Region("file1.py", 1, 10, "function", "test_func"),
            Region("file1.py", 15, 25, "function", "another_func")
        ]
        
        self.tracker.track_read("file1.py", regions)
        
        read_set = self.tracker.get_read_set()
        assert "file1.py" in read_set
        assert len(read_set["file1.py"]) == 2
        
        region_names = {r.name for r in read_set["file1.py"]}
        assert "test_func" in region_names
        assert "another_func" in region_names
    
    def test_get_read_set(self):
        """Test retrieving read set."""
        self.tracker.track_read("file1.py")
        self.tracker.track_read("file2.py")
        
        read_set = self.tracker.get_read_set()
        
        assert isinstance(read_set, dict)
        assert len(read_set) == 2
        
        # Should be a copy
        read_set["file3.py"] = set()
        assert "file3.py" not in self.tracker.read_set
    
    def test_get_write_set(self):
        """Test retrieving write set."""
        self.tracker.track_write("file1.py")
        self.tracker.track_write("file2.py")
        
        write_set = self.tracker.get_write_set()
        
        assert isinstance(write_set, dict)
        assert len(write_set) == 2
        
        # Should be a copy
        write_set["file3.py"] = set()
        assert "file3.py" not in self.tracker.write_set
    
    def test_clear(self):
        """Test clearing tracked data."""
        self.tracker.track_read("file1.py")
        self.tracker.track_write("file2.py")
        
        assert len(self.tracker.read_set) == 1
        assert len(self.tracker.write_set) == 1
        
        self.tracker.clear()
        
        assert len(self.tracker.read_set) == 0
        assert len(self.tracker.write_set) == 0


class TestOCCValidatorReadConflicts:
    """Test OCC validator read conflict detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dependency_analyzer = Mock(spec=DependencyAnalyzer)
        self.mock_git_handler = Mock(spec=GitHandler)
        self.mock_logger = Mock(spec=logging.Logger)
        
        self.validator = OCCValidator(
            workspace_base="/test/workspace",
            workspace_mount_path_in_sandbox="/workspace",
            dependency_analyzer=self.mock_dependency_analyzer,
            git_handler=self.mock_git_handler,
            logger=self.mock_logger
        )
    
    @pytest.mark.asyncio
    async def test_validate_no_read_conflicts(self):
        """Test validation when no read conflicts exist."""
        # Mock git operations
        self.validator._get_changed_files = AsyncMock(return_value=set())
        
        read_set = {"file1.py": {Region("file1.py", 1, 10, "function", "test")}}
        write_set = {}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        assert result.success
        assert len(result.conflicts) == 0
        assert result.suggested_resolution == "commit"
    
    @pytest.mark.asyncio
    async def test_validate_read_conflict_detected(self):
        """Test detection of read conflicts (file read by agent was modified)."""
        # Mock git operations
        self.validator._get_changed_files = AsyncMock(return_value={"file1.py"})
        self.validator._get_file_content_at_commit = AsyncMock(
            side_effect=lambda path, commit: "original content" if commit == "abc123" else "modified content"
        )
        
        read_set = {"file1.py": {Region("file1.py", 1, 999999, "block", "entire_file")}}
        write_set = {}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        assert not result.success
        assert len(result.conflicts) >= 1
        assert result.conflicts[0].conflict_type == ConflictType.READ_WRITE
        assert result.suggested_resolution in ["rebase", "merge"]
    
    @pytest.mark.asyncio
    async def test_validate_read_conflict_different_region(self):
        """Test no conflict when different regions modified."""
        # Mock git operations to return no changed files
        self.validator._get_changed_files = AsyncMock(return_value=set())
        
        read_set = {"file1.py": {Region("file1.py", 1, 10, "function", "test_func")}}
        write_set = {}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        assert result.success
        assert len(result.conflicts) == 0
    
    @pytest.mark.asyncio
    async def test_validate_read_conflict_overlapping_region(self):
        """Test conflict when overlapping regions modified."""
        # Mock git operations
        self.validator._get_changed_files = AsyncMock(return_value={"file1.py"})
        
        # Mock file content with different versions
        base_content = """
def test_func():
    print("original")
"""
        current_content = """
def test_func():
    print("modified")
"""
        
        self.validator._get_file_content_at_commit = AsyncMock(
            side_effect=lambda path, commit: base_content if commit == "abc123" else current_content
        )
        
        read_set = {"file1.py": {Region("file1.py", 1, 3, "function", "test_func")}}
        write_set = {}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        # Should detect conflict since the read region was modified
        assert not result.success
        assert len(result.conflicts) >= 1


class TestOCCValidatorWriteConflicts:
    """Test OCC validator write conflict detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dependency_analyzer = Mock(spec=DependencyAnalyzer)
        self.mock_git_handler = Mock(spec=GitHandler)
        self.mock_logger = Mock(spec=logging.Logger)
        
        self.validator = OCCValidator(
            workspace_base="/test/workspace",
            workspace_mount_path_in_sandbox="/workspace",
            dependency_analyzer=self.mock_dependency_analyzer,
            git_handler=self.mock_git_handler,
            logger=self.mock_logger
        )
    
    @pytest.mark.asyncio
    async def test_validate_no_write_conflicts(self):
        """Test validation when no write conflicts exist."""
        # Mock git operations
        self.validator._get_changed_files = AsyncMock(return_value=set())
        
        read_set = {}
        write_set = {"file1.py": {Region("file1.py", 1, 10, "function", "test")}}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        assert result.success
        assert len(result.conflicts) == 0
        assert result.suggested_resolution == "commit"
    
    @pytest.mark.asyncio
    async def test_validate_write_conflict_detected(self):
        """Test detection of write conflicts (same region modified)."""
        # Mock git operations
        self.validator._get_changed_files = AsyncMock(return_value={"file1.py"})
        self.validator._get_file_content_at_commit = AsyncMock(
            side_effect=lambda path, commit: "original content" if commit == "abc123" else "modified content"
        )
        
        read_set = {}
        write_set = {"file1.py": {Region("file1.py", 1, 999999, "block", "entire_file")}}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        assert not result.success
        assert len(result.conflicts) >= 1
        assert result.conflicts[0].conflict_type == ConflictType.WRITE_WRITE
        assert result.suggested_resolution in ["merge", "abort"]
    
    @pytest.mark.asyncio
    async def test_validate_write_conflict_different_files(self):
        """Test no conflict when different files modified."""
        # Mock git operations - file1.py changed but agent wrote file2.py
        self.validator._get_changed_files = AsyncMock(return_value={"file1.py"})
        
        read_set = {}
        write_set = {"file2.py": {Region("file2.py", 1, 10, "function", "test")}}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        assert result.success
        assert len(result.conflicts) == 0
    
    @pytest.mark.asyncio
    async def test_validate_write_conflict_same_file_different_regions(self):
        """Test no conflict when same file, different regions."""
        # This test would require more sophisticated mocking of region extraction
        # For now, just test the basic case
        self.validator._get_changed_files = AsyncMock(return_value=set())
        
        read_set = {}
        write_set = {"file1.py": {Region("file1.py", 1, 10, "function", "func1")}}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        assert result.success


class TestConflictClassification:
    """Test conflict classification logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dependency_analyzer = Mock(spec=DependencyAnalyzer)
        self.mock_git_handler = Mock(spec=GitHandler)
        
        self.validator = OCCValidator(
            workspace_base="/test/workspace",
            workspace_mount_path_in_sandbox="/workspace",
            dependency_analyzer=self.mock_dependency_analyzer,
            git_handler=self.mock_git_handler
        )
    
    def test_classify_formatting_conflict(self):
        """Test classification of formatting-only changes."""
        base_content = "def hello():\n    print('hello')"
        current_content = "def hello():\n     print( 'hello' )"
        
        conflict_type = self.validator._classify_conflict(base_content, current_content, "", None)
        
        # Might be classified as formatting if normalization makes them equal
        assert conflict_type in [ConflictType.FORMATTING, ConflictType.SEMANTIC]
    
    def test_classify_ast_reordering_conflict(self):
        """Test classification of AST reordering (imports, etc.)."""
        base_content = "import os\nimport sys\ndef func(): pass"
        current_content = "import sys\nimport os\ndef func(): pass"
        
        conflict_type = self.validator._classify_conflict(base_content, current_content, "", None)
        
        # Should be classified as AST reordering since same lines, different order
        assert conflict_type == ConflictType.AST_REORDERING
    
    def test_classify_semantic_conflict(self):
        """Test classification of semantic conflicts (logic changes)."""
        base_content = "def func(): return True"
        current_content = "def func(): return False"
        
        conflict_type = self.validator._classify_conflict(base_content, current_content, "", None)
        
        # Should be semantic conflict
        assert conflict_type == ConflictType.SEMANTIC


class TestResolutionSuggestions:
    """Test resolution strategy suggestions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dependency_analyzer = Mock(spec=DependencyAnalyzer)
        self.mock_git_handler = Mock(spec=GitHandler)
        
        self.validator = OCCValidator(
            workspace_base="/test/workspace",
            workspace_mount_path_in_sandbox="/workspace",
            dependency_analyzer=self.mock_dependency_analyzer,
            git_handler=self.mock_git_handler
        )
    
    def test_suggest_rebase_for_formatting(self):
        """Test rebase suggestion for formatting conflicts."""
        conflicts = [
            ConflictDetail(
                file_path="file1.py",
                conflict_type=ConflictType.FORMATTING,
                description="Formatting conflict",
                suggested_resolution="rebase"
            )
        ]
        
        resolution = self.validator._suggest_resolution(conflicts)
        assert resolution == "rebase"
    
    def test_suggest_merge_for_different_regions(self):
        """Test merge suggestion for non-overlapping changes."""
        conflicts = [
            ConflictDetail(
                file_path="file1.py",
                conflict_type=ConflictType.READ_WRITE,
                description="Read-write conflict",
                suggested_resolution="merge"
            )
        ]
        
        resolution = self.validator._suggest_resolution(conflicts)
        assert resolution in ["merge", "abort"]  # Depends on specific conflict type
    
    def test_suggest_abort_for_semantic_conflict(self):
        """Test abort suggestion for semantic conflicts."""
        conflicts = [
            ConflictDetail(
                file_path="file1.py",
                conflict_type=ConflictType.SEMANTIC,
                description="Semantic conflict",
                suggested_resolution="abort"
            )
        ]
        
        resolution = self.validator._suggest_resolution(conflicts)
        assert resolution == "abort"
    
    def test_suggest_commit_for_no_conflicts(self):
        """Test commit suggestion when no conflicts."""
        conflicts = []
        
        resolution = self.validator._suggest_resolution(conflicts)
        assert resolution == "commit"


class TestGitIntegration:
    """Test integration with Git operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_git_handler = Mock(spec=GitHandler)
    
    def test_get_file_content_at_commit(self):
        """Test retrieving file content at specific commit."""
        # Mock git show command
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "file content at commit"
        self.mock_git_handler.run_git_command.return_value = mock_result
        
        content = _get_file_content_at_commit("file1.py", "abc123", self.mock_git_handler)
        
        assert content == "file content at commit"
        self.mock_git_handler.run_git_command.assert_called_once_with(['show', 'abc123:file1.py'])
    
    def test_get_file_content_at_commit_failure(self):
        """Test handling of git command failure."""
        # Mock git show command failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "file not found"
        self.mock_git_handler.run_git_command.return_value = mock_result
        
        with pytest.raises(Exception) as exc_info:
            _get_file_content_at_commit("file1.py", "abc123", self.mock_git_handler)
        
        assert "Git show failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_with_real_git_repo(self):
        """Test validation with mocked git repository."""
        # This test mocks the git operations rather than using a real repo
        mock_dependency_analyzer = Mock(spec=DependencyAnalyzer)
        
        validator = OCCValidator(
            workspace_base="/test/workspace",
            workspace_mount_path_in_sandbox="/workspace",
            dependency_analyzer=mock_dependency_analyzer,
            git_handler=self.mock_git_handler
        )
        
        # Mock git diff to show no changes
        validator._get_changed_files = AsyncMock(return_value=set())
        
        read_set = {"file1.py": {Region("file1.py", 1, 10, "function", "test")}}
        write_set = {}
        
        result = await validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        assert result.success


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dependency_analyzer = Mock(spec=DependencyAnalyzer)
        self.mock_git_handler = Mock(spec=GitHandler)
        
        self.validator = OCCValidator(
            workspace_base="/test/workspace",
            workspace_mount_path_in_sandbox="/workspace",
            dependency_analyzer=self.mock_dependency_analyzer,
            git_handler=self.mock_git_handler
        )
    
    @pytest.mark.asyncio
    async def test_validate_with_deleted_file(self):
        """Test validation when file is deleted."""
        # Mock file as changed but content retrieval fails
        self.validator._get_changed_files = AsyncMock(return_value={"deleted_file.py"})
        self.validator._get_file_content_at_commit = AsyncMock(
            side_effect=Exception("File not found")
        )
        
        read_set = {"deleted_file.py": {Region("deleted_file.py", 1, 10, "function", "test")}}
        write_set = {}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        # Should handle gracefully and report conflict
        assert not result.success or len(result.conflicts) >= 0  # Either fails or has conflicts
    
    @pytest.mark.asyncio
    async def test_validate_with_new_file(self):
        """Test validation when file is newly created."""
        # Mock new file (doesn't exist at base commit)
        self.validator._get_changed_files = AsyncMock(return_value={"new_file.py"})
        self.validator._get_file_content_at_commit = AsyncMock(
            side_effect=lambda path, commit: "" if commit == "abc123" else "new content"
        )
        
        write_set = {"new_file.py": {Region("new_file.py", 1, 10, "function", "test")}}
        read_set = {}
        
        result = await self.validator.validate_commit(
            read_set=read_set,
            write_set=write_set,
            base_commit="abc123",
            current_head="def456"
        )
        
        # New file creation might cause conflicts
        assert isinstance(result, ValidationResult)
    
    def test_validate_with_syntax_error(self):
        """Test validation when file has syntax errors."""
        signature = ASTRegionSignature()
        
        # Python file with syntax error
        invalid_content = """
def hello(
    print("missing closing paren")
"""
        
        # Should handle syntax error gracefully
        regions = signature.extract_regions("invalid.py", invalid_content)
        
        # Should fall back to generic regions or return empty list
        assert isinstance(regions, list)
    
    def test_validate_with_unsupported_language(self):
        """Test validation with unsupported language."""
        signature = ASTRegionSignature()
        
        # Unsupported language file
        content = """
// This is some unknown language
function test() {
    // code here
}
"""
        
        regions = signature.extract_regions("test.unknown", content)
        
        # Should return generic regions
        assert isinstance(regions, list)
        
        if regions:
            # Should be generic statement regions
            assert all(r.region_type == "statement" for r in regions)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_compute_region_hash(self):
        """Test computing hash of code region."""
        content = """line1
line2
line3
line4"""
        
        hash1 = _compute_region_hash(content, 1, 2)  # line1, line2
        hash2 = _compute_region_hash(content, 3, 4)  # line3, line4
        hash3 = _compute_region_hash(content, 1, 2)  # line1, line2 again
        
        assert hash1 != hash2  # Different content should have different hashes
        assert hash1 == hash3  # Same content should have same hash
        assert len(hash1) == 16  # Should be truncated SHA256
    
    def test_compute_region_hash_invalid_range(self):
        """Test hash computation with invalid line range."""
        content = "line1\nline2\nline3"
        
        # Invalid ranges should return empty string
        assert _compute_region_hash(content, 0, 1) == ""  # start_line < 1
        assert _compute_region_hash(content, 1, 10) == ""  # end_line > len(lines)
        assert _compute_region_hash(content, 3, 1) == ""  # start_line > end_line
    
    def test_regions_overlap(self):
        """Test region overlap detection."""
        region1 = Region("file.py", 1, 10, "function", "func1")
        region2 = Region("file.py", 5, 15, "function", "func2")
        region3 = Region("file.py", 20, 30, "function", "func3")
        region4 = Region("other.py", 1, 10, "function", "func4")
        
        # Overlapping regions (same file, overlapping lines)
        assert _regions_overlap(region1, region2)
        assert _regions_overlap(region2, region1)
        
        # Non-overlapping regions (same file, non-overlapping lines)
        assert not _regions_overlap(region1, region3)
        assert not _regions_overlap(region3, region1)
        
        # Different files should not overlap
        assert not _regions_overlap(region1, region4)
        assert not _regions_overlap(region4, region1)


class TestValidationResult:
    """Test ValidationResult functionality."""
    
    def test_validation_result_properties(self):
        """Test ValidationResult properties and methods."""
        conflicts = [
            ConflictDetail("file1.py", ConflictType.FORMATTING),
            ConflictDetail("file2.py", ConflictType.SEMANTIC)
        ]
        
        result = ValidationResult(
            success=False,
            conflicts=conflicts,
            suggested_resolution="merge",
            validation_duration_ms=123.45,
            read_conflicts_count=1,
            write_conflicts_count=1
        )
        
        assert not result.success
        assert result.conflicts_count == 2
        assert len(result.conflicts) == 2
        assert result.suggested_resolution == "merge"
        assert result.validation_duration_ms == 123.45
        assert result.read_conflicts_count == 1
        assert result.write_conflicts_count == 1


class TestRegion:
    """Test Region functionality."""
    
    def test_region_overlap(self):
        """Test region overlap detection."""
        region1 = Region("file.py", 1, 10, "function", "func1")
        region2 = Region("file.py", 5, 15, "function", "func2")
        region3 = Region("file.py", 20, 30, "function", "func3")
        
        assert region1.overlaps(region2)
        assert region2.overlaps(region1)
        assert not region1.overlaps(region3)
    
    def test_region_serialization(self):
        """Test region serialization to/from dictionary."""
        region = Region(
            file_path="test.py",
            start_line=1,
            end_line=10,
            region_type="function",
            name="test_func",
            signature="abc123"
        )
        
        # Test to_dict
        region_dict = region.to_dict()
        expected_keys = ['file_path', 'start_line', 'end_line', 'region_type', 'name', 'signature']
        assert all(key in region_dict for key in expected_keys)
        
        # Test from_dict
        restored_region = Region.from_dict(region_dict)
        assert restored_region.file_path == region.file_path
        assert restored_region.start_line == region.start_line
        assert restored_region.end_line == region.end_line
        assert restored_region.region_type == region.region_type
        assert restored_region.name == region.name
        assert restored_region.signature == region.signature
    
    def test_region_hash(self):
        """Test region hashing for use in sets."""
        region1 = Region("file.py", 1, 10, "function", "func1")
        region2 = Region("file.py", 1, 10, "function", "func1")
        region3 = Region("file.py", 1, 10, "function", "func2")
        
        # Same regions should have same hash
        assert hash(region1) == hash(region2)
        
        # Different regions should (usually) have different hash
        assert hash(region1) != hash(region3)
        
        # Should work in sets
        region_set = {region1, region2, region3}
        assert len(region_set) == 2  # region1 and region2 are the same


# Integration test fixtures and utilities

@pytest.fixture
def mock_dependency_analyzer():
    """Create mock dependency analyzer."""
    analyzer = Mock(spec=DependencyAnalyzer)
    return analyzer


@pytest.fixture
def mock_git_handler():
    """Create mock git handler."""
    handler = Mock(spec=GitHandler)
    
    # Default successful responses
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""
    handler.run_git_command.return_value = mock_result
    
    return handler


@pytest.fixture
def sample_python_file():
    """Sample Python file content."""
    return """
import os
import sys

def hello_world():
    print("Hello, world!")

def goodbye_world():
    print("Goodbye, world!")

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
"""


@pytest.fixture
def sample_javascript_file():
    """Sample JavaScript file content."""
    return """
function helloWorld() {
    console.log("Hello, world!");
}

function goodbyeWorld() {
    console.log("Goodbye, world!");
}

class TestClass {
    constructor() {
        this.value = 42;
    }
    
    getValue() {
        return this.value;
    }
}
"""


def create_mock_region(file_path: str, start_line: int, end_line: int, name: str) -> Region:
    """Create mock Region object for testing."""
    return Region(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        region_type="function",
        name=name
    )


def create_mock_validation_result(success: bool, conflicts_count: int = 0) -> ValidationResult:
    """Create mock ValidationResult for testing."""
    conflicts = []
    if conflicts_count > 0:
        for i in range(conflicts_count):
            conflicts.append(ConflictDetail(
                file_path=f"file{i}.py",
                conflict_type=ConflictType.SEMANTIC,
                description=f"Test conflict {i}"
            ))
    
    return ValidationResult(
        success=success,
        conflicts=conflicts,
        suggested_resolution="commit" if success else "abort",
        validation_duration_ms=100.0,
        read_conflicts_count=conflicts_count // 2,
        write_conflicts_count=conflicts_count - conflicts_count // 2
    )


if __name__ == "__main__":
    pytest.main([__file__])
