"""Unit tests for the Dependency Analyzer module.

This is a basic test structure demonstrating how to test the dependency analyzer.
For a full implementation, comprehensive tests would be needed for all components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import os

# Import the components to test
try:
    from openhands.core.dependency_analyzer import (
        DependencyAnalyzer,
        ImportStatement,
        FileDependencies,
        DependencyAnalyzerConfig
    )
    from openhands.core.dependency_analyzer.parsers import (
        PythonDependencyParser,
        get_parser,
        detect_language
    )
    from openhands.core.dependency_analyzer.cache import DependencyCacheManager
    from openhands.core.dependency_analyzer.exceptions import (
        DependencyAnalysisError,
        UnsupportedLanguageError
    )
    DEPENDENCY_ANALYZER_AVAILABLE = True
except ImportError:
    DEPENDENCY_ANALYZER_AVAILABLE = False


@pytest.mark.skipif(not DEPENDENCY_ANALYZER_AVAILABLE, reason="DependencyAnalyzer not available")
class TestDependencyAnalyzer:
    """Test cases for the main DependencyAnalyzer class."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_files = {
                'main.py': '''
import os
from typing import List
from . import utils
import subprocess
''',
                'utils.py': '''
import json
from .helpers import helper_func
''',
                'helpers.py': '''
import sys
'''
            }
            
            for filename, content in test_files.items():
                file_path = Path(temp_dir) / filename
                file_path.write_text(content)
            
            yield temp_dir

    @pytest.fixture
    def analyzer_config(self):
        """Create a test configuration."""
        return DependencyAnalyzerConfig(
            use_llm_fallback=False,  # Disable LLM for unit tests
            cache_backend='memory',
            parallel_analysis=False,  # Disable for simpler testing
            skip_standard_libraries=True,
            skip_third_party_packages=True
        )

    @pytest.fixture
    def mock_analyzer(self, temp_workspace, analyzer_config):
        """Create a DependencyAnalyzer instance for testing."""
        return DependencyAnalyzer(
            workspace_base=temp_workspace,
            workspace_mount_path_in_sandbox='/workspace',
            config=analyzer_config
        )

    def test_initialization(self, mock_analyzer):
        """Test that DependencyAnalyzer initializes correctly."""
        assert mock_analyzer is not None
        assert mock_analyzer.config.use_llm_fallback is False
        assert mock_analyzer.config.cache_backend == 'memory'

    @pytest.mark.asyncio
    async def test_analyze_file_basic(self, mock_analyzer, temp_workspace):
        """Test basic file analysis."""
        main_py = os.path.join(temp_workspace, 'main.py')
        
        # Mock file operations to avoid actual file I/O complexity
        with patch('openhands.runtime.utils.files.read_file') as mock_read, \
             patch('openhands.runtime.utils.files.resolve_path') as mock_resolve:
            
            mock_resolve.return_value = main_py
            mock_read.return_value = Mock(content='''
import os
from typing import List
from . import utils
''')
            
            result = await mock_analyzer.analyze_file('main.py')
            
            assert isinstance(result, FileDependencies)
            assert result.file_path == main_py
            assert result.language == 'python'
            assert len(result.imports) > 0

    @pytest.mark.asyncio
    async def test_unsupported_language(self, mock_analyzer):
        """Test handling of unsupported file types."""
        with patch('openhands.runtime.utils.files.resolve_path') as mock_resolve:
            mock_resolve.return_value = '/path/to/file.unknown'
            
            with pytest.raises(UnsupportedLanguageError):
                await mock_analyzer.analyze_file('file.unknown')

    def test_detect_language(self):
        """Test language detection from file extensions."""
        assert detect_language('test.py') == 'python'
        assert detect_language('test.js') == 'javascript'
        assert detect_language('test.ts') == 'typescript'
        assert detect_language('test.java') == 'java'
        assert detect_language('test.unknown') is None


@pytest.mark.skipif(not DEPENDENCY_ANALYZER_AVAILABLE, reason="DependencyAnalyzer not available")
class TestPythonParser:
    """Test cases for the Python dependency parser."""

    @pytest.fixture
    def python_parser(self):
        """Create a Python parser instance."""
        return PythonDependencyParser()

    @pytest.mark.asyncio
    async def test_parse_simple_imports(self, python_parser):
        """Test parsing of simple import statements."""
        code = '''
import os
import sys
from typing import List
from . import utils
'''
        
        imports = await python_parser.parse('test.py', code)
        
        # Should find local imports only (os, sys, typing filtered out)
        local_imports = [imp for imp in imports if not python_parser.is_standard_library(imp.module, 'python')]
        
        assert len(local_imports) >= 1  # At least the relative import
        
        # Find the relative import
        relative_import = next((imp for imp in imports if imp.module == '.'), None)
        assert relative_import is not None
        assert relative_import.import_type == 'static'
        assert relative_import.source == 'ast'

    @pytest.mark.asyncio
    async def test_parse_dynamic_imports(self, python_parser):
        """Test parsing of dynamic import statements."""
        code = '''
import importlib
importlib.import_module('dynamic_module')
__import__('another_module')
'''
        
        imports = await python_parser.parse('test.py', code)
        
        # Look for dynamic imports
        dynamic_imports = [imp for imp in imports if imp.import_type == 'dynamic']
        
        assert len(dynamic_imports) >= 2
        assert any(imp.module == 'dynamic_module' for imp in dynamic_imports)
        assert any(imp.module == 'another_module' for imp in dynamic_imports)

    @pytest.mark.asyncio
    async def test_parse_syntax_error(self, python_parser):
        """Test handling of syntax errors."""
        code = '''
import os
def broken_function(
    # Missing closing parenthesis
'''
        
        imports = await python_parser.parse('test.py', code)
        
        # Should return empty list on syntax error
        assert imports == []

    def test_is_standard_library(self, python_parser):
        """Test standard library detection."""
        assert python_parser.is_standard_library('os', 'python') is True
        assert python_parser.is_standard_library('sys', 'python') is True
        assert python_parser.is_standard_library('json', 'python') is True
        assert python_parser.is_standard_library('mymodule', 'python') is False
        assert python_parser.is_standard_library('.utils', 'python') is False


@pytest.mark.skipif(not DEPENDENCY_ANALYZER_AVAILABLE, reason="DependencyAnalyzer not available")
class TestCacheManager:
    """Test cases for the cache manager."""

    @pytest.fixture
    def cache_manager(self):
        """Create a cache manager instance."""
        return DependencyCacheManager(
            cache_backend='memory',
            ttl_seconds=3600
        )

    @pytest.fixture
    def sample_file_dependencies(self):
        """Create sample FileDependencies for testing."""
        return FileDependencies(
            file_path='/test/file.py',
            imports=[
                ImportStatement(
                    module='os',
                    import_type='static',
                    line_number=1,
                    confidence=1.0,
                    source='ast'
                )
            ],
            resolved_paths=[],
            language='python',
            file_hash='abcd1234',
            parser_used='ast'
        )

    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache_manager, sample_file_dependencies):
        """Test basic cache set and get operations."""
        file_path = '/test/file.py'
        file_hash = 'abcd1234'
        
        # Initially should be empty
        result = await cache_manager.get(file_path, file_hash)
        assert result is None
        
        # Set value
        await cache_manager.set(file_path, file_hash, sample_file_dependencies)
        
        # Should be able to retrieve
        result = await cache_manager.get(file_path, file_hash)
        assert result is not None
        assert result.file_path == file_path
        assert len(result.imports) == 1

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_manager, sample_file_dependencies):
        """Test cache invalidation."""
        file_path = '/test/file.py'
        file_hash = 'abcd1234'
        
        # Set value
        await cache_manager.set(file_path, file_hash, sample_file_dependencies)
        
        # Verify it's there
        result = await cache_manager.get(file_path, file_hash)
        assert result is not None
        
        # Invalidate
        await cache_manager.invalidate(file_path)
        
        # Should be gone
        result = await cache_manager.get(file_path, file_hash)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        stats = await cache_manager.get_stats()
        
        assert 'backend' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_ratio' in stats
        assert stats['backend'] == 'memory'


@pytest.mark.skipif(not DEPENDENCY_ANALYZER_AVAILABLE, reason="DependencyAnalyzer not available")
class TestImportStatement:
    """Test cases for the ImportStatement model."""

    def test_import_statement_creation(self):
        """Test creation of ImportStatement objects."""
        stmt = ImportStatement(
            module='test_module',
            import_type='static',
            line_number=5,
            confidence=0.9,
            source='ast'
        )
        
        assert stmt.module == 'test_module'
        assert stmt.import_type == 'static'
        assert stmt.line_number == 5
        assert stmt.confidence == 0.9
        assert stmt.source == 'ast'
        assert stmt.resolved_path is None

    def test_import_statement_validation(self):
        """Test validation of ImportStatement fields."""
        # Confidence should be between 0.0 and 1.0
        with pytest.raises(ValueError):
            ImportStatement(
                module='test',
                import_type='static',
                line_number=1,
                confidence=1.5,  # Invalid confidence
                source='ast'
            )


@pytest.mark.skipif(not DEPENDENCY_ANALYZER_AVAILABLE, reason="DependencyAnalyzer not available")
class TestIntegrationHelpers:
    """Test cases for integration helper functions."""

    def test_is_dependency_analysis_enabled(self):
        """Test that dependency analysis is detected as available."""
        from openhands.core.dependency_analyzer.integration import is_dependency_analysis_enabled
        
        assert is_dependency_analysis_enabled() is True

    def test_get_default_config(self):
        """Test getting default analyzer configuration."""
        from openhands.core.dependency_analyzer.integration import get_default_analyzer_config
        
        config = get_default_analyzer_config()
        
        assert isinstance(config, DependencyAnalyzerConfig)
        assert config.use_llm_fallback is True
        assert config.cache_backend == 'memory'
        assert config.skip_standard_libraries is True


# Example of how to run these tests:
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
