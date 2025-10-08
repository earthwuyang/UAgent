"""OpenHands Dependency Analyzer Module.

This module provides dependency analysis capabilities for file locking in multi-agent systems.
It combines AST parsing, tree-sitter, and LLM-based analysis to extract import relationships
and build dependency graphs.

Features:
- Multi-language support: Python, JavaScript/TypeScript, Java
- AST parsing for fast, reliable dependency extraction
- LLM fallback for complex cases (dynamic imports, syntax errors)
- Dependency graph building with cycle detection
- File hash-based caching with Redis support
- Import path resolution
- Integration with OpenHands MultiAgentCoordinator

Usage:
    from openhands.core.dependency_analyzer import DependencyAnalyzer
    
    analyzer = DependencyAnalyzer(
        workspace_base='/path/to/workspace',
        workspace_mount_path_in_sandbox='/workspace'
    )
    
    # Analyze a single file
    deps = await analyzer.analyze_file('myproject/main.py')
    
    # Get lock set for file locking
    lock_set = await analyzer.get_dependencies('myproject/main.py', transitive=True)
"""

from .core import DependencyAnalyzer
from .models import (
    ImportStatement,
    FileDependencies,
    DependencyGraph,
    DependencyAnalysisResult,
    DependencyAnalyzerConfig
)
from .parsers import (
    BaseDependencyParser,
    PythonDependencyParser,
    JavaScriptDependencyParser,
    TypeScriptDependencyParser,
    JavaDependencyParser,
    get_parser,
    get_parser_by_language,
    detect_language
)
from .cache import DependencyCacheManager
from .graph_builder import DependencyGraphBuilder
from .import_resolver import ImportResolver
from .llm_fallback import LLMDependencyExtractor
from .exceptions import (
    DependencyAnalysisError,
    ParserError,
    ImportResolutionError,
    CacheError,
    GraphBuildError,
    UnsupportedLanguageError
)

# Version information
__version__ = '1.0.0'
__author__ = 'OpenHands Team'

# Export main classes and functions
__all__ = [
    # Main analyzer
    'DependencyAnalyzer',
    
    # Data models
    'ImportStatement',
    'FileDependencies', 
    'DependencyGraph',
    'DependencyAnalysisResult',
    'DependencyAnalyzerConfig',
    
    # Language parsers
    'BaseDependencyParser',
    'PythonDependencyParser',
    'JavaScriptDependencyParser',
    'TypeScriptDependencyParser', 
    'JavaDependencyParser',
    'get_parser',
    'get_parser_by_language',
    'detect_language',
    
    # Core components
    'DependencyCacheManager',
    'DependencyGraphBuilder',
    'ImportResolver',
    'LLMDependencyExtractor',
    
    # Exceptions
    'DependencyAnalysisError',
    'ParserError',
    'ImportResolutionError',
    'CacheError', 
    'GraphBuildError',
    'UnsupportedLanguageError',
    
    # Version
    '__version__',
]

# Default configuration
DEFAULT_CONFIG = DependencyAnalyzerConfig(
    use_llm_fallback=True,
    cache_backend='memory',
    cache_ttl_seconds=3600,
    parallel_analysis=True,
    max_workers=10,
    skip_standard_libraries=True,
    skip_third_party_packages=True
)
