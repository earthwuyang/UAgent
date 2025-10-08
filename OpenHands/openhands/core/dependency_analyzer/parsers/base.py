"""Abstract base class for language parsers."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import re
import sys

from openhands.core.dependency_analyzer.models import ImportStatement


class BaseDependencyParser(ABC):
    """Abstract base class for language-specific dependency parsers."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize parser with optional logger."""
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    async def parse(self, file_path: str, content: str) -> List[ImportStatement]:
        """Parse file content and extract imports.
        
        Args:
            file_path: Path to the file being parsed
            content: Content of the file
            
        Returns:
            List of ImportStatement objects found in the file
        """
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if parser supports given language.
        
        Args:
            language: Language name (python, javascript, typescript, java)
            
        Returns:
            True if parser supports the language
        """
        pass

    def _create_import_statement(
        self,
        module: str,
        import_type: str = 'static',
        line_number: int = 1,
        confidence: float = 1.0,
        source: str = 'ast'
    ) -> ImportStatement:
        """Helper to create ImportStatement with defaults.
        
        Args:
            module: Module name
            import_type: Type of import (static, dynamic, conditional)
            line_number: Line number in source file
            confidence: Confidence score (0.0-1.0)
            source: Parser source (ast, tree_sitter, llm)
            
        Returns:
            ImportStatement object
        """
        return ImportStatement(
            module=module,
            import_type=import_type,
            line_number=line_number,
            confidence=confidence,
            source=source
        )

    def extract_line_number(self, content: str, match_start: int) -> int:
        """Extract line number from content position.
        
        Args:
            content: Full file content
            match_start: Start position of match
            
        Returns:
            Line number (1-based)
        """
        return content[:match_start].count('\n') + 1

    def normalize_import_path(self, path: str) -> str:
        """Normalize import paths by removing quotes and whitespace.
        
        Args:
            path: Raw import path from code
            
        Returns:
            Normalized import path
        """
        # Remove quotes and whitespace
        path = path.strip().strip('\'"')
        return path

    def is_standard_library(self, module: str, language: str) -> bool:
        """Check if module is standard library (should be skipped).
        
        Args:
            module: Module name
            language: Programming language
            
        Returns:
            True if module is standard library
        """
        if language == 'python':
            return self._is_python_stdlib(module)
        elif language in ('javascript', 'typescript'):
            return self._is_js_builtin(module)
        elif language == 'java':
            return self._is_java_stdlib(module)
        return False

    def is_third_party(self, module: str, language: str) -> bool:
        """Check if module is third-party package (should be skipped).
        
        Args:
            module: Module name
            language: Programming language
            
        Returns:
            True if module is third-party package
        """
        if language == 'python':
            return self._is_python_third_party(module)
        elif language in ('javascript', 'typescript'):
            return self._is_js_third_party(module)
        elif language == 'java':
            return self._is_java_third_party(module)
        return False

    def _is_python_stdlib(self, module: str) -> bool:
        """Check if module is Python standard library."""
        # Get top-level module name
        top_module = module.split('.')[0]
        
        # Python 3.10+ has stdlib_module_names
        if hasattr(sys, 'stdlib_module_names'):
            return top_module in sys.stdlib_module_names
        
        # Fallback to hardcoded list for older Python versions
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 're', 'math', 'random',
            'collections', 'itertools', 'functools', 'operator', 'pathlib',
            'typing', 'dataclasses', 'enum', 'abc', 'contextlib', 'warnings',
            'logging', 'unittest', 'http', 'urllib', 'email', 'html', 'xml',
            'sqlite3', 'csv', 'io', 'tempfile', 'shutil', 'subprocess',
            'threading', 'multiprocessing', 'queue', 'socket', 'ssl',
            'hashlib', 'hmac', 'secrets', 'base64', 'binascii', 'struct',
            'codecs', 'locale', 'calendar', 'zoneinfo', 'pickle', 'copy',
            'pprint', 'reprlib', 'types', 'weakref', 'gc', 'inspect',
            'site', 'importlib', 'pkgutil', 'modulefinder', 'runpy',
            'argparse', 'getopt', 'getpass', 'curses', 'platform',
            'errno', 'signal', 'mmap', 'ctypes', 'array', 'decimal',
            'fractions', 'statistics', 'cmath', 'unicodedata', 'stringprep',
            'readline', 'rlcompleter', 'turtle', 'cmd', 'shlex'
        }
        return top_module in stdlib_modules

    def _is_python_third_party(self, module: str) -> bool:
        """Check if module is Python third-party package."""
        # Get top-level module name
        top_module = module.split('.')[0]
        
        # Common third-party packages (not exhaustive)
        third_party_packages = {
            'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'tensorflow',
            'torch', 'keras', 'requests', 'flask', 'django', 'fastapi',
            'pydantic', 'sqlalchemy', 'alembic', 'celery', 'redis', 'boto3',
            'click', 'typer', 'rich', 'tqdm', 'pytest', 'black', 'flake8',
            'mypy', 'isort', 'pylint', 'bandit', 'safety', 'poetry',
            'setuptools', 'wheel', 'pip', 'virtualenv', 'conda', 'jupyter',
            'ipython', 'notebook', 'streamlit', 'gradio', 'plotly', 'seaborn',
            'pillow', 'opencv', 'beautifulsoup4', 'lxml', 'xmltodict',
            'pyyaml', 'toml', 'configparser', 'python-dotenv', 'jinja2',
            'mako', 'marshmallow', 'attrs', 'cattrs', 'dacite', 'pydantic'
        }
        return top_module in third_party_packages

    def _is_js_builtin(self, module: str) -> bool:
        """Check if module is JavaScript/Node.js builtin."""
        # Node.js built-in modules
        builtin_modules = {
            'assert', 'buffer', 'child_process', 'cluster', 'console',
            'constants', 'crypto', 'dgram', 'dns', 'domain', 'events',
            'fs', 'http', 'http2', 'https', 'inspector', 'module',
            'net', 'os', 'path', 'perf_hooks', 'process', 'punycode',
            'querystring', 'readline', 'repl', 'stream', 'string_decoder',
            'sys', 'timers', 'tls', 'trace_events', 'tty', 'url', 'util',
            'v8', 'vm', 'wasi', 'worker_threads', 'zlib'
        }
        return module in builtin_modules

    def _is_js_third_party(self, module: str) -> bool:
        """Check if module is JavaScript third-party package."""
        # If it doesn't start with ./ or ../, it's likely a package
        # But we need to be more sophisticated here
        if module.startswith(('./', '../', '/')):
            return False
        
        # If it's not a Node.js builtin and doesn't start with relative path,
        # it's likely a third-party package from npm
        return not self._is_js_builtin(module)

    def _is_java_stdlib(self, module: str) -> bool:
        """Check if module is Java standard library."""
        java_stdlib_prefixes = (
            'java.', 'javax.', 'org.w3c.', 'org.xml.', 'org.ietf.',
            'org.omg.', 'sun.', 'com.sun.', 'jdk.', 'javafx.'
        )
        return module.startswith(java_stdlib_prefixes)

    def _is_java_third_party(self, module: str) -> bool:
        """Check if module is Java third-party package."""
        # Common third-party Java package prefixes
        third_party_prefixes = (
            'org.apache.', 'org.springframework.', 'com.google.',
            'com.fasterxml.', 'org.junit.', 'org.mockito.',
            'org.slf4j.', 'ch.qos.logback.', 'org.hibernate.',
            'com.amazonaws.', 'io.netty.', 'org.eclipse.',
            'com.github.', 'org.jetbrains.', 'kotlin.'
        )
        return module.startswith(third_party_prefixes)
