"""
RepoMaster Integration - Advanced semantic code analysis and repository understanding
Deep code comprehension, pattern detection, and intelligent code generation
"""

import ast
import os
import re
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import json

from .meta_agent import Task, TaskType


class CodeElementType(Enum):
    """Types of code elements that can be analyzed"""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    MODULE = "module"
    INTERFACE = "interface"
    ENUM = "enum"
    CONSTANT = "constant"


class AnalysisDepth(Enum):
    """Depth levels for code analysis"""
    SURFACE = "surface"      # Basic structure and imports
    SEMANTIC = "semantic"    # Function/class relationships
    DEEP = "deep"           # Control flow, data flow, patterns
    COMPREHENSIVE = "comprehensive"  # Full analysis with ML insights


@dataclass
class CodeElement:
    """Represents a single code element with semantic information"""
    id: str
    name: str
    element_type: CodeElementType
    file_path: str
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    complexity: int = 0
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodePattern:
    """Represents a detected code pattern"""
    id: str
    name: str
    pattern_type: str
    description: str
    occurrences: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    suggestions: List[str] = field(default_factory=list)


@dataclass
class Repository:
    """Represents a complete repository with semantic understanding"""
    id: str
    path: str
    name: str
    language: str
    structure: Dict[str, Any] = field(default_factory=dict)
    elements: List[CodeElement] = field(default_factory=list)
    patterns: List[CodePattern] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: datetime = field(default_factory=datetime.now)


class SemanticAnalyzer:
    """Advanced semantic analysis engine for code understanding"""

    def __init__(self):
        self.language_parsers = {
            'python': PythonParser(),
            'javascript': JavaScriptParser(),
            'typescript': TypeScriptParser(),
            'java': JavaParser(),
            'cpp': CppParser()
        }
        self.pattern_detectors = self._initialize_pattern_detectors()

    def _initialize_pattern_detectors(self) -> List:
        """Initialize pattern detection modules"""
        return [
            DesignPatternDetector(),
            CodeSmellDetector(),
            PerformancePatternDetector(),
            SecurityPatternDetector()
        ]

    async def analyze_repository(
        self,
        repo_path: str,
        depth: AnalysisDepth = AnalysisDepth.SEMANTIC
    ) -> Repository:
        """Perform comprehensive semantic analysis of repository"""
        repo_id = hashlib.md5(repo_path.encode()).hexdigest()
        repo_name = Path(repo_path).name

        # Detect primary language
        language = await self._detect_primary_language(repo_path)

        # Create repository object
        repository = Repository(
            id=repo_id,
            path=repo_path,
            name=repo_name,
            language=language
        )

        # Analyze repository structure
        repository.structure = await self._analyze_structure(repo_path)

        # Extract code elements
        repository.elements = await self._extract_code_elements(repo_path, language, depth)

        # Detect patterns
        repository.patterns = await self._detect_patterns(repository.elements)

        # Build dependency graph
        repository.dependencies = await self._build_dependency_graph(repository.elements)

        # Calculate metrics
        repository.metrics = await self._calculate_metrics(repository)

        return repository

    async def _detect_primary_language(self, repo_path: str) -> str:
        """Detect the primary programming language in repository"""
        language_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp'
        }

        file_counts = {}
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in language_extensions:
                    lang = language_extensions[ext]
                    file_counts[lang] = file_counts.get(lang, 0) + 1

        if not file_counts:
            return 'unknown'

        return max(file_counts.items(), key=lambda x: x[1])[0]

    async def _analyze_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure and organization"""
        structure = {
            'directories': [],
            'files': [],
            'total_files': 0,
            'total_lines': 0,
            'build_files': [],
            'config_files': [],
            'documentation': []
        }

        for root, dirs, files in os.walk(repo_path):
            rel_root = os.path.relpath(root, repo_path)
            if rel_root != '.':
                structure['directories'].append(rel_root)

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                structure['files'].append(rel_path)
                structure['total_files'] += 1

                # Count lines
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        structure['total_lines'] += lines
                except:
                    pass

                # Categorize special files
                file_lower = file.lower()
                if file_lower in ['makefile', 'dockerfile', 'package.json', 'setup.py', 'requirements.txt']:
                    structure['build_files'].append(rel_path)
                elif file_lower.endswith(('.cfg', '.conf', '.ini', '.yaml', '.yml', '.json')):
                    structure['config_files'].append(rel_path)
                elif file_lower.endswith(('.md', '.rst', '.txt')) or 'readme' in file_lower:
                    structure['documentation'].append(rel_path)

        return structure

    async def _extract_code_elements(
        self,
        repo_path: str,
        language: str,
        depth: AnalysisDepth
    ) -> List[CodeElement]:
        """Extract and analyze code elements from repository"""
        parser = self.language_parsers.get(language)
        if not parser:
            return []

        elements = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if parser.can_parse(file):
                    file_path = os.path.join(root, file)
                    file_elements = await parser.parse_file(file_path, depth)
                    elements.extend(file_elements)

        return elements

    async def _detect_patterns(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect code patterns using multiple detectors"""
        all_patterns = []

        for detector in self.pattern_detectors:
            patterns = await detector.detect_patterns(elements)
            all_patterns.extend(patterns)

        return all_patterns

    async def _build_dependency_graph(self, elements: List[CodeElement]) -> Dict[str, List[str]]:
        """Build dependency graph from code elements"""
        dependency_graph = {}

        # Build element lookup
        element_lookup = {elem.id: elem for elem in elements}

        for element in elements:
            dependencies = []

            # Add explicit dependencies
            for dep_id in element.dependencies:
                if dep_id in element_lookup:
                    dependencies.append(dep_id)

            # Add implicit dependencies (same file, inheritance, etc.)
            for other_element in elements:
                if (other_element.id != element.id and
                    other_element.file_path == element.file_path and
                    self._has_implicit_dependency(element, other_element)):
                    dependencies.append(other_element.id)

            dependency_graph[element.id] = dependencies

        return dependency_graph

    def _has_implicit_dependency(self, element: CodeElement, other: CodeElement) -> bool:
        """Check if there's an implicit dependency between elements"""
        # Method depends on its class
        if (element.element_type == CodeElementType.METHOD and
            other.element_type == CodeElementType.CLASS and
            element.line_start > other.line_start and
            element.line_end < other.line_end):
            return True

        # Function uses variables/constants defined earlier
        if (element.element_type == CodeElementType.FUNCTION and
            other.element_type in [CodeElementType.VARIABLE, CodeElementType.CONSTANT] and
            other.line_start < element.line_start):
            return True

        return False

    async def _calculate_metrics(self, repository: Repository) -> Dict[str, Any]:
        """Calculate comprehensive repository metrics"""
        metrics = {
            'complexity': await self._calculate_complexity(repository.elements),
            'maintainability': await self._calculate_maintainability(repository),
            'test_coverage': await self._estimate_test_coverage(repository),
            'code_quality': await self._calculate_code_quality(repository),
            'architecture_score': await self._calculate_architecture_score(repository)
        }

        return metrics

    async def _calculate_complexity(self, elements: List[CodeElement]) -> Dict[str, Any]:
        """Calculate code complexity metrics"""
        total_complexity = sum(elem.complexity for elem in elements)
        function_elements = [e for e in elements if e.element_type == CodeElementType.FUNCTION]

        if not function_elements:
            return {'total': 0, 'average': 0, 'max': 0}

        avg_complexity = total_complexity / len(function_elements)
        max_complexity = max(elem.complexity for elem in function_elements)

        return {
            'total': total_complexity,
            'average': avg_complexity,
            'max': max_complexity,
            'distribution': await self._complexity_distribution(function_elements)
        }

    async def _complexity_distribution(self, elements: List[CodeElement]) -> Dict[str, int]:
        """Calculate complexity distribution"""
        distribution = {'low': 0, 'medium': 0, 'high': 0}

        for elem in elements:
            if elem.complexity <= 5:
                distribution['low'] += 1
            elif elem.complexity <= 10:
                distribution['medium'] += 1
            else:
                distribution['high'] += 1

        return distribution

    async def _calculate_maintainability(self, repository: Repository) -> Dict[str, Any]:
        """Calculate maintainability metrics for the repository"""
        elements = repository.elements
        if not elements:
            return {'score': 0.5, 'factors': {}}

        # Calculate various maintainability factors
        avg_complexity = sum(elem.complexity for elem in elements) / len(elements)

        # Count different element types
        functions = [e for e in elements if e.element_type == CodeElementType.FUNCTION]
        classes = [e for e in elements if e.element_type == CodeElementType.CLASS]

        # Calculate maintainability factors
        complexity_factor = max(0, 1 - (avg_complexity / 20))  # Normalize complexity
        structure_factor = min(len(classes) / max(len(functions), 1), 1.0)  # Class to function ratio

        # Documentation factor (based on docstrings)
        documented_elements = sum(1 for elem in elements if elem.docstring and len(elem.docstring.strip()) > 0)
        documentation_factor = documented_elements / max(len(functions), 1)

        # Overall maintainability score (weighted average)
        score = (
            complexity_factor * 0.4 +
            structure_factor * 0.3 +
            documentation_factor * 0.3
        )

        return {
            'score': min(max(score, 0), 1),  # Clamp between 0 and 1
            'factors': {
                'complexity': complexity_factor,
                'structure': structure_factor,
                'documentation': documentation_factor,
                'avg_complexity': avg_complexity,
                'function_count': len(functions),
                'class_count': len(classes)
            }
        }


class PythonParser:
    """Python-specific code parser with AST analysis"""

    def can_parse(self, filename: str) -> bool:
        """Check if file can be parsed by this parser"""
        return filename.endswith('.py')

    async def parse_file(self, file_path: str, depth: AnalysisDepth) -> List[CodeElement]:
        """Parse Python file and extract code elements"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            elements = []

            # Extract elements using AST visitor
            visitor = PythonASTVisitor(file_path, depth)
            visitor.visit(tree)
            elements.extend(visitor.elements)

            return elements

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []


class PythonASTVisitor(ast.NodeVisitor):
    """AST visitor for extracting Python code elements"""

    def __init__(self, file_path: str, depth: AnalysisDepth):
        self.file_path = file_path
        self.depth = depth
        self.elements = []
        self.current_class = None

    def visit_FunctionDef(self, node):
        """Visit function definition"""
        element_id = f"{self.file_path}:{node.name}:{node.lineno}"

        element = CodeElement(
            id=element_id,
            name=node.name,
            element_type=CodeElementType.METHOD if self.current_class else CodeElementType.FUNCTION,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=self._extract_signature(node),
            docstring=ast.get_docstring(node) or "",
            complexity=self._calculate_cyclomatic_complexity(node),
            dependencies=self._extract_function_dependencies(node)
        )

        self.elements.append(element)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit class definition"""
        element_id = f"{self.file_path}:{node.name}:{node.lineno}"
        old_class = self.current_class
        self.current_class = node.name

        element = CodeElement(
            id=element_id,
            name=node.name,
            element_type=CodeElementType.CLASS,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node) or "",
            dependencies=self._extract_class_dependencies(node)
        )

        self.elements.append(element)
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Import(self, node):
        """Visit import statement"""
        for alias in node.names:
            element_id = f"{self.file_path}:import:{alias.name}:{node.lineno}"
            element = CodeElement(
                id=element_id,
                name=alias.name,
                element_type=CodeElementType.IMPORT,
                file_path=self.file_path,
                line_start=node.lineno,
                line_end=node.lineno
            )
            self.elements.append(element)

    def visit_ImportFrom(self, node):
        """Visit from-import statement"""
        module_name = node.module or ""
        for alias in node.names:
            element_id = f"{self.file_path}:import:{module_name}.{alias.name}:{node.lineno}"
            element = CodeElement(
                id=element_id,
                name=f"{module_name}.{alias.name}",
                element_type=CodeElementType.IMPORT,
                file_path=self.file_path,
                line_start=node.lineno,
                line_end=node.lineno
            )
            self.elements.append(element)

    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        return f"{node.name}({', '.join(args)})"

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _extract_function_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract function dependencies"""
        dependencies = []

        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                dependencies.append(child.id)
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                dependencies.append(child.func.id)

        return list(set(dependencies))

    def _extract_class_dependencies(self, node: ast.ClassDef) -> List[str]:
        """Extract class dependencies including inheritance"""
        dependencies = []

        # Add base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                dependencies.append(base.id)

        return dependencies


class DesignPatternDetector:
    """Detects common design patterns in code"""

    async def detect_patterns(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect design patterns in code elements"""
        patterns = []

        patterns.extend(await self._detect_singleton_pattern(elements))
        patterns.extend(await self._detect_factory_pattern(elements))
        patterns.extend(await self._detect_observer_pattern(elements))
        patterns.extend(await self._detect_decorator_pattern(elements))

        return patterns

    async def _detect_singleton_pattern(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect Singleton pattern"""
        patterns = []
        classes = [e for e in elements if e.element_type == CodeElementType.CLASS]

        for class_elem in classes:
            # Look for singleton indicators
            if ('instance' in class_elem.name.lower() or
                '__new__' in str(class_elem.metadata) or
                'singleton' in class_elem.name.lower()):

                pattern = CodePattern(
                    id=f"singleton_{class_elem.id}",
                    name="Singleton Pattern",
                    pattern_type="creational",
                    description="Class implementing singleton pattern",
                    occurrences=[{
                        'element_id': class_elem.id,
                        'file': class_elem.file_path,
                        'line': class_elem.line_start
                    }],
                    confidence=0.7
                )
                patterns.append(pattern)

        return patterns

    async def _detect_factory_pattern(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect Factory pattern"""
        patterns = []
        functions = [e for e in elements if e.element_type == CodeElementType.FUNCTION]

        for func in functions:
            if ('create' in func.name.lower() or
                'factory' in func.name.lower() or
                'make' in func.name.lower()):

                pattern = CodePattern(
                    id=f"factory_{func.id}",
                    name="Factory Pattern",
                    pattern_type="creational",
                    description="Function implementing factory pattern",
                    occurrences=[{
                        'element_id': func.id,
                        'file': func.file_path,
                        'line': func.line_start
                    }],
                    confidence=0.6
                )
                patterns.append(pattern)

        return patterns

    async def _detect_observer_pattern(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect Observer pattern"""
        patterns = []
        classes = [e for e in elements if e.element_type == CodeElementType.CLASS]

        for class_elem in classes:
            observer_indicators = ['notify', 'subscribe', 'observer', 'listener']
            if any(indicator in class_elem.name.lower() for indicator in observer_indicators):
                pattern = CodePattern(
                    id=f"observer_{class_elem.id}",
                    name="Observer Pattern",
                    pattern_type="behavioral",
                    description="Class implementing observer pattern",
                    occurrences=[{
                        'element_id': class_elem.id,
                        'file': class_elem.file_path,
                        'line': class_elem.line_start
                    }],
                    confidence=0.5
                )
                patterns.append(pattern)

        return patterns

    async def _detect_decorator_pattern(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect Decorator pattern"""
        patterns = []
        functions = [e for e in elements if e.element_type == CodeElementType.FUNCTION]

        for func in functions:
            if ('decorator' in func.name.lower() or
                'wrap' in func.name.lower() or
                func.name.startswith('_')):

                pattern = CodePattern(
                    id=f"decorator_{func.id}",
                    name="Decorator Pattern",
                    pattern_type="structural",
                    description="Function implementing decorator pattern",
                    occurrences=[{
                        'element_id': func.id,
                        'file': func.file_path,
                        'line': func.line_start
                    }],
                    confidence=0.4
                )
                patterns.append(pattern)

        return patterns


class CodeSmellDetector:
    """Detects code smells and anti-patterns"""

    async def detect_patterns(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect code smells"""
        patterns = []

        patterns.extend(await self._detect_long_functions(elements))
        patterns.extend(await self._detect_large_classes(elements))
        patterns.extend(await self._detect_duplicate_code(elements))

        return patterns

    async def _detect_long_functions(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect functions that are too long"""
        patterns = []
        functions = [e for e in elements if e.element_type == CodeElementType.FUNCTION]

        for func in functions:
            lines = func.line_end - func.line_start + 1
            if lines > 50:  # Threshold for long function
                pattern = CodePattern(
                    id=f"long_function_{func.id}",
                    name="Long Function",
                    pattern_type="code_smell",
                    description=f"Function has {lines} lines (recommended: <50)",
                    occurrences=[{
                        'element_id': func.id,
                        'file': func.file_path,
                        'line': func.line_start,
                        'metrics': {'lines': lines}
                    }],
                    confidence=0.8,
                    suggestions=["Consider breaking this function into smaller functions"]
                )
                patterns.append(pattern)

        return patterns

    async def _detect_large_classes(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect classes that are too large"""
        patterns = []
        classes = [e for e in elements if e.element_type == CodeElementType.CLASS]

        for class_elem in classes:
            lines = class_elem.line_end - class_elem.line_start + 1
            if lines > 200:  # Threshold for large class
                pattern = CodePattern(
                    id=f"large_class_{class_elem.id}",
                    name="Large Class",
                    pattern_type="code_smell",
                    description=f"Class has {lines} lines (recommended: <200)",
                    occurrences=[{
                        'element_id': class_elem.id,
                        'file': class_elem.file_path,
                        'line': class_elem.line_start,
                        'metrics': {'lines': lines}
                    }],
                    confidence=0.8,
                    suggestions=["Consider splitting this class into multiple classes"]
                )
                patterns.append(pattern)

        return patterns

    async def _detect_duplicate_code(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect potential duplicate code"""
        patterns = []
        functions = [e for e in elements if e.element_type == CodeElementType.FUNCTION]

        # Group functions by similar signatures
        signature_groups = {}
        for func in functions:
            signature_key = self._normalize_signature(func.signature)
            if signature_key not in signature_groups:
                signature_groups[signature_key] = []
            signature_groups[signature_key].append(func)

        # Check for groups with multiple functions (potential duplicates)
        for signature, funcs in signature_groups.items():
            if len(funcs) > 1:
                pattern = CodePattern(
                    id=f"duplicate_code_{hash(signature)}",
                    name="Potential Duplicate Code",
                    pattern_type="code_smell",
                    description=f"Found {len(funcs)} functions with similar signatures",
                    occurrences=[{
                        'element_id': func.id,
                        'file': func.file_path,
                        'line': func.line_start
                    } for func in funcs],
                    confidence=0.6,
                    suggestions=["Review these functions for potential code duplication"]
                )
                patterns.append(pattern)

        return patterns

    def _normalize_signature(self, signature: str) -> str:
        """Normalize function signature for comparison"""
        # Remove function name and keep parameter structure
        if '(' in signature:
            params = signature.split('(')[1].split(')')[0]
            return f"({params})"
        return signature


class PerformancePatternDetector:
    """Detects performance-related patterns and anti-patterns"""

    async def detect_patterns(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect performance patterns"""
        patterns = []

        patterns.extend(await self._detect_nested_loops(elements))
        patterns.extend(await self._detect_inefficient_operations(elements))

        return patterns

    async def _detect_nested_loops(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect nested loops that might impact performance"""
        # This is a simplified implementation
        # In practice, would need actual AST analysis
        patterns = []
        return patterns

    async def _detect_inefficient_operations(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect potentially inefficient operations"""
        patterns = []
        return patterns


class SecurityPatternDetector:
    """Detects security-related patterns and vulnerabilities"""

    async def detect_patterns(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect security patterns"""
        patterns = []

        patterns.extend(await self._detect_sql_injection_risks(elements))
        patterns.extend(await self._detect_hardcoded_secrets(elements))

        return patterns

    async def _detect_sql_injection_risks(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect potential SQL injection vulnerabilities"""
        patterns = []
        return patterns

    async def _detect_hardcoded_secrets(self, elements: List[CodeElement]) -> List[CodePattern]:
        """Detect hardcoded secrets in code"""
        patterns = []
        return patterns


# Placeholder parsers for other languages
class JavaScriptParser:
    def can_parse(self, filename: str) -> bool:
        return filename.endswith('.js')

    async def parse_file(self, file_path: str, depth: AnalysisDepth) -> List[CodeElement]:
        return []


class TypeScriptParser:
    def can_parse(self, filename: str) -> bool:
        return filename.endswith('.ts')

    async def parse_file(self, file_path: str, depth: AnalysisDepth) -> List[CodeElement]:
        return []


class JavaParser:
    def can_parse(self, filename: str) -> bool:
        return filename.endswith('.java')

    async def parse_file(self, file_path: str, depth: AnalysisDepth) -> List[CodeElement]:
        return []


class CppParser:
    def can_parse(self, filename: str) -> bool:
        return filename.endswith(('.cpp', '.cxx', '.cc', '.c', '.h', '.hpp'))

    async def parse_file(self, file_path: str, depth: AnalysisDepth) -> List[CodeElement]:
        return []


class RepoMaster:
    """
    Main RepoMaster orchestrator for repository analysis and understanding
    Provides intelligent code search, analysis, and generation capabilities
    """

    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.analyzed_repositories: Dict[str, Repository] = {}
        self.code_index: Dict[str, List[str]] = {}  # Search index

    async def analyze_repository(
        self,
        repo_path: str,
        depth: AnalysisDepth = AnalysisDepth.SEMANTIC,
        force_reanalyze: bool = False
    ) -> str:
        """Analyze repository and return analysis ID"""
        repo_id = hashlib.md5(repo_path.encode()).hexdigest()

        if repo_id in self.analyzed_repositories and not force_reanalyze:
            return repo_id

        repository = await self.semantic_analyzer.analyze_repository(repo_path, depth)
        self.analyzed_repositories[repo_id] = repository

        # Build search index
        await self._build_search_index(repository)

        return repo_id

    async def _build_search_index(self, repository: Repository):
        """Build search index for repository"""
        index_key = repository.id

        # Index code elements
        searchable_items = []
        for element in repository.elements:
            searchable_items.extend([
                element.name,
                element.signature,
                element.docstring,
                element.file_path
            ])

        # Index patterns
        for pattern in repository.patterns:
            searchable_items.extend([
                pattern.name,
                pattern.description
            ])

        self.code_index[index_key] = searchable_items

    async def semantic_search(
        self,
        repo_id: str,
        query: str,
        search_type: str = "all"
    ) -> Dict[str, Any]:
        """Perform semantic search within repository"""
        repository = self.analyzed_repositories.get(repo_id)
        if not repository:
            return {"error": "Repository not found"}

        query_terms = query.lower().split()
        results = {
            "elements": [],
            "patterns": [],
            "files": [],
            "suggestions": []
        }

        # Search code elements
        if search_type in ["all", "elements"]:
            for element in repository.elements:
                score = self._calculate_search_score(element, query_terms)
                if score > 0.3:
                    results["elements"].append({
                        "element": element,
                        "score": score,
                        "context": self._get_element_context(element, repository)
                    })

        # Search patterns
        if search_type in ["all", "patterns"]:
            for pattern in repository.patterns:
                score = self._calculate_pattern_search_score(pattern, query_terms)
                if score > 0.3:
                    results["patterns"].append({
                        "pattern": pattern,
                        "score": score
                    })

        # Sort results by score
        results["elements"].sort(key=lambda x: x["score"], reverse=True)
        results["patterns"].sort(key=lambda x: x["score"], reverse=True)

        # Generate search suggestions
        results["suggestions"] = await self._generate_search_suggestions(query, repository)

        return results

    def _calculate_search_score(self, element: CodeElement, query_terms: List[str]) -> float:
        """Calculate search relevance score for code element"""
        score = 0.0
        searchable_text = (
            element.name + " " +
            element.signature + " " +
            element.docstring + " " +
            element.file_path
        ).lower()

        for term in query_terms:
            if term in searchable_text:
                score += 1.0

        # Boost score for exact name matches
        if any(term == element.name.lower() for term in query_terms):
            score += 2.0

        return min(score / len(query_terms), 1.0)

    def _calculate_pattern_search_score(self, pattern: CodePattern, query_terms: List[str]) -> float:
        """Calculate search relevance score for pattern"""
        score = 0.0
        searchable_text = (pattern.name + " " + pattern.description).lower()

        for term in query_terms:
            if term in searchable_text:
                score += 1.0

        return min(score / len(query_terms), 1.0)

    def _get_element_context(self, element: CodeElement, repository: Repository) -> Dict[str, Any]:
        """Get contextual information for code element"""
        context = {
            "file_structure": self._get_file_structure_context(element, repository),
            "dependencies": self._get_dependency_context(element, repository),
            "usage_patterns": self._get_usage_patterns(element, repository)
        }
        return context

    def _get_file_structure_context(self, element: CodeElement, repository: Repository) -> Dict[str, Any]:
        """Get file structure context for element"""
        same_file_elements = [
            e for e in repository.elements
            if e.file_path == element.file_path
        ]

        return {
            "file_path": element.file_path,
            "total_elements": len(same_file_elements),
            "element_types": list(set(e.element_type.value for e in same_file_elements))
        }

    def _get_dependency_context(self, element: CodeElement, repository: Repository) -> Dict[str, Any]:
        """Get dependency context for element"""
        dependencies = repository.dependencies.get(element.id, [])
        dependents = [
            elem_id for elem_id, deps in repository.dependencies.items()
            if element.id in deps
        ]

        return {
            "dependencies_count": len(dependencies),
            "dependents_count": len(dependents),
            "coupling_score": (len(dependencies) + len(dependents)) / len(repository.elements)
        }

    def _get_usage_patterns(self, element: CodeElement, repository: Repository) -> List[str]:
        """Get usage patterns for element"""
        patterns = []
        for pattern in repository.patterns:
            for occurrence in pattern.occurrences:
                if occurrence.get('element_id') == element.id:
                    patterns.append(pattern.name)
        return patterns

    async def _generate_search_suggestions(self, query: str, repository: Repository) -> List[str]:
        """Generate search suggestions based on repository content"""
        suggestions = []

        # Suggest similar element names
        all_names = [elem.name for elem in repository.elements]
        for name in all_names:
            if self._is_similar(query, name):
                suggestions.append(f"Did you mean: {name}?")

        # Suggest pattern searches
        pattern_names = [pattern.name for pattern in repository.patterns]
        suggestions.extend([f"Search patterns: {name}" for name in pattern_names[:3]])

        return suggestions[:5]

    def _is_similar(self, query: str, name: str) -> bool:
        """Check if query is similar to name"""
        query_lower = query.lower()
        name_lower = name.lower()

        # Simple similarity check
        return (
            query_lower in name_lower or
            name_lower in query_lower or
            len(set(query_lower) & set(name_lower)) > len(query_lower) * 0.6
        )

    async def get_repository_summary(self, repo_id: str) -> Dict[str, Any]:
        """Get comprehensive repository summary"""
        repository = self.analyzed_repositories.get(repo_id)
        if not repository:
            return {"error": "Repository not found"}

        return {
            "repository_info": {
                "name": repository.name,
                "language": repository.language,
                "analyzed_at": repository.analyzed_at.isoformat()
            },
            "structure_summary": repository.structure,
            "code_metrics": repository.metrics,
            "elements_summary": {
                "total_elements": len(repository.elements),
                "by_type": self._summarize_elements_by_type(repository.elements),
                "complexity_distribution": await self._get_complexity_distribution(repository.elements)
            },
            "patterns_summary": {
                "total_patterns": len(repository.patterns),
                "by_type": self._summarize_patterns_by_type(repository.patterns),
                "high_confidence": len([p for p in repository.patterns if p.confidence > 0.7])
            },
            "recommendations": await self._generate_repository_recommendations(repository)
        }

    def _summarize_elements_by_type(self, elements: List[CodeElement]) -> Dict[str, int]:
        """Summarize elements by type"""
        summary = {}
        for element in elements:
            element_type = element.element_type.value
            summary[element_type] = summary.get(element_type, 0) + 1
        return summary

    def _summarize_patterns_by_type(self, patterns: List[CodePattern]) -> Dict[str, int]:
        """Summarize patterns by type"""
        summary = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            summary[pattern_type] = summary.get(pattern_type, 0) + 1
        return summary

    async def _get_complexity_distribution(self, elements: List[CodeElement]) -> Dict[str, int]:
        """Get complexity distribution of elements"""
        distribution = {"low": 0, "medium": 0, "high": 0}

        for element in elements:
            if element.complexity <= 5:
                distribution["low"] += 1
            elif element.complexity <= 10:
                distribution["medium"] += 1
            else:
                distribution["high"] += 1

        return distribution

    async def _generate_repository_recommendations(self, repository: Repository) -> List[str]:
        """Generate recommendations for repository improvement"""
        recommendations = []

        # Check complexity
        high_complexity_elements = [
            e for e in repository.elements if e.complexity > 10
        ]
        if high_complexity_elements:
            recommendations.append(
                f"Consider refactoring {len(high_complexity_elements)} high-complexity functions"
            )

        # Check patterns
        code_smell_patterns = [
            p for p in repository.patterns if p.pattern_type == "code_smell"
        ]
        if code_smell_patterns:
            recommendations.append(
                f"Address {len(code_smell_patterns)} code smell issues identified"
            )

        # Check test coverage
        test_files = [
            f for f in repository.structure["files"]
            if "test" in f.lower()
        ]
        if len(test_files) < len(repository.structure["files"]) * 0.3:
            recommendations.append("Consider adding more test files to improve coverage")

        # Check documentation
        documented_elements = [
            e for e in repository.elements if e.docstring
        ]
        if len(documented_elements) < len(repository.elements) * 0.5:
            recommendations.append("Add documentation to improve code maintainability")

        return recommendations