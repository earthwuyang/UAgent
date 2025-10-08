"""
OCC Validator - Optimistic Concurrency Control for Multi-Agent Code Editing.

This module implements database-style optimistic concurrency control for multi-agent
parallel code editing. It tracks read-set and write-set during agent execution and
validates conflicts at commit time using AST region signatures.
"""

import ast
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Literal, Union

try:
    import tree_sitter
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_java
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from openhands.core.dependency_analyzer.core import DependencyAnalyzer
from openhands.core.dependency_analyzer.parsers.python_parser import PythonDependencyParser
from openhands.runtime.utils.git_handler import GitHandler


@dataclass
class Region:
    """Represents a code region for conflict detection."""
    file_path: str
    start_line: int
    end_line: int
    region_type: Literal['function', 'class', 'method', 'block', 'statement']
    name: str
    signature: str = ""
    
    def __hash__(self) -> int:
        return hash((self.file_path, self.start_line, self.end_line, self.name))
    
    def overlaps(self, other: 'Region') -> bool:
        """Check if this region overlaps with another region."""
        if self.file_path != other.file_path:
            return False
        return not (self.end_line < other.start_line or self.start_line > other.end_line)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'region_type': self.region_type,
            'name': self.name,
            'signature': self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Region':
        """Create from dictionary."""
        return cls(**data)


class ConflictType(Enum):
    """Types of conflicts that can occur during validation."""
    FORMATTING = "formatting"
    AST_REORDERING = "ast_reordering"
    SEMANTIC = "semantic"
    READ_WRITE = "read_write"
    WRITE_WRITE = "write_write"


@dataclass
class ConflictDetail:
    """Detailed information about a detected conflict."""
    file_path: str
    conflict_type: ConflictType
    base_region: Optional[Region] = None
    current_region: Optional[Region] = None
    agent_region: Optional[Region] = None
    description: str = ""
    suggested_resolution: Literal['rebase', 'merge', 'abort'] = 'merge'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'file_path': self.file_path,
            'conflict_type': self.conflict_type.value,
            'base_region': self.base_region.to_dict() if self.base_region else None,
            'current_region': self.current_region.to_dict() if self.current_region else None,
            'agent_region': self.agent_region.to_dict() if self.agent_region else None,
            'description': self.description,
            'suggested_resolution': self.suggested_resolution
        }


@dataclass
class ValidationResult:
    """Result of OCC validation."""
    success: bool
    conflicts: List[ConflictDetail] = field(default_factory=list)
    suggested_resolution: Literal['commit', 'rebase', 'merge', 'abort'] = 'commit'
    validation_duration_ms: float = 0.0
    read_conflicts_count: int = 0
    write_conflicts_count: int = 0
    
    @property
    def conflicts_count(self) -> int:
        """Total number of conflicts."""
        return len(self.conflicts)


class ASTRegionSignature:
    """Computes and compares AST region signatures for robust conflict detection."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize AST region signature computer.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._signature_cache: Dict[str, str] = {}
        
        # Initialize tree-sitter parsers if available
        self._tree_sitter_parsers = {}
        self._tree_sitter_warned = False
        if TREE_SITTER_AVAILABLE:
            try:
                # Use tree_sitter_languages for proper language setup
                try:
                    from tree_sitter_languages import get_language
                    self._tree_sitter_parsers['python'] = get_language('python')
                    self._tree_sitter_parsers['javascript'] = get_language('javascript')
                    self._tree_sitter_parsers['java'] = get_language('java')
                    self.logger.info("Tree-sitter parsers initialized successfully")
                except ImportError:
                    # Fallback to direct tree-sitter module
                    self._tree_sitter_parsers['python'] = tree_sitter.Language(tree_sitter_python.language(), 'python')
                    self._tree_sitter_parsers['javascript'] = tree_sitter.Language(tree_sitter_javascript.language(), 'javascript')
                    self._tree_sitter_parsers['java'] = tree_sitter.Language(tree_sitter_java.language(), 'java')
                    self.logger.info("Tree-sitter parsers initialized using direct module")
            except Exception as e:
                self.logger.warning(f"Failed to initialize tree-sitter parsers: {e}. AST-based signatures will use fallback methods.")
                self._tree_sitter_warned = True
        else:
            if not self._tree_sitter_warned:
                self.logger.warning("Tree-sitter not available. Install tree-sitter and tree-sitter-languages for enhanced AST parsing. Falling back to basic AST analysis.")
                self._tree_sitter_warned = True
    
    def compute_signature(self, file_path: str, content: str, region: Region) -> str:
        """Compute SHA256 hash signature of an AST region.
        
        Args:
            file_path: Path to the file
            content: File content
            region: Region to compute signature for
            
        Returns:
            SHA256 hash signature of the region
        """
        try:
            # Extract region content
            lines = content.splitlines()
            if region.start_line < 1 or region.end_line > len(lines):
                self.logger.warning(f"Region out of bounds: {region}")
                return ""
            
            region_content = '\n'.join(lines[region.start_line - 1:region.end_line])
            
            # Create cache key
            cache_key = f"{file_path}:{region.start_line}:{region.end_line}:{hash(region_content)}"
            if cache_key in self._signature_cache:
                return self._signature_cache[cache_key]
            
            # Compute signature based on language
            language = self._detect_language(file_path)
            signature = self._compute_ast_signature(region_content, language)
            
            # Cache result
            self._signature_cache[cache_key] = signature
            return signature
            
        except Exception as e:
            self.logger.warning(f"Failed to compute signature for region {region}: {e}")
            return ""
    
    def extract_regions(self, file_path: str, content: str) -> List[Region]:
        """Extract all regions from a file.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            List of extracted regions
        """
        language = self._detect_language(file_path)
        
        if language == 'python':
            return self._extract_python_regions(file_path, content)
        elif language in ('javascript', 'typescript'):
            return self._extract_javascript_regions(file_path, content)
        elif language == 'java':
            return self._extract_java_regions(file_path, content)
        else:
            return self._extract_generic_regions(file_path, content)
    
    def compare_signatures(self, base_signature: str, current_signature: str) -> bool:
        """Check if two signatures match.
        
        Args:
            base_signature: Base signature
            current_signature: Current signature
            
        Returns:
            True if signatures match
        """
        return base_signature == current_signature
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        if file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith(('.js', '.jsx')):
            return 'javascript'
        elif file_path.endswith(('.ts', '.tsx')):
            return 'typescript'
        elif file_path.endswith('.java'):
            return 'java'
        else:
            return 'unknown'
    
    def _compute_ast_signature(self, content: str, language: str) -> str:
        """Compute AST-based signature for code content."""
        try:
            if language == 'python':
                return self._compute_python_ast_signature(content)
            elif language in ('javascript', 'typescript'):
                return self._compute_javascript_ast_signature(content)
            elif language == 'java':
                return self._compute_java_ast_signature(content)
            else:
                # Fallback to normalized content hash
                return self._compute_normalized_signature(content)
        except Exception as e:
            self.logger.debug(f"AST parsing failed for {language}: {e}, using normalized signature")
            return self._compute_normalized_signature(content)
    
    def _compute_python_ast_signature(self, content: str) -> str:
        """Compute Python AST signature."""
        try:
            tree = ast.parse(content)
            # Extract meaningful AST components (ignore formatting)
            ast_components = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    ast_components.append(f"{type(node).__name__}:{node.name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        ast_components.append(f"import:{alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        ast_components.append(f"from:{node.module}:{alias.name}")
                elif isinstance(node, ast.Assign):
                    if hasattr(node.targets[0], 'id'):
                        ast_components.append(f"assign:{node.targets[0].id}")
            
            # Sort for consistent ordering
            ast_components.sort()
            signature_content = '|'.join(ast_components)
            return hashlib.sha256(signature_content.encode()).hexdigest()[:16]
            
        except SyntaxError:
            # If AST parsing fails, fall back to normalized content
            return self._compute_normalized_signature(content)
    
    def _compute_javascript_ast_signature(self, content: str) -> str:
        """Compute JavaScript/TypeScript AST signature."""
        if TREE_SITTER_AVAILABLE and 'javascript' in self._tree_sitter_parsers:
            return self._compute_tree_sitter_signature(content, 'javascript')
        else:
            return self._compute_javascript_regex_signature(content)
    
    def _compute_java_ast_signature(self, content: str) -> str:
        """Compute Java AST signature."""
        if TREE_SITTER_AVAILABLE and 'java' in self._tree_sitter_parsers:
            return self._compute_tree_sitter_signature(content, 'java')
        else:
            return self._compute_java_regex_signature(content)
    
    def _compute_tree_sitter_signature(self, content: str, language: str) -> str:
        """Compute signature using tree-sitter parser."""
        try:
            parser = tree_sitter.Parser()
            parser.set_language(self._tree_sitter_parsers[language])
            tree = parser.parse(bytes(content, 'utf8'))
            
            # Extract meaningful nodes
            ast_components = []
            
            def walk_tree(node):
                if node.type in ('function_declaration', 'class_declaration', 'method_definition'):
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        ast_components.append(f"{node.type}:{name_node.text.decode()}")
                
                for child in node.children:
                    walk_tree(child)
            
            walk_tree(tree.root_node)
            ast_components.sort()
            signature_content = '|'.join(ast_components)
            return hashlib.sha256(signature_content.encode()).hexdigest()[:16]
            
        except Exception as e:
            self.logger.debug(f"Tree-sitter parsing failed: {e}")
            return self._compute_normalized_signature(content)
    
    def _compute_javascript_regex_signature(self, content: str) -> str:
        """Compute JavaScript signature using regex patterns."""
        ast_components = []
        
        # Extract function declarations
        func_pattern = r'\bfunction\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content):
            ast_components.append(f"function:{match.group(1)}")
        
        # Extract class declarations
        class_pattern = r'\bclass\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            ast_components.append(f"class:{match.group(1)}")
        
        # Extract arrow functions with names
        arrow_pattern = r'\b(\w+)\s*=\s*\([^)]*\)\s*=>'
        for match in re.finditer(arrow_pattern, content):
            ast_components.append(f"arrow_function:{match.group(1)}")
        
        ast_components.sort()
        signature_content = '|'.join(ast_components)
        return hashlib.sha256(signature_content.encode()).hexdigest()[:16]
    
    def _compute_java_regex_signature(self, content: str) -> str:
        """Compute Java signature using regex patterns."""
        ast_components = []
        
        # Extract class declarations
        class_pattern = r'\bclass\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            ast_components.append(f"class:{match.group(1)}")
        
        # Extract method declarations
        method_pattern = r'\b(?:public|private|protected|\s)+(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{'
        for match in re.finditer(method_pattern, content):
            ast_components.append(f"method:{match.group(1)}")
        
        ast_components.sort()
        signature_content = '|'.join(ast_components)
        return hashlib.sha256(signature_content.encode()).hexdigest()[:16]
    
    def _compute_normalized_signature(self, content: str) -> str:
        """Compute signature based on normalized content (remove whitespace/comments)."""
        # Remove comments and normalize whitespace
        lines = content.splitlines()
        normalized_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith(('#', '//', '/*')):
                normalized_lines.append(line)
        
        normalized_content = '\n'.join(normalized_lines)
        return hashlib.sha256(normalized_content.encode()).hexdigest()[:16]
    
    def _extract_python_regions(self, file_path: str, content: str) -> List[Region]:
        """Extract regions from Python file using AST."""
        regions = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    regions.append(Region(
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno + 1),
                        region_type='function',
                        name=node.name
                    ))
                elif isinstance(node, ast.ClassDef):
                    regions.append(Region(
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno + 1),
                        region_type='class',
                        name=node.name
                    ))
                elif isinstance(node, ast.AsyncFunctionDef):
                    regions.append(Region(
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno + 1),
                        region_type='function',
                        name=node.name
                    ))
        
        except SyntaxError as e:
            self.logger.warning(f"Failed to parse Python AST for {file_path}: {e}")
            # Fallback to line-based regions
            return self._extract_generic_regions(file_path, content)
        
        return regions
    
    def _extract_javascript_regions(self, file_path: str, content: str) -> List[Region]:
        """Extract regions from JavaScript/TypeScript file."""
        if TREE_SITTER_AVAILABLE and 'javascript' in self._tree_sitter_parsers:
            return self._extract_tree_sitter_regions(file_path, content, 'javascript')
        else:
            return self._extract_javascript_regex_regions(file_path, content)
    
    def _extract_java_regions(self, file_path: str, content: str) -> List[Region]:
        """Extract regions from Java file."""
        if TREE_SITTER_AVAILABLE and 'java' in self._tree_sitter_parsers:
            return self._extract_tree_sitter_regions(file_path, content, 'java')
        else:
            return self._extract_java_regex_regions(file_path, content)
    
    def _extract_tree_sitter_regions(self, file_path: str, content: str, language: str) -> List[Region]:
        """Extract regions using tree-sitter parser."""
        regions = []
        
        try:
            parser = tree_sitter.Parser()
            parser.set_language(self._tree_sitter_parsers[language])
            tree = parser.parse(bytes(content, 'utf8'))
            
            def extract_regions(node):
                if node.type in ('function_declaration', 'method_definition'):
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        regions.append(Region(
                            file_path=file_path,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            region_type='function',
                            name=name_node.text.decode()
                        ))
                elif node.type == 'class_declaration':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        regions.append(Region(
                            file_path=file_path,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            region_type='class',
                            name=name_node.text.decode()
                        ))
                
                for child in node.children:
                    extract_regions(child)
            
            extract_regions(tree.root_node)
            
        except Exception as e:
            self.logger.debug(f"Tree-sitter region extraction failed: {e}")
            return self._extract_generic_regions(file_path, content)
        
        return regions
    
    def _extract_javascript_regex_regions(self, file_path: str, content: str) -> List[Region]:
        """Extract JavaScript regions using regex patterns."""
        regions = []
        lines = content.splitlines()
        
        # Function declarations
        func_pattern = r'\bfunction\s+(\w+)\s*\('
        for i, line in enumerate(lines, 1):
            match = re.search(func_pattern, line)
            if match:
                regions.append(Region(
                    file_path=file_path,
                    start_line=i,
                    end_line=self._find_block_end(lines, i - 1),
                    region_type='function',
                    name=match.group(1)
                ))
        
        # Class declarations
        class_pattern = r'\bclass\s+(\w+)'
        for i, line in enumerate(lines, 1):
            match = re.search(class_pattern, line)
            if match:
                regions.append(Region(
                    file_path=file_path,
                    start_line=i,
                    end_line=self._find_block_end(lines, i - 1),
                    region_type='class',
                    name=match.group(1)
                ))
        
        return regions
    
    def _extract_java_regex_regions(self, file_path: str, content: str) -> List[Region]:
        """Extract Java regions using regex patterns."""
        regions = []
        lines = content.splitlines()
        
        # Class declarations
        class_pattern = r'\bclass\s+(\w+)'
        for i, line in enumerate(lines, 1):
            match = re.search(class_pattern, line)
            if match:
                regions.append(Region(
                    file_path=file_path,
                    start_line=i,
                    end_line=self._find_block_end(lines, i - 1),
                    region_type='class',
                    name=match.group(1)
                ))
        
        # Method declarations
        method_pattern = r'\b(?:public|private|protected|\s)+(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{'
        for i, line in enumerate(lines, 1):
            match = re.search(method_pattern, line)
            if match:
                regions.append(Region(
                    file_path=file_path,
                    start_line=i,
                    end_line=self._find_block_end(lines, i - 1),
                    region_type='method',
                    name=match.group(1)
                ))
        
        return regions
    
    def _extract_generic_regions(self, file_path: str, content: str) -> List[Region]:
        """Extract generic regions (e.g., line-based for unsupported languages)."""
        regions = []
        lines = content.splitlines()
        
        # Create one region per non-empty line
        for i, line in enumerate(lines, 1):
            if line.strip():
                regions.append(Region(
                    file_path=file_path,
                    start_line=i,
                    end_line=i,
                    region_type='statement',
                    name=f"line_{i}"
                ))
        
        return regions
    
    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a code block starting from start_idx.
        
        Enhanced to ignore braces within strings and simple comments.
        """
        brace_count = 0
        in_block = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            # Strip simple line comments and process
            cleaned_line = self._strip_strings_and_comments(line)
            
            for char in cleaned_line:
                if char == '{':
                    brace_count += 1
                    in_block = True
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and in_block:
                        return i + 1
        
        # If no closing brace found, return last line
        return len(lines)
    
    def _strip_strings_and_comments(self, line: str) -> str:
        """Strip strings and comments from a line to avoid false brace matches.
        
        This is a simple heuristic that handles common cases.
        """
        result = []
        in_string = False
        string_char = None
        i = 0
        
        while i < len(line):
            char = line[i]
            
            # Handle escape sequences in strings
            if in_string and char == '\\' and i + 1 < len(line):
                i += 2  # Skip escape sequence
                continue
            
            # Toggle string state
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
                i += 1
                continue
            elif in_string and char == string_char:
                in_string = False
                string_char = None
                i += 1
                continue
            
            # Skip line comments (// or #)
            if not in_string:
                if i + 1 < len(line) and line[i:i+2] == '//':
                    break  # Rest of line is comment
                if char == '#':
                    break  # Rest of line is comment (Python)
            
            # Add non-string characters
            if not in_string:
                result.append(char)
            
            i += 1
        
        return ''.join(result)


class ReadWriteTracker:
    """Tracks file/region access during agent execution."""
    
    def __init__(self, base_commit: str = ""):
        """Initialize tracker.
        
        Args:
            base_commit: Base commit SHA when tracking started
        """
        self.read_set: Dict[str, Set[Region]] = {}
        self.write_set: Dict[str, Set[Region]] = {}
        self.base_commit = base_commit
        
    def track_read(self, file_path: str, regions: Optional[List[Region]] = None):
        """Record file/region read.
        
        Args:
            file_path: Path to file being read
            regions: Specific regions read (None = entire file)
        """
        if file_path not in self.read_set:
            self.read_set[file_path] = set()
        
        if regions:
            self.read_set[file_path].update(regions)
        else:
            # Add a placeholder region for entire file
            self.read_set[file_path].add(Region(
                file_path=file_path,
                start_line=1,
                end_line=999999,  # Placeholder for entire file
                region_type='block',
                name='entire_file'
            ))
    
    def track_write(self, file_path: str, regions: Optional[List[Region]] = None):
        """Record file/region write.
        
        Args:
            file_path: Path to file being written
            regions: Specific regions written (None = entire file)
        """
        if file_path not in self.write_set:
            self.write_set[file_path] = set()
        
        if regions:
            self.write_set[file_path].update(regions)
        else:
            # Add a placeholder region for entire file
            self.write_set[file_path].add(Region(
                file_path=file_path,
                start_line=1,
                end_line=999999,  # Placeholder for entire file
                region_type='block',
                name='entire_file'
            ))
    
    def get_read_set(self) -> Dict[str, Set[Region]]:
        """Get all tracked reads."""
        return self.read_set.copy()
    
    def get_write_set(self) -> Dict[str, Set[Region]]:
        """Get all tracked writes."""
        return self.write_set.copy()
    
    def clear(self):
        """Reset tracking."""
        self.read_set.clear()
        self.write_set.clear()


class OCCValidator:
    """Main validator class for OCC validation."""
    
    def __init__(
        self,
        workspace_base: str,
        workspace_mount_path_in_sandbox: str,
        dependency_analyzer: DependencyAnalyzer,
        git_handler: GitHandler,
        logger: Optional[logging.Logger] = None,
        event_stream: Optional[Any] = None
    ):
        """Initialize OCC validator.
        
        Args:
            workspace_base: Base workspace path
            workspace_mount_path_in_sandbox: Workspace path in sandbox
            dependency_analyzer: Dependency analyzer for transitive impact analysis (reserved for future use)
            git_handler: Git handler instance
            logger: Optional logger
            event_stream: Optional event stream for emitting OCC events
        """
        self.workspace_base = workspace_base
        self.workspace_mount_path_in_sandbox = workspace_mount_path_in_sandbox
        self.dependency_analyzer = dependency_analyzer
        self.git_handler = git_handler
        self.logger = logger or logging.getLogger(__name__)
        self.event_stream = event_stream
        
        # Initialize AST signature computer
        self.ast_signature = ASTRegionSignature(logger=self.logger)
        
        # Statistics
        self._validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'conflicts_detected': 0,
            'avg_validation_time_ms': 0.0
        }
    
    async def validate_commit(
        self,
        read_set: Dict[str, Set[Region]],
        write_set: Dict[str, Set[Region]],
        base_commit: str,
        current_head: str,
        agent_id: str = ""
    ) -> ValidationResult:
        """Main validation method.
        
        Args:
            read_set: Files/regions read by agent
            write_set: Files/regions modified by agent
            base_commit: Base commit when agent started
            current_head: Current HEAD commit
            
        Returns:
            ValidationResult with conflicts and suggested resolution
        """
        start_time = time.time()
        self._validation_stats['total_validations'] += 1
        
        try:
            self.logger.info(f"Starting OCC validation: {base_commit} -> {current_head}")
            
            # Emit validation started event
            if self.event_stream:
                from openhands.events.observation.occ import OCCValidationStartedObservation
                self.event_stream.add_event(OCCValidationStartedObservation(
                    agent_id=agent_id,
                    base_commit=base_commit,
                    current_head=current_head,
                    read_set_size=len(read_set),
                    write_set_size=len(write_set),
                    timestamp=time.time()
                ))
            
            conflicts = []
            
            # Check read conflicts
            read_conflicts = await self._check_read_conflicts(read_set, base_commit, current_head)
            conflicts.extend(read_conflicts)
            
            # Check write conflicts
            write_conflicts = await self._check_write_conflicts(write_set, base_commit, current_head)
            conflicts.extend(write_conflicts)
            
            # Determine success and resolution
            success = len(conflicts) == 0
            if success:
                suggested_resolution = 'commit'
            else:
                suggested_resolution = self._suggest_resolution(conflicts)
            
            # Create result
            validation_duration = (time.time() - start_time) * 1000
            result = ValidationResult(
                success=success,
                conflicts=conflicts,
                suggested_resolution=suggested_resolution,
                validation_duration_ms=validation_duration,
                read_conflicts_count=len(read_conflicts),
                write_conflicts_count=len(write_conflicts)
            )
            
            # Update statistics
            if success:
                self._validation_stats['successful_validations'] += 1
            else:
                self._validation_stats['conflicts_detected'] += len(conflicts)
            
            # Update average validation time
            prev_avg = self._validation_stats['avg_validation_time_ms']
            total_validations = self._validation_stats['total_validations']
            self._validation_stats['avg_validation_time_ms'] = (
                (prev_avg * (total_validations - 1) + validation_duration) / total_validations
            )
            
            self.logger.info(
                f"OCC validation completed: {result.success}, "
                f"{result.conflicts_count} conflicts, "
                f"{validation_duration:.1f}ms"
            )
            
            # Emit validation completed event
            if self.event_stream:
                from openhands.events.observation.occ import OCCValidationCompletedObservation
                self.event_stream.add_event(OCCValidationCompletedObservation(
                    agent_id=agent_id,
                    success=result.success,
                    conflicts_count=result.conflicts_count,
                    read_conflicts=result.read_conflicts_count,
                    write_conflicts=result.write_conflicts_count,
                    suggested_resolution=result.suggested_resolution,
                    validation_duration_ms=validation_duration,
                    timestamp=time.time()
                ))
            
            return result
            
        except Exception as e:
            self.logger.error(f"OCC validation failed: {e}")
            return ValidationResult(
                success=False,
                conflicts=[],
                suggested_resolution='abort',
                validation_duration_ms=(time.time() - start_time) * 1000
            )
    
    async def _check_read_conflicts(
        self,
        read_set: Dict[str, Set[Region]],
        base_commit: str,
        current_head: str
    ) -> List[ConflictDetail]:
        """Check if read files/regions have been modified.
        
        Args:
            read_set: Files/regions read by agent
            base_commit: Base commit
            current_head: Current HEAD commit
            
        Returns:
            List of read conflicts detected
        """
        conflicts = []
        
        try:
            # Get list of changed files between commits
            changed_files = await self._get_changed_files(base_commit, current_head)
            
            for file_path in read_set:
                if file_path in changed_files:
                    # File was read by agent and modified by others
                    try:
                        base_content = await self._get_file_content_at_commit(file_path, base_commit)
                        current_content = await self._get_file_content_at_commit(file_path, current_head)
                        
                        # Extract regions from both versions
                        base_regions = self.ast_signature.extract_regions(file_path, base_content)
                        current_regions = self.ast_signature.extract_regions(file_path, current_content)
                        
                        # Check if any read regions were modified
                        read_regions = read_set[file_path]
                        for read_region in read_regions:
                            if read_region.name == 'entire_file':
                                # Entire file was read, any change is a conflict
                                conflict_type = self._classify_conflict(base_content, current_content, "", None)
                                conflicts.append(ConflictDetail(
                                    file_path=file_path,
                                    conflict_type=ConflictType.READ_WRITE,
                                    base_region=None,
                                    current_region=None,
                                    agent_region=read_region,
                                    description=f"Agent read {file_path} which was later modified",
                                    suggested_resolution='rebase'
                                ))
                                break
                            else:
                                # Check specific regions
                                for current_region in current_regions:
                                    if (read_region.overlaps(current_region) and 
                                        self._region_was_modified(read_region, base_regions, current_regions, base_content, current_content)):
                                        
                                        conflict_type = self._classify_region_conflict(
                                            read_region, base_regions, current_regions, base_content, current_content
                                        )
                                        
                                        conflicts.append(ConflictDetail(
                                            file_path=file_path,
                                            conflict_type=ConflictType.READ_WRITE,
                                            base_region=read_region,
                                            current_region=current_region,
                                            agent_region=read_region,
                                            description=f"Agent read region '{read_region.name}' which was later modified",
                                            suggested_resolution='rebase' if conflict_type == ConflictType.FORMATTING else 'merge'
                                        ))
                                        break
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to check read conflict for {file_path}: {e}")
                        # Add generic conflict
                        conflicts.append(ConflictDetail(
                            file_path=file_path,
                            conflict_type=ConflictType.READ_WRITE,
                            description=f"Agent read {file_path} which was later modified (details unavailable)",
                            suggested_resolution='merge'
                        ))
        
        except Exception as e:
            self.logger.error(f"Failed to check read conflicts: {e}")
        
        return conflicts
    
    async def _check_write_conflicts(
        self,
        write_set: Dict[str, Set[Region]],
        base_commit: str,
        current_head: str
    ) -> List[ConflictDetail]:
        """Check if write regions overlap with changes in current_head.
        
        Args:
            write_set: Files/regions modified by agent
            base_commit: Base commit
            current_head: Current HEAD commit
            
        Returns:
            List of write conflicts detected
        """
        conflicts = []
        
        try:
            # Get list of changed files between commits
            changed_files = await self._get_changed_files(base_commit, current_head)
            
            for file_path in write_set:
                if file_path in changed_files:
                    # File was modified by agent and also modified by others
                    try:
                        base_content = await self._get_file_content_at_commit(file_path, base_commit)
                        current_content = await self._get_file_content_at_commit(file_path, current_head)
                        
                        # Extract regions from both versions
                        base_regions = self.ast_signature.extract_regions(file_path, base_content)
                        current_regions = self.ast_signature.extract_regions(file_path, current_content)
                        
                        # Check if any write regions overlap with modified regions
                        write_regions = write_set[file_path]
                        for write_region in write_regions:
                            if write_region.name == 'entire_file':
                                # Entire file was modified, any change is a conflict
                                conflict_type = self._classify_conflict(base_content, current_content, "", None)
                                conflicts.append(ConflictDetail(
                                    file_path=file_path,
                                    conflict_type=ConflictType.WRITE_WRITE,
                                    base_region=None,
                                    current_region=None,
                                    agent_region=write_region,
                                    description=f"Agent modified {file_path} which was also modified by others",
                                    suggested_resolution='abort' if conflict_type == ConflictType.SEMANTIC else 'merge'
                                ))
                                break
                            else:
                                # Check specific regions
                                for current_region in current_regions:
                                    if (write_region.overlaps(current_region) and
                                        self._region_was_modified(write_region, base_regions, current_regions, base_content, current_content)):
                                        
                                        conflict_type = self._classify_region_conflict(
                                            write_region, base_regions, current_regions, base_content, current_content
                                        )
                                        
                                        resolution = 'rebase' if conflict_type in (ConflictType.FORMATTING, ConflictType.AST_REORDERING) else 'abort'
                                        
                                        conflicts.append(ConflictDetail(
                                            file_path=file_path,
                                            conflict_type=ConflictType.WRITE_WRITE,
                                            base_region=write_region,
                                            current_region=current_region,
                                            agent_region=write_region,
                                            description=f"Agent modified region '{write_region.name}' which was also modified by others",
                                            suggested_resolution=resolution
                                        ))
                                        break
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to check write conflict for {file_path}: {e}")
                        # Add generic conflict
                        conflicts.append(ConflictDetail(
                            file_path=file_path,
                            conflict_type=ConflictType.WRITE_WRITE,
                            description=f"Agent modified {file_path} which was also modified by others (details unavailable)",
                            suggested_resolution='abort'
                        ))
        
        except Exception as e:
            self.logger.error(f"Failed to check write conflicts: {e}")
        
        return conflicts
    
    def _classify_conflict(
        self,
        base_content: str,
        current_content: str,
        agent_content: str,
        region: Optional[Region]
    ) -> ConflictType:
        """Classify conflict type based on content changes.
        
        Args:
            base_content: Base version content
            current_content: Current version content
            agent_content: Agent's version content (may be empty for read conflicts)
            region: Specific region (None for entire file)
            
        Returns:
            ConflictType classification
        """
        # Check if only whitespace/formatting changed
        base_normalized = re.sub(r'\s+', ' ', base_content.strip())
        current_normalized = re.sub(r'\s+', ' ', current_content.strip())
        
        if base_normalized == current_normalized:
            return ConflictType.FORMATTING
        
        # Check if only imports/declarations were reordered
        if self._is_ast_reordering(base_content, current_content):
            return ConflictType.AST_REORDERING
        
        # Default to semantic conflict
        return ConflictType.SEMANTIC
    
    def _classify_region_conflict(
        self,
        region: Region,
        base_regions: List[Region],
        current_regions: List[Region],
        base_content: str,
        current_content: str
    ) -> ConflictType:
        """Classify conflict type for a specific region."""
        try:
            # Extract region content from both versions
            base_lines = base_content.splitlines()
            current_lines = current_content.splitlines()
            
            # Find matching base region
            base_region_content = ""
            for base_region in base_regions:
                if base_region.name == region.name:
                    if base_region.start_line <= len(base_lines) and base_region.end_line <= len(base_lines):
                        base_region_content = '\n'.join(base_lines[base_region.start_line-1:base_region.end_line])
                    break
            
            # Find matching current region
            current_region_content = ""
            for current_region in current_regions:
                if current_region.name == region.name:
                    if current_region.start_line <= len(current_lines) and current_region.end_line <= len(current_lines):
                        current_region_content = '\n'.join(current_lines[current_region.start_line-1:current_region.end_line])
                    break
            
            return self._classify_conflict(base_region_content, current_region_content, "", region)
            
        except Exception as e:
            self.logger.debug(f"Failed to classify region conflict: {e}")
            return ConflictType.SEMANTIC
    
    def _is_ast_reordering(self, base_content: str, current_content: str) -> bool:
        """Check if changes are just AST reordering (imports, function order, etc.)."""
        try:
            # Simple heuristic: check if set of lines is the same (ignoring order)
            base_lines = set(line.strip() for line in base_content.splitlines() if line.strip())
            current_lines = set(line.strip() for line in current_content.splitlines() if line.strip())
            
            # If sets are identical, it's likely just reordering
            return base_lines == current_lines
            
        except Exception:
            return False
    
    def _region_was_modified(
        self,
        region: Region,
        base_regions: List[Region],
        current_regions: List[Region],
        base_content: str,
        current_content: str
    ) -> bool:
        """Check if a region was modified between base and current."""
        try:
            # Find corresponding regions in base and current
            base_region = None
            current_region = None
            
            for r in base_regions:
                # Match by name, type, and precise line-range overlap
                if (r.name == region.name and 
                    r.region_type == region.region_type and 
                    r.overlaps(region)):
                    base_region = r
                    break
            
            for r in current_regions:
                # Match by name, type, and precise line-range overlap
                if (r.name == region.name and 
                    r.region_type == region.region_type and 
                    r.overlaps(region)):
                    current_region = r
                    break
            
            if not base_region or not current_region:
                # Region missing in one version = modified
                return True
            
            # Compare signatures
            base_signature = self.ast_signature.compute_signature(region.file_path, base_content, base_region)
            current_signature = self.ast_signature.compute_signature(region.file_path, current_content, current_region)
            
            return not self.ast_signature.compare_signatures(base_signature, current_signature)
            
        except Exception as e:
            self.logger.debug(f"Failed to check if region was modified: {e}")
            return True  # Assume modified on error
    
    def _suggest_resolution(self, conflicts: List[ConflictDetail]) -> Literal['commit', 'rebase', 'merge', 'abort']:
        """Suggest resolution strategy based on conflicts.
        
        Args:
            conflicts: List of detected conflicts
            
        Returns:
            Suggested resolution strategy
        """
        if not conflicts:
            return 'commit'
        
        # Check conflict types
        has_semantic = any(c.conflict_type == ConflictType.SEMANTIC for c in conflicts)
        has_write_write = any(c.conflict_type == ConflictType.WRITE_WRITE for c in conflicts)
        
        if has_semantic or has_write_write:
            return 'abort'  # Manual intervention required
        
        # Only formatting/reordering conflicts
        has_formatting_only = all(c.conflict_type in (ConflictType.FORMATTING, ConflictType.AST_REORDERING) for c in conflicts)
        
        if has_formatting_only:
            return 'rebase'  # Safe to auto-rebase
        
        # Mixed conflict types
        return 'merge'
    
    async def _get_changed_files(self, base_commit: str, current_head: str) -> Set[str]:
        """Get list of files changed between commits.
        
        Args:
            base_commit: Base commit
            current_head: Current HEAD commit
            
        Returns:
            Set of changed file paths
        """
        try:
            # Use git diff to get changed files
            result = self.git_handler.run_git_command(['diff', '--name-only', base_commit, current_head])
            if result.returncode == 0:
                changed_files = set(line.strip() for line in result.stdout.strip().split('\n') if line.strip())
                return changed_files
            else:
                self.logger.warning(f"Git diff failed: {result.stderr}")
                return set()
                
        except Exception as e:
            self.logger.error(f"Failed to get changed files: {e}")
            return set()
    
    async def _get_file_content_at_commit(self, file_path: str, commit: str) -> str:
        """Get file content at specific commit.
        
        Args:
            file_path: Path to file
            commit: Git commit SHA
            
        Returns:
            File content at commit
        """
        return _get_file_content_at_commit(file_path, commit, self.git_handler)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        return self._validation_stats.copy()


# Helper functions

def _get_file_content_at_commit(file_path: str, commit: str, git_handler: GitHandler) -> str:
    """Get file content at specific commit.
    
    Args:
        file_path: Path to file
        commit: Git commit SHA
        git_handler: Git handler instance
        
    Returns:
        File content at commit
        
    Raises:
        Exception: If git command fails
    """
    try:
        result = git_handler.run_git_command(['show', f'{commit}:{file_path}'])
        if result.returncode == 0:
            return result.stdout
        else:
            raise Exception(f"Git show failed: {result.stderr}")
            
    except Exception as e:
        raise Exception(f"Failed to get file content at commit {commit}: {e}")


def _compute_region_hash(content: str, start_line: int, end_line: int) -> str:
    """Compute hash of code region.
    
    Args:
        content: File content
        start_line: Start line number (1-based)
        end_line: End line number (1-based)
        
    Returns:
        SHA256 hash of region content
    """
    lines = content.splitlines()
    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        return ""
    
    region_content = '\n'.join(lines[start_line - 1:end_line])
    return hashlib.sha256(region_content.encode()).hexdigest()[:16]


def _regions_overlap(region1: Region, region2: Region) -> bool:
    """Check if two regions overlap.
    
    Args:
        region1: First region
        region2: Second region
        
    Returns:
        True if regions overlap
    """
    return region1.overlaps(region2)
