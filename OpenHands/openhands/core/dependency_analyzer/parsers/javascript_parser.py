"""JavaScript/TypeScript dependency parser using tree-sitter with regex fallback."""

import logging
import re
from typing import List, Optional

try:
    from tree_sitter_language_pack import get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from .base import BaseDependencyParser
from openhands.core.dependency_analyzer.models import ImportStatement


class JavaScriptDependencyParser(BaseDependencyParser):
    """JavaScript/TypeScript dependency parser using tree-sitter with regex fallback."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize JavaScript parser."""
        super().__init__(logger)
        self._js_parser = None
        self._ts_parser = None

    def supports_language(self, language: str) -> bool:
        """Check if parser supports given language."""
        return language in ('javascript', 'typescript')

    async def parse(self, file_path: str, content: str) -> List[ImportStatement]:
        """Parse JavaScript/TypeScript file content and extract imports.
        
        Args:
            file_path: Path to the file being parsed
            content: Content of the file
            
        Returns:
            List of ImportStatement objects found in the file
        """
        language = 'typescript' if file_path.endswith(('.ts', '.tsx')) else 'javascript'
        
        # Try tree-sitter first
        if TREE_SITTER_AVAILABLE:
            try:
                imports = await self._parse_with_tree_sitter(file_path, content, language)
                if imports:  # If tree-sitter found imports, use them
                    return self._filter_imports(imports, language)
            except Exception as e:
                self.logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
        
        # Fallback to regex parsing
        self.logger.debug(f"Using regex fallback for {file_path}")
        imports = await self._parse_with_regex(file_path, content, language)
        return self._filter_imports(imports, language)

    async def _parse_with_tree_sitter(self, file_path: str, content: str, language: str) -> List[ImportStatement]:
        """Parse using tree-sitter."""
        parser = self._get_parser(language)
        if not parser:
            return []
        
        # Parse the code
        tree = parser.parse(bytes(content, 'utf-8'))
        root_node = tree.root_node
        
        imports = []
        
        # Walk through the AST to find import statements
        self._extract_imports_from_node(root_node, content, imports, language)
        
        return imports

    def _get_parser(self, language: str):
        """Get tree-sitter parser for language."""
        try:
            if language == 'javascript':
                if self._js_parser is None:
                    self._js_parser = get_parser('javascript')
                return self._js_parser
            elif language == 'typescript':
                if self._ts_parser is None:
                    self._ts_parser = get_parser('typescript')
                return self._ts_parser
        except Exception as e:
            self.logger.warning(f"Failed to get {language} parser: {e}")
            return None

    def _extract_imports_from_node(self, node, content: str, imports: List[ImportStatement], language: str):
        """Recursively extract imports from tree-sitter nodes."""
        # Check for import statements
        if node.type == 'import_statement':
            import_stmt = self._handle_import_statement(node, content, language)
            if import_stmt:
                imports.append(import_stmt)
        
        # Check for require calls
        elif node.type == 'call_expression':
            require_stmt = self._handle_require_call(node, content, language)
            if require_stmt:
                imports.append(require_stmt)
        
        # Check for dynamic imports
        elif node.type == 'call_expression' and self._is_dynamic_import(node, content):
            dynamic_stmt = self._handle_dynamic_import(node, content, language)
            if dynamic_stmt:
                imports.append(dynamic_stmt)
        
        # Recursively process child nodes
        for child in node.children:
            self._extract_imports_from_node(child, content, imports, language)

    def _handle_import_statement(self, node, content: str, language: str) -> Optional[ImportStatement]:
        """Handle ES6 import statements."""
        # Find the source node (string literal with module path)
        source_node = None
        for child in node.children:
            if child.type == 'string':
                source_node = child
                break
        
        if not source_node:
            return None
        
        # Extract module path
        module_path = content[source_node.start_byte:source_node.end_byte]
        module_path = self.normalize_import_path(module_path)
        
        line_number = node.start_point[0] + 1
        
        return self._create_import_statement(
            module=module_path,
            import_type='static',
            line_number=line_number,
            confidence=1.0,
            source='tree_sitter'
        )

    def _handle_require_call(self, node, content: str, language: str) -> Optional[ImportStatement]:
        """Handle CommonJS require() calls."""
        # Check if this is a require call
        function_node = None
        for child in node.children:
            if child.type == 'identifier':
                func_name = content[child.start_byte:child.end_byte]
                if func_name == 'require':
                    function_node = child
                    break
        
        if not function_node:
            return None
        
        # Find the argument (module path)
        arguments_node = None
        for child in node.children:
            if child.type == 'arguments':
                arguments_node = child
                break
        
        if not arguments_node:
            return None
        
        # Get the first argument (module path)
        module_node = None
        for child in arguments_node.children:
            if child.type == 'string':
                module_node = child
                break
        
        if not module_node:
            # Dynamic require with variable
            line_number = node.start_point[0] + 1
            return self._create_import_statement(
                module='<dynamic:variable>',
                import_type='dynamic',
                line_number=line_number,
                confidence=0.5,
                source='tree_sitter'
            )
        
        # Extract module path
        module_path = content[module_node.start_byte:module_node.end_byte]
        module_path = self.normalize_import_path(module_path)
        
        line_number = node.start_point[0] + 1
        
        return self._create_import_statement(
            module=module_path,
            import_type='static',
            line_number=line_number,
            confidence=1.0,
            source='tree_sitter'
        )

    def _is_dynamic_import(self, node, content: str) -> bool:
        """Check if this is a dynamic import() call."""
        # Check if function is 'import'
        for child in node.children:
            if child.type == 'import':
                return True
        return False

    def _handle_dynamic_import(self, node, content: str, language: str) -> Optional[ImportStatement]:
        """Handle dynamic import() calls."""
        # Find the argument (module path)
        arguments_node = None
        for child in node.children:
            if child.type == 'arguments':
                arguments_node = child
                break
        
        if not arguments_node:
            return None
        
        # Get the first argument (module path)
        module_node = None
        for child in arguments_node.children:
            if child.type == 'string':
                module_node = child
                break
        
        line_number = node.start_point[0] + 1
        
        if not module_node:
            # Dynamic import with variable
            return self._create_import_statement(
                module='<dynamic:variable>',
                import_type='dynamic',
                line_number=line_number,
                confidence=0.3,
                source='tree_sitter'
            )
        
        # Extract module path
        module_path = content[module_node.start_byte:module_node.end_byte]
        module_path = self.normalize_import_path(module_path)
        
        return self._create_import_statement(
            module=module_path,
            import_type='dynamic',
            line_number=line_number,
            confidence=0.9,
            source='tree_sitter'
        )

    async def _parse_with_regex(self, file_path: str, content: str, language: str) -> List[ImportStatement]:
        """Fallback regex-based parsing for JavaScript/TypeScript."""
        imports = []
        
        # Regex patterns for different import types
        patterns = [
            # ES6 imports: import ... from 'module'
            (r"import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)(?:\s*,\s*(?:\{[^}]*\}|\*\s+as\s+\w+|\w+))*\s+)?from\s+['\"]([^'\"]+)['\"]", 'static', 1.0),
            # CommonJS require: require('module')
            (r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", 'static', 0.8),
            # Dynamic import: import('module')
            (r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", 'dynamic', 0.9),
            # Dynamic require with variable: require(variable)
            (r"require\s*\(\s*([^'\"][^)]*)\s*\)", 'dynamic', 0.4),
        ]
        
        for pattern, import_type, base_confidence in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                module_path = match.group(1)
                line_number = self.extract_line_number(content, match.start())
                
                # Adjust confidence for regex-based extraction
                confidence = base_confidence * 0.7 if import_type == 'static' else base_confidence * 0.5
                
                import_stmt = self._create_import_statement(
                    module=module_path,
                    import_type=import_type,
                    line_number=line_number,
                    confidence=confidence,
                    source='tree_sitter'  # Still mark as tree_sitter since it's the intended source
                )
                
                imports.append(import_stmt)
        
        return imports

    def _filter_imports(self, imports: List[ImportStatement], language: str) -> List[ImportStatement]:
        """Filter out standard library and third-party imports."""
        filtered_imports = []
        
        for import_stmt in imports:
            module = import_stmt.module
            
            # Skip dynamic imports with variables (can't resolve)
            if module.startswith('<dynamic:'):
                continue
            
            # Skip standard library and third-party packages
            if (self.is_standard_library(module, language) or 
                self.is_third_party(module, language)):
                self.logger.debug(f"Skipping standard/third-party import: {module}")
                continue
            
            filtered_imports.append(import_stmt)
        
        return filtered_imports


class TypeScriptDependencyParser(JavaScriptDependencyParser):
    """TypeScript dependency parser (extends JavaScript parser)."""

    def supports_language(self, language: str) -> bool:
        """Check if parser supports given language."""
        return language == 'typescript'
