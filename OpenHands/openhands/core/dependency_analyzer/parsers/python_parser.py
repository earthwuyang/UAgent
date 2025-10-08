"""Python dependency parser using AST."""

import ast
import logging
from typing import List, Optional

from .base import BaseDependencyParser
from openhands.core.dependency_analyzer.models import ImportStatement


class PythonDependencyParser(BaseDependencyParser):
    """Python dependency parser using built-in AST module."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize Python parser."""
        super().__init__(logger)

    def supports_language(self, language: str) -> bool:
        """Check if parser supports given language."""
        return language == 'python'

    async def parse(self, file_path: str, content: str) -> List[ImportStatement]:
        """Parse Python file content and extract imports using AST.
        
        Args:
            file_path: Path to the file being parsed
            content: Content of the Python file
            
        Returns:
            List of ImportStatement objects found in the file
        """
        imports = []
        
        try:
            # Parse the Python code into AST
            tree = ast.parse(content, filename=file_path)
            
            # Walk through all nodes in the AST
            for node in ast.walk(tree):
                # Check if we're inside a conditional block
                is_conditional = self._is_conditional_context(node, tree)
                
                if isinstance(node, ast.Import):
                    # Handle 'import module' statements
                    imports.extend(self._handle_import_node(node, is_conditional))
                
                elif isinstance(node, ast.ImportFrom):
                    # Handle 'from module import name' statements
                    imports.extend(self._handle_import_from_node(node, is_conditional))
                
                elif isinstance(node, ast.Call):
                    # Handle dynamic imports like importlib.import_module()
                    dynamic_import = self._handle_dynamic_import(node)
                    if dynamic_import:
                        imports.append(dynamic_import)
            
        except SyntaxError as e:
            # If AST parsing fails due to syntax error, log and return empty list
            self.logger.warning(f"Syntax error parsing {file_path}: {e}")
            return []
        except Exception as e:
            # Log any other parsing errors but don't crash
            self.logger.error(f"Error parsing {file_path}: {e}")
            return []
        
        # Filter out standard library and third-party imports if configured
        filtered_imports = []
        for import_stmt in imports:
            if (not self.is_standard_library(import_stmt.module, 'python') and 
                not self.is_third_party(import_stmt.module, 'python')):
                filtered_imports.append(import_stmt)
            else:
                self.logger.debug(f"Skipping standard/third-party import: {import_stmt.module}")
        
        return filtered_imports

    def _handle_import_node(self, node: ast.Import, is_conditional: bool) -> List[ImportStatement]:
        """Handle 'import module' statements.
        
        Args:
            node: AST Import node
            is_conditional: Whether import is in conditional context
            
        Returns:
            List of ImportStatement objects
        """
        imports = []
        import_type = 'conditional' if is_conditional else 'static'
        confidence = 0.9 if is_conditional else 1.0
        
        for alias in node.names:
            module_name = alias.name
            
            import_stmt = self._create_import_statement(
                module=module_name,
                import_type=import_type,
                line_number=node.lineno,
                confidence=confidence,
                source='ast'
            )
            
            # Add metadata
            if alias.asname:
                import_stmt.metadata['alias'] = alias.asname
            
            imports.append(import_stmt)
        
        return imports

    def _handle_import_from_node(self, node: ast.ImportFrom, is_conditional: bool) -> List[ImportStatement]:
        """Handle 'from module import name' statements.
        
        Args:
            node: AST ImportFrom node
            is_conditional: Whether import is in conditional context
            
        Returns:
            List of ImportStatement objects
        """
        imports = []
        
        if node.module is None:
            # Handle relative imports with no module (from . import name)
            if node.level > 0:
                module_name = '.' * node.level
            else:
                # Skip malformed imports
                return imports
        else:
            # Handle absolute and relative imports
            if node.level > 0:
                # Relative import: from ..module import name
                module_name = '.' * node.level + node.module
            else:
                # Absolute import: from module import name
                module_name = node.module
        
        import_type = 'conditional' if is_conditional else 'static'
        confidence = 0.9 if is_conditional else 1.0
        
        import_stmt = self._create_import_statement(
            module=module_name,
            import_type=import_type,
            line_number=node.lineno,
            confidence=confidence,
            source='ast'
        )
        
        # Add metadata about what's being imported
        if node.names:
            imported_names = []
            for alias in node.names:
                name = alias.name
                if alias.asname:
                    name += f" as {alias.asname}"
                imported_names.append(name)
            import_stmt.metadata['imported_names'] = imported_names
            import_stmt.metadata['relative_level'] = node.level
        
        imports.append(import_stmt)
        return imports

    def _handle_dynamic_import(self, node: ast.Call) -> Optional[ImportStatement]:
        """Handle dynamic imports like importlib.import_module().
        
        Args:
            node: AST Call node
            
        Returns:
            ImportStatement if dynamic import found, None otherwise
        """
        # Check for importlib.import_module()
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'importlib' and
            node.func.attr == 'import_module'):
            
            return self._extract_import_module_call(node)
        
        # Check for __import__()
        elif (isinstance(node.func, ast.Name) and
              node.func.id == '__import__'):
            
            return self._extract_builtin_import_call(node)
        
        return None

    def _extract_import_module_call(self, node: ast.Call) -> Optional[ImportStatement]:
        """Extract module name from importlib.import_module() call.
        
        Args:
            node: AST Call node for importlib.import_module()
            
        Returns:
            ImportStatement if module name can be extracted
        """
        if not node.args:
            return None
        
        first_arg = node.args[0]
        
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            # String literal: importlib.import_module('module_name')
            module_name = first_arg.value
            confidence = 0.8
        elif isinstance(first_arg, ast.Str):  # Python < 3.8 compatibility
            module_name = first_arg.s
            confidence = 0.8
        else:
            # Variable or complex expression: importlib.import_module(variable)
            module_name = f"<dynamic:{ast.unparse(first_arg) if hasattr(ast, 'unparse') else 'unknown'}>"
            confidence = 0.3
        
        import_stmt = self._create_import_statement(
            module=module_name,
            import_type='dynamic',
            line_number=node.lineno,
            confidence=confidence,
            source='ast'
        )
        
        import_stmt.metadata['call_type'] = 'importlib.import_module'
        return import_stmt

    def _extract_builtin_import_call(self, node: ast.Call) -> Optional[ImportStatement]:
        """Extract module name from __import__() call.
        
        Args:
            node: AST Call node for __import__()
            
        Returns:
            ImportStatement if module name can be extracted
        """
        if not node.args:
            return None
        
        first_arg = node.args[0]
        
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            # String literal: __import__('module_name')
            module_name = first_arg.value
            confidence = 0.7
        elif isinstance(first_arg, ast.Str):  # Python < 3.8 compatibility
            module_name = first_arg.s
            confidence = 0.7
        else:
            # Variable or complex expression: __import__(variable)
            module_name = f"<dynamic:{ast.unparse(first_arg) if hasattr(ast, 'unparse') else 'unknown'}>"
            confidence = 0.2
        
        import_stmt = self._create_import_statement(
            module=module_name,
            import_type='dynamic',
            line_number=node.lineno,
            confidence=confidence,
            source='ast'
        )
        
        import_stmt.metadata['call_type'] = '__import__'
        return import_stmt

    def _is_conditional_context(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if a node is in a conditional context (if, try, etc.).
        
        Args:
            node: AST node to check
            tree: Full AST tree
            
        Returns:
            True if node is in conditional context
        """
        # This is a simplified check - we could make it more sophisticated
        # by walking up the AST tree to find parent nodes
        
        # For now, we'll check if the node is directly inside specific parent types
        for parent in ast.walk(tree):
            if hasattr(parent, 'body'):
                if node in parent.body and isinstance(parent, (ast.If, ast.Try, ast.ExceptHandler, ast.With)):
                    return True
            if hasattr(parent, 'orelse'):
                if node in parent.orelse and isinstance(parent, ast.If):
                    return True
            if hasattr(parent, 'handlers'):
                if isinstance(parent, ast.Try):
                    for handler in parent.handlers:
                        if hasattr(handler, 'body') and node in handler.body:
                            return True
        
        return False
