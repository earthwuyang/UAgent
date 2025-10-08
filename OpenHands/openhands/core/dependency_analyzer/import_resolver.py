"""Import path resolution to convert module names to absolute file paths."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Set, List
from functools import lru_cache

import openhands.runtime.utils.files as file_utils


class ImportResolver:
    """Resolves import paths to absolute file paths."""

    def __init__(
        self,
        workspace_base: str,
        workspace_mount_path_in_sandbox: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize ImportResolver.
        
        Args:
            workspace_base: Root directory of workspace on host
            workspace_mount_path_in_sandbox: Sandbox mount path  
            logger: Optional logger
        """
        self.workspace_base = workspace_base
        self.workspace_mount_path_in_sandbox = workspace_mount_path_in_sandbox
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache for file index and resolved imports
        self._file_index: Optional[Dict[str, str]] = None
        self._resolved_cache: Dict[str, Optional[str]] = {}

    async def resolve_import(
        self,
        import_str: str,
        source_file: str,
        language: str
    ) -> Optional[str]:
        """Resolve import string to absolute file path.
        
        Args:
            import_str: Import string from code
            source_file: Absolute path to file containing import
            language: Programming language
            
        Returns:
            Absolute file path or None if not found
        """
        # Create cache key
        cache_key = f"{language}:{source_file}:{import_str}"
        
        # Check cache first
        if cache_key in self._resolved_cache:
            return self._resolved_cache[cache_key]
        
        resolved_path = None
        
        try:
            if language == 'python':
                resolved_path = await self.resolve_python_import(import_str, source_file)
            elif language in ('javascript', 'typescript'):
                resolved_path = await self.resolve_javascript_import(import_str, source_file)
            elif language == 'java':
                resolved_path = await self.resolve_java_import(import_str, source_file)
            else:
                self.logger.warning(f"Unsupported language for import resolution: {language}")
        
        except Exception as e:
            self.logger.debug(f"Failed to resolve import '{import_str}' in {source_file}: {e}")
        
        # Cache result
        self._resolved_cache[cache_key] = resolved_path
        return resolved_path

    async def resolve_python_import(self, import_str: str, source_file: str) -> Optional[str]:
        """Resolve Python import to file path.
        
        Args:
            import_str: Python import string
            source_file: Source file absolute path
            
        Returns:
            Absolute file path or None if not found
        """
        # Handle relative imports
        if import_str.startswith('.'):
            return await self._resolve_python_relative_import(import_str, source_file)
        
        # Handle absolute imports
        return await self._resolve_python_absolute_import(import_str, source_file)

    async def _resolve_python_relative_import(self, import_str: str, source_file: str) -> Optional[str]:
        """Resolve Python relative import."""
        # Count dots to determine level
        level = 0
        for char in import_str:
            if char == '.':
                level += 1
            else:
                break
        
        # Get parent directory based on level
        source_dir = Path(source_file).parent
        parent_dir = source_dir
        
        for _ in range(level):
            parent_dir = parent_dir.parent
        
        # Extract module name after dots
        module_name = import_str[level:]
        
        if not module_name:
            # Import from current level (from . import something)
            # Check for __init__.py
            init_file = parent_dir / '__init__.py'
            if init_file.exists() and self._is_within_workspace(str(init_file)):
                return str(init_file)
            return None
        
        # Try different file patterns
        candidates = [
            parent_dir / f"{module_name}.py",
            parent_dir / module_name / "__init__.py",
        ]
        
        # Also try with dots replaced by slashes for submodules
        if '.' in module_name:
            submodule_path = module_name.replace('.', os.sep)
            candidates.extend([
                parent_dir / f"{submodule_path}.py",
                parent_dir / submodule_path / "__init__.py",
            ])
        
        for candidate in candidates:
            if candidate.exists() and self._is_within_workspace(str(candidate)):
                return str(candidate)
        
        return None

    async def _resolve_python_absolute_import(self, import_str: str, source_file: str) -> Optional[str]:
        """Resolve Python absolute import."""
        # Build file index if not already built
        if self._file_index is None:
            await self.build_file_index()
        
        # Try exact match first
        if import_str in self._file_index:
            return self._file_index[import_str]
        
        # Try with different patterns
        workspace_path = Path(self.workspace_base)
        
        # Convert dots to path separators
        module_path = import_str.replace('.', os.sep)
        
        candidates = [
            workspace_path / f"{module_path}.py",
            workspace_path / module_path / "__init__.py",
            workspace_path / "src" / f"{module_path}.py",
            workspace_path / "src" / module_path / "__init__.py",
        ]
        
        for candidate in candidates:
            if candidate.exists() and self._is_within_workspace(str(candidate)):
                return str(candidate)
        
        return None

    async def resolve_javascript_import(self, import_str: str, source_file: str) -> Optional[str]:
        """Resolve JavaScript/TypeScript import to file path.
        
        Args:
            import_str: JavaScript import string
            source_file: Source file absolute path
            
        Returns:
            Absolute file path or None if not found
        """
        source_dir = Path(source_file).parent
        
        # Handle relative imports
        if import_str.startswith(('./', '../')):
            return await self._resolve_js_relative_import(import_str, source_dir)
        
        # Handle absolute imports from root
        if import_str.startswith('/'):
            return await self._resolve_js_absolute_import(import_str[1:])  # Remove leading slash
        
        # Package imports (node_modules) - skip these
        return None

    async def _resolve_js_relative_import(self, import_str: str, source_dir: Path) -> Optional[str]:
        """Resolve JavaScript relative import."""
        # Resolve relative path
        target_path = source_dir / import_str
        target_path = target_path.resolve()
        
        # Try different extensions
        extensions = ['.js', '.jsx', '.ts', '.tsx', '.json']
        
        # First try as-is (might already have extension)
        if target_path.exists() and self._is_within_workspace(str(target_path)):
            return str(target_path)
        
        # Try with extensions
        for ext in extensions:
            candidate = target_path.with_suffix(ext)
            if candidate.exists() and self._is_within_workspace(str(candidate)):
                return str(candidate)
        
        # Try as directory with index file
        for ext in extensions:
            index_file = target_path / f"index{ext}"
            if index_file.exists() and self._is_within_workspace(str(index_file)):
                return str(index_file)
        
        return None

    async def _resolve_js_absolute_import(self, import_str: str) -> Optional[str]:
        """Resolve JavaScript absolute import from workspace root."""
        workspace_path = Path(self.workspace_base)
        target_path = workspace_path / import_str
        
        # Try different extensions
        extensions = ['.js', '.jsx', '.ts', '.tsx', '.json']
        
        # Try as-is
        if target_path.exists() and self._is_within_workspace(str(target_path)):
            return str(target_path)
        
        # Try with extensions
        for ext in extensions:
            candidate = target_path.with_suffix(ext)
            if candidate.exists() and self._is_within_workspace(str(candidate)):
                return str(candidate)
        
        # Try as directory with index file
        for ext in extensions:
            index_file = target_path / f"index{ext}"
            if index_file.exists() and self._is_within_workspace(str(index_file)):
                return str(index_file)
        
        return None

    async def resolve_java_import(self, import_str: str, source_file: str) -> Optional[str]:
        """Resolve Java import to file path.
        
        Args:
            import_str: Java import string (package.Class)
            source_file: Source file absolute path
            
        Returns:
            Absolute file path or None if not found
        """
        # Convert package name to file path
        # com.example.MyClass -> com/example/MyClass.java
        class_path = import_str.replace('.', os.sep) + '.java'
        
        workspace_path = Path(self.workspace_base)
        
        # Common Java source directory patterns
        source_dirs = [
            workspace_path,
            workspace_path / "src",
            workspace_path / "src" / "main" / "java",
            workspace_path / "app" / "src" / "main" / "java",
        ]
        
        for source_dir in source_dirs:
            candidate = source_dir / class_path
            if candidate.exists() and self._is_within_workspace(str(candidate)):
                return str(candidate)
        
        return None

    async def build_file_index(self) -> None:
        """Build index of files in workspace for fast lookup."""
        self.logger.debug("Building file index for import resolution")
        
        self._file_index = {}
        workspace_path = Path(self.workspace_base)
        
        try:
            # Walk through workspace and index Python modules
            for file_path in workspace_path.rglob("*.py"):
                if file_path.is_file():
                    # Calculate module name from file path
                    relative_path = file_path.relative_to(workspace_path)
                    
                    # Convert path to module name
                    if file_path.name == "__init__.py":
                        # Package __init__.py file
                        module_name = ".".join(relative_path.parent.parts)
                    else:
                        # Regular module file
                        module_parts = list(relative_path.parts[:-1])  # Exclude filename
                        module_parts.append(file_path.stem)  # Add filename without extension
                        module_name = ".".join(module_parts)
                    
                    # Store in index
                    if module_name and not module_name.startswith('.'):
                        self._file_index[module_name] = str(file_path)
        
        except Exception as e:
            self.logger.warning(f"Failed to build file index: {e}")
            self._file_index = {}

    def _is_within_workspace(self, file_path: str) -> bool:
        """Check if file path is within workspace boundaries."""
        try:
            resolved_path = file_utils.resolve_path(
                file_path,
                self.workspace_mount_path_in_sandbox,
                self.workspace_base,
                self.workspace_mount_path_in_sandbox
            )
            return resolved_path is not None
        except Exception:
            return False

    async def invalidate_cache(self, file_path: Optional[str] = None) -> None:
        """Invalidate resolver cache.
        
        Args:
            file_path: If provided, only invalidate cache entries for this file
        """
        if file_path is None:
            # Clear entire cache
            self._resolved_cache.clear()
            self._file_index = None
        else:
            # Clear cache entries for specific file
            keys_to_remove = [
                key for key in self._resolved_cache.keys()
                if file_path in key
            ]
            for key in keys_to_remove:
                del self._resolved_cache[key]
