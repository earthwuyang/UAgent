"""Main DependencyAnalyzer class that orchestrates dependency analysis."""

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from openhands.core.config.llm_config import LLMConfig
import openhands.runtime.utils.files as file_utils

from .models import (
    FileDependencies,
    DependencyGraph,
    DependencyAnalyzerConfig,
    DependencyAnalysisResult
)
from .parsers import get_parser, detect_language
from .cache import DependencyCacheManager
from .graph_builder import DependencyGraphBuilder
from .import_resolver import ImportResolver
from .llm_fallback import LLMDependencyExtractor
from .exceptions import (
    DependencyAnalysisError,
    UnsupportedLanguageError,
    ParserError
)

# Import custom observations if available
try:
    from openhands.events.observation.dependency import (
        DependencyAnalysisStartedObservation,
        DependencyAnalysisCompletedObservation,
        DependencyGraphUpdatedObservation,
        DependencyLockSetComputedObservation
    )
    from openhands.events.event import EventSource
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False
    EventSource = None


class DependencyAnalyzer:
    """Main class for analyzing code dependencies and building dependency graphs."""

    def __init__(
        self,
        workspace_base: str,
        workspace_mount_path_in_sandbox: str,
        working_directory: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        config: Optional[DependencyAnalyzerConfig] = None,
        logger: Optional[logging.Logger] = None,
        event_stream: Optional[Any] = None
    ):
        """Initialize DependencyAnalyzer.
        
        Args:
            workspace_base: Root directory of workspace on host
            workspace_mount_path_in_sandbox: Sandbox mount path
            working_directory: Working directory (defaults to workspace_mount_path_in_sandbox)
            llm_config: Optional LLM configuration for fallback analysis
            config: Optional configuration for analyzer behavior
            logger: Optional logger
            event_stream: Optional event stream for emitting observations
        """
        self.workspace_base = workspace_base
        self.workspace_mount_path_in_sandbox = workspace_mount_path_in_sandbox
        self.working_directory = working_directory or workspace_mount_path_in_sandbox
        self.llm_config = llm_config
        self.config = config or DependencyAnalyzerConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.event_stream = event_stream
        
        # Initialize components
        self._cache_manager = DependencyCacheManager(
            cache_backend=self.config.cache_backend,
            redis_url=self.config.redis_url,
            ttl_seconds=self.config.cache_ttl_seconds,
            logger=self.logger
        )
        
        self._graph_builder = DependencyGraphBuilder(logger=self.logger)
        
        self._import_resolver = ImportResolver(
            workspace_base=workspace_base,
            workspace_mount_path_in_sandbox=workspace_mount_path_in_sandbox,
            logger=self.logger
        )
        
        # Initialize LLM fallback if enabled
        self._llm_extractor: Optional[LLMDependencyExtractor] = None
        if self.config.use_llm_fallback and llm_config:
            try:
                self._llm_extractor = LLMDependencyExtractor(
                    llm_config=llm_config,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM fallback: {e}")
                self._llm_extractor = None
        
        # Current dependency graph
        self._current_graph: Optional[DependencyGraph] = None
        
        self.logger.info(f"Initialized DependencyAnalyzer for workspace: {workspace_base}")

    async def analyze_file(
        self,
        file_path: str,
        content: Optional[str] = None
    ) -> FileDependencies:
        """Analyze a single file and extract its dependencies.
        
        Args:
            file_path: Absolute or relative path to file
            content: Optional file content (if not provided, reads from disk)
            
        Returns:
            FileDependencies object with analysis results
        """
        # Resolve file path
        resolved_path = file_utils.resolve_path(
            file_path,
            self.working_directory,
            self.workspace_base,
            self.workspace_mount_path_in_sandbox
        )
        
        if not resolved_path:
            raise DependencyAnalysisError(f"Cannot resolve file path: {file_path}")
        
        resolved_path_str = str(resolved_path)
        
        # Detect language
        language = detect_language(resolved_path_str)
        if not language:
            raise UnsupportedLanguageError(f"Cannot detect language for file: {resolved_path_str}")
        
        # Emit start event
        if self.event_stream and EVENTS_AVAILABLE:
            start_observation = DependencyAnalysisStartedObservation(
                file_path=resolved_path_str,
                language=language
            )
            self.event_stream.add_event(start_observation, EventSource.AGENT)
        
        start_time = time.time()
        
        try:
            # Read file content if not provided
            if content is None:
                content_result = await file_utils.read_file(
                    file_path,
                    workdir=self.working_directory,
                    workspace_base=self.workspace_base,
                    workspace_mount_path_in_sandbox=self.workspace_mount_path_in_sandbox
                )
                if not content_result.content:
                    raise DependencyAnalysisError(f"Cannot read file: {resolved_path_str}")
                content = content_result.content
            
            # Check file size limit
            if len(content) > self.config.max_file_size_bytes:
                self.logger.warning(
                    f"File {resolved_path_str} exceeds size limit "
                    f"({len(content)} > {self.config.max_file_size_bytes} bytes)"
                )
                return self._create_empty_file_dependencies(resolved_path_str, language, content)
            
            # Compute file hash for caching
            file_hash = self._compute_file_hash(content)
            
            # Check cache first
            cached_deps = await self._cache_manager.get(resolved_path_str, file_hash)
            if cached_deps:
                self.logger.debug(f"Using cached dependencies for {resolved_path_str}")
                return cached_deps
            
            # Parse with appropriate parser
            parser_used = 'unknown'
            imports = []
            errors = []
            
            try:
                # Try language-specific parser first
                parser = get_parser(resolved_path_str)
                if parser:
                    # Configure parser with filtering options
                    parser.skip_standard_libraries = self.config.skip_standard_libraries
                    parser.skip_third_party_packages = self.config.skip_third_party_packages
                    
                    imports = await parser.parse(resolved_path_str, content)
                    parser_used = 'ast' if language == 'python' else 'tree_sitter'
                else:
                    raise ParserError(resolved_path_str, language, "No parser available")
                
            except Exception as e:
                self.logger.warning(f"Primary parser failed for {resolved_path_str}: {e}")
                errors.append(f"Primary parser error: {e}")
                
                # Try LLM fallback if available
                if self._llm_extractor:
                    try:
                        llm_imports = await self._llm_extractor.extract_dependencies(
                            resolved_path_str, content, language, context=str(e)
                        )
                        imports.extend(llm_imports)
                        parser_used = 'llm'
                    except Exception as llm_error:
                        self.logger.warning(f"LLM fallback failed for {resolved_path_str}: {llm_error}")
                        errors.append(f"LLM fallback error: {llm_error}")
            
            # Additional filtering in analyzer if not done by parser
            if not hasattr(parser, 'skip_standard_libraries'):
                filtered_imports = []
                for import_stmt in imports:
                    skip = False
                    if self.config.skip_standard_libraries and self._is_standard_library(import_stmt.module, language):
                        skip = True
                    elif self.config.skip_third_party_packages and self._is_third_party_package(import_stmt.module, language):
                        skip = True
                    
                    if not skip:
                        filtered_imports.append(import_stmt)
                    else:
                        self.logger.debug(f"Skipping filtered import: {import_stmt.module}")
                
                imports = filtered_imports
            
            # Resolve import paths
            resolved_paths = []
            for import_stmt in imports:
                try:
                    resolved_import_path = await self._import_resolver.resolve_import(
                        import_stmt.module,
                        resolved_path_str,
                        language
                    )
                    if resolved_import_path:
                        import_stmt.resolved_path = resolved_import_path
                        resolved_paths.append(resolved_import_path)
                except Exception as e:
                    self.logger.debug(f"Failed to resolve import '{import_stmt.module}': {e}")
            
            # Create FileDependencies object
            analysis_duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            file_deps = FileDependencies(
                file_path=resolved_path_str,
                imports=imports,
                resolved_paths=resolved_paths,
                language=language,
                file_hash=file_hash,
                analysis_duration_ms=analysis_duration,
                parser_used=parser_used,
                errors=errors
            )
            
            # Cache the result
            await self._cache_manager.set(resolved_path_str, file_hash, file_deps)
            
            # Emit completion event
            if self.event_stream and EVENTS_AVAILABLE:
                completion_observation = DependencyAnalysisCompletedObservation(
                    file_path=resolved_path_str,
                    dependencies_found=len(resolved_paths),
                    analysis_duration_ms=analysis_duration,
                    parser_used=parser_used
                )
                self.event_stream.add_event(completion_observation, EventSource.AGENT)
            
            self.logger.debug(
                f"Analyzed {resolved_path_str}: {len(imports)} imports, "
                f"{len(resolved_paths)} resolved paths in {analysis_duration:.1f}ms"
            )
            
            return file_deps
            
        except Exception as e:
            self.logger.error(f"Failed to analyze file {resolved_path_str}: {e}")
            raise DependencyAnalysisError(f"Analysis failed for {resolved_path_str}: {e}")

    async def analyze_workspace(
        self,
        root_path: Optional[str] = None,
        file_patterns: Optional[List[str]] = None
    ) -> DependencyGraph:
        """Analyze all files in workspace matching patterns.
        
        Args:
            root_path: Root path to analyze (defaults to workspace_base)
            file_patterns: File patterns to match (defaults to supported extensions)
            
        Returns:
            DependencyGraph object with complete workspace analysis
        """
        if root_path is None:
            root_path = self.workspace_base
        
        if file_patterns is None:
            file_patterns = ['*.py', '*.js', '*.jsx', '*.ts', '*.tsx', '*.java']
        
        # Find all matching files
        files_to_analyze = []
        root_path_obj = Path(root_path)
        
        for pattern in file_patterns:
            matching_files = list(root_path_obj.rglob(pattern))
            files_to_analyze.extend([str(f) for f in matching_files if f.is_file()])
        
        self.logger.info(f"Found {len(files_to_analyze)} files to analyze in {root_path}")
        
        # Analyze files in parallel if enabled
        if self.config.parallel_analysis and len(files_to_analyze) > 1:
            file_dependencies = await self._analyze_files_parallel(files_to_analyze)
        else:
            file_dependencies = await self._analyze_files_sequential(files_to_analyze)
        
        # Build dependency graph
        dependency_graph = await self._graph_builder.build_graph(file_dependencies)
        self._current_graph = dependency_graph
        
        # Emit graph updated event
        if self.event_stream and EVENTS_AVAILABLE:
            graph_observation = DependencyGraphUpdatedObservation(
                total_files=dependency_graph.total_files,
                total_dependencies=dependency_graph.total_dependencies,
                cycles_detected=len(dependency_graph.cycles),
                languages=list(dependency_graph.languages)
            )
            self.event_stream.add_event(graph_observation, EventSource.AGENT)
        
        return dependency_graph

    async def get_dependencies(
        self,
        file_path: str,
        transitive: bool = False
    ) -> List[str]:
        """Get dependencies for a file.
        
        Args:
            file_path: File path to analyze
            transitive: If True, return all transitive dependencies
            
        Returns:
            List of file paths this file depends on
        """
        # Resolve file path
        resolved_path = file_utils.resolve_path(
            file_path,
            self.working_directory,
            self.workspace_base,
            self.workspace_mount_path_in_sandbox
        )
        
        if not resolved_path:
            raise DependencyAnalysisError(f"Cannot resolve file path: {file_path}")
        
        resolved_path_str = str(resolved_path)
        
        # If we need transitive dependencies, we need the full graph
        if transitive:
            if not self._current_graph or resolved_path_str not in self._current_graph.nodes:
                # Need to build graph first
                await self.analyze_workspace()
            
            if self._current_graph:
                return self._current_graph.get_transitive_dependencies(resolved_path_str)
        
        # For direct dependencies only, just analyze the file
        file_deps = await self.analyze_file(file_path)
        return file_deps.resolved_paths

    async def get_dependents(self, file_path: str) -> List[str]:
        """Get files that depend on this file.
        
        Args:
            file_path: File path to check
            
        Returns:
            List of file paths that depend on the given file
        """
        resolved_path = file_utils.resolve_path(
            file_path,
            self.working_directory,
            self.workspace_base,
            self.workspace_mount_path_in_sandbox
        )
        
        if not resolved_path:
            raise DependencyAnalysisError(f"Cannot resolve file path: {file_path}")
        
        resolved_path_str = str(resolved_path)
        
        # Need full graph for reverse dependencies
        if not self._current_graph or resolved_path_str not in self._current_graph.nodes:
            await self.analyze_workspace()
        
        if self._current_graph:
            return self._current_graph.get_dependents(resolved_path_str)
        
        return []

    async def get_lock_set(self, file_path: str) -> Set[str]:
        """Get all files that should be locked when locking this file.
        
        Args:
            file_path: File path to get lock set for
            
        Returns:
            Set of file paths to lock (includes file itself + dependencies + cycles)
        """
        resolved_path = file_utils.resolve_path(
            file_path,
            self.working_directory,
            self.workspace_base,
            self.workspace_mount_path_in_sandbox
        )
        
        if not resolved_path:
            raise DependencyAnalysisError(f"Cannot resolve file path: {file_path}")
        
        resolved_path_str = str(resolved_path)
        start_time = time.time()
        
        # Ensure we have current graph
        if not self._current_graph or resolved_path_str not in self._current_graph.nodes:
            await self.analyze_workspace()
        
        lock_set = {resolved_path_str}
        
        if self._current_graph:
            lock_set = self._graph_builder.get_lock_set(resolved_path_str)
        
        computation_time = (time.time() - start_time) * 1000
        includes_cycles = any(resolved_path_str in cycle for cycle in (self._current_graph.cycles if self._current_graph else []))
        
        # Emit lock set computed event
        if self.event_stream and EVENTS_AVAILABLE:
            lock_observation = DependencyLockSetComputedObservation(
                file_path=resolved_path_str,
                lock_set_size=len(lock_set),
                includes_cycles=includes_cycles,
                computation_time_ms=computation_time
            )
            self.event_stream.add_event(lock_observation, EventSource.AGENT)
        
        return lock_set

    async def invalidate_cache(self, file_path: str) -> None:
        """Invalidate cache for a specific file.
        
        Args:
            file_path: File path to invalidate
        """
        resolved_path = file_utils.resolve_path(
            file_path,
            self.working_directory,
            self.workspace_base,
            self.workspace_mount_path_in_sandbox
        )
        
        if resolved_path:
            resolved_path_str = str(resolved_path)
            await self._cache_manager.invalidate(resolved_path_str)
            await self._import_resolver.invalidate_cache(resolved_path_str)
            
            # Mark graph as stale
            self._current_graph = None
            
            self.logger.debug(f"Invalidated cache for {resolved_path_str}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the analyzer.
        
        Returns:
            Dictionary with analyzer statistics
        """
        cache_stats = await self._cache_manager.get_stats()
        
        stats = {
            'workspace_base': self.workspace_base,
            'config': self.config.model_dump(),
            'cache': cache_stats,
            'current_graph': None,
            'llm_enabled': self._llm_extractor is not None
        }
        
        if self._current_graph:
            stats['current_graph'] = {
                'total_files': self._current_graph.total_files,
                'total_dependencies': self._current_graph.total_dependencies,
                'cycles': len(self._current_graph.cycles),
                'languages': list(self._current_graph.languages),
                'created_at': self._current_graph.created_at
            }
        
        if self._llm_extractor:
            stats['llm_usage'] = self._llm_extractor.get_usage_stats()
        
        return stats

    def _compute_file_hash(self, content: str) -> str:
        """Compute SHA256 hash of file content.
        
        Args:
            content: File content
            
        Returns:
            Hex digest of hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _create_empty_file_dependencies(
        self,
        file_path: str,
        language: str,
        content: str
    ) -> FileDependencies:
        """Create empty FileDependencies object for oversized files.
        
        Args:
            file_path: Absolute file path
            language: Programming language
            content: File content
            
        Returns:
            Empty FileDependencies object
        """
        return FileDependencies(
            file_path=file_path,
            imports=[],
            resolved_paths=[],
            language=language,
            file_hash=self._compute_file_hash(content),
            parser_used='skipped',
            errors=[f"File size exceeds limit ({len(content)} bytes)"]
        )

    def _is_standard_library(self, module: str, language: str) -> bool:
        """Check if module is standard library."""
        # Import the base parser to reuse its logic
        from .parsers.base import BaseDependencyParser
        dummy_parser = BaseDependencyParser()
        return dummy_parser.is_standard_library(module, language)

    def _is_third_party_package(self, module: str, language: str) -> bool:
        """Check if module is third-party package."""
        # Import the base parser to reuse its logic
        from .parsers.base import BaseDependencyParser
        dummy_parser = BaseDependencyParser()
        return dummy_parser.is_third_party(module, language)

    async def _analyze_files_parallel(
        self,
        files_to_analyze: List[str]
    ) -> Dict[str, FileDependencies]:
        """Analyze files in parallel.
        
        Args:
            files_to_analyze: List of file paths to analyze
            
        Returns:
            Dictionary mapping file paths to FileDependencies
        """
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def analyze_with_semaphore(file_path: str) -> tuple[str, FileDependencies]:
            async with semaphore:
                try:
                    deps = await self.analyze_file(file_path)
                    return file_path, deps
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {file_path}: {e}")
                    language = detect_language(file_path) or 'unknown'
                    empty_deps = FileDependencies(
                        file_path=file_path,
                        imports=[],
                        resolved_paths=[],
                        language=language,
                        file_hash='',
                        parser_used='failed',
                        errors=[str(e)]
                    )
                    return file_path, empty_deps
        
        tasks = [analyze_with_semaphore(file_path) for file_path in files_to_analyze]
        results = await asyncio.gather(*tasks)
        
        return dict(results)

    async def _analyze_files_sequential(
        self,
        files_to_analyze: List[str]
    ) -> Dict[str, FileDependencies]:
        """Analyze files sequentially.
        
        Args:
            files_to_analyze: List of file paths to analyze
            
        Returns:
            Dictionary mapping file paths to FileDependencies
        """
        file_dependencies = {}
        
        for file_path in files_to_analyze:
            try:
                deps = await self.analyze_file(file_path)
                file_dependencies[file_path] = deps
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
                language = detect_language(file_path) or 'unknown'
                file_dependencies[file_path] = FileDependencies(
                    file_path=file_path,
                    imports=[],
                    resolved_paths=[],
                    language=language,
                    file_hash='',
                    parser_used='failed',
                    errors=[str(e)]
                )
        
        return file_dependencies
