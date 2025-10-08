"""Integration layer for connecting DependencyAnalyzer with MultiAgentCoordinator and session."""

import logging
from typing import Dict, Optional, Set, Any

from openhands.core.config.llm_config import LLMConfig

from .core import DependencyAnalyzer
from .models import DependencyAnalyzerConfig


class DependencyAnalyzerService:
    """Singleton service that manages DependencyAnalyzer instances per session."""
    
    _instances: Dict[str, DependencyAnalyzer] = {}
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def get_analyzer(
        cls,
        session_id: str,
        workspace_base: str,
        workspace_mount_path_in_sandbox: str,
        llm_config: Optional[LLMConfig] = None,
        config: Optional[DependencyAnalyzerConfig] = None,
        event_stream: Optional[Any] = None
    ) -> DependencyAnalyzer:
        """Get or create DependencyAnalyzer for session.
        
        Args:
            session_id: Unique session identifier
            workspace_base: Root directory of workspace on host
            workspace_mount_path_in_sandbox: Sandbox mount path
            llm_config: Optional LLM configuration
            config: Optional analyzer configuration
            event_stream: Optional event stream for observations
            
        Returns:
            DependencyAnalyzer instance for the session
        """
        if session_id not in cls._instances:
            try:
                analyzer = DependencyAnalyzer(
                    workspace_base=workspace_base,
                    workspace_mount_path_in_sandbox=workspace_mount_path_in_sandbox,
                    llm_config=llm_config,
                    config=config,
                    logger=cls._logger,
                    event_stream=event_stream
                )
                cls._instances[session_id] = analyzer
                cls._logger.info(f"Created DependencyAnalyzer for session: {session_id}")
            except Exception as e:
                cls._logger.error(f"Failed to create DependencyAnalyzer for session {session_id}: {e}")
                raise
        
        return cls._instances[session_id]
    
    @classmethod
    def cleanup_analyzer(cls, session_id: str) -> None:
        """Cleanup analyzer for session.
        
        Args:
            session_id: Session identifier to cleanup
        """
        if session_id in cls._instances:
            try:
                # Could add cleanup logic here if needed
                del cls._instances[session_id]
                cls._logger.info(f"Cleaned up DependencyAnalyzer for session: {session_id}")
            except Exception as e:
                cls._logger.warning(f"Error during cleanup for session {session_id}: {e}")
    
    @classmethod
    async def analyze_file_for_locking(
        cls,
        session_id: str,
        file_path: str,
        workspace_base: str,
        workspace_mount_path_in_sandbox: str,
        llm_config: Optional[LLMConfig] = None,
        transitive: bool = True
    ) -> Set[str]:
        """Convenience method for file lock manager to get lock set.
        
        Args:
            session_id: Session identifier
            file_path: File path to analyze for locking
            workspace_base: Root directory of workspace
            workspace_mount_path_in_sandbox: Sandbox mount path
            llm_config: Optional LLM configuration
            transitive: Whether to include transitive dependencies
            
        Returns:
            Set of file paths that should be locked
        """
        try:
            analyzer = cls.get_analyzer(
                session_id=session_id,
                workspace_base=workspace_base,
                workspace_mount_path_in_sandbox=workspace_mount_path_in_sandbox,
                llm_config=llm_config
            )
            
            if transitive:
                # Get complete lock set (includes cycles)
                lock_set = await analyzer.get_lock_set(file_path)
            else:
                # Get only direct dependencies
                dependencies = await analyzer.get_dependencies(file_path, transitive=False)
                lock_set = {file_path}
                lock_set.update(dependencies)
            
            return lock_set
            
        except Exception as e:
            cls._logger.error(f"Failed to analyze file for locking {file_path} in session {session_id}: {e}")
            # Fallback: just lock the file itself
            return {file_path}
    
    @classmethod
    async def invalidate_file_cache(
        cls,
        session_id: str,
        file_path: str
    ) -> None:
        """Invalidate cache for a file when it changes.
        
        Args:
            session_id: Session identifier
            file_path: File path that changed
        """
        if session_id in cls._instances:
            try:
                analyzer = cls._instances[session_id]
                await analyzer.invalidate_cache(file_path)
                cls._logger.debug(f"Invalidated cache for {file_path} in session {session_id}")
            except Exception as e:
                cls._logger.warning(f"Failed to invalidate cache for {file_path} in session {session_id}: {e}")
    
    @classmethod
    def get_session_stats(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a session's analyzer.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Statistics dictionary or None if session not found
        """
        if session_id in cls._instances:
            try:
                analyzer = cls._instances[session_id]
                # Note: get_stats is async, but this method is sync for convenience
                # Consider making this async if needed
                return {
                    'session_id': session_id,
                    'workspace_base': analyzer.workspace_base,
                    'config': analyzer.config.model_dump(),
                    'llm_enabled': analyzer._llm_extractor is not None,
                    'current_graph_files': analyzer._current_graph.total_files if analyzer._current_graph else 0
                }
            except Exception as e:
                cls._logger.warning(f"Failed to get stats for session {session_id}: {e}")
        
        return None
    
    @classmethod
    def get_all_session_ids(cls) -> Set[str]:
        """Get all active session IDs.
        
        Returns:
            Set of active session IDs
        """
        return set(cls._instances.keys())
    
    @classmethod
    def get_instance_count(cls) -> int:
        """Get number of active analyzer instances.
        
        Returns:
            Number of active instances
        """
        return len(cls._instances)


# Integration helpers for MultiAgentCoordinator

async def get_file_dependencies_for_locking(
    session_id: str,
    file_path: str,
    workspace_base: str,
    workspace_mount_path_in_sandbox: str,
    llm_config: Optional[LLMConfig] = None,
    transitive: bool = True
) -> Set[str]:
    """Helper function to get file dependencies for locking.
    
    This is the main function that should be called by MultiAgentCoordinator
    or FileLockManager when they need to determine which files to lock.
    
    Args:
        session_id: Unique session identifier
        file_path: File path to analyze
        workspace_base: Root directory of workspace
        workspace_mount_path_in_sandbox: Sandbox mount path
        llm_config: Optional LLM configuration
        transitive: Whether to include transitive dependencies and cycles
        
    Returns:
        Set of absolute file paths that should be locked
    """
    return await DependencyAnalyzerService.analyze_file_for_locking(
        session_id=session_id,
        file_path=file_path,
        workspace_base=workspace_base,
        workspace_mount_path_in_sandbox=workspace_mount_path_in_sandbox,
        llm_config=llm_config,
        transitive=transitive
    )


def cleanup_session_analyzer(session_id: str) -> None:
    """Helper function to cleanup analyzer when session ends.
    
    Args:
        session_id: Session identifier to cleanup
    """
    DependencyAnalyzerService.cleanup_analyzer(session_id)


async def invalidate_file_dependencies(session_id: str, file_path: str) -> None:
    """Helper function to invalidate dependencies when file changes.
    
    Args:
        session_id: Session identifier
        file_path: File path that changed
    """
    await DependencyAnalyzerService.invalidate_file_cache(session_id, file_path)


def is_dependency_analysis_enabled() -> bool:
    """Check if dependency analysis is available.
    
    Returns:
        True if dependency analysis is enabled and working
    """
    try:
        # Try to import core components
        from .core import DependencyAnalyzer
        from .models import DependencyAnalyzerConfig
        return True
    except ImportError:
        return False


def get_default_analyzer_config() -> DependencyAnalyzerConfig:
    """Get default analyzer configuration.
    
    Returns:
        Default DependencyAnalyzerConfig
    """
    return DependencyAnalyzerConfig(
        use_llm_fallback=True,
        cache_backend='memory',
        cache_ttl_seconds=3600,
        parallel_analysis=True,
        max_workers=5,  # Conservative default
        skip_standard_libraries=True,
        skip_third_party_packages=True
    )


# Export main integration functions
__all__ = [
    'DependencyAnalyzerService',
    'get_file_dependencies_for_locking',
    'cleanup_session_analyzer', 
    'invalidate_file_dependencies',
    'is_dependency_analysis_enabled',
    'get_default_analyzer_config'
]
