"""
OCC Validator Integration - Integration layer for OCC Validator with multi-agent coordinator.

This module provides integration between the OCC Validator and the multi-agent coordinator,
including agent-specific tracking and validation management.
"""

import logging
import time
from typing import Dict, Optional, Tuple, Any, Set

from openhands.core.occ_validator import (
    OCCValidator, 
    ReadWriteTracker, 
    ValidationResult,
    Region
)
from openhands.core.dependency_analyzer.core import DependencyAnalyzer
from openhands.runtime.utils.git_handler import GitHandler


class AgentReadWriteTracker:
    """Wrapper around ReadWriteTracker for agent-specific tracking."""
    
    def __init__(self, agent_id: str):
        """Initialize agent tracker.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id
        self.tracker: Optional[ReadWriteTracker] = None
        self.start_time: float = 0.0
        self._active = False
        
    def start_tracking(self, base_commit: str):
        """Initialize tracking for agent.
        
        Args:
            base_commit: Base commit SHA when tracking started
        """
        self.tracker = ReadWriteTracker(base_commit=base_commit)
        self.start_time = time.time()
        self._active = True
        
    def stop_tracking(self) -> Tuple[Dict[str, Set[Region]], Dict[str, Set[Region]]]:
        """Stop tracking and return read/write sets.
        
        Returns:
            Tuple of (read_set, write_set)
        """
        if not self.tracker or not self._active:
            return {}, {}
        
        self._active = False
        read_set = self.tracker.get_read_set()
        write_set = self.tracker.get_write_set()
        
        return read_set, write_set
    
    def track_file_read(self, file_path: str):
        """Hook for file read operations.
        
        Args:
            file_path: Path to file being read
        """
        if self.tracker and self._active:
            self.tracker.track_read(file_path)
    
    def track_file_write(self, file_path: str):
        """Hook for file write operations.
        
        Args:
            file_path: Path to file being written
        """
        if self.tracker and self._active:
            self.tracker.track_write(file_path)
    
    def track_region_read(self, file_path: str, regions: list[Region]):
        """Hook for region read operations.
        
        Args:
            file_path: Path to file being read
            regions: Specific regions being read
        """
        if self.tracker and self._active:
            self.tracker.track_read(file_path, regions)
    
    def track_region_write(self, file_path: str, regions: list[Region]):
        """Hook for region write operations.
        
        Args:
            file_path: Path to file being written
            regions: Specific regions being written
        """
        if self.tracker and self._active:
            self.tracker.track_write(file_path, regions)
    
    @property
    def is_active(self) -> bool:
        """Check if tracking is active."""
        return self._active and self.tracker is not None
    
    @property
    def tracking_duration(self) -> float:
        """Get tracking duration in seconds."""
        if self.start_time > 0:
            return time.time() - self.start_time
        return 0.0


class OCCValidatorManager:
    """Manage OCC validation for multiple agents."""
    
    def __init__(
        self,
        workspace_base: str,
        workspace_mount_path_in_sandbox: str,
        dependency_analyzer: DependencyAnalyzer,
        git_handler: GitHandler,
        logger: Optional[logging.Logger] = None,
        event_stream: Optional[Any] = None,
        enable_tracking: bool = True
    ):
        """Initialize OCC validator manager.
        
        Args:
            workspace_base: Base workspace path
            workspace_mount_path_in_sandbox: Workspace path in sandbox
            dependency_analyzer: Dependency analyzer instance
            git_handler: Git handler instance
            logger: Optional logger
            event_stream: Optional event stream
            enable_tracking: Whether to enable OCC tracking
        """
        self.workspace_base = workspace_base
        self.workspace_mount_path_in_sandbox = workspace_mount_path_in_sandbox
        self.dependency_analyzer = dependency_analyzer
        self.git_handler = git_handler
        self.logger = logger or logging.getLogger(__name__)
        self.event_stream = event_stream
        self.enable_tracking = enable_tracking
        
        # Per-agent validators and trackers
        self.validators: Dict[str, OCCValidator] = {}
        self.trackers: Dict[str, AgentReadWriteTracker] = {}
        
        # Global statistics
        self._manager_stats = {
            'total_agents_tracked': 0,
            'active_agents': 0,
            'total_validations': 0,
            'successful_validations': 0,
            'conflicted_validations': 0
        }
    
    def create_validator_for_agent(self, agent_id: str, base_commit: str) -> OCCValidator:
        """Create validator for agent.
        
        Args:
            agent_id: Unique identifier for the agent
            base_commit: Base commit SHA
            
        Returns:
            OCCValidator instance for the agent
        """
        if not self.enable_tracking:
            # Return a dummy validator that always succeeds
            return self._create_dummy_validator()
        
        # Create new validator for agent
        validator = OCCValidator(
            workspace_base=self.workspace_base,
            workspace_mount_path_in_sandbox=self.workspace_mount_path_in_sandbox,
            dependency_analyzer=self.dependency_analyzer,
            git_handler=self.git_handler,
            logger=self.logger,
            event_stream=self.event_stream
        )
        
        self.validators[agent_id] = validator
        self.logger.info(f"Created OCC validator for agent {agent_id} at base commit {base_commit}")
        
        return validator
    
    def start_tracking(self, agent_id: str, base_commit: str):
        """Start tracking for agent.
        
        Args:
            agent_id: Unique identifier for the agent
            base_commit: Base commit SHA
        """
        if not self.enable_tracking:
            self.logger.debug(f"OCC tracking disabled, skipping tracking for agent {agent_id}")
            return
        
        # Create tracker for agent
        tracker = AgentReadWriteTracker(agent_id)
        tracker.start_tracking(base_commit)
        
        self.trackers[agent_id] = tracker
        self._manager_stats['total_agents_tracked'] += 1
        self._manager_stats['active_agents'] += 1
        
        self.logger.info(f"Started OCC tracking for agent {agent_id} at base commit {base_commit}")
        
        # Emit tracking started event
        if self.event_stream:
            try:
                from openhands.events.observation.occ import OCCTrackingStartedObservation
                self.event_stream.add_event(OCCTrackingStartedObservation(
                    agent_id=agent_id,
                    base_commit=base_commit,
                    timestamp=time.time()
                ))
            except ImportError:
                self.logger.debug("OCC observation events not available")
    
    async def validate_agent_commit(
        self, 
        agent_id: str, 
        current_head: str
    ) -> ValidationResult:
        """Validate agent's changes.
        
        Args:
            agent_id: Unique identifier for the agent
            current_head: Current HEAD commit
            
        Returns:
            ValidationResult with conflicts and suggested resolution
        """
        if not self.enable_tracking:
            # Return success if tracking is disabled
            return ValidationResult(success=True, suggested_resolution='commit')
        
        self._manager_stats['total_validations'] += 1
        
        # Get tracker and validator for agent
        tracker = self.trackers.get(agent_id)
        validator = self.validators.get(agent_id)
        
        if not tracker or not validator:
            self.logger.error(f"No tracker or validator found for agent {agent_id}")
            return ValidationResult(
                success=False,
                suggested_resolution='abort',
                validation_duration_ms=0.0
            )
        
        # Stop tracking and get read/write sets
        read_set, write_set = tracker.stop_tracking()
        
        if not tracker.tracker:
            self.logger.error(f"Tracker not initialized for agent {agent_id}")
            return ValidationResult(
                success=False,
                suggested_resolution='abort',
                validation_duration_ms=0.0
            )
        
        base_commit = tracker.tracker.base_commit
        
        self.logger.info(
            f"Validating agent {agent_id}: {len(read_set)} read files, "
            f"{len(write_set)} write files, base={base_commit}, head={current_head}"
        )
        
        try:
            # Perform validation
            result = await validator.validate_commit(
                read_set=read_set,
                write_set=write_set,
                base_commit=base_commit,
                current_head=current_head,
                agent_id=agent_id
            )
            
            # Update statistics
            if result.success:
                self._manager_stats['successful_validations'] += 1
            else:
                self._manager_stats['conflicted_validations'] += 1
            
            # Emit conflict events if any
            if result.conflicts and self.event_stream:
                try:
                    from openhands.events.observation.occ import OCCConflictDetectedObservation
                    for conflict in result.conflicts:
                        self.event_stream.add_event(OCCConflictDetectedObservation(
                            agent_id=agent_id,
                            file_path=conflict.file_path,
                            conflict_type=conflict.conflict_type.value,
                            description=conflict.description,
                            suggested_resolution=conflict.suggested_resolution,
                            base_region=conflict.base_region.to_dict() if conflict.base_region else None,
                            current_region=conflict.current_region.to_dict() if conflict.current_region else None,
                            agent_region=conflict.agent_region.to_dict() if conflict.agent_region else None,
                            timestamp=time.time()
                        ))
                except ImportError:
                    self.logger.debug("OCC observation events not available")
            
            self.logger.info(
                f"Agent {agent_id} validation completed: success={result.success}, "
                f"conflicts={result.conflicts_count}, resolution={result.suggested_resolution}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed for agent {agent_id}: {e}")
            return ValidationResult(
                success=False,
                suggested_resolution='abort',
                validation_duration_ms=0.0
            )
    
    def cleanup_agent(self, agent_id: str):
        """Remove agent's validator and tracker.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        # Clean up tracker
        if agent_id in self.trackers:
            tracker = self.trackers[agent_id]
            if tracker.is_active:
                tracker.stop_tracking()
                self._manager_stats['active_agents'] = max(0, self._manager_stats['active_agents'] - 1)
            del self.trackers[agent_id]
        
        # Clean up validator
        if agent_id in self.validators:
            del self.validators[agent_id]
        
        self.logger.info(f"Cleaned up OCC resources for agent {agent_id}")
    
    def get_agent_tracker(self, agent_id: str) -> Optional[AgentReadWriteTracker]:
        """Get tracker for agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            AgentReadWriteTracker or None if not found
        """
        return self.trackers.get(agent_id)
    
    def get_agent_validator(self, agent_id: str) -> Optional[OCCValidator]:
        """Get validator for agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            OCCValidator or None if not found
        """
        return self.validators.get(agent_id)
    
    def get_active_agents(self) -> list[str]:
        """Get list of active agent IDs.
        
        Returns:
            List of active agent IDs
        """
        return [
            agent_id for agent_id, tracker in self.trackers.items()
            if tracker.is_active
        ]
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics.
        
        Returns:
            Dictionary with manager statistics
        """
        stats = self._manager_stats.copy()
        stats['active_agents_count'] = len(self.get_active_agents())
        stats['total_trackers'] = len(self.trackers)
        stats['total_validators'] = len(self.validators)
        
        return stats
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Dictionary with agent statistics
        """
        stats = {
            'agent_id': agent_id,
            'has_tracker': agent_id in self.trackers,
            'has_validator': agent_id in self.validators,
            'is_active': False,
            'tracking_duration': 0.0,
            'read_files_count': 0,
            'write_files_count': 0
        }
        
        # Get tracker stats
        tracker = self.trackers.get(agent_id)
        if tracker:
            stats['is_active'] = tracker.is_active
            stats['tracking_duration'] = tracker.tracking_duration
            
            if tracker.tracker:
                stats['read_files_count'] = len(tracker.tracker.read_set)
                stats['write_files_count'] = len(tracker.tracker.write_set)
        
        # Get validator stats
        validator = self.validators.get(agent_id)
        if validator:
            validator_stats = validator.get_validation_stats()
            stats.update({
                'validator_total_validations': validator_stats.get('total_validations', 0),
                'validator_successful_validations': validator_stats.get('successful_validations', 0),
                'validator_conflicts_detected': validator_stats.get('conflicts_detected', 0),
                'validator_avg_validation_time_ms': validator_stats.get('avg_validation_time_ms', 0.0)
            })
        
        return stats
    
    def _create_dummy_validator(self) -> OCCValidator:
        """Create a dummy validator that always returns success.
        
        This is used when OCC tracking is disabled.
        
        Returns:
            OCCValidator that always succeeds
        """
        class DummyOCCValidator:
            """Dummy validator that always returns success."""
            
            async def validate_commit(self, **kwargs) -> ValidationResult:
                return ValidationResult(success=True, suggested_resolution='commit')
            
            def get_validation_stats(self) -> Dict[str, Any]:
                return {
                    'total_validations': 0,
                    'successful_validations': 0,
                    'conflicts_detected': 0,
                    'avg_validation_time_ms': 0.0
                }
        
        return DummyOCCValidator()


# Runtime hooks for file tracking

class FileTrackingContext:
    """Context manager for file tracking during agent execution."""
    
    def __init__(self, manager: OCCValidatorManager, agent_id: str):
        """Initialize context.
        
        Args:
            manager: OCC validator manager
            agent_id: Agent identifier
        """
        self.manager = manager
        self.agent_id = agent_id
        self.tracker: Optional[AgentReadWriteTracker] = None
    
    def __enter__(self):
        """Enter context and enable tracking."""
        self.tracker = self.manager.get_agent_tracker(self.agent_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and disable tracking."""
        pass  # Tracking continues until validation
    
    def track_file_read(self, file_path: str):
        """Track file read operation.
        
        Args:
            file_path: Path to file being read
        """
        if self.tracker:
            self.tracker.track_file_read(file_path)
    
    def track_file_write(self, file_path: str):
        """Track file write operation.
        
        Args:
            file_path: Path to file being written
        """
        if self.tracker:
            self.tracker.track_file_write(file_path)


# Integration hooks for MultiAgentCoordinator

def install_occ_hooks(coordinator, occ_manager: OCCValidatorManager):
    """Install OCC tracking hooks in MultiAgentCoordinator.
    
    This function wraps coordinator methods to add OCC tracking without
    changing their public API signatures.
    
    Args:
        coordinator: MultiAgentCoordinator instance
        occ_manager: OCCValidatorManager instance
    """
    import functools
    
    # Store reference to OCC manager
    coordinator.occ_validator_manager = occ_manager
    
    # Hook into agent lifecycle methods
    original_spawn_agent = coordinator.spawn_coding_agent
    original_cleanup_agent = coordinator._cleanup_agent
    
    @functools.wraps(original_spawn_agent)
    async def spawn_agent_with_occ(goal: str, files: list[str], **kwargs):
        """Spawn agent with OCC tracking.
        
        Preserves the original spawn_coding_agent signature exactly.
        Base commit is obtained internally from git.
        """
        # Get current HEAD commit for OCC tracking
        base_commit = "HEAD"
        try:
            result = occ_manager.git_handler.run_git_command(['rev-parse', 'HEAD'])
            if result.returncode == 0:
                base_commit = result.stdout.strip()
        except Exception as e:
            occ_manager.logger.warning(f"Failed to get current commit for OCC: {e}")
        
        # Spawn agent normally with original signature
        agent_id = await original_spawn_agent(goal, files, **kwargs)
        
        # Start OCC tracking after agent is spawned
        occ_manager.start_tracking(agent_id, base_commit)
        
        return agent_id
    
    @functools.wraps(original_cleanup_agent)
    def cleanup_agent_with_occ(agent_id: str):
        """Cleanup agent with OCC cleanup.
        
        Preserves the original _cleanup_agent signature.
        """
        # Cleanup OCC resources first
        occ_manager.cleanup_agent(agent_id)
        
        # Then cleanup agent normally
        return original_cleanup_agent(agent_id)
    
    # Replace methods
    coordinator.spawn_coding_agent = spawn_agent_with_occ
    coordinator._cleanup_agent = cleanup_agent_with_occ


# Configuration helpers

class OCCConfig:
    """Configuration for OCC Validator integration."""
    
    def __init__(
        self,
        enable_occ_validation: bool = True,
        occ_validation_mode: str = 'strict',
        occ_auto_rebase: bool = False,
        occ_track_read_set: bool = True,
        occ_track_write_set: bool = True,
        occ_max_validation_time_ms: float = 10000.0,
        occ_conflict_resolution_strategy: str = 'conservative'
    ):
        """Initialize OCC configuration.
        
        Args:
            enable_occ_validation: Whether to enable OCC validation
            occ_validation_mode: Validation mode ('strict' or 'permissive')
            occ_auto_rebase: Whether to auto-rebase on conflicts
            occ_track_read_set: Whether to track read operations
            occ_track_write_set: Whether to track write operations
            occ_max_validation_time_ms: Maximum validation time in milliseconds
            occ_conflict_resolution_strategy: Conflict resolution strategy
        """
        self.enable_occ_validation = enable_occ_validation
        self.occ_validation_mode = occ_validation_mode
        self.occ_auto_rebase = occ_auto_rebase
        self.occ_track_read_set = occ_track_read_set
        self.occ_track_write_set = occ_track_write_set
        self.occ_max_validation_time_ms = occ_max_validation_time_ms
        self.occ_conflict_resolution_strategy = occ_conflict_resolution_strategy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_occ_validation': self.enable_occ_validation,
            'occ_validation_mode': self.occ_validation_mode,
            'occ_auto_rebase': self.occ_auto_rebase,
            'occ_track_read_set': self.occ_track_read_set,
            'occ_track_write_set': self.occ_track_write_set,
            'occ_max_validation_time_ms': self.occ_max_validation_time_ms,
            'occ_conflict_resolution_strategy': self.occ_conflict_resolution_strategy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCCConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def default(cls) -> 'OCCConfig':
        """Create default configuration."""
        return cls()
    
    @classmethod
    def permissive(cls) -> 'OCCConfig':
        """Create permissive configuration (fewer conflicts)."""
        return cls(
            occ_validation_mode='permissive',
            occ_auto_rebase=True,
            occ_conflict_resolution_strategy='aggressive'
        )
    
    @classmethod
    def disabled(cls) -> 'OCCConfig':
        """Create disabled configuration."""
        return cls(enable_occ_validation=False)
