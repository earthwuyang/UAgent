"""
OCC Validator Observation Events.

This module defines observation events for the OCC Validator system that are emitted
during the optimistic concurrency control validation lifecycle.
"""

import time
from typing import Dict, List, Optional, Any

from openhands.events.observation import Observation


class OCCTrackingStartedObservation(Observation):
    """Emitted when OCC tracking starts for an agent."""
    
    def __init__(
        self,
        agent_id: str,
        base_commit: str,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        """Initialize observation.
        
        Args:
            agent_id: Unique identifier for the agent
            base_commit: Base commit SHA when tracking started
            timestamp: Event timestamp (defaults to current time)
            **kwargs: Additional observation parameters
        """
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.base_commit = base_commit
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            'agent_id': self.agent_id,
            'base_commit': self.base_commit,
            'timestamp': self.timestamp
        })
        return result
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"OCCTrackingStarted(agent={self.agent_id}, base={self.base_commit[:8]})"


class OCCFileAccessObservation(Observation):
    """Emitted when agent reads or writes a file."""
    
    def __init__(
        self,
        agent_id: str,
        file_path: str,
        access_type: str,
        regions: Optional[List[Dict]] = None,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        """Initialize observation.
        
        Args:
            agent_id: Unique identifier for the agent
            file_path: Path to the file being accessed
            access_type: Type of access ('read' or 'write')
            regions: List of regions being accessed (serialized Region objects)
            timestamp: Event timestamp (defaults to current time)
            **kwargs: Additional observation parameters
        """
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.file_path = file_path
        self.access_type = access_type
        self.regions = regions or []
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            'agent_id': self.agent_id,
            'file_path': self.file_path,
            'access_type': self.access_type,
            'regions': self.regions,
            'timestamp': self.timestamp
        })
        return result
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        region_count = len(self.regions)
        regions_info = f", {region_count} regions" if region_count > 0 else ""
        return f"OCCFileAccess(agent={self.agent_id}, {self.access_type} {self.file_path}{regions_info})"


class OCCValidationStartedObservation(Observation):
    """Emitted when OCC validation begins."""
    
    def __init__(
        self,
        agent_id: str,
        base_commit: str,
        current_head: str,
        read_set_size: int,
        write_set_size: int,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        """Initialize observation.
        
        Args:
            agent_id: Unique identifier for the agent
            base_commit: Base commit SHA when agent started
            current_head: Current HEAD commit SHA
            read_set_size: Number of files in read set
            write_set_size: Number of files in write set
            timestamp: Event timestamp (defaults to current time)
            **kwargs: Additional observation parameters
        """
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.base_commit = base_commit
        self.current_head = current_head
        self.read_set_size = read_set_size
        self.write_set_size = write_set_size
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            'agent_id': self.agent_id,
            'base_commit': self.base_commit,
            'current_head': self.current_head,
            'read_set_size': self.read_set_size,
            'write_set_size': self.write_set_size,
            'timestamp': self.timestamp
        })
        return result
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"OCCValidationStarted(agent={self.agent_id}, "
                f"base={self.base_commit[:8]}, head={self.current_head[:8]}, "
                f"read={self.read_set_size}, write={self.write_set_size})")


class OCCValidationCompletedObservation(Observation):
    """Emitted when OCC validation completes."""
    
    def __init__(
        self,
        agent_id: str,
        success: bool,
        conflicts_count: int,
        read_conflicts: int,
        write_conflicts: int,
        suggested_resolution: str,
        validation_duration_ms: float,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        """Initialize observation.
        
        Args:
            agent_id: Unique identifier for the agent
            success: True if validation succeeded (no conflicts)
            conflicts_count: Total number of conflicts detected
            read_conflicts: Number of read conflicts
            write_conflicts: Number of write conflicts
            suggested_resolution: Suggested resolution strategy
            validation_duration_ms: Validation duration in milliseconds
            timestamp: Event timestamp (defaults to current time)
            **kwargs: Additional observation parameters
        """
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.success = success
        self.conflicts_count = conflicts_count
        self.read_conflicts = read_conflicts
        self.write_conflicts = write_conflicts
        self.suggested_resolution = suggested_resolution
        self.validation_duration_ms = validation_duration_ms
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            'agent_id': self.agent_id,
            'success': self.success,
            'conflicts_count': self.conflicts_count,
            'read_conflicts': self.read_conflicts,
            'write_conflicts': self.write_conflicts,
            'suggested_resolution': self.suggested_resolution,
            'validation_duration_ms': self.validation_duration_ms,
            'timestamp': self.timestamp
        })
        return result
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "SUCCESS" if self.success else f"CONFLICTS({self.conflicts_count})"
        return (f"OCCValidationCompleted(agent={self.agent_id}, {status}, "
                f"resolution={self.suggested_resolution}, {self.validation_duration_ms:.1f}ms)")


class OCCConflictDetectedObservation(Observation):
    """Emitted for each conflict detected during validation."""
    
    def __init__(
        self,
        agent_id: str,
        file_path: str,
        conflict_type: str,
        description: str,
        suggested_resolution: str,
        base_region: Optional[Dict] = None,
        current_region: Optional[Dict] = None,
        agent_region: Optional[Dict] = None,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        """Initialize observation.
        
        Args:
            agent_id: Unique identifier for the agent
            file_path: Path to the conflicted file
            conflict_type: Type of conflict (FORMATTING, AST_REORDERING, SEMANTIC, READ_WRITE, WRITE_WRITE)
            description: Human-readable description of the conflict
            suggested_resolution: Suggested resolution strategy (rebase, merge, abort)
            base_region: Base version region (serialized Region object)
            current_region: Current version region (serialized Region object)
            agent_region: Agent's version region (serialized Region object)
            timestamp: Event timestamp (defaults to current time)
            **kwargs: Additional observation parameters
        """
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.file_path = file_path
        self.conflict_type = conflict_type
        self.description = description
        self.suggested_resolution = suggested_resolution
        self.base_region = base_region
        self.current_region = current_region
        self.agent_region = agent_region
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            'agent_id': self.agent_id,
            'file_path': self.file_path,
            'conflict_type': self.conflict_type,
            'description': self.description,
            'suggested_resolution': self.suggested_resolution,
            'base_region': self.base_region,
            'current_region': self.current_region,
            'agent_region': self.agent_region,
            'timestamp': self.timestamp
        })
        return result
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"OCCConflictDetected(agent={self.agent_id}, {self.conflict_type} "
                f"in {self.file_path}, resolution={self.suggested_resolution})")


class OCCAutoRebaseStartedObservation(Observation):
    """Emitted when automatic rebase starts."""
    
    def __init__(
        self,
        agent_id: str,
        base_commit: str,
        target_commit: str,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        """Initialize observation.
        
        Args:
            agent_id: Unique identifier for the agent
            base_commit: Base commit SHA
            target_commit: Target commit SHA to rebase onto
            timestamp: Event timestamp (defaults to current time)
            **kwargs: Additional observation parameters
        """
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.base_commit = base_commit
        self.target_commit = target_commit
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            'agent_id': self.agent_id,
            'base_commit': self.base_commit,
            'target_commit': self.target_commit,
            'timestamp': self.timestamp
        })
        return result
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"OCCAutoRebaseStarted(agent={self.agent_id}, "
                f"base={self.base_commit[:8]}, target={self.target_commit[:8]})")


class OCCAutoRebaseCompletedObservation(Observation):
    """Emitted when automatic rebase completes."""
    
    def __init__(
        self,
        agent_id: str,
        success: bool,
        new_base_commit: str,
        conflicts_resolved: int,
        error_message: Optional[str] = None,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        """Initialize observation.
        
        Args:
            agent_id: Unique identifier for the agent
            success: True if rebase succeeded
            new_base_commit: New base commit SHA after rebase
            conflicts_resolved: Number of conflicts resolved during rebase
            error_message: Error message if rebase failed
            timestamp: Event timestamp (defaults to current time)
            **kwargs: Additional observation parameters
        """
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.success = success
        self.new_base_commit = new_base_commit
        self.conflicts_resolved = conflicts_resolved
        self.error_message = error_message
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            'agent_id': self.agent_id,
            'success': self.success,
            'new_base_commit': self.new_base_commit,
            'conflicts_resolved': self.conflicts_resolved,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        })
        return result
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "SUCCESS" if self.success else f"FAILED({self.error_message})"
        return (f"OCCAutoRebaseCompleted(agent={self.agent_id}, {status}, "
                f"new_base={self.new_base_commit[:8]}, resolved={self.conflicts_resolved})")


class OCCRegionAnalysisObservation(Observation):
    """Emitted when detailed region analysis is performed."""
    
    def __init__(
        self,
        agent_id: str,
        file_path: str,
        regions_extracted: int,
        signatures_computed: int,
        parsing_language: str,
        analysis_duration_ms: float,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        """Initialize observation.
        
        Args:
            agent_id: Unique identifier for the agent
            file_path: Path to the analyzed file
            regions_extracted: Number of regions extracted from the file
            signatures_computed: Number of AST signatures computed
            parsing_language: Language used for parsing (python, javascript, java, etc.)
            analysis_duration_ms: Analysis duration in milliseconds
            timestamp: Event timestamp (defaults to current time)
            **kwargs: Additional observation parameters
        """
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.file_path = file_path
        self.regions_extracted = regions_extracted
        self.signatures_computed = signatures_computed
        self.parsing_language = parsing_language
        self.analysis_duration_ms = analysis_duration_ms
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            'agent_id': self.agent_id,
            'file_path': self.file_path,
            'regions_extracted': self.regions_extracted,
            'signatures_computed': self.signatures_computed,
            'parsing_language': self.parsing_language,
            'analysis_duration_ms': self.analysis_duration_ms,
            'timestamp': self.timestamp
        })
        return result
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"OCCRegionAnalysis(agent={self.agent_id}, {self.file_path}, "
                f"{self.regions_extracted} regions, {self.parsing_language}, {self.analysis_duration_ms:.1f}ms)")


class OCCCacheHitObservation(Observation):
    """Emitted when OCC validation uses cached results."""
    
    def __init__(
        self,
        agent_id: str,
        cache_type: str,
        cache_key: str,
        hit_count: int,
        total_requests: int,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        """Initialize observation.
        
        Args:
            agent_id: Unique identifier for the agent
            cache_type: Type of cache (signature_cache, validation_cache, etc.)
            cache_key: Cache key that was hit
            hit_count: Number of cache hits for this key
            total_requests: Total number of requests for this key
            timestamp: Event timestamp (defaults to current time)
            **kwargs: Additional observation parameters
        """
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.cache_type = cache_type
        self.cache_key = cache_key
        self.hit_count = hit_count
        self.total_requests = total_requests
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            'agent_id': self.agent_id,
            'cache_type': self.cache_type,
            'cache_key': self.cache_key,
            'hit_count': self.hit_count,
            'total_requests': self.total_requests,
            'timestamp': self.timestamp
        })
        return result
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        hit_rate = (self.hit_count / self.total_requests * 100) if self.total_requests > 0 else 0
        return (f"OCCCacheHit(agent={self.agent_id}, {self.cache_type}, "
                f"{self.cache_key[:16]}..., hit_rate={hit_rate:.1f}%)")


# Event registry for easier access
OCC_OBSERVATION_EVENTS = {
    'tracking_started': OCCTrackingStartedObservation,
    'file_access': OCCFileAccessObservation,
    'validation_started': OCCValidationStartedObservation,
    'validation_completed': OCCValidationCompletedObservation,
    'conflict_detected': OCCConflictDetectedObservation,
    'auto_rebase_started': OCCAutoRebaseStartedObservation,
    'auto_rebase_completed': OCCAutoRebaseCompletedObservation,
    'region_analysis': OCCRegionAnalysisObservation,
    'cache_hit': OCCCacheHitObservation,
}
