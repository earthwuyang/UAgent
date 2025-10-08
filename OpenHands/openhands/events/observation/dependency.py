"""Custom observation classes for dependency analysis events."""

from dataclasses import dataclass
from typing import List

from openhands.events.observation.observation import Observation
from openhands.core.schema.observation import ObservationType


@dataclass
class DependencyAnalysisStartedObservation(Observation):
    """Emitted when dependency analysis starts."""
    
    file_path: str
    language: str
    observation: str = ObservationType.DEPENDENCY_ANALYSIS
    
    @property
    def message(self) -> str:
        return f"Starting dependency analysis for {self.file_path} ({self.language})"


@dataclass
class DependencyAnalysisCompletedObservation(Observation):
    """Emitted when dependency analysis completes."""
    
    file_path: str
    dependencies_found: int
    analysis_duration_ms: float
    parser_used: str
    observation: str = ObservationType.DEPENDENCY_ANALYSIS
    
    @property
    def message(self) -> str:
        return (
            f"Dependency analysis complete for {self.file_path}: "
            f"{self.dependencies_found} dependencies found using {self.parser_used} "
            f"in {self.analysis_duration_ms:.1f}ms"
        )


@dataclass
class DependencyGraphUpdatedObservation(Observation):
    """Emitted when dependency graph is updated."""
    
    total_files: int
    total_dependencies: int
    cycles_detected: int
    languages: List[str]
    observation: str = ObservationType.DEPENDENCY_GRAPH
    
    @property
    def message(self) -> str:
        langs = ", ".join(self.languages) if self.languages else "none"
        return (
            f"Dependency graph updated: {self.total_files} files, "
            f"{self.total_dependencies} dependencies, {self.cycles_detected} cycles "
            f"detected. Languages: {langs}"
        )


@dataclass
class DependencyGraphInvalidatedObservation(Observation):
    """Emitted when dependency graph cache is invalidated."""
    
    file_path: str
    reason: str
    observation: str = ObservationType.DEPENDENCY_GRAPH
    
    @property
    def message(self) -> str:
        return f"Dependency graph invalidated for {self.file_path}: {self.reason}"


@dataclass
class DependencyLockSetComputedObservation(Observation):
    """Emitted when lock set is computed for a file."""
    
    file_path: str
    lock_set_size: int
    includes_cycles: bool
    computation_time_ms: float
    observation: str = ObservationType.DEPENDENCY_GRAPH
    
    @property
    def message(self) -> str:
        cycle_info = " (includes cycles)" if self.includes_cycles else ""
        return (
            f"Lock set computed for {self.file_path}: {self.lock_set_size} files to lock{cycle_info} "
            f"in {self.computation_time_ms:.1f}ms"
        )
