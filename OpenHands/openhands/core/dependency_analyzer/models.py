"""Data models for dependency analysis using Pydantic BaseModel."""

import time
from typing import Any, Dict, List, Optional, Set, Tuple, Literal
from pydantic import BaseModel, Field
import networkx as nx

from openhands.core.config.llm_config import LLMConfig


class ImportStatement(BaseModel):
    """Represents a single import statement found in code."""
    
    module: str = Field(..., description="Module name (e.g., 'os', './utils', 'java.util.List')")
    import_type: Literal['static', 'dynamic', 'conditional'] = Field(
        ..., description="Type of import: static, dynamic, or conditional"
    )
    line_number: int = Field(..., description="Line number where import is found")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    source: Literal['ast', 'llm', 'tree_sitter'] = Field(
        ..., description="Parser that extracted this import"
    )
    resolved_path: Optional[str] = Field(
        default=None, description="Absolute file path if resolved"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"


class FileDependencies(BaseModel):
    """Represents dependencies found in a single file."""
    
    file_path: str = Field(..., description="Absolute path to the analyzed file")
    imports: List[ImportStatement] = Field(
        default_factory=list, description="List of import statements found"
    )
    resolved_paths: List[str] = Field(
        default_factory=list, description="List of absolute file paths this file depends on"
    )
    language: str = Field(..., description="Programming language (python, javascript, typescript, java)")
    file_hash: str = Field(..., description="SHA256 hash of file content for cache invalidation")
    analyzed_at: float = Field(
        default_factory=time.time, description="Unix timestamp when analysis was performed"
    )
    analysis_duration_ms: float = Field(
        default=0.0, description="Time taken to analyze this file in milliseconds"
    )
    parser_used: str = Field(
        ..., description="Primary parser used (ast, tree_sitter, llm)"
    )
    errors: List[str] = Field(
        default_factory=list, description="Any errors encountered during analysis"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"


class DependencyGraph(BaseModel):
    """Represents a dependency graph of multiple files."""
    
    nodes: Dict[str, FileDependencies] = Field(
        default_factory=dict, description="Mapping of file path to FileDependencies"
    )
    edges: List[Tuple[str, str, Dict[str, Any]]] = Field(
        default_factory=list, description="List of edges (from, to, metadata)"
    )
    cycles: List[List[str]] = Field(
        default_factory=list, description="List of cycles (each cycle is list of file paths)"
    )
    created_at: float = Field(
        default_factory=time.time, description="Unix timestamp when graph was created"
    )
    total_files: int = Field(default=0, description="Total number of files in graph")
    total_dependencies: int = Field(default=0, description="Total number of dependency relationships")
    languages: Set[str] = Field(
        default_factory=set, description="Set of languages detected in graph"
    )

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for graph algorithms."""
        graph = nx.DiGraph()
        
        # Add nodes
        for file_path, file_deps in self.nodes.items():
            graph.add_node(file_path, file_deps=file_deps, language=file_deps.language)
        
        # Add edges
        for from_path, to_path, metadata in self.edges:
            graph.add_edge(from_path, to_path, **metadata)
        
        return graph
    
    def get_transitive_dependencies(self, file_path: str) -> List[str]:
        """Get all transitive dependencies using BFS."""
        if file_path not in self.nodes:
            return []
        
        graph = self.to_networkx()
        try:
            return list(nx.descendants(graph, file_path))
        except nx.NetworkXError:
            return []
    
    def get_dependents(self, file_path: str) -> List[str]:
        """Get files that depend on this file (reverse dependencies)."""
        if file_path not in self.nodes:
            return []
        
        graph = self.to_networkx()
        try:
            return list(nx.ancestors(graph, file_path))
        except nx.NetworkXError:
            return []
    
    def has_cycle(self, file_path: str) -> bool:
        """Check if file is part of any cycle."""
        for cycle in self.cycles:
            if file_path in cycle:
                return True
        return False

    class Config:
        """Pydantic configuration."""
        extra = "forbid"


class DependencyAnalysisResult(BaseModel):
    """Result of dependency analysis operation."""
    
    success: bool = Field(..., description="Whether analysis was successful")
    file_dependencies: Optional[FileDependencies] = Field(
        default=None, description="File dependencies if analysis succeeded"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if analysis failed"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warning messages from analysis"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"


class DependencyAnalyzerConfig(BaseModel):
    """Configuration for DependencyAnalyzer."""
    
    use_llm_fallback: bool = Field(
        default=True, description="Use LLM as fallback when AST parsing fails"
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None, description="LLM configuration for fallback analysis"
    )
    cache_backend: Literal['memory', 'redis'] = Field(
        default='memory', description="Cache backend to use"
    )
    redis_url: Optional[str] = Field(
        default=None, description="Redis connection URL if using Redis cache"
    )
    cache_ttl_seconds: int = Field(
        default=3600, description="Cache TTL in seconds (1 hour default)"
    )
    max_file_size_bytes: int = Field(
        default=1_000_000, description="Maximum file size to analyze (1MB default)"
    )
    parallel_analysis: bool = Field(
        default=True, description="Enable parallel analysis for workspace analysis"
    )
    max_workers: int = Field(
        default=10, description="Maximum worker threads for parallel analysis"
    )
    skip_standard_libraries: bool = Field(
        default=True, description="Skip standard library imports"
    )
    skip_third_party_packages: bool = Field(
        default=True, description="Skip third-party package imports"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"
