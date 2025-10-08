"""Dependency graph construction and analysis using NetworkX."""

import asyncio
import logging
import time
from typing import Dict, List, Set, Optional, Tuple, Any

import networkx as nx

from .models import FileDependencies, DependencyGraph
from .exceptions import GraphBuildError


class DependencyGraphBuilder:
    """Builds and analyzes dependency graphs using NetworkX."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize graph builder.
        
        Args:
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self._graph: Optional[nx.DiGraph] = None

    async def build_graph(
        self, 
        file_dependencies: Dict[str, FileDependencies]
    ) -> DependencyGraph:
        """Build dependency graph from analyzed files.
        
        Args:
            file_dependencies: Mapping of file path to FileDependencies
            
        Returns:
            DependencyGraph object
        """
        try:
            self.logger.debug(f"Building dependency graph for {len(file_dependencies)} files")
            start_time = time.time()
            
            # Create new NetworkX DiGraph
            self._graph = nx.DiGraph()
            
            # Add nodes (files) to graph
            for file_path, file_deps in file_dependencies.items():
                self._add_file_node(file_path, file_deps)
            
            # Add edges (dependencies) to graph
            for file_path, file_deps in file_dependencies.items():
                self._add_dependency_edges(file_path, file_deps)
            
            # Detect cycles
            cycles = self._detect_cycles()
            
            # Build DependencyGraph model
            dependency_graph = await self._create_dependency_graph(
                file_dependencies, cycles
            )
            
            build_time = time.time() - start_time
            self.logger.info(
                f"Built dependency graph: {dependency_graph.total_files} files, "
                f"{dependency_graph.total_dependencies} dependencies, "
                f"{len(cycles)} cycles detected in {build_time:.2f}s"
            )
            
            return dependency_graph
            
        except Exception as e:
            self.logger.error(f"Failed to build dependency graph: {e}")
            raise GraphBuildError(f"Graph construction failed: {e}")

    def _add_file_node(self, file_path: str, file_deps: FileDependencies) -> None:
        """Add file to graph as node.
        
        Args:
            file_path: Absolute file path
            file_deps: FileDependencies object
        """
        if not self._graph:
            return
        
        # Add node with metadata
        self._graph.add_node(
            file_path,
            file_deps=file_deps,
            language=file_deps.language,
            file_hash=file_deps.file_hash,
            import_count=len(file_deps.imports)
        )

    def _add_dependency_edges(
        self, 
        file_path: str, 
        file_deps: FileDependencies
    ) -> None:
        """Add edges for file dependencies.
        
        Args:
            file_path: Source file path
            file_deps: FileDependencies object
        """
        if not self._graph:
            return
        
        for resolved_path in file_deps.resolved_paths:
            if resolved_path and resolved_path != file_path:
                # Find corresponding import statement
                import_stmt = None
                for stmt in file_deps.imports:
                    if stmt.resolved_path == resolved_path:
                        import_stmt = stmt
                        break
                
                # Create edge metadata
                edge_metadata = {
                    'import_type': import_stmt.import_type if import_stmt else 'unknown',
                    'confidence': import_stmt.confidence if import_stmt else 0.5,
                    'source': import_stmt.source if import_stmt else 'unknown',
                    'line_number': import_stmt.line_number if import_stmt else 0,
                    'module': import_stmt.module if import_stmt else 'unknown'
                }
                
                # Add edge from source to dependency
                self._graph.add_edge(file_path, resolved_path, **edge_metadata)

    def _detect_cycles(self) -> List[List[str]]:
        """Detect cycles using NetworkX algorithms.
        
        Returns:
            List of cycles (each cycle is list of file paths)
        """
        if not self._graph:
            return []
        
        try:
            # Find strongly connected components with more than one node
            sccs = list(nx.strongly_connected_components(self._graph))
            cycles = [list(scc) for scc in sccs if len(scc) > 1]
            
            if cycles:
                self.logger.warning(f"Detected {len(cycles)} cycles in dependency graph")
                for i, cycle in enumerate(cycles):
                    self.logger.debug(f"Cycle {i+1}: {' -> '.join(cycle)}")
            
            return cycles
            
        except Exception as e:
            self.logger.warning(f"Error detecting cycles: {e}")
            return []

    async def _create_dependency_graph(
        self,
        file_dependencies: Dict[str, FileDependencies],
        cycles: List[List[str]]
    ) -> DependencyGraph:
        """Create DependencyGraph model from NetworkX graph.
        
        Args:
            file_dependencies: Original file dependencies
            cycles: Detected cycles
            
        Returns:
            DependencyGraph object
        """
        if not self._graph:
            raise GraphBuildError("No graph to convert")
        
        # Extract edges with metadata
        edges = []
        for from_node, to_node, edge_data in self._graph.edges(data=True):
            edges.append((from_node, to_node, dict(edge_data)))
        
        # Collect languages
        languages = set()
        for file_deps in file_dependencies.values():
            languages.add(file_deps.language)
        
        return DependencyGraph(
            nodes=file_dependencies,
            edges=edges,
            cycles=cycles,
            total_files=len(file_dependencies),
            total_dependencies=len(edges),
            languages=languages
        )

    def get_transitive_dependencies(self, file_path: str) -> List[str]:
        """Get all transitive dependencies using BFS.
        
        Args:
            file_path: Source file path
            
        Returns:
            List of file paths in dependency order
        """
        if not self._graph or file_path not in self._graph:
            return []
        
        try:
            descendants = nx.descendants(self._graph, file_path)
            return list(descendants)
        except nx.NetworkXError as e:
            self.logger.warning(f"Error getting transitive dependencies for {file_path}: {e}")
            return []

    def get_dependents(self, file_path: str) -> List[str]:
        """Get files that depend on this file (reverse dependencies).
        
        Args:
            file_path: Target file path
            
        Returns:
            List of file paths that depend on the target
        """
        if not self._graph or file_path not in self._graph:
            return []
        
        try:
            ancestors = nx.ancestors(self._graph, file_path)
            return list(ancestors)
        except nx.NetworkXError as e:
            self.logger.warning(f"Error getting dependents for {file_path}: {e}")
            return []

    def get_lock_set(self, file_path: str) -> Set[str]:
        """Get all files that should be locked when locking this file.
        
        This includes:
        - The file itself
        - All transitive dependencies
        - If file is in a cycle, all files in the cycle
        
        Args:
            file_path: Source file path
            
        Returns:
            Set of file paths to lock
        """
        if not self._graph:
            return {file_path}
        
        lock_set = {file_path}
        
        try:
            # Add all transitive dependencies
            transitive_deps = self.get_transitive_dependencies(file_path)
            lock_set.update(transitive_deps)
            
            # If file is in a cycle, add all files in the cycle
            cycle = self._get_cycle_containing_file(file_path)
            if cycle:
                lock_set.update(cycle)
            
            return lock_set
            
        except Exception as e:
            self.logger.warning(f"Error computing lock set for {file_path}: {e}")
            return {file_path}

    def _get_cycle_containing_file(self, file_path: str) -> Optional[List[str]]:
        """Get the cycle that contains the given file.
        
        Args:
            file_path: File path to check
            
        Returns:
            List of files in cycle, or None if file is not in a cycle
        """
        if not self._graph:
            return None
        
        try:
            # Find strongly connected component containing the file
            for scc in nx.strongly_connected_components(self._graph):
                if file_path in scc and len(scc) > 1:
                    return list(scc)
            return None
            
        except Exception as e:
            self.logger.warning(f"Error finding cycle for {file_path}: {e}")
            return None

    def get_topological_order(self) -> List[str]:
        """Get topological ordering of files for lock acquisition.
        
        Returns:
            List of file paths in topological order.
            If graph has cycles, returns alphabetical order.
        """
        if not self._graph:
            return []
        
        try:
            # Try topological sort
            return list(nx.topological_sort(self._graph))
            
        except nx.NetworkXError:
            # Graph has cycles, return alphabetical order
            self.logger.debug("Graph has cycles, using alphabetical order for locking")
            return sorted(self._graph.nodes())

    async def add_file(self, file_deps: FileDependencies) -> None:
        """Add file to existing graph.
        
        Args:
            file_deps: FileDependencies to add
        """
        if not self._graph:
            self._graph = nx.DiGraph()
        
        file_path = file_deps.file_path
        
        # Add/update node
        self._add_file_node(file_path, file_deps)
        
        # Add/update edges
        self._add_dependency_edges(file_path, file_deps)
        
        self.logger.debug(f"Added/updated file in graph: {file_path}")

    async def remove_file(self, file_path: str) -> None:
        """Remove file from graph.
        
        Args:
            file_path: File path to remove
        """
        if not self._graph or file_path not in self._graph:
            return
        
        # Remove node (this automatically removes all connected edges)
        self._graph.remove_node(file_path)
        
        self.logger.debug(f"Removed file from graph: {file_path}")

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the current graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if not self._graph:
            return {
                'nodes': 0,
                'edges': 0,
                'density': 0.0,
                'is_dag': True,
                'cycles': 0,
                'components': 0
            }
        
        try:
            # Basic stats
            num_nodes = self._graph.number_of_nodes()
            num_edges = self._graph.number_of_edges()
            density = nx.density(self._graph) if num_nodes > 0 else 0.0
            
            # Check if it's a DAG (Directed Acyclic Graph)
            is_dag = nx.is_directed_acyclic_graph(self._graph)
            
            # Count cycles
            cycles = len([scc for scc in nx.strongly_connected_components(self._graph) if len(scc) > 1])
            
            # Count connected components
            components = nx.number_weakly_connected_components(self._graph)
            
            return {
                'nodes': num_nodes,
                'edges': num_edges,
                'density': density,
                'is_dag': is_dag,
                'cycles': cycles,
                'components': components
            }
            
        except Exception as e:
            self.logger.warning(f"Error computing graph stats: {e}")
            return {
                'nodes': 0,
                'edges': 0,
                'density': 0.0,
                'is_dag': True,
                'cycles': 0,
                'components': 0,
                'error': str(e)
            }
