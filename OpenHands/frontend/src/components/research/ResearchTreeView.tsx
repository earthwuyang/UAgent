/**
 * Research Tree View Component
 *
 * ReactFlow-based visualization of the research tree with enhanced controls and filters.
 */

import React, { useMemo, useCallback, useState, useRef, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  BackgroundVariant,
  NodeTypes,
  useReactFlow,
  MarkerType,
  Position,
  MiniMap,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from 'dagre';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Filter,
  Maximize2,
  Search,
  TreePine,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { useResearchTreeStore, ResearchNode as ResearchNodeType } from '#/state/research-tree-store';
import { ResearchNode } from './ResearchNode';
import { Loader } from '#/components/shared/loader';
import { cn } from '#/utils/utils';

const nodeTypes: NodeTypes = {
  researchNode: ResearchNode,
};

// Stable layout function - uses JSON.stringify to create stable cache keys
const layoutCache = new Map<string, { nodes: Node[]; edges: Edge[] }>();

const getLayoutedElements = (nodes: Node[], edges: Edge[]) => {
  // Create a stable cache key based on node IDs and positions
  const cacheKey = JSON.stringify({
    nodeIds: nodes.map(n => n.id).sort(),
    edgeIds: edges.map(e => `${e.source}-${e.target}`).sort(),
  });

  // Return cached result if available
  if (layoutCache.has(cacheKey)) {
    const cached = layoutCache.get(cacheKey)!;
    // Return nodes and edges with updated data but same positions
    return {
      nodes: nodes.map((node, idx) => ({
        ...node,
        position: cached.nodes[idx]?.position || node.position,
        targetPosition: Position.Top,
        sourcePosition: Position.Bottom,
      })),
      edges,
    };
  }

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: 'TB', ranksep: 120, nodesep: 100 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: 280, height: 220 });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - 140,
        y: nodeWithPosition.y - 110,
      },
      targetPosition: Position.Top,
      sourcePosition: Position.Bottom,
    };
  });

  const result = { nodes: layoutedNodes, edges };
  
  // Cache the result (limit cache size)
  if (layoutCache.size > 50) {
    const firstKey = layoutCache.keys().next().value;
    layoutCache.delete(firstKey);
  }
  layoutCache.set(cacheKey, result);

  return result;
};

const filterOptions = {
  type: [
    { value: 'root', label: 'Root' },
    { value: 'idea', label: 'Idea' },
    { value: 'hypothesis', label: 'Hypothesis' },
    { value: 'plan', label: 'Plan' },
    { value: 'web_search', label: 'Web Search' },
    { value: 'code_search', label: 'Code Search' },
    { value: 'experiment', label: 'Experiment' },
    { value: 'result', label: 'Result' },
  ],
  status: [
    { value: 'pending', label: 'Pending' },
    { value: 'running', label: 'Running' },
    { value: 'complete', label: 'Complete' },
    { value: 'failed', label: 'Failed' },
    { value: 'cancelled', label: 'Cancelled' },
  ],
};

export function ResearchTreeView() {
  const [showFilterPanel, setShowFilterPanel] = useState(false);
  const [showMiniMap, setShowMiniMap] = useState(true);

  // Use separate store selectors to avoid creating new objects on every render
  const storeNodes = useResearchTreeStore((state) => state.nodes);
  const storeEdges = useResearchTreeStore((state) => state.edges);
  const selectedNodeId = useResearchTreeStore((state) => state.selectedNodeId);
  const selectNode = useResearchTreeStore((state) => state.selectNode);
  const filterType = useResearchTreeStore((state) => state.filterType);
  const filterStatus = useResearchTreeStore((state) => state.filterStatus);
  const searchQuery = useResearchTreeStore((state) => state.searchQuery);
  const setFilterType = useResearchTreeStore((state) => state.setFilterType);
  const setFilterStatus = useResearchTreeStore((state) => state.setFilterStatus);
  const setSearchQuery = useResearchTreeStore((state) => state.setSearchQuery);
  const clearFilters = useResearchTreeStore((state) => state.clearFilters);
  const isLoading = useResearchTreeStore((state) => state.isLoading);

  const hasFilters = Boolean(filterType || filterStatus || searchQuery.trim());

  // Memoize filtered nodes to prevent infinite re-renders
  // This prevents getFilteredNodes() from returning a new array reference on every render
  const filteredNodes = useMemo(() => {
    if (!hasFilters) {
      return Array.from(storeNodes.values());
    }
    return Array.from(storeNodes.values()).filter((node) => {
      const typeMatch = !filterType || node.type === filterType;
      const statusMatch = !filterStatus || node.status === filterStatus;
      const searchMatch = !searchQuery.trim() || 
        node.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        node.description?.toLowerCase().includes(searchQuery.toLowerCase());
      return typeMatch && statusMatch && searchMatch;
    });
  }, [storeNodes, filterType, filterStatus, searchQuery, hasFilters]);

  const filteredIds = useMemo(() => new Set(filteredNodes.map((node) => node.id)), [filteredNodes]);
  const totalNodes = storeNodes.size;

  const { fitView, zoomIn, zoomOut } = useReactFlow();

  // Memoize the raw flow nodes and edges to prevent unnecessary re-layouts
  const { rawNodes, rawEdges } = useMemo(() => {
    const flowNodes: Node[] = Array.from(storeNodes.values()).map((node) => {
      const isMatch = filteredIds.has(node.id);
      const isFilteredOut = hasFilters && !isMatch;
      const nodeData: ResearchNodeType = {
        ...node,
        isFilteredMatch: isMatch,
        isFilteredOut,
      };

      return {
        id: node.id,
        type: 'researchNode',
        data: nodeData,
        position: { x: 0, y: 0 },
        selected: node.id === selectedNodeId,
        className: cn(
          'research-node-wrapper',
          isFilteredOut && 'filtered-out'
        ),
        style: hasFilters && !isMatch ? { opacity: 0.3 } : undefined,
      };
    });

    const flowEdges: Edge[] = storeEdges.map((edge, index) => {
      const edgeMatches = filteredIds.has(edge.parent_id) && filteredIds.has(edge.child_id);
      return {
        id: `edge-${edge.parent_id}-${edge.child_id}-${index}`,
        source: edge.parent_id,
        target: edge.child_id,
        type: 'smoothstep',
        animated: false,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 18,
          height: 18,
        },
        style: {
          strokeWidth: 2,
          stroke: edgeMatches ? '#94a3b8' : '#475569',
          opacity: hasFilters ? (edgeMatches ? 1 : 0.15) : 0.9,
        },
      };
    });

    return { rawNodes: flowNodes, rawEdges: flowEdges };
  }, [storeNodes, storeEdges, selectedNodeId, filteredIds, hasFilters]);

  // Apply layout only when structure changes
  const { nodes, edges } = useMemo(() => {
    return getLayoutedElements(rawNodes, rawEdges);
  }, [rawNodes, rawEdges]);

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  const handleClearFilters = useCallback(() => {
    clearFilters();
  }, [clearFilters]);

  const handleFitView = useCallback(() => {
    fitView({ padding: 0.2, duration: 400 });
  }, [fitView]);

  // Auto-fit only once when nodes first load
  const hasAutoFitted = useRef(false);
  useEffect(() => {
    if (nodes.length > 0 && !hasAutoFitted.current) {
      const timeout = window.setTimeout(() => {
        fitView({ padding: 0.2, duration: 500 });
        hasAutoFitted.current = true;
      }, 150);
      return () => window.clearTimeout(timeout);
    }
    return undefined;
  }, [nodes.length, fitView]);

  return (
    <div className="research-tree-view">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView={false}
        minZoom={0.1}
        maxZoom={2}
        defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={true}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#47556920" />

        <Panel position="bottom-right" className="research-control-panel">
          <button
            type="button"
            className="research-control-button"
            title="Zoom in"
            onClick={() => zoomIn()}
          >
            <ZoomIn size={16} />
          </button>
          <button
            type="button"
            className="research-control-button"
            title="Zoom out"
            onClick={() => zoomOut()}
          >
            <ZoomOut size={16} />
          </button>
          <button
            type="button"
            className="research-control-button"
            title="Fit view"
            onClick={handleFitView}
          >
            <Maximize2 size={16} />
          </button>
          <button
            type="button"
            className={cn('research-control-button', showMiniMap && 'active')}
            title={showMiniMap ? 'Hide overview' : 'Show overview'}
            onClick={() => setShowMiniMap((prev) => !prev)}
          >
            <TreePine size={16} />
          </button>
          <button
            type="button"
            className={cn('research-control-button', showFilterPanel && 'active')}
            title="Filters"
            onClick={() => setShowFilterPanel((prev) => !prev)}
          >
            <Filter size={16} />
          </button>
        </Panel>

        {showFilterPanel && (
          <Panel position="top-right" className="research-filter-panel">
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="filter-panel-inner"
            >
                <header className="filter-panel-header">
                  <h4>Filters</h4>
                  <button type="button" onClick={() => setShowFilterPanel(false)}>
                    <X size={14} />
                  </button>
                </header>

                <div className="filter-panel-body">
                  <label className="filter-field">
                    <span>Type</span>
                    <select
                      className="research-filter-dropdown"
                      value={filterType ?? ''}
                      onChange={(event) =>
                        setFilterType(event.target.value ? event.target.value : null)
                      }
                    >
                      <option value="">All types</option>
                      {filterOptions.type.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="filter-field">
                    <span>Status</span>
                    <select
                      className="research-filter-dropdown"
                      value={filterStatus ?? ''}
                      onChange={(event) =>
                        setFilterStatus(event.target.value ? event.target.value : null)
                      }
                    >
                      <option value="">All statuses</option>
                      {filterOptions.status.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="filter-field">
                    <span>Search</span>
                    <input
                      type="search"
                      className="research-filter-dropdown"
                      placeholder="Find by title or content"
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                    />
                  </label>
                </div>

                <footer className="filter-panel-footer">
                  <button type="button" className="clear-button" onClick={handleClearFilters}>
                    Clear filters
                  </button>
                </footer>
            </motion.div>
          </Panel>
        )}

        {showMiniMap && (
          <MiniMap
            className="research-tree-minimap"
            nodeColor={(node) => {
              const data = node.data as ResearchNodeType;
              switch (data.status) {
                case 'running':
                  return '#3b82f6';
                case 'complete':
                  return '#10b981';
                case 'failed':
                  return '#ef4444';
                default:
                  return '#64748b';
              }
            }}
            nodeStrokeWidth={3}
            pannable
            zoomable
          />
        )}
      </ReactFlow>

      <AnimatePresence>
        {isLoading && (
          <motion.div
            className="research-tree-loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <Loader size="large" />
            <p>Loading research tree...</p>
          </motion.div>
        )}
      </AnimatePresence>

      {!isLoading && totalNodes === 0 && (
        <div
          className="research-tree-empty"
          role="status"
          onClick={(event) => event.stopPropagation()}
        >
          <TreePine size={32} className="icon" />
          <h4>No Research Data Yet</h4>
          <p>Start a research session to see the tree visualization.</p>
        </div>
      )}

      {!isLoading && totalNodes > 0 && filteredNodes.length === 0 && (
        <div
          className="research-tree-no-results"
          role="status"
          onClick={(event) => event.stopPropagation()}
        >
          <Search size={28} className="icon" />
          <h4>No nodes match your filters</h4>
          <p>Adjust your criteria or clear filters to see all nodes again.</p>
          <button type="button" className="clear-button" onClick={handleClearFilters}>
            Clear filters
          </button>
        </div>
      )}
    </div>
  );
}
