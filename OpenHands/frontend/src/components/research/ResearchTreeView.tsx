/**
 * Research Tree View Component
 *
 * ReactFlow-based visualization of the research tree.
 * Displays nodes and edges with automatic layout.
 */

import React, { useMemo, useCallback, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  BackgroundVariant,
  NodeTypes,
  useReactFlow,
  MarkerType,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useResearchTreeStore } from '#/state/research-tree-store';
import { ResearchNode } from './ResearchNode';
import dagre from 'dagre';

const nodeTypes: NodeTypes = {
  researchNode: ResearchNode,
};

// Dagre layout algorithm
const getLayoutedElements = (nodes: Node[], edges: Edge[]) => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: 'TB', ranksep: 100, nodesep: 80 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: 250, height: 180 });
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
        x: nodeWithPosition.x - 125, // Center node (width/2)
        y: nodeWithPosition.y - 90, // Center node (height/2)
      },
      targetPosition: Position.Top,
      sourcePosition: Position.Bottom,
    };
  });

  return { nodes: layoutedNodes, edges };
};

export function ResearchTreeView() {
  const {
    nodes: storeNodes,
    edges: storeEdges,
    selectedNodeId,
    selectNode,
  } = useResearchTreeStore();

  const { fitView } = useReactFlow();

  // Convert store nodes to ReactFlow nodes
  const { nodes, edges } = useMemo(() => {
    const flowNodes: Node[] = Array.from(storeNodes.values()).map((node) => ({
      id: node.id,
      type: 'researchNode',
      data: node,
      position: { x: 0, y: 0 }, // Will be set by layout
      selected: node.id === selectedNodeId,
    }));

    const flowEdges: Edge[] = storeEdges.map((edge, index) => ({
      id: `edge-${edge.parent_id}-${edge.child_id}-${index}`,
      source: edge.parent_id,
      target: edge.child_id,
      type: 'smoothstep',
      animated: false,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 20,
        height: 20,
      },
      style: {
        strokeWidth: 2,
        stroke: '#64748b',
      },
    }));

    // Apply layout
    return getLayoutedElements(flowNodes, flowEdges);
  }, [storeNodes, storeEdges, selectedNodeId]);

  // Handle node click
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  // Handle pane click (deselect)
  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  // Fit view when nodes change
  useEffect(() => {
    if (nodes.length > 0) {
      // Delay to allow nodes to render
      setTimeout(() => {
        fitView({ padding: 0.2, duration: 500 });
      }, 100);
    }
  }, [nodes.length, fitView]);

  return (
    <div className="research-tree-view">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView
        minZoom={0.1}
        maxZoom={2}
        defaultEdgeOptions={{
          type: 'smoothstep',
          animated: false,
        }}
      >
        <Controls
          showInteractive={false}
          position="bottom-right"
        />
        <Background
          variant={BackgroundVariant.Dots}
          gap={16}
          size={1}
          color="#e5e7eb"
        />
      </ReactFlow>
    </div>
  );
}
