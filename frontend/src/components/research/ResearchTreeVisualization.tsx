/**
 * ResearchTreeVisualization Component
 * ReactFlow-based tree visualization for research process with detailed sidebar
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  NodeTypes,
  ReactFlowProvider,
  useReactFlow,
  ControlButton,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Research process node interfaces
interface ResearchNode {
  id: string;
  type: 'engine' | 'step' | 'result';
  engine: string;
  phase: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  title: string;
  description?: string;
  timestamp?: string;
  progress?: number;
  metadata?: any;
  parent_id?: string;
  children?: string[];
}

interface ProgressEvent {
  event_type: string;
  session_id: string;
  timestamp: string;
  data: {
    engine: string;
    phase?: string;
    metadata?: any;
  };
  source: string;
  progress_percentage?: number;
  message?: string;
}

// Research Node Component
const ResearchNodeComponent: React.FC<{ data: { node: ResearchNode, isSelected?: boolean } }> = ({ data }) => {
  const { node, isSelected } = data;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-50 border-green-200 text-green-800';
      case 'running': return 'bg-blue-50 border-blue-200 text-blue-800';
      case 'error': return 'bg-red-50 border-red-200 text-red-800';
      case 'pending': return 'bg-gray-50 border-gray-200 text-gray-800';
      default: return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✅';
      case 'running': return '🔄';
      case 'error': return '❌';
      case 'pending': return '⏳';
      default: return '⚪';
    }
  };

  const getEngineIcon = (engine: string) => {
    switch (engine) {
      case 'deep_research': return '🔍';
      case 'code_research': return '💻';
      case 'scientific_research': return '🧪';
      case 'smart_router': return '🎯';
      default: return '⚙️';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'engine': return '🏭';
      case 'step': return '📝';
      case 'result': return '📊';
      default: return '📋';
    }
  };

  return (
    <div className={`
      min-w-[250px] max-w-[300px] p-3 rounded-lg border-2 shadow-md transition-all duration-300
      ${getStatusColor(node.status)}
      ${isSelected ? 'ring-2 ring-blue-500 scale-105 shadow-lg' : 'hover:shadow-lg hover:scale-[1.02]'}
      ${node.status === 'running' ? 'animate-pulse' : ''}
    `}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <span className="text-lg">
            {getEngineIcon(node.engine)}
          </span>
          <span className="text-sm">
            {getTypeIcon(node.type)}
          </span>
          <div className="px-2 py-1 bg-white bg-opacity-70 rounded text-xs font-medium">
            {node.engine.replace('_', ' ').toUpperCase()}
          </div>
        </div>
        <div className="flex items-center space-x-1">
          <span className="text-lg">{getStatusIcon(node.status)}</span>
          <div className="text-xs font-medium">
            {node.status.toUpperCase()}
          </div>
        </div>
      </div>

      {/* Phase */}
      <div className="text-xs text-gray-600 bg-white bg-opacity-50 rounded px-2 py-1 mb-2">
        📋 {node.phase || 'General'}
      </div>

      {/* Title */}
      <div className="text-sm font-semibold leading-tight mb-2">
        {node.title.length > 80 ? node.title.substring(0, 80) + '...' : node.title}
      </div>

      {/* Description */}
      {node.description && (
        <div className="text-xs text-gray-600 border-t border-gray-300 pt-2 mt-2">
          {node.description.length > 60 ? node.description.substring(0, 60) + '...' : node.description}
        </div>
      )}

      {/* Progress Bar */}
      {node.progress !== undefined && (
        <div className="mt-2">
          <div className="flex justify-between text-xs mb-1">
            <span>Progress</span>
            <span>{Math.round(node.progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${node.progress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Timestamp */}
      {node.timestamp && (
        <div className="text-xs text-gray-500 mt-1">
          ⏰ {new Date(node.timestamp).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

// Detailed Sidebar Component
const DetailedSidebar: React.FC<{
  selectedNode: ResearchNode | null;
  onClose: () => void;
  events: ProgressEvent[];
}> = ({ selectedNode, onClose, events }) => {
  const nodeEvents = events.filter(event =>
    selectedNode && event.data.engine === selectedNode.engine
  );

  if (!selectedNode) return null;

  return (
    <div className="fixed right-0 top-0 h-full w-80 bg-white shadow-xl border-l border-gray-200 z-50 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="text-lg font-bold">Node Details</h3>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 text-xl font-bold"
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Basic Info */}
        <div className="space-y-2">
          <div>
            <label className="text-sm font-medium text-gray-600">Engine</label>
            <div className="text-sm">{selectedNode.engine.replace('_', ' ').toUpperCase()}</div>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-600">Type</label>
            <div className="text-sm">{selectedNode.type.toUpperCase()}</div>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-600">Phase</label>
            <div className="text-sm">{selectedNode.phase || 'General'}</div>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-600">Status</label>
            <div className="text-sm">{selectedNode.status.toUpperCase()}</div>
          </div>
        </div>

        {/* Title & Description */}
        <div className="space-y-2">
          <div>
            <label className="text-sm font-medium text-gray-600">Title</label>
            <div className="text-sm">{selectedNode.title}</div>
          </div>
          {selectedNode.description && (
            <div>
              <label className="text-sm font-medium text-gray-600">Description</label>
              <div className="text-sm">{selectedNode.description}</div>
            </div>
          )}
        </div>

        {/* Progress */}
        {selectedNode.progress !== undefined && (
          <div>
            <label className="text-sm font-medium text-gray-600">Progress</label>
            <div className="mt-1">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${selectedNode.progress}%` }}
                ></div>
              </div>
              <div className="text-sm text-gray-600 mt-1">{Math.round(selectedNode.progress)}%</div>
            </div>
          </div>
        )}

        {/* Metadata */}
        {selectedNode.metadata && Object.keys(selectedNode.metadata).length > 0 && (
          <div>
            <label className="text-sm font-medium text-gray-600">Metadata</label>
            <div className="text-xs bg-gray-50 p-2 rounded border mt-1">
              <pre>{JSON.stringify(selectedNode.metadata, null, 2)}</pre>
            </div>
          </div>
        )}

        {/* Related Events */}
        <div>
          <label className="text-sm font-medium text-gray-600">Recent Events ({nodeEvents.length})</label>
          <div className="space-y-2 mt-2 max-h-40 overflow-y-auto">
            {nodeEvents.slice(-5).map((event, index) => (
              <div key={index} className="text-xs bg-gray-50 p-2 rounded border">
                <div className="font-medium">{event.event_type}</div>
                {event.message && <div className="text-gray-600">{event.message}</div>}
                <div className="text-gray-500 mt-1">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Timestamp */}
        {selectedNode.timestamp && (
          <div>
            <label className="text-sm font-medium text-gray-600">Created</label>
            <div className="text-sm">{new Date(selectedNode.timestamp).toLocaleString()}</div>
          </div>
        )}
      </div>
    </div>
  );
};

const nodeTypes: NodeTypes = {
  researchNode: ResearchNodeComponent,
};

// Convert research progress to tree nodes
function convertToResearchNodes(events: ProgressEvent[]): ResearchNode[] {
  const nodes: ResearchNode[] = [];
  const engineNodes = new Set<string>();

  events.forEach((event, index) => {
    const nodeId = `${event.data.engine}-${event.data.phase || 'main'}-${index}`;

    // Create engine node if not exists
    const engineId = `engine-${event.data.engine}`;
    if (!engineNodes.has(engineId)) {
      nodes.push({
        id: engineId,
        type: 'engine',
        engine: event.data.engine,
        phase: 'main',
        status: event.event_type === 'research_completed' ? 'completed' :
               event.event_type === 'research_error' ? 'error' : 'running',
        title: `${event.data.engine.replace('_', ' ').toUpperCase()} Engine`,
        description: `Main ${event.data.engine} research engine`,
        timestamp: event.timestamp,
        progress: event.progress_percentage
      });
      engineNodes.add(engineId);
    }

    // Create step node
    nodes.push({
      id: nodeId,
      type: 'step',
      engine: event.data.engine,
      phase: event.data.phase || 'processing',
      status: event.event_type === 'research_completed' ? 'completed' :
             event.event_type === 'research_error' ? 'error' : 'running',
      title: event.message || `${event.event_type.replace('_', ' ').toUpperCase()}`,
      description: event.data.phase ? `Phase: ${event.data.phase}` : undefined,
      timestamp: event.timestamp,
      progress: event.progress_percentage,
      metadata: event.data.metadata,
      parent_id: engineId
    });
  });

  return nodes;
}

// Convert to ReactFlow format
function convertToFlowNodes(researchNodes: ResearchNode[]): Node[] {
  const engines = researchNodes.filter(n => n.type === 'engine');
  const steps = researchNodes.filter(n => n.type === 'step');

  const nodes: Node[] = [];

  // Position engines
  engines.forEach((engine, index) => {
    nodes.push({
      id: engine.id,
      type: 'researchNode',
      position: { x: index * 350, y: 0 },
      data: { node: engine, isSelected: false }
    });

    // Position steps under each engine
    const engineSteps = steps.filter(s => s.parent_id === engine.id);
    engineSteps.forEach((step, stepIndex) => {
      nodes.push({
        id: step.id,
        type: 'researchNode',
        position: { x: index * 350, y: (stepIndex + 1) * 150 },
        data: { node: step, isSelected: false }
      });
    });
  });

  return nodes;
}

function convertToFlowEdges(researchNodes: ResearchNode[]): Edge[] {
  const edges: Edge[] = [];

  researchNodes.forEach(node => {
    if (node.parent_id) {
      edges.push({
        id: `${node.parent_id}-${node.id}`,
        source: node.parent_id,
        target: node.id,
        type: 'smoothstep',
        animated: node.status === 'running'
      });
    }
  });

  return edges;
}

interface ResearchTreeVisualizationProps {
  events?: ProgressEvent[];
  sessionId?: string;
  selectedNodeId?: string;
  onNodeSelect?: (nodeId: string) => void;
}

const FlowContent: React.FC<ResearchTreeVisualizationProps> = ({
  events: propEvents,
  sessionId,
  selectedNodeId,
  onNodeSelect
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<ResearchNode | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [internalEvents, setInternalEvents] = useState<ProgressEvent[]>([]);
  const { fitView } = useReactFlow();

  // Use provided events or fetch from sessionId
  const events = propEvents || internalEvents;

  // Fetch events from WebSocket if sessionId is provided and no events are passed
  useEffect(() => {
    if (!sessionId || propEvents) return;

    const connectWS = () => {
      const wsUrl = `ws://localhost:8012/ws/research/${sessionId}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('Tree visualization WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'research_event') {
            setInternalEvents(prev => [...prev, data.event]);
          }
        } catch (error) {
          console.error('Error parsing tree WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        console.log('Tree visualization WebSocket disconnected');
        // Reconnect after 3 seconds
        setTimeout(connectWS, 3000);
      };

      ws.onerror = (error) => {
        console.error('Tree visualization WebSocket error:', error);
      };

      return ws;
    };

    const ws = connectWS();
    return () => ws?.close();
  }, [sessionId, propEvents]);

  const flowData = useMemo(() => {
    if (!events || events.length === 0) return { nodes: [], edges: [] };

    try {
      const researchNodes = convertToResearchNodes(events);
      const flowNodes = convertToFlowNodes(researchNodes).map(flowNode => ({
        ...flowNode,
        data: {
          ...flowNode.data,
          isSelected: flowNode.id === selectedNodeId
        }
      }));
      const flowEdges = convertToFlowEdges(researchNodes);

      return { nodes: flowNodes, edges: flowEdges, researchNodes };
    } catch (error) {
      console.error('Error converting research data:', error);
      return { nodes: [], edges: [], researchNodes: [] };
    }
  }, [events, selectedNodeId]);

  useEffect(() => {
    setNodes(flowData.nodes);
    setEdges(flowData.edges);

    if (flowData.nodes.length > 0) {
      setTimeout(() => fitView({ padding: 0.2, duration: 800 }), 300);
    }
  }, [flowData, setNodes, setEdges, fitView]);

  const onNodeClick = useCallback(
    (event: React.MouseEvent, node: Node) => {
      onNodeSelect?.(node.id);

      // Find and show detailed info for this node
      const researchNode = flowData.researchNodes?.find(n => n.id === node.id);
      if (researchNode) {
        setSelectedNode(researchNode);
        setSidebarOpen(true);
      }
    },
    [onNodeSelect, flowData.researchNodes]
  );

  if (!events || events.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="text-6xl mb-4">🌳</div>
          <h3 className="text-xl font-bold text-gray-700 mb-2">No Research Tree</h3>
          <p className="text-gray-500">Start a research to see the process visualization</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView={false}
        attributionPosition="bottom-left"
        proOptions={{ hideAttribution: true }}
        nodesDraggable={true}
        nodesConnectable={false}
        elementsSelectable={true}
        minZoom={0.1}
        maxZoom={4}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          className="opacity-30"
        />
        <Controls className="bg-white border border-gray-200 rounded-lg shadow-lg">
          <ControlButton onClick={() => fitView({ padding: 0.2, duration: 800 })}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M3 3h6v6H3V3zm12 0h6v6h-6V3zM3 15h6v6H3v-6zm12 0h6v6h-6v-6zm-4-4h2v2h-2v-2zm-4 0h2v2H7v-2zm8 0h2v2h-2v-2zm-4-4h2v2h-2V7z"/>
            </svg>
          </ControlButton>
        </Controls>
      </ReactFlow>

      {/* Detailed Sidebar */}
      {sidebarOpen && (
        <DetailedSidebar
          selectedNode={selectedNode}
          onClose={() => {
            setSidebarOpen(false);
            setSelectedNode(null);
          }}
          events={events}
        />
      )}
    </>
  );
};

const ResearchTreeVisualization: React.FC<ResearchTreeVisualizationProps> = (props) => {
  return (
    <div className="w-full h-full bg-white relative">
      <ReactFlowProvider>
        <FlowContent {...props} />
      </ReactFlowProvider>
    </div>
  );
};

export default ResearchTreeVisualization;