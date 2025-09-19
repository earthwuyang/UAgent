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
  Handle,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import LLMConversationLogs from './LLMConversationLogs';
import { WS_BASE_URL } from '../../config';

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
  depth?: number;
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
      {/* ReactFlow Handles */}
      <Handle type="target" position={Position.Top} id="target" />
      <Handle type="source" position={Position.Bottom} id="source" />
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
  sessionId?: string;
}> = ({ selectedNode, onClose, events, sessionId }) => {
  const [activeTab, setActiveTab] = useState<'details' | 'llm' | 'events'>('details');

  const nodeEvents = events.filter(event =>
    selectedNode && event.data.engine === selectedNode.engine
  );

  if (!selectedNode) return null;

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-white shadow-xl border-l border-gray-200 z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="text-lg font-bold">🔍 Node Details</h3>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 text-xl font-bold"
        >
          ×
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b bg-gray-50">
        <button
          onClick={() => setActiveTab('details')}
          className={`flex-1 px-4 py-2 text-sm font-medium ${
            activeTab === 'details'
              ? 'text-blue-600 border-b-2 border-blue-600 bg-white'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          📋 Details
        </button>
        <button
          onClick={() => setActiveTab('llm')}
          data-tab="llm"
          className={`flex-1 px-4 py-2 text-sm font-medium ${
            activeTab === 'llm'
              ? 'text-blue-600 border-b-2 border-blue-600 bg-white'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          🤖 LLM Chat
        </button>
        <button
          onClick={() => setActiveTab('events')}
          className={`flex-1 px-4 py-2 text-sm font-medium ${
            activeTab === 'events'
              ? 'text-blue-600 border-b-2 border-blue-600 bg-white'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          📊 Events
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'details' && (
          <div className="p-4 space-y-4 h-full overflow-y-auto">
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
        )}

        {activeTab === 'llm' && sessionId && (
          <LLMConversationLogs
            sessionId={sessionId}
            nodeId={selectedNode.id}
            engine={selectedNode.engine}
          />
        )}

        {activeTab === 'events' && (
          <div className="p-4 space-y-4 h-full overflow-y-auto">
            <div>
              <label className="text-sm font-medium text-gray-600">Recent Events ({nodeEvents.length})</label>
              <div className="space-y-2 mt-2">
                {nodeEvents.length === 0 ? (
                  <div className="text-center text-gray-500 py-8">
                    <div className="text-2xl mb-2">📭</div>
                    <p>No events for this node</p>
                  </div>
                ) : (
                  nodeEvents.map((event, index) => (
                    <div key={index} className="text-xs bg-gray-50 p-3 rounded border">
                      <div className="flex items-center justify-between mb-1">
                        <div className="font-medium">{event.event_type.replace('_', ' ').toUpperCase()}</div>
                        <div className="text-gray-500">
                          {new Date(event.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                      {event.message && (
                        <div className="text-gray-600 mb-1">{event.message}</div>
                      )}
                      {event.data.phase && (
                        <div className="text-blue-600 text-xs">Phase: {event.data.phase}</div>
                      )}
                      {event.progress_percentage !== undefined && (
                        <div className="text-green-600 text-xs">Progress: {event.progress_percentage}%</div>
                      )}
                      {event.data.metadata && (
                        <div className="mt-2 pt-2 border-t border-gray-200">
                          <div className="text-xs text-gray-500">
                            <pre className="whitespace-pre-wrap">{JSON.stringify(event.data.metadata, null, 2)}</pre>
                          </div>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
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
  const nodeMap = new Map<string, ResearchNode>()
  const sessionRoots = new Map<string, string>()

  events.forEach((event, index) => {
    const engine = event.data.engine || 'engine'
    const sessionKey = event.session_id || 'session'
    const engineKey = `${sessionKey}-${engine}`
    const rootId = `${engineKey}-root`

    if (!sessionRoots.has(engineKey)) {
      sessionRoots.set(engineKey, rootId)
      nodeMap.set(rootId, {
        id: rootId,
        type: 'engine',
        engine,
        phase: 'root',
        status: 'running',
        title: `${engine.replace('_', ' ').toUpperCase()} Engine`,
        description: `Execution tree for ${engine}`,
        timestamp: event.timestamp,
        progress: event.progress_percentage,
        metadata: { node_id: rootId },
        children: [],
        parent_id: undefined,
        depth: 0,
      })
    }

    const metadata = { ...(event.data.metadata || {}) }
    const nodeId = metadata.node_id || `${sessionKey}-${engine}-${event.data.phase || 'step'}-${index}`
    const parentId = metadata.parent_id || sessionRoots.get(engineKey)!
    const nodeType = metadata.node_type === 'result' ? 'result' : 'step'
    const status = event.event_type === 'research_completed'
      ? 'completed'
      : event.event_type === 'research_error'
      ? 'error'
      : 'running'

    const existingNode = nodeMap.get(nodeId)
    if (existingNode) {
      existingNode.status = status
      existingNode.progress = event.progress_percentage ?? existingNode.progress
      existingNode.timestamp = event.timestamp
      existingNode.metadata = { ...existingNode.metadata, ...metadata }
      if (status === 'completed' && existingNode.type !== 'engine') {
        existingNode.status = 'completed'
      }
    } else {
      const parentNode = nodeMap.get(parentId)
      const title = metadata.title || event.message || event.event_type.replace('_', ' ').toUpperCase()
      const description = metadata.description || (event.data.phase ? `Phase: ${event.data.phase}` : undefined)

      const node: ResearchNode = {
        id: nodeId,
        type: nodeType,
        engine,
        phase: metadata.phase || event.data.phase || 'processing',
        status,
        title,
        description,
        timestamp: event.timestamp,
        progress: event.progress_percentage,
        metadata,
        parent_id: parentId,
        children: [],
      }

      nodeMap.set(nodeId, node)

      if (parentNode) {
        parentNode.children = parentNode.children || []
        if (!parentNode.children.includes(nodeId)) {
          parentNode.children.push(nodeId)
        }
      }
    }

    if (event.event_type === 'research_completed') {
      const rootNode = nodeMap.get(sessionRoots.get(engineKey)!)
      if (rootNode) {
        rootNode.status = 'completed'
        rootNode.progress = 100
        rootNode.timestamp = event.timestamp
      }
    }
  })

  return Array.from(nodeMap.values())
}

// Convert to ReactFlow format
function convertToFlowNodes(researchNodes: ResearchNode[]): Node[] {
  const nodeMap = new Map<string, ResearchNode>()
  researchNodes.forEach((node) => {
    node.children = node.children || []
    nodeMap.set(node.id, node)
  })

  const roots = researchNodes.filter((node) => !node.parent_id)
  const nodePositions = new Map<string, { x: number; y: number }>()
  const horizontalSpacing = 240
  const verticalSpacing = 160

  const assignLayout = (nodeId: string, depth: number, baseX: number): { minX: number; maxX: number } => {
    const node = nodeMap.get(nodeId)
    if (!node) {
      return { minX: baseX, maxX: baseX }
    }

    const children = node.children?.map((childId) => nodeMap.get(childId)).filter(Boolean) as ResearchNode[]

    if (!children || children.length === 0) {
      const x = baseX
      const y = depth * verticalSpacing
      node.depth = depth
      nodePositions.set(nodeId, { x, y })
      return { minX: x - horizontalSpacing / 2, maxX: x + horizontalSpacing / 2 }
    }

    let currentX = baseX
    const childRanges: Array<{ minX: number; maxX: number }> = []

    children.forEach((child, index) => {
      const childBaseX = currentX + index * horizontalSpacing
      const range = assignLayout(child.id, depth + 1, childBaseX)
      childRanges.push(range)
      currentX = range.maxX
    })

    const minX = childRanges[0].minX
    const maxX = childRanges[childRanges.length - 1].maxX
    const nodeX = (minX + maxX) / 2
    const nodeY = depth * verticalSpacing

    node.depth = depth
    nodePositions.set(nodeId, { x: nodeX, y: nodeY })

    return { minX, maxX }
  }

  roots.forEach((root, index) => {
    const startX = index * 500
    assignLayout(root.id, 0, startX)
  })

  return researchNodes.map((node) => {
    const position = nodePositions.get(node.id) || { x: 0, y: 0 }
    return {
      id: node.id,
      type: 'researchNode',
      position,
      data: { node, isSelected: false }
    }
  })
}

function convertToFlowEdges(researchNodes: ResearchNode[]): Edge[] {
  const edges: Edge[] = [];

  researchNodes.forEach(node => {
    if (node.parent_id) {
      edges.push({
        id: `${node.parent_id}-${node.id}`,
        source: node.parent_id,
        target: node.id,
        sourceHandle: 'source',
        targetHandle: 'target',
        type: 'smoothstep',
        animated: node.status === 'running'
      })
    }
  })

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

    let isCleanup = false;
    let reconnectTimeout: number | null = null;

    const connectWS = () => {
      if (isCleanup) return;

      const wsUrl = `${WS_BASE_URL}/ws/research/${sessionId}`;
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

      ws.onclose = (event) => {
        console.log('Tree visualization WebSocket disconnected');

        // Only reconnect if not a clean close and component hasn't unmounted
        if (!isCleanup && event.code !== 1000) {
          reconnectTimeout = setTimeout(connectWS, 3000);
        }
      };

      ws.onerror = (error) => {
        console.error('Tree visualization WebSocket error:', error);
      };

      return ws;
    };

    const ws = connectWS();
    return () => {
      isCleanup = true;

      // Clear any pending reconnection timeout
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }

      // Close WebSocket connection with clean close code
      if (ws?.readyState === WebSocket.OPEN) {
        ws.close(1000, 'Component unmounting');
      }
    };
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

  const onNodeDoubleClick = useCallback(
    (event: React.MouseEvent, node: Node) => {
      // Find and show detailed info for this node with LLM tab pre-selected
      const researchNode = flowData.researchNodes?.find(n => n.id === node.id);
      if (researchNode) {
        setSelectedNode(researchNode);
        setSidebarOpen(true);
        // Auto-select LLM tab for double-click
        setTimeout(() => {
          const llmTab = document.querySelector('[data-tab="llm"]') as HTMLButtonElement;
          if (llmTab) {
            llmTab.click();
          }
        }, 100);
      }
    },
    [flowData.researchNodes]
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
        onNodeDoubleClick={onNodeDoubleClick}
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
          sessionId={sessionId}
        />
      )}
    </>
  );
};

const ResearchTreeVisualization: React.FC<ResearchTreeVisualizationProps> = (props) => {
  return (
    <div className="w-full bg-white relative" style={{ height: '600px' }}>
      <ReactFlowProvider>
        <FlowContent {...props} />
      </ReactFlowProvider>
    </div>
  );
};

export default ResearchTreeVisualization;
