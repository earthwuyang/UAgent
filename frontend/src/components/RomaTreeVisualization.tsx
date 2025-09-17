import React, { useCallback, useEffect, useMemo } from 'react'
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
} from 'reactflow'
import 'reactflow/dist/style.css'

// ROMA-style task node interfaces
interface TaskNode {
  task_id: string
  goal: string
  task_type: string
  node_type: 'PLAN' | 'EXECUTE'
  agent_name?: string
  layer: number
  parent_node_id?: string
  status: TaskStatus
  output_summary?: string
  timestamp_created?: string
  timestamp_updated?: string
  timestamp_completed?: string
  planned_sub_task_ids?: string[]
  model_display?: string
}

type TaskStatus =
  | 'PENDING'
  | 'READY'
  | 'RUNNING'
  | 'PLAN_DONE'
  | 'AGGREGATING'
  | 'DONE'
  | 'FAILED'
  | 'NEEDS_REPLAN'
  | 'CANCELLED'

interface APIResponse {
  overall_project_goal?: string
  all_nodes: Record<string, TaskNode>
  graphs: Record<string, { edges: Array<{ source: string, target: string }> }>
}

// ROMA-style Task Node Component
const TaskNodeComponent: React.FC<{ data: { node: TaskNode, isSelected?: boolean } }> = ({ data }) => {
  const { node, isSelected } = data

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'DONE': return 'bg-green-50 border-green-200 text-green-800'
      case 'RUNNING': return 'bg-orange-50 border-orange-200 text-orange-800'
      case 'FAILED': return 'bg-red-50 border-red-200 text-red-800'
      case 'READY': return 'bg-blue-50 border-blue-200 text-blue-800'
      case 'PENDING': return 'bg-gray-50 border-gray-200 text-gray-800'
      default: return 'bg-gray-50 border-gray-200 text-gray-800'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'DONE': return '✅'
      case 'RUNNING': return '🔄'
      case 'FAILED': return '❌'
      case 'READY': return '▶️'
      case 'PENDING': return '⏳'
      default: return '⚪'
    }
  }

  const isPlanNode = node.node_type === 'PLAN'

  return (
    <div className={`
      min-w-[280px] max-w-[320px] p-4 rounded-xl border-2 shadow-lg transition-all duration-300
      ${getStatusColor(node.status)}
      ${isSelected ? 'ring-2 ring-blue-500 scale-105 shadow-xl' : 'hover:shadow-xl hover:scale-[1.02]'}
      ${node.status === 'RUNNING' ? 'animate-pulse' : ''}
    `}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <span className="text-lg">
            {isPlanNode ? '🧠' : '⚙️'}
          </span>
          <div className="px-2 py-1 bg-white bg-opacity-70 rounded text-xs font-medium">
            {node.task_type}
          </div>
          {node.layer > 0 && (
            <div className="px-2 py-1 bg-gray-200 rounded text-xs">
              L{node.layer}
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-lg">{getStatusIcon(node.status)}</span>
          <div className="text-xs font-medium">
            {node.status}
          </div>
        </div>
      </div>

      {/* Goal */}
      <div className="text-sm font-semibold leading-tight mb-3">
        {node.goal.length > 120 ? node.goal.substring(0, 120) + '...' : node.goal}
      </div>

      {/* Agent */}
      {node.agent_name && (
        <div className="text-xs text-gray-600 bg-white bg-opacity-50 rounded px-2 py-1 mb-2">
          🤖 {node.agent_name}
        </div>
      )}

      {/* Model */}
      {node.model_display && node.model_display !== "Not processed" && (
        <div className="text-xs text-gray-600 bg-white bg-opacity-30 rounded px-2 py-1 mb-2">
          💻 {node.model_display}
        </div>
      )}

      {/* Output Summary */}
      {node.output_summary && (
        <div className="text-xs text-gray-600 border-t border-gray-300 pt-2 mt-2">
          📋 {node.output_summary.length > 100 ? node.output_summary.substring(0, 100) + '...' : node.output_summary}
        </div>
      )}

      {/* Timestamp */}
      {node.timestamp_created && (
        <div className="text-xs text-gray-500 mt-1">
          ⏰ {new Date(node.timestamp_created).toLocaleTimeString()}
        </div>
      )}
    </div>
  )
}

const nodeTypes: NodeTypes = {
  taskNode: TaskNodeComponent,
}

// Convert ROMA data to ReactFlow format
function convertToFlowNodes(graphNodes: Record<string, TaskNode>): Node[] {
  const tasks = Object.values(graphNodes)

  // Group nodes by layer for hierarchical layout
  const nodesByLayer: Record<number, TaskNode[]> = {}
  tasks.forEach(task => {
    const layer = task.layer || 0
    if (!nodesByLayer[layer]) {
      nodesByLayer[layer] = []
    }
    nodesByLayer[layer].push(task)
  })

  return tasks.map((task) => {
    const layer = task.layer || 0
    const nodesInLayer = nodesByLayer[layer] || []
    const indexInLayer = nodesInLayer.findIndex(n => n.task_id === task.task_id)
    const totalInLayer = nodesInLayer.length

    // Calculate position based on layer and position within layer
    const layerHeight = 200
    const nodeSpacing = 350
    const y = layer * layerHeight

    // Center nodes horizontally within their layer
    const totalWidth = Math.max(0, (totalInLayer - 1) * nodeSpacing)
    const startX = -totalWidth / 2
    const x = startX + (indexInLayer * nodeSpacing)

    return {
      id: task.task_id,
      type: 'taskNode',
      position: { x, y },
      data: {
        node: task,
        label: task.goal,
        isSelected: false
      }
    }
  })
}

function convertToFlowEdges(
  graphNodes: Record<string, TaskNode>,
  graphs: Record<string, { edges: Array<{ source: string, target: string }> }> = {}
): Edge[] {
  const edges: Edge[] = []

  // Use graph edges if available, otherwise derive from parent relationships
  const graphEdges = graphs.main_graph?.edges || []

  if (graphEdges.length > 0) {
    graphEdges.forEach(edge => {
      edges.push({
        id: `${edge.source}-${edge.target}`,
        source: edge.source,
        target: edge.target,
        type: 'smoothstep',
        animated: graphNodes[edge.target]?.status === 'RUNNING'
      })
    })
  } else {
    // Fallback: derive from parent_node_id
    Object.values(graphNodes).forEach(task => {
      if (task.parent_node_id && graphNodes[task.parent_node_id]) {
        edges.push({
          id: `${task.parent_node_id}-${task.task_id}`,
          source: task.parent_node_id,
          target: task.task_id,
          type: 'smoothstep',
          animated: task.status === 'RUNNING'
        })
      }
    })
  }

  return edges
}

interface RomaTreeVisualizationProps {
  data: APIResponse | null
  selectedNodeId?: string
  onNodeSelect?: (nodeId: string) => void
}

const FlowContent: React.FC<RomaTreeVisualizationProps> = ({ data, selectedNodeId, onNodeSelect }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const { fitView } = useReactFlow()

  const flowData = useMemo(() => {
    if (!data?.all_nodes) return { nodes: [], edges: [] }

    try {
      const flowNodes = convertToFlowNodes(data.all_nodes).map(flowNode => ({
        ...flowNode,
        data: {
          ...flowNode.data,
          isSelected: flowNode.id === selectedNodeId
        }
      }))

      const flowEdges = convertToFlowEdges(data.all_nodes, data.graphs || {})

      return { nodes: flowNodes, edges: flowEdges }
    } catch (error) {
      console.error('Error converting graph data:', error)
      return { nodes: [], edges: [] }
    }
  }, [data, selectedNodeId])

  useEffect(() => {
    setNodes(flowData.nodes)
    setEdges(flowData.edges)

    if (flowData.nodes.length > 0) {
      setTimeout(() => fitView({ padding: 0.2, duration: 800 }), 300)
    }
  }, [flowData, setNodes, setEdges, fitView])

  const onNodeClick = useCallback(
    (event: React.MouseEvent, node: Node) => {
      onNodeSelect?.(node.id)
    },
    [onNodeSelect]
  )

  if (!data?.all_nodes || Object.keys(data.all_nodes).length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="text-6xl mb-4">🌳</div>
          <h3 className="text-xl font-bold text-gray-700 mb-2">No Research Tree</h3>
          <p className="text-gray-500">Start a new research goal to see the execution tree</p>
        </div>
      </div>
    )
  }

  return (
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
  )
}

const RomaTreeVisualization: React.FC<RomaTreeVisualizationProps> = (props) => {
  return (
    <div className="w-full h-full bg-white">
      <ReactFlowProvider>
        <FlowContent {...props} />
      </ReactFlowProvider>
    </div>
  )
}

export default RomaTreeVisualization