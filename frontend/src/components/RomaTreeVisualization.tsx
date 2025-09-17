import React, { useCallback, useEffect, useMemo, useState } from 'react'
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

  // Enhanced status display for failed nodes
  const getFailureIndicator = () => {
    if (node.status === 'FAILED') {
      return (
        <div className="mt-2 p-2 bg-red-100 border-l-4 border-red-500 rounded">
          <div className="flex items-center">
            <span className="text-red-600 text-xs font-medium">⚠️ Failed</span>
          </div>
          <div className="text-xs text-red-600 mt-1">
            Click for details
          </div>
        </div>
      )
    }
    return null
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

      {/* Failure Indicator */}
      {getFailureIndicator()}
    </div>
  )
}

// Execution Steps Modal Component
const ExecutionStepsModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  goalId: string;
  nodeId: string;
  nodeTitle: string;
}> = ({ isOpen, onClose, goalId, nodeId, nodeTitle }) => {
  const [executionData, setExecutionData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchExecutionSteps = useCallback(async () => {
    if (!isOpen || !goalId || !nodeId) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`/api/research-tree/goals/${goalId}/nodes/${nodeId}/execution-steps`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setExecutionData(data)
    } catch (err) {
      console.error('Failed to fetch execution steps:', err)
      setError(err instanceof Error ? err.message : 'Failed to load execution steps')
    } finally {
      setLoading(false)
    }
  }, [isOpen, goalId, nodeId])

  useEffect(() => {
    fetchExecutionSteps()
  }, [fetchExecutionSteps])

  // Auto-refresh for running nodes
  useEffect(() => {
    if (!isOpen || !executionData?.status_summary?.status || executionData.status_summary.status !== 'RUNNING') return

    const interval = setInterval(fetchExecutionSteps, 2000) // Refresh every 2 seconds
    return () => clearInterval(interval)
  }, [isOpen, executionData?.status_summary?.status, fetchExecutionSteps])

  if (!isOpen) return null

  const getStepIcon = (type: string, level?: string) => {
    switch (type) {
      case 'error': return '❌'
      case 'log':
        switch (level) {
          case 'INFO': return '✅'
          case 'DEBUG': return '🔍'
          case 'WARNING': return '⚠️'
          case 'ERROR': return '❌'
          default: return 'ℹ️'
        }
      case 'processing_step': return '⚙️'
      default: return '📝'
    }
  }

  const getStepColor = (type: string, level?: string) => {
    switch (type) {
      case 'error': return 'bg-red-50 border-red-200 text-red-800'
      case 'log':
        switch (level) {
          case 'INFO': return 'bg-green-50 border-green-200 text-green-800'
          case 'DEBUG': return 'bg-blue-50 border-blue-200 text-blue-800'
          case 'WARNING': return 'bg-yellow-50 border-yellow-200 text-yellow-800'
          case 'ERROR': return 'bg-red-50 border-red-200 text-red-800'
          default: return 'bg-gray-50 border-gray-200 text-gray-800'
        }
      case 'processing_step': return 'bg-indigo-50 border-indigo-200 text-indigo-800'
      default: return 'bg-gray-50 border-gray-200 text-gray-800'
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-xl font-bold">Execution Steps</h2>
            <p className="text-gray-600 mt-1">{nodeTitle}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="ml-3">Loading execution steps...</span>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded p-4 mb-4">
              <div className="text-red-800 font-medium">Error loading execution steps</div>
              <div className="text-red-600 text-sm mt-1">{error}</div>
              <button
                onClick={fetchExecutionSteps}
                className="mt-2 text-sm bg-red-100 hover:bg-red-200 text-red-800 px-3 py-1 rounded"
              >
                Retry
              </button>
            </div>
          )}

          {executionData && (
            <>
              {/* Status Summary */}
              <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold">Status: {executionData.status_summary.status}</span>
                  <span className="text-sm text-gray-600">
                    {executionData.status_summary.total_steps} steps
                  </span>
                </div>
                {executionData.status_summary.has_errors && (
                  <div className="text-sm text-red-600 mt-2">
                    ⚠️ {executionData.detailed_errors.length} error(s) occurred (Retry count: {executionData.status_summary.retry_count})
                  </div>
                )}
                {executionData.status_summary.last_error && (
                  <div className="text-sm text-red-600 mt-1 font-mono bg-red-50 p-2 rounded">
                    Last error: {executionData.status_summary.last_error}
                  </div>
                )}
              </div>

              {/* Execution Steps Timeline */}
              <div className="space-y-3">
                {executionData.execution_steps.map((step: any, index: number) => (
                  <div
                    key={index}
                    className={`border rounded-lg p-3 ${getStepColor(step.type, step.level)}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3">
                        <span className="text-lg flex-shrink-0">
                          {getStepIcon(step.type, step.level)}
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium">{step.message}</div>
                          {step.details && (
                            <div className="text-sm mt-1">{step.details}</div>
                          )}
                          {step.context && Object.keys(step.context).length > 0 && (
                            <details className="text-xs mt-2">
                              <summary className="cursor-pointer hover:text-blue-600">
                                View context
                              </summary>
                              <pre className="mt-1 p-2 bg-black bg-opacity-10 rounded overflow-auto">
                                {JSON.stringify(step.context, null, 2)}
                              </pre>
                            </details>
                          )}
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 flex-shrink-0 ml-4">
                        {step.timestamp ? new Date(step.timestamp).toLocaleTimeString() : ''}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Detailed Error Summary */}
              {executionData.detailed_errors.length > 0 && (
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <h3 className="font-semibold text-red-800 mb-3">Error Summary</h3>
                  <div className="space-y-2">
                    {executionData.detailed_errors.map((error: any, index: number) => (
                      <div key={index} className="text-sm">
                        <div className="font-medium text-red-700">
                          {error.error_type}: {error.error_message}
                        </div>
                        <div className="text-red-600 text-xs">
                          Retry #{error.retry_count} • Duration: {error.execution_time.toFixed(2)}s •
                          {error.timestamp ? new Date(error.timestamp).toLocaleString() : ''}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="border-t p-4 flex justify-between items-center">
          <div className="text-sm text-gray-600">
            {executionData?.status_summary?.status === 'RUNNING' && (
              <span className="flex items-center">
                <div className="animate-pulse w-2 h-2 bg-orange-400 rounded-full mr-2"></div>
                Auto-refreshing every 2 seconds
              </span>
            )}
          </div>
          <div className="space-x-2">
            <button
              onClick={fetchExecutionSteps}
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Refresh
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
            >
              Close
            </button>
          </div>
        </div>
      </div>
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
  goalId?: string
  onNodeSelect?: (nodeId: string) => void
  onNodeDoubleClick?: (nodeId: string) => void
}

const FlowContent: React.FC<RomaTreeVisualizationProps> = ({ data, selectedNodeId, goalId, onNodeSelect, onNodeDoubleClick }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [executionModalOpen, setExecutionModalOpen] = useState(false)
  const [selectedExecutionNode, setSelectedExecutionNode] = useState<{ nodeId: string, nodeTitle: string } | null>(null)
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

  const onNodeDoubleClickHandler = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault()

      // Open execution modal for this node
      if (goalId && data?.all_nodes?.[node.id]) {
        setSelectedExecutionNode({
          nodeId: node.id,
          nodeTitle: data.all_nodes[node.id].goal
        })
        setExecutionModalOpen(true)
      }

      // Still call the original handler if provided
      onNodeDoubleClick?.(node.id)
    },
    [onNodeDoubleClick, goalId, data]
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
    <>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onNodeDoubleClick={onNodeDoubleClickHandler}
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

      {/* Execution Steps Modal */}
      {goalId && selectedExecutionNode && (
        <ExecutionStepsModal
          isOpen={executionModalOpen}
          onClose={() => {
            setExecutionModalOpen(false)
            setSelectedExecutionNode(null)
          }}
          goalId={goalId}
          nodeId={selectedExecutionNode.nodeId}
          nodeTitle={selectedExecutionNode.nodeTitle}
        />
      )}
    </>
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