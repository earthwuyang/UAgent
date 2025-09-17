import React, { useState, useEffect } from 'react'
import RomaTreeVisualization from './RomaTreeVisualization'

// Research Tree API functions
const API_BASE = '/api/research-tree'

const researchAPI = {
  startGoal: async (goalData: any) => {
    const response = await fetch(`${API_BASE}/goals/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(goalData)
    })
    if (!response.ok) throw new Error('Failed to start research goal')
    return response.json()
  },

  getTreeStatus: async (goalId: string) => {
    const response = await fetch(`${API_BASE}/goals/${goalId}/status`)
    if (!response.ok) throw new Error('Failed to get tree status')
    return response.json()
  },

  getVisualization: async (goalId: string) => {
    const response = await fetch(`${API_BASE}/goals/${goalId}/visualization`)
    if (!response.ok) throw new Error('Failed to get visualization')
    return response.json()
  },

  getInsights: async (goalId: string) => {
    const response = await fetch(`${API_BASE}/goals/${goalId}/insights`)
    if (!response.ok) throw new Error('Failed to get insights')
    return response.json()
  },

  listActiveGoals: async () => {
    const response = await fetch(`${API_BASE}/goals/active`)
    if (!response.ok) throw new Error('Failed to list active goals')
    return response.json()
  },

  getSystemMetrics: async () => {
    const response = await fetch(`${API_BASE}/system/metrics`)
    if (!response.ok) throw new Error('Failed to get system metrics')
    return response.json()
  },

  exportResults: async (goalId: string) => {
    const response = await fetch(`${API_BASE}/goals/${goalId}/export`)
    if (!response.ok) throw new Error('Failed to export results')
    return response.json()
  },

  expandNode: async (goalId: string, nodeId: string) => {
    const response = await fetch(`${API_BASE}/goals/${goalId}/manual-experiment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'expand_node',
        node_id: nodeId,
        expansion_type: 'automatic'
      })
    })
    if (!response.ok) throw new Error('Failed to expand node')
    return response.json()
  },

  runExperiment: async (goalId: string, nodeId: string, experimentType: string) => {
    const response = await fetch(`${API_BASE}/goals/${goalId}/manual-experiment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'run_experiment',
        node_id: nodeId,
        experiment_type: experimentType
      })
    })
    if (!response.ok) throw new Error('Failed to run experiment')
    return response.json()
  },

  pruneNode: async (goalId: string, nodeId: string) => {
    const response = await fetch(`${API_BASE}/goals/${goalId}/manual-experiment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'prune_node',
        node_id: nodeId
      })
    })
    if (!response.ok) throw new Error('Failed to prune node')
    return response.json()
  },

  getNodeReport: async (goalId: string, nodeId: string) => {
    const response = await fetch(`${API_BASE}/goals/${goalId}/nodes/${nodeId}/report`)
    if (!response.ok) throw new Error('Failed to get node report')
    return response.json()
  }
}

interface TreeNode {
  id: string
  title: string
  type: string
  status: string
  confidence: number
  children: TreeNode[]
  depth: number
  visits: number
}

interface ResearchGoal {
  goal_id: string
  title: string
  description: string
  created_at: string
  experiments_run: number
}

export default function HierarchicalResearchDashboard() {
  // State management
  const [activeGoals, setActiveGoals] = useState<ResearchGoal[]>([])
  const [selectedGoal, setSelectedGoal] = useState<string>('')
  const [treeStatus, setTreeStatus] = useState<any>(null)
  const [treeVisualization, setTreeVisualization] = useState<any>(null)
  const [insights, setInsights] = useState<any>(null)
  const [systemMetrics, setSystemMetrics] = useState<any>(null)
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null)
  const [showNodeDetails, setShowNodeDetails] = useState(false)
  const [nodeReport, setNodeReport] = useState<any>(null)
  const [showNodeReport, setShowNodeReport] = useState(false)

  // Form state for new research goals
  const [newGoal, setNewGoal] = useState({
    title: 'Hierarchical AI Agent Architecture Optimization',
    description: 'Research optimal architectures for hierarchical AI agents that can perform complex reasoning and planning tasks',
    success_criteria: [
      'Identify key architectural components for hierarchical agents',
      'Develop performance benchmarks for agent architectures',
      'Validate architecture effectiveness through empirical testing'
    ],
    max_depth: 6,
    max_experiments: 150
  })

  // UI state
  const [activeTab, setActiveTab] = useState('dashboard')
  const [loading, setLoading] = useState<string>('')
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Load data on component mount
  useEffect(() => {
    loadInitialData()

    if (autoRefresh) {
      const interval = setInterval(() => {
        if (selectedGoal) {
          refreshGoalData(selectedGoal)
        }
        loadActiveGoals()
        loadSystemMetrics()
      }, 10000) // Refresh every 10 seconds

      return () => clearInterval(interval)
    }
  }, [selectedGoal, autoRefresh])

  const loadInitialData = async () => {
    await Promise.all([
      loadActiveGoals(),
      loadSystemMetrics()
    ])
  }

  const loadActiveGoals = async () => {
    try {
      const response = await researchAPI.listActiveGoals()
      setActiveGoals(response.active_goals)

      // Auto-select first goal if none selected
      if (response.active_goals.length > 0 && !selectedGoal) {
        const firstGoal = response.active_goals[0].goal_id
        setSelectedGoal(firstGoal)
        await refreshGoalData(firstGoal)
      }
    } catch (error) {
      console.error('Failed to load active goals:', error)
    }
  }

  const loadSystemMetrics = async () => {
    try {
      const metrics = await researchAPI.getSystemMetrics()
      setSystemMetrics(metrics)
    } catch (error) {
      console.error('Failed to load system metrics:', error)
    }
  }

  const refreshGoalData = async (goalId: string) => {
    try {
      const [status, viz, insightsData] = await Promise.all([
        researchAPI.getTreeStatus(goalId),
        researchAPI.getVisualization(goalId),
        researchAPI.getInsights(goalId)
      ])

      setTreeStatus(status)
      setTreeVisualization(viz)
      setInsights(insightsData)
    } catch (error) {
      console.error('Failed to refresh goal data:', error)
    }
  }

  const handleStartNewGoal = async () => {
    setLoading('starting')
    try {
      const response = await researchAPI.startGoal(newGoal)
      alert(`Research goal "${response.title}" started successfully!`)

      setSelectedGoal(response.goal_id)
      await loadActiveGoals()
      await refreshGoalData(response.goal_id)
    } catch (error) {
      alert(`Failed to start research goal: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handleGoalSelection = async (goalId: string) => {
    setSelectedGoal(goalId)
    setLoading('loading')
    try {
      await refreshGoalData(goalId)
    } finally {
      setLoading('')
    }
  }

  const handleExportResults = async () => {
    if (!selectedGoal) return

    try {
      const results = await researchAPI.exportResults(selectedGoal)

      // Create download
      const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `research_results_${selectedGoal.slice(0, 8)}.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      alert(`Failed to export results: ${error}`)
    }
  }

  // Node action handlers
  const handleExpandNode = async (nodeId: string) => {
    if (!selectedGoal) return

    setLoading('expanding')
    try {
      const result = await researchAPI.expandNode(selectedGoal, nodeId)
      alert(`Node expansion started! ${result.message}`)

      // Refresh tree data
      await refreshGoalData(selectedGoal)
    } catch (error) {
      alert(`Failed to expand node: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handleRunExperiment = async (nodeId: string) => {
    if (!selectedGoal || !selectedNode) return

    setLoading('running-experiment')
    try {
      // Determine experiment type based on node type
      const experimentType = selectedNode.type === 'literature' ? 'literature_analysis' :
                           selectedNode.type === 'code_analysis' ? 'code_study' :
                           selectedNode.type === 'hypothesis' ? 'computational' :
                           'simulation'

      const result = await researchAPI.runExperiment(selectedGoal, nodeId, experimentType)
      alert(`Experiment started! ${result.message}`)

      // Refresh tree data
      await refreshGoalData(selectedGoal)
    } catch (error) {
      alert(`Failed to run experiment: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handleViewFullReport = async (nodeId: string) => {
    if (!selectedGoal) return

    setLoading('loading-report')
    try {
      const report = await researchAPI.getNodeReport(selectedGoal, nodeId)
      setNodeReport(report)
      setShowNodeReport(true)
    } catch (error) {
      console.error('Failed to load node report:', error)
      alert(`Failed to load node report: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handlePruneNode = async (nodeId: string) => {
    if (!selectedGoal || !selectedNode) return

    const confirmed = window.confirm(
      `Are you sure you want to prune "${selectedNode.title}"? This will remove this branch from the research tree.`
    )

    if (!confirmed) return

    setLoading('pruning')
    try {
      const result = await researchAPI.pruneNode(selectedGoal, nodeId)
      alert(`Node pruned successfully! ${result.message}`)

      // Close details sidebar and refresh
      setShowNodeDetails(false)
      setSelectedNode(null)
      await refreshGoalData(selectedGoal)
    } catch (error) {
      alert(`Failed to prune node: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handleNodeClick = (node: TreeNode) => {
    setSelectedNode(node)
    setShowNodeDetails(true)
  }

  const renderTreeVisualization = (node: TreeNode, depth = 0) => {
    const getStatusColor = (status: string) => {
      switch (status) {
        case 'completed': return 'bg-green-100 border-green-500 text-green-800'
        case 'running': return 'bg-blue-100 border-blue-500 text-blue-800'
        case 'failed': return 'bg-red-100 border-red-500 text-red-800'
        case 'pruned': return 'bg-gray-100 border-gray-500 text-gray-600'
        default: return 'bg-yellow-100 border-yellow-500 text-yellow-800'
      }
    }

    const getTypeIcon = (type: string) => {
      switch (type) {
        case 'root': return '🎯'
        case 'hypothesis': return '💡'
        case 'experiment': return '🧪'
        case 'literature': return '📚'
        case 'code_analysis': return '💻'
        case 'synthesis': return '🔬'
        case 'validation': return '✅'
        default: return '🔍'
      }
    }

    const isSelected = selectedNode?.id === node.id

    return (
      <div key={node.id} className={`ml-${depth * 4} mb-2`}>
        <div
          className={`p-3 rounded-lg border-2 cursor-pointer transition-all duration-200 hover:shadow-md ${
            isSelected
              ? 'border-blue-600 bg-blue-50 shadow-lg transform scale-105'
              : getStatusColor(node.status)
          }`}
          onClick={() => handleNodeClick(node)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-lg">{getTypeIcon(node.type)}</span>
              <div>
                <div className="font-medium text-sm">{node.title}</div>
                <div className="text-xs opacity-75">
                  {node.type} • Depth {node.depth} • Visits {node.visits}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm font-bold">
                {(node.confidence * 100).toFixed(1)}%
              </div>
              <div className="text-xs opacity-75">confidence</div>
            </div>
          </div>

          {/* Confidence bar */}
          <div className="mt-2 bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${
                isSelected ? 'bg-blue-600' : 'bg-blue-500'
              }`}
              style={{ width: `${node.confidence * 100}%` }}
            />
          </div>

          {/* Selection indicator */}
          {isSelected && (
            <div className="absolute -right-1 -top-1">
              <div className="w-3 h-3 bg-blue-600 rounded-full border-2 border-white"></div>
            </div>
          )}
        </div>

        {/* Render children */}
        {node.children && node.children.map(child =>
          renderTreeVisualization(child, depth + 1)
        )}
      </div>
    )
  }

  const renderNodeDetailsSidebar = () => {
    if (!selectedNode) return null

    const getStatusBadge = (status: string) => {
      const colors = {
        completed: 'bg-green-100 text-green-800',
        running: 'bg-blue-100 text-blue-800',
        failed: 'bg-red-100 text-red-800',
        pruned: 'bg-gray-100 text-gray-800',
        pending: 'bg-yellow-100 text-yellow-800',
        promising: 'bg-purple-100 text-purple-800'
      }
      return colors[status as keyof typeof colors] || colors.pending
    }

    const getTypeDescription = (type: string) => {
      const descriptions = {
        root: 'Initial research goal and starting point',
        hypothesis: 'Testable research hypothesis',
        experiment: 'Scientific experiment or validation',
        literature: 'Literature review and analysis',
        code_analysis: 'Code repository analysis',
        synthesis: 'Result synthesis and integration',
        validation: 'Validation and verification study'
      }
      return descriptions[type as keyof typeof descriptions] || 'Research node'
    }

    const getUCBScoreColor = (confidence: number) => {
      if (confidence >= 0.8) return 'text-green-600'
      if (confidence >= 0.6) return 'text-blue-600'
      if (confidence >= 0.4) return 'text-yellow-600'
      return 'text-red-600'
    }

    return (
      <div className="space-y-6 max-h-full overflow-y-auto">
        {/* Header */}
        <div className="border-b pb-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold">Node Details</h3>
            <button
              onClick={() => setShowNodeDetails(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          </div>
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-2xl">
              {selectedNode.type === 'root' ? '🎯' :
               selectedNode.type === 'hypothesis' ? '💡' :
               selectedNode.type === 'experiment' ? '🧪' :
               selectedNode.type === 'literature' ? '📚' :
               selectedNode.type === 'code_analysis' ? '💻' :
               selectedNode.type === 'synthesis' ? '🔬' :
               selectedNode.type === 'validation' ? '✅' : '🔍'}
            </span>
            <div>
              <h4 className="font-medium">{selectedNode.title}</h4>
              <span className={`px-2 py-1 rounded-full text-xs ${getStatusBadge(selectedNode.status)}`}>
                {selectedNode.status}
              </span>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-xs text-gray-600">Confidence</div>
            <div className={`text-xl font-bold ${getUCBScoreColor(selectedNode.confidence)}`}>
              {(selectedNode.confidence * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-xs text-gray-600">Tree Depth</div>
            <div className="text-xl font-bold text-gray-800">{selectedNode.depth}</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-xs text-gray-600">Visits</div>
            <div className="text-xl font-bold text-gray-800">{selectedNode.visits}</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-xs text-gray-600">Children</div>
            <div className="text-xl font-bold text-gray-800">{selectedNode.children?.length || 0}</div>
          </div>
        </div>

        {/* Node Type Information */}
        <div>
          <h5 className="font-medium mb-2">Node Type</h5>
          <div className="bg-blue-50 p-3 rounded">
            <div className="font-medium text-sm capitalize">{selectedNode.type.replace('_', ' ')}</div>
            <div className="text-xs text-gray-600 mt-1">
              {getTypeDescription(selectedNode.type)}
            </div>
          </div>
        </div>

        {/* Confidence Progress Bar */}
        <div>
          <h5 className="font-medium mb-2">Confidence Score</h5>
          <div className="bg-gray-200 rounded-full h-4 mb-2">
            <div
              className="bg-gradient-to-r from-blue-400 to-blue-600 h-4 rounded-full transition-all duration-500"
              style={{ width: `${selectedNode.confidence * 100}%` }}
            />
          </div>
          <div className="text-xs text-gray-600">
            {selectedNode.confidence < 0.3 ? 'Low confidence - needs more validation' :
             selectedNode.confidence < 0.6 ? 'Moderate confidence - promising direction' :
             selectedNode.confidence < 0.8 ? 'High confidence - strong evidence' :
             'Very high confidence - excellent results'}
          </div>
        </div>

        {/* UCB Analysis */}
        <div>
          <h5 className="font-medium mb-2">Tree Search Analysis</h5>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Exploitation Score:</span>
              <span className="font-medium">{selectedNode.visits > 0 ? ((selectedNode.confidence * selectedNode.visits) / selectedNode.visits).toFixed(3) : '0.000'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span>Exploration Priority:</span>
              <span className="font-medium">
                {selectedNode.visits === 0 ? 'MAX' :
                 selectedNode.visits < 3 ? 'High' :
                 selectedNode.visits < 10 ? 'Medium' : 'Low'}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span>Selection Frequency:</span>
              <span className="font-medium">{selectedNode.visits} times</span>
            </div>
          </div>
        </div>

        {/* Simulated Experiment Results */}
        {selectedNode.status === 'completed' && (
          <div>
            <h5 className="font-medium mb-2">Experiment Results</h5>
            <div className="bg-green-50 p-3 rounded space-y-2">
              <div className="text-sm">
                <div className="font-medium">Success Rate: {(Math.random() * 40 + 60).toFixed(1)}%</div>
                <div className="text-gray-600">Execution Time: {(Math.random() * 15 + 5).toFixed(1)}min</div>
              </div>

              {selectedNode.type === 'literature' && (
                <div className="text-xs text-gray-600">
                  📚 Found {Math.floor(Math.random() * 20 + 10)} relevant papers
                  <br />
                  📊 {Math.floor(Math.random() * 5 + 3)} key insights extracted
                </div>
              )}

              {selectedNode.type === 'code_analysis' && (
                <div className="text-xs text-gray-600">
                  💻 Analyzed {Math.floor(Math.random() * 15 + 5)} repositories
                  <br />
                  🔍 {Math.floor(Math.random() * 8 + 2)} patterns identified
                </div>
              )}

              {selectedNode.type === 'experiment' && (
                <div className="text-xs text-gray-600">
                  🧪 {Math.floor(Math.random() * 500 + 100)} test cases executed
                  <br />
                  📈 {(Math.random() * 0.3 + 0.7).toFixed(3)} validation score
                </div>
              )}
            </div>
          </div>
        )}

        {/* Generated Insights */}
        {selectedNode.status === 'completed' && (
          <div>
            <h5 className="font-medium mb-2">Key Insights</h5>
            <div className="space-y-2">
              {[
                `Strong evidence supporting ${selectedNode.type} approach`,
                `Results exceed baseline by ${(Math.random() * 25 + 10).toFixed(1)}%`,
                `Methodology validated across multiple test scenarios`,
                selectedNode.confidence > 0.8 ? 'Recommended for follow-up investigation' : 'Requires additional validation'
              ].map((insight, index) => (
                <div key={index} className="text-xs bg-gray-50 p-2 rounded">
                  💡 {insight}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Child Nodes Preview */}
        {selectedNode.children && selectedNode.children.length > 0 && (
          <div>
            <h5 className="font-medium mb-2">Child Nodes ({selectedNode.children.length})</h5>
            <div className="space-y-1">
              {selectedNode.children.slice(0, 3).map((childId, index) => (
                <div key={index} className="text-xs bg-gray-50 p-2 rounded flex items-center space-x-2">
                  <span>🔗</span>
                  <span className="text-gray-600">Child node {index + 1}</span>
                </div>
              ))}
              {selectedNode.children.length > 3 && (
                <div className="text-xs text-gray-500 text-center">
                  ... and {selectedNode.children.length - 3} more
                </div>
              )}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="border-t pt-4">
          <h5 className="font-medium mb-2">Actions</h5>
          <div className="space-y-2">
            <button
              onClick={() => handleExpandNode(selectedNode.id)}
              disabled={loading === 'expanding'}
              className="w-full px-3 py-2 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading === 'expanding' ? 'Expanding...' : '🔍 Expand Node'}
            </button>
            <button
              onClick={() => handleRunExperiment(selectedNode.id)}
              disabled={loading === 'experiment' || selectedNode.status === 'running'}
              className="w-full px-3 py-2 bg-green-500 text-white rounded text-sm hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading === 'experiment' ? 'Running...' :
               selectedNode.status === 'running' ? 'Experiment Running' : '🧪 Run Experiment'}
            </button>
            <button
              onClick={() => handleViewFullReport(selectedNode.id)}
              disabled={loading === 'report'}
              className="w-full px-3 py-2 bg-purple-500 text-white rounded text-sm hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading === 'report' ? 'Loading...' : '📊 View Full Report'}
            </button>
            {selectedNode.confidence < 0.4 && (
              <button
                onClick={() => handlePruneNode(selectedNode.id)}
                disabled={loading === 'pruning'}
                className="w-full px-3 py-2 bg-red-500 text-white rounded text-sm hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading === 'pruning' ? 'Pruning...' : '✂️ Prune Branch'}
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }


  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <div className="space-y-6">
            {/* System Overview */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Hierarchical Research System</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-500">
                    {systemMetrics?.system_overview?.active_research_goals || 0}
                  </div>
                  <div className="text-sm text-gray-600">Active Goals</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-500">
                    {systemMetrics?.system_overview?.total_experiments_run || 0}
                  </div>
                  <div className="text-sm text-gray-600">Experiments Run</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-500">
                    {((systemMetrics?.system_overview?.success_rate || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">Success Rate</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-500">
                    {systemMetrics?.system_overview?.currently_running || 0}
                  </div>
                  <div className="text-sm text-gray-600">Running Now</div>
                </div>
              </div>
            </div>

            {/* Active Research Goals */}
            <div className="bg-white p-6 rounded-lg shadow">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Active Research Goals</h3>
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`px-3 py-1 rounded text-sm ${autoRefresh ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'}`}
                >
                  Auto-refresh: {autoRefresh ? 'ON' : 'OFF'}
                </button>
              </div>

              {activeGoals.length === 0 ? (
                <p className="text-gray-500">No active research goals</p>
              ) : (
                <div className="space-y-3">
                  {activeGoals.map((goal) => (
                    <div
                      key={goal.goal_id}
                      className={`border rounded p-3 cursor-pointer transition-colors ${
                        selectedGoal === goal.goal_id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => handleGoalSelection(goal.goal_id)}
                    >
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="font-medium">{goal.title}</div>
                          <div className="text-sm text-gray-600 mt-1">{goal.description}</div>
                          <div className="text-xs text-gray-500 mt-2">
                            Experiments: {goal.experiments_run} • ID: {goal.goal_id.slice(0, 8)}...
                          </div>
                        </div>
                        {selectedGoal === goal.goal_id && (
                          <div className="ml-4">
                            <span className="px-2 py-1 bg-blue-500 text-white rounded text-xs">Selected</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Current Goal Status */}
            {treeStatus && (
              <div className="bg-white p-6 rounded-lg shadow">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold">Research Tree Status</h3>
                  <button
                    onClick={handleExportResults}
                    className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
                  >
                    Export Results
                  </button>
                </div>

                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="text-center">
                    <div className="text-xl font-bold">{treeStatus.tree_stats.total_nodes}</div>
                    <div className="text-sm text-gray-600">Total Nodes</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xl font-bold">{treeStatus.tree_stats.completed_nodes}</div>
                    <div className="text-sm text-gray-600">Completed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xl font-bold">
                      {(treeStatus.tree_stats.success_rate * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">Success Rate</div>
                  </div>
                </div>

                {/* Best Results */}
                <div>
                  <h4 className="font-medium mb-2">Top Results</h4>
                  <div className="space-y-2">
                    {treeStatus.best_results.slice(0, 3).map((result: any, index: number) => (
                      <div key={result.node_id} className="flex justify-between items-center bg-gray-50 p-2 rounded">
                        <div>
                          <div className="font-medium text-sm">{result.title}</div>
                          <div className="text-xs text-gray-600">{result.node_type}</div>
                        </div>
                        <div className="text-right">
                          <div className="font-bold text-sm">{(result.confidence * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )

      case 'start-goal':
        return (
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Start New Research Goal</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Research Title</label>
                <input
                  type="text"
                  value={newGoal.title}
                  onChange={(e) => setNewGoal({...newGoal, title: e.target.value})}
                  className="w-full p-2 border rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Description</label>
                <textarea
                  value={newGoal.description}
                  onChange={(e) => setNewGoal({...newGoal, description: e.target.value})}
                  rows={3}
                  className="w-full p-2 border rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Success Criteria (one per line)</label>
                <textarea
                  value={newGoal.success_criteria.join('\n')}
                  onChange={(e) => setNewGoal({...newGoal, success_criteria: e.target.value.split('\n').filter(c => c.trim())})}
                  rows={4}
                  className="w-full p-2 border rounded"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Max Tree Depth</label>
                  <input
                    type="number"
                    value={newGoal.max_depth}
                    onChange={(e) => setNewGoal({...newGoal, max_depth: parseInt(e.target.value)})}
                    min={3}
                    max={10}
                    className="w-full p-2 border rounded"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Max Experiments</label>
                  <input
                    type="number"
                    value={newGoal.max_experiments}
                    onChange={(e) => setNewGoal({...newGoal, max_experiments: parseInt(e.target.value)})}
                    min={50}
                    max={1000}
                    className="w-full p-2 border rounded"
                  />
                </div>
              </div>

              <button
                onClick={handleStartNewGoal}
                disabled={loading === 'starting'}
                className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
              >
                {loading === 'starting' ? 'Starting Research Tree...' : 'Start Hierarchical Research'}
              </button>
            </div>
          </div>
        )

      case 'tree-view':
        return (
          <div className="flex gap-4 h-screen">
            {/* ROMA-Style Tree Execution View */}
            <div className={`${showNodeDetails ? 'w-2/3' : 'w-full'} bg-white rounded-lg border border-gray-300 transition-all duration-300`}>
              {/* Header */}
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-4 rounded-t-lg border-b border-gray-300">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-xl font-bold text-white">🌳 Research Execution Tree</h3>
                    <p className="text-blue-100 text-sm">Real-time • Hierarchical • Interactive</p>
                  </div>
                  <div className="flex space-x-2">
                    <div className="px-2 py-1 bg-green-500 rounded text-xs text-white">LIVE</div>
                    {selectedNode && (
                      <button
                        onClick={() => setShowNodeDetails(!showNodeDetails)}
                        className="px-3 py-1 bg-white bg-opacity-20 text-white rounded text-xs hover:bg-opacity-30"
                      >
                        {showNodeDetails ? 'Hide Details' : 'Show Details'}
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* ReactFlow Tree Visualization */}
              <div className="h-full">
                <RomaTreeVisualization
                  data={treeVisualization}
                  selectedNodeId={selectedNode?.id}
                  onNodeSelect={(nodeId) => {
                    const node = treeVisualization?.all_nodes?.[nodeId]
                    if (node) {
                      setSelectedNode({
                        id: nodeId,
                        title: node.goal,
                        type: node.task_type,
                        status: node.status,
                        confidence: 0.8, // placeholder
                        depth: node.layer,
                        visits: 5, // placeholder
                        children: []
                      })
                      setShowNodeDetails(true)
                    }
                  }}
                />
              </div>
            </div>

            {/* Node Details Sidebar */}
            {showNodeDetails && selectedNode && (
              <div className="w-1/3 bg-white rounded-lg border border-gray-300 transition-all duration-300">
                {renderNodeDetailsSidebar()}
              </div>
            )}
          </div>
        )

      case 'insights':
        return (
          <div className="space-y-6">
            {insights ? (
              <>
                <div className="bg-white p-6 rounded-lg shadow">
                  <h3 className="text-lg font-semibold mb-4">AI-Generated Insights</h3>

                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium mb-2">Meta-Insights</h4>
                      <ul className="space-y-1">
                        {insights.meta_insights.map((insight: string, index: number) => (
                          <li key={index} className="text-sm bg-blue-50 p-2 rounded">
                            💡 {insight}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">Research Trajectory</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-lg font-bold">{insights.research_trajectory.total_experiments}</div>
                          <div className="text-xs text-gray-600">Total Experiments</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-bold">{(insights.research_trajectory.success_rate * 100).toFixed(1)}%</div>
                          <div className="text-xs text-gray-600">Success Rate</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-bold">{insights.research_trajectory.exploration_breadth}</div>
                          <div className="text-xs text-gray-600">Exploration Breadth</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-bold">{insights.research_trajectory.max_depth_reached}</div>
                          <div className="text-xs text-gray-600">Max Depth</div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">Recommendations</h4>
                      <ul className="space-y-1">
                        {insights.recommendations.map((rec: string, index: number) => (
                          <li key={index} className="text-sm bg-green-50 p-2 rounded">
                            🎯 {rec}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white p-6 rounded-lg shadow">
                  <h3 className="text-lg font-semibold mb-4">Experiment Insights</h3>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {insights.experiment_insights.map((insight: string, index: number) => (
                      <div key={index} className="text-sm bg-gray-50 p-2 rounded">
                        🔬 {insight}
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-white p-6 rounded-lg shadow">
                <p className="text-gray-500">Select a research goal to view insights</p>
              </div>
            )}
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">🌳 Hierarchical AI Research</h1>
              <p className="text-gray-600">Tree Search-Based Scientific Experiment System</p>
            </div>
            <div className="text-sm text-gray-500">
              Parallel • Intelligent • Self-Optimizing
            </div>
          </div>
        </div>
      </header>

      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'dashboard', label: '📊 Dashboard', icon: '📊' },
              { id: 'start-goal', label: '🎯 Start Research', icon: '🎯' },
              { id: 'tree-view', label: '🌳 Tree View', icon: '🌳' },
              { id: 'insights', label: '💡 AI Insights', icon: '💡' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          {loading === 'loading' ? (
            <div className="bg-white p-6 rounded-lg shadow text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p>Loading research data...</p>
            </div>
          ) : (
            renderTabContent()
          )}
        </div>
      </main>

      {/* Node Report Modal */}
      {showNodeReport && nodeReport && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto mx-4">
            <div className="sticky top-0 bg-white border-b px-6 py-4 flex justify-between items-center">
              <h2 className="text-xl font-bold">📊 Full Node Report</h2>
              <button
                onClick={() => setShowNodeReport(false)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Node Overview */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-2">Node Overview</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">ID:</span> {nodeReport.node_id}
                  </div>
                  <div>
                    <span className="font-medium">Type:</span> {nodeReport.node_type}
                  </div>
                  <div>
                    <span className="font-medium">Status:</span>
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      nodeReport.status === 'completed' ? 'bg-green-100 text-green-800' :
                      nodeReport.status === 'running' ? 'bg-blue-100 text-blue-800' :
                      nodeReport.status === 'failed' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {nodeReport.status}
                    </span>
                  </div>
                  <div>
                    <span className="font-medium">Confidence:</span> {(nodeReport.confidence * 100).toFixed(1)}%
                  </div>
                  <div>
                    <span className="font-medium">Title:</span> {nodeReport.title}
                  </div>
                  <div>
                    <span className="font-medium">Retry Count:</span> {nodeReport.retry_count || 0}
                  </div>
                </div>
              </div>

              {/* Error Information */}
              {(nodeReport.last_error || nodeReport.error_history?.length > 0) && (
                <div>
                  <h3 className="font-semibold mb-2 text-red-700">🚨 Error Information</h3>
                  <div className="bg-red-50 p-4 rounded-lg space-y-3">
                    {nodeReport.last_error && (
                      <div>
                        <div className="font-medium text-red-800">Latest Error:</div>
                        <div className="text-sm text-red-700 bg-red-100 p-2 rounded">{nodeReport.last_error}</div>
                      </div>
                    )}
                    {nodeReport.error_history?.length > 0 && (
                      <div>
                        <div className="font-medium text-red-800">Error History ({nodeReport.error_history.length} errors):</div>
                        <div className="space-y-2 max-h-40 overflow-y-auto">
                          {nodeReport.error_history.slice(-3).map((error: any, index: number) => (
                            <div key={index} className="text-xs text-red-700 bg-red-100 p-2 rounded">
                              <div className="font-medium">{error.error_type}: {error.error_message}</div>
                              <div className="text-gray-600">{error.timestamp} (Attempt #{error.retry_count})</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Execution Logs */}
              {nodeReport.execution_logs?.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">📋 Execution Logs</h3>
                  <div className="bg-gray-100 p-4 rounded-lg max-h-60 overflow-y-auto">
                    <div className="space-y-2">
                      {nodeReport.execution_logs.map((log: any, index: number) => (
                        <div key={index} className={`text-xs p-2 rounded ${
                          log.level === 'ERROR' ? 'bg-red-100 text-red-800' :
                          log.level === 'WARNING' ? 'bg-yellow-100 text-yellow-800' :
                          log.level === 'INFO' ? 'bg-blue-100 text-blue-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          <div className="flex justify-between">
                            <span className="font-mono">[{log.level}]</span>
                            <span className="text-gray-600">{new Date(log.timestamp).toLocaleTimeString()}</span>
                          </div>
                          <div className="mt-1">{log.message}</div>
                          {log.context && Object.keys(log.context).length > 0 && (
                            <details className="mt-1">
                              <summary className="cursor-pointer text-gray-600">Context</summary>
                              <pre className="text-xs mt-1 bg-white p-1 rounded">{JSON.stringify(log.context, null, 2)}</pre>
                            </details>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Timing Information */}
              {nodeReport.timestamps && (
                <div>
                  <h3 className="font-semibold mb-2">⏱️ Timing Information</h3>
                  <div className="bg-blue-50 p-4 rounded-lg space-y-2 text-sm">
                    <div>
                      <span className="font-medium">Created:</span> {nodeReport.timestamps.created_at ? new Date(nodeReport.timestamps.created_at).toLocaleString() : 'Unknown'}
                    </div>
                    <div>
                      <span className="font-medium">Started:</span> {nodeReport.timestamps.started_at ? new Date(nodeReport.timestamps.started_at).toLocaleString() : 'Not started'}
                    </div>
                    <div>
                      <span className="font-medium">Completed:</span> {nodeReport.timestamps.completed_at ? new Date(nodeReport.timestamps.completed_at).toLocaleString() : 'Not completed'}
                    </div>
                    {nodeReport.timestamps.total_execution_time && (
                      <div>
                        <span className="font-medium">Total Execution Time:</span> {nodeReport.timestamps.total_execution_time.toFixed(2)}s
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Experiment Configuration */}
              {nodeReport.experiment_config && Object.keys(nodeReport.experiment_config).length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">⚙️ Experiment Configuration</h3>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <pre className="text-xs whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(nodeReport.experiment_config, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {/* Results with Enhanced Debugging */}
              {nodeReport.results?.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">🧪 Experiment Results</h3>
                  <div className="space-y-4">
                    {nodeReport.results.map((result: any, index: number) => (
                      <div key={index} className={`p-4 rounded-lg ${result.success ? 'bg-green-50' : 'bg-red-50'}`}>
                        <div className="text-sm space-y-3">
                          <div className="flex justify-between items-center">
                            <div className="font-medium">Result #{result.result_index + 1}</div>
                            <div className="flex space-x-2">
                              <span className={`px-2 py-1 rounded text-xs ${result.success ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'}`}>
                                {result.success ? '✅ Success' : '❌ Failed'}
                              </span>
                              <span className="px-2 py-1 bg-gray-200 text-gray-800 rounded text-xs">
                                {result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : 'No confidence'}
                              </span>
                            </div>
                          </div>

                          {result.execution_time && (
                            <div>
                              <span className="font-medium">Execution Time:</span> {result.execution_time.toFixed(2)}s
                            </div>
                          )}

                          {result.insights?.length > 0 && (
                            <div>
                              <div className="font-medium">Insights:</div>
                              <ul className="list-disc list-inside space-y-1 ml-2">
                                {result.insights.map((insight: string, i: number) => (
                                  <li key={i} className="text-xs">{insight}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {result.search_queries_used?.length > 0 && (
                            <div>
                              <div className="font-medium">Search Queries Used:</div>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {result.search_queries_used.map((query: string, i: number) => (
                                  <span key={i} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                                    "{query}"
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {result.api_calls?.length > 0 && (
                            <div>
                              <div className="font-medium">API Calls ({result.api_calls.length}):</div>
                              <div className="space-y-1 max-h-32 overflow-y-auto">
                                {result.api_calls.map((call: any, i: number) => (
                                  <div key={i} className="text-xs bg-white p-2 rounded">
                                    <div className="font-mono">{call.api}</div>
                                    <div className="text-gray-600">{new Date(call.timestamp).toLocaleTimeString()}</div>
                                    {call.results_count !== undefined && (
                                      <div className="text-gray-600">Results: {call.results_count}</div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {result.processing_steps?.length > 0 && (
                            <details>
                              <summary className="cursor-pointer font-medium">Processing Steps ({result.processing_steps.length})</summary>
                              <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
                                {result.processing_steps.map((step: any, i: number) => (
                                  <div key={i} className="text-xs bg-white p-2 rounded">
                                    <div className="font-medium">{step.step}</div>
                                    <div className="text-gray-600">{new Date(step.timestamp).toLocaleTimeString()}</div>
                                    {step.query && <div className="text-gray-600">Query: {step.query}</div>}
                                    {step.papers_count !== undefined && <div className="text-gray-600">Papers: {step.papers_count}</div>}
                                  </div>
                                ))}
                              </div>
                            </details>
                          )}

                          {result.error_details && (
                            <div className="bg-red-100 p-2 rounded">
                              <div className="font-medium text-red-800">Error Details:</div>
                              <div className="text-xs text-red-700">{result.error_details.error_type}: {result.error_details.error_message}</div>
                            </div>
                          )}

                          {result.stack_trace && (
                            <details>
                              <summary className="cursor-pointer font-medium text-red-700">Stack Trace</summary>
                              <pre className="text-xs mt-1 bg-red-100 p-2 rounded overflow-x-auto">{result.stack_trace}</pre>
                            </details>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tree Search Metrics */}
              {nodeReport.tree_search_metrics && (
                <div>
                  <h3 className="font-semibold mb-2">🌳 Tree Search Metrics</h3>
                  <div className="bg-purple-50 p-4 rounded-lg space-y-2 text-sm">
                    <div>
                      <span className="font-medium">UCB Score:</span> {nodeReport.tree_search_metrics.ucb_score?.toFixed(3) || 'N/A'}
                    </div>
                    <div>
                      <span className="font-medium">Exploration Bonus:</span> {nodeReport.tree_search_metrics.exploration_bonus?.toFixed(3) || 'N/A'}
                    </div>
                    <div>
                      <span className="font-medium">Exploitation Score:</span> {nodeReport.tree_search_metrics.exploitation_score?.toFixed(3) || 'N/A'}
                    </div>
                    <div>
                      <span className="font-medium">Total Reward:</span> {nodeReport.tree_search_metrics.total_reward?.toFixed(3) || 'N/A'}
                    </div>
                  </div>
                </div>
              )}

              {/* Raw Debug Data */}
              <details>
                <summary className="cursor-pointer font-semibold">🔍 Raw Debug Data</summary>
                <div className="mt-2 bg-gray-100 p-4 rounded-lg">
                  <pre className="text-xs whitespace-pre-wrap overflow-x-auto">
                    {JSON.stringify(nodeReport, null, 2)}
                  </pre>
                </div>
              </details>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}