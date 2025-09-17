import React, { useState, useEffect } from 'react'
import {
  executeWorkflow,
  getWorkflowStatus,
  getWorkflowResults,
  listActiveWorkflows,
  cancelWorkflow,
  startResearchProject,
  analyzeCodeRepository,
  intelligentSearch,
  unifiedSearch,
  getWorkflowTemplates,
  getSystemStatus,
  getSystemHealth
} from '../services/api'

interface WorkflowStatus {
  workflow_id: string
  workflow_type: string
  status: string
  components_used: string[]
  execution_time?: number
  created_at: string
  completed_at?: string
}

interface SystemStatus {
  components: any
  workflows: any
  templates_available: string[]
}

export default function UnifiedDashboard() {
  // Workflow Management State
  const [activeWorkflows, setActiveWorkflows] = useState<WorkflowStatus[]>([])
  const [selectedWorkflow, setSelectedWorkflow] = useState<string>('')
  const [workflowResults, setWorkflowResults] = useState<any>(null)
  const [templates, setTemplates] = useState<any>(null)

  // Research Project State
  const [researchTitle, setResearchTitle] = useState('AI Agent Architecture Study')
  const [researchDescription, setResearchDescription] = useState('Comprehensive study of AI agent architectures and design patterns')
  const [researchQuestions, setResearchQuestions] = useState('What are the key components of effective AI agents?\nHow do different architectures compare?')

  // Code Analysis State
  const [repoPath, setRepoPath] = useState('/path/to/repository')
  const [analysisDepth, setAnalysisDepth] = useState('semantic')

  // Search State
  const [searchQuery, setSearchQuery] = useState('machine learning agent frameworks')
  const [searchTypes, setSearchTypes] = useState(['web', 'academic', 'code'])
  const [searchResults, setSearchResults] = useState<any>(null)

  // System State
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [systemHealth, setSystemHealth] = useState<any>(null)

  // UI State
  const [activeTab, setActiveTab] = useState('dashboard')
  const [loading, setLoading] = useState<string>('')

  // Load initial data
  useEffect(() => {
    loadInitialData()
    const interval = setInterval(refreshActiveWorkflows, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const loadInitialData = async () => {
    try {
      await Promise.all([
        refreshActiveWorkflows(),
        loadTemplates(),
        loadSystemStatus(),
        loadSystemHealth()
      ])
    } catch (error) {
      console.error('Failed to load initial data:', error)
    }
  }

  const refreshActiveWorkflows = async () => {
    try {
      const response = await listActiveWorkflows()
      setActiveWorkflows(response.active_workflows || [])
    } catch (error) {
      console.error('Failed to load active workflows:', error)
    }
  }

  const loadTemplates = async () => {
    try {
      const response = await getWorkflowTemplates()
      setTemplates(response)
    } catch (error) {
      console.error('Failed to load templates:', error)
    }
  }

  const loadSystemStatus = async () => {
    try {
      const response = await getSystemStatus()
      setSystemStatus(response)
    } catch (error) {
      console.error('Failed to load system status:', error)
    }
  }

  const loadSystemHealth = async () => {
    try {
      const response = await getSystemHealth()
      setSystemHealth(response)
    } catch (error) {
      console.error('Failed to load system health:', error)
    }
  }

  const handleStartResearch = async () => {
    setLoading('research')
    try {
      const projectData = {
        title: researchTitle,
        description: researchDescription,
        research_questions: researchQuestions.split('\n').filter(q => q.trim()),
        collaboration_enabled: true,
        auto_iteration: true
      }
      const response = await startResearchProject(projectData)
      alert(`Research project started! Workflow ID: ${response.workflow_id}`)
      await refreshActiveWorkflows()
    } catch (error) {
      alert(`Failed to start research: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handleAnalyzeCode = async () => {
    setLoading('code')
    try {
      const repoData = {
        repository_path: repoPath,
        analysis_depth: analysisDepth,
        pattern_detection: true,
        collaboration_enabled: true
      }
      const response = await analyzeCodeRepository(repoData)
      alert(`Code analysis started! Workflow ID: ${response.workflow_id}`)
      await refreshActiveWorkflows()
    } catch (error) {
      alert(`Failed to analyze code: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handleIntelligentSearch = async () => {
    setLoading('search')
    try {
      const searchData = {
        query: searchQuery,
        context: { domain: 'ai_ml' }
      }
      const response = await intelligentSearch(searchData)
      setSearchResults(response)
    } catch (error) {
      alert(`Search failed: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handleUnifiedSearch = async () => {
    setLoading('unified-search')
    try {
      const searchData = {
        query: searchQuery,
        search_types: searchTypes,
        limit: 20
      }
      const response = await unifiedSearch(searchData)
      setSearchResults(response)
    } catch (error) {
      alert(`Unified search failed: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handleExecuteTemplate = async (templateName: string) => {
    setLoading(`template-${templateName}`)
    try {
      const workflowConfig = {
        template_name: templateName,
        inputs: {
          title: 'Template-based workflow',
          query: searchQuery,
          repository_path: repoPath
        }
      }
      const response = await executeWorkflow(workflowConfig)
      alert(`Workflow started! ID: ${response.workflow_id}`)
      await refreshActiveWorkflows()
    } catch (error) {
      alert(`Failed to execute template: ${error}`)
    } finally {
      setLoading('')
    }
  }

  const handleCheckWorkflowResults = async (workflowId: string) => {
    try {
      const response = await getWorkflowResults(workflowId)
      setWorkflowResults(response)
      setSelectedWorkflow(workflowId)
    } catch (error) {
      alert(`Failed to get results: ${error}`)
    }
  }

  const handleCancelWorkflow = async (workflowId: string) => {
    try {
      await cancelWorkflow(workflowId)
      alert('Workflow cancelled')
      await refreshActiveWorkflows()
    } catch (error) {
      alert(`Failed to cancel workflow: ${error}`)
    }
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <div className="space-y-6">
            {/* System Health Overview */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">System Health</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className={`text-2xl font-bold ${systemHealth?.status === 'healthy' ? 'text-green-500' : 'text-red-500'}`}>
                    {systemHealth?.status || 'Unknown'}
                  </div>
                  <div className="text-sm text-gray-600">Overall Status</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-500">
                    {systemStatus?.workflows?.active || 0}
                  </div>
                  <div className="text-sm text-gray-600">Active Workflows</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-500">
                    {systemStatus?.components?.meta_agent?.total_agents || 0}
                  </div>
                  <div className="text-sm text-gray-600">Available Agents</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-500">
                    {systemStatus?.templates_available?.length || 0}
                  </div>
                  <div className="text-sm text-gray-600">Templates</div>
                </div>
              </div>
            </div>

            {/* Active Workflows */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Active Workflows</h3>
              {activeWorkflows.length === 0 ? (
                <p className="text-gray-500">No active workflows</p>
              ) : (
                <div className="space-y-3">
                  {activeWorkflows.map((workflow) => (
                    <div key={workflow.workflow_id} className="border rounded p-3 flex justify-between items-center">
                      <div>
                        <div className="font-medium">{workflow.workflow_type}</div>
                        <div className="text-sm text-gray-600">
                          Status: <span className={`font-medium ${
                            workflow.status === 'completed' ? 'text-green-600' :
                            workflow.status === 'failed' ? 'text-red-600' :
                            'text-blue-600'
                          }`}>{workflow.status}</span>
                        </div>
                        <div className="text-xs text-gray-500">
                          ID: {workflow.workflow_id.slice(0, 8)}...
                        </div>
                      </div>
                      <div className="space-x-2">
                        <button
                          onClick={() => handleCheckWorkflowResults(workflow.workflow_id)}
                          className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
                        >
                          Results
                        </button>
                        <button
                          onClick={() => handleCancelWorkflow(workflow.workflow_id)}
                          className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Quick Actions */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {templates?.templates_available?.map((template: string) => (
                  <button
                    key={template}
                    onClick={() => handleExecuteTemplate(template)}
                    disabled={loading === `template-${template}`}
                    className="p-3 bg-gray-100 rounded hover:bg-gray-200 text-sm font-medium disabled:opacity-50"
                  >
                    {loading === `template-${template}` ? 'Starting...' : template.replace('_', ' ').toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )

      case 'research':
        return (
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Start Research Project</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Research Title</label>
                <input
                  type="text"
                  value={researchTitle}
                  onChange={(e) => setResearchTitle(e.target.value)}
                  className="w-full p-2 border rounded"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Description</label>
                <textarea
                  value={researchDescription}
                  onChange={(e) => setResearchDescription(e.target.value)}
                  rows={3}
                  className="w-full p-2 border rounded"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Research Questions (one per line)</label>
                <textarea
                  value={researchQuestions}
                  onChange={(e) => setResearchQuestions(e.target.value)}
                  rows={4}
                  className="w-full p-2 border rounded"
                />
              </div>
              <button
                onClick={handleStartResearch}
                disabled={loading === 'research'}
                className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
              >
                {loading === 'research' ? 'Starting Research...' : 'Start Research Project'}
              </button>
            </div>
          </div>
        )

      case 'code':
        return (
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Code Repository Analysis</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Repository Path</label>
                <input
                  type="text"
                  value={repoPath}
                  onChange={(e) => setRepoPath(e.target.value)}
                  className="w-full p-2 border rounded"
                  placeholder="/absolute/path/to/repository"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Analysis Depth</label>
                <select
                  value={analysisDepth}
                  onChange={(e) => setAnalysisDepth(e.target.value)}
                  className="w-full p-2 border rounded"
                >
                  <option value="surface">Surface</option>
                  <option value="semantic">Semantic</option>
                  <option value="deep">Deep</option>
                  <option value="comprehensive">Comprehensive</option>
                </select>
              </div>
              <button
                onClick={handleAnalyzeCode}
                disabled={loading === 'code'}
                className="px-6 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
              >
                {loading === 'code' ? 'Analyzing...' : 'Analyze Repository'}
              </button>
            </div>
          </div>
        )

      case 'search':
        return (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Multi-Modal Search</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Search Query</label>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full p-2 border rounded"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Search Types</label>
                  <div className="flex space-x-4">
                    {['web', 'academic', 'code', 'documentation', 'news'].map(type => (
                      <label key={type} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={searchTypes.includes(type)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSearchTypes([...searchTypes, type])
                            } else {
                              setSearchTypes(searchTypes.filter(t => t !== type))
                            }
                          }}
                          className="mr-2"
                        />
                        {type}
                      </label>
                    ))}
                  </div>
                </div>
                <div className="space-x-4">
                  <button
                    onClick={handleIntelligentSearch}
                    disabled={loading === 'search'}
                    className="px-6 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50"
                  >
                    {loading === 'search' ? 'Searching...' : 'Intelligent Search'}
                  </button>
                  <button
                    onClick={handleUnifiedSearch}
                    disabled={loading === 'unified-search'}
                    className="px-6 py-2 bg-indigo-500 text-white rounded hover:bg-indigo-600 disabled:opacity-50"
                  >
                    {loading === 'unified-search' ? 'Searching...' : 'Unified Search'}
                  </button>
                </div>
              </div>
            </div>

            {/* Search Results */}
            {searchResults && (
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-semibold mb-4">Search Results</h3>
                <div className="space-y-4">
                  {Object.entries(searchResults.results || searchResults).map(([source, results]: [string, any]) => {
                    if (!Array.isArray(results) || results.length === 0) return null
                    return (
                      <div key={source}>
                        <h4 className="font-medium text-gray-800 mb-2 capitalize">{source} Results</h4>
                        <div className="space-y-2">
                          {results.slice(0, 5).map((result: any, index: number) => (
                            <div key={index} className="border-l-4 border-blue-300 pl-3">
                              <a
                                href={result.url || result.link}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-600 hover:underline font-medium"
                              >
                                {result.title}
                              </a>
                              <p className="text-sm text-gray-600 mt-1">{result.snippet || result.description}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )

      case 'results':
        return (
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Workflow Results</h3>
            {selectedWorkflow ? (
              <div>
                <p className="text-sm text-gray-600 mb-4">Workflow ID: {selectedWorkflow}</p>
                {workflowResults ? (
                  <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-96 text-sm">
                    {JSON.stringify(workflowResults, null, 2)}
                  </pre>
                ) : (
                  <p className="text-gray-500">Loading results...</p>
                )}
              </div>
            ) : (
              <p className="text-gray-500">Select a workflow from the dashboard to view results</p>
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
              <h1 className="text-3xl font-bold text-gray-900">uagent</h1>
              <p className="text-gray-600">Unified AI Agent Platform</p>
            </div>
            <div className="text-sm text-gray-500">
              ROMA + AI-Scientist + AgentLab + RepoMaster
            </div>
          </div>
        </div>
      </header>

      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'dashboard', label: 'Dashboard' },
              { id: 'research', label: 'Research' },
              { id: 'code', label: 'Code Analysis' },
              { id: 'search', label: 'Search' },
              { id: 'results', label: 'Results' }
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
          {renderTabContent()}
        </div>
      </main>
    </div>
  )
}