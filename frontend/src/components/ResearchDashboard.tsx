import React from 'react'
import { useQuery } from 'react-query'
import { History, Search, Trash2, ExternalLink, Clock, CheckCircle } from 'lucide-react'
import UAgentAPI from '../services/api'
import { clsx } from 'clsx'

const ResearchDashboard: React.FC = () => {
  const { data: sessions, isLoading, refetch } = useQuery(
    'researchSessions',
    UAgentAPI.getResearchSessions,
    { refetchInterval: 10000 } // Refetch every 10 seconds
  )

  const handleDeleteSession = async (researchId: string) => {
    try {
      await UAgentAPI.deleteResearchSession(researchId)
      refetch()
    } catch (error) {
      console.error('Failed to delete session:', error)
    }
  }

  const getEngineColor = (type: string) => {
    switch (type) {
      case 'deep_research':
        return 'bg-blue-100 text-blue-800'
      case 'code_research':
        return 'bg-green-100 text-green-800'
      case 'scientific_research':
        return 'bg-purple-100 text-purple-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600'
      case 'in_progress':
        return 'text-yellow-600'
      case 'failed':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return CheckCircle
      case 'in_progress':
        return Clock
      default:
        return Clock
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Research Dashboard</h1>
          <p className="text-gray-600">View and manage your research sessions</p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={() => refetch()}
            className="btn btn-outline flex items-center space-x-2"
          >
            <Search className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Stats */}
      {sessions && (
        <div className="grid md:grid-cols-4 gap-4">
          <div className="card p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-primary-100 rounded-lg">
                <History className="h-5 w-5 text-primary-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">{sessions.total}</div>
                <div className="text-sm text-gray-600">Total Sessions</div>
              </div>
            </div>
          </div>

          <div className="card p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Search className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {sessions.sessions.filter(s => s.type === 'deep_research').length}
                </div>
                <div className="text-sm text-gray-600">Deep Research</div>
              </div>
            </div>
          </div>

          <div className="card p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <Search className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {sessions.sessions.filter(s => s.type === 'code_research').length}
                </div>
                <div className="text-sm text-gray-600">Code Research</div>
              </div>
            </div>
          </div>

          <div className="card p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Search className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {sessions.sessions.filter(s => s.type === 'scientific_research').length}
                </div>
                <div className="text-sm text-gray-600">Scientific Research</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Sessions List */}
      <div className="card">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Recent Sessions</h2>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin h-8 w-8 border-4 border-primary-500 border-t-transparent rounded-full" />
          </div>
        ) : sessions && sessions.sessions.length > 0 ? (
          <div className="divide-y divide-gray-200">
            {sessions.sessions.map((session) => {
              const StatusIcon = getStatusIcon(session.status)

              return (
                <div key={session.research_id} className="p-6 hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 space-y-3">
                      {/* Header */}
                      <div className="flex items-center space-x-3">
                        <StatusIcon className={clsx('h-5 w-5', getStatusColor(session.status))} />
                        <span className={clsx('badge', getEngineColor(session.type))}>
                          {session.type.replace('_', ' ')}
                        </span>
                        <span className="text-sm text-gray-500">
                          {session.created_at || 'Unknown time'}
                        </span>
                      </div>

                      {/* Query */}
                      <div>
                        <h3 className="font-medium text-gray-900 mb-1">
                          {session.query}
                        </h3>
                        <p className="text-sm text-gray-600">
                          Research ID: {session.research_id}
                        </p>
                      </div>

                      {/* Status */}
                      <div className="flex items-center space-x-4">
                        <span className={clsx('text-sm font-medium', getStatusColor(session.status))}>
                          {session.status.charAt(0).toUpperCase() + session.status.slice(1)}
                        </span>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center space-x-2 ml-4">
                      <button
                        onClick={() => {
                          // Navigate to session details
                          window.open(`/research/sessions/${session.research_id}`, '_blank')
                        }}
                        className="btn btn-outline btn-sm flex items-center space-x-1"
                      >
                        <ExternalLink className="h-3 w-3" />
                        <span>View</span>
                      </button>
                      <button
                        onClick={() => handleDeleteSession(session.research_id)}
                        className="btn btn-outline btn-sm text-red-600 border-red-300 hover:bg-red-50 flex items-center space-x-1"
                      >
                        <Trash2 className="h-3 w-3" />
                        <span>Delete</span>
                      </button>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="text-center py-12">
            <History className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Research Sessions</h3>
            <p className="text-gray-600 mb-6">
              Start your first research session from the home page.
            </p>
            <a href="/" className="btn btn-primary">
              Start Research
            </a>
          </div>
        )}
      </div>

      {/* Recent Activity Summary */}
      {sessions && sessions.sessions.length > 0 && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h2>
          <div className="space-y-3">
            {sessions.sessions.slice(0, 5).map((session) => (
              <div key={session.research_id} className="flex items-center space-x-3 text-sm">
                <div className={clsx('w-2 h-2 rounded-full', {
                  'bg-green-500': session.status === 'completed',
                  'bg-yellow-500': session.status === 'in_progress',
                  'bg-red-500': session.status === 'failed',
                  'bg-gray-500': !['completed', 'in_progress', 'failed'].includes(session.status),
                })} />
                <span className="text-gray-600">
                  {session.status === 'completed' ? 'Completed' :
                   session.status === 'in_progress' ? 'Started' :
                   'Attempted'} research:
                </span>
                <span className="font-medium text-gray-900 truncate max-w-md">
                  "{session.query}"
                </span>
                <span className="text-gray-500 ml-auto">
                  {session.created_at || 'Recently'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ResearchDashboard