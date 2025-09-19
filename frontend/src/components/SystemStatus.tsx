import React from 'react'
import { useQuery } from 'react-query'
import { CheckCircle, XCircle, AlertCircle, Cpu, Database, Zap } from 'lucide-react'
import UAgentAPI from '../services/api'
import { clsx } from 'clsx'

const SystemStatus: React.FC = () => {
  const { data: health, isLoading: healthLoading } = useQuery(
    'health',
    UAgentAPI.getHealth,
    { refetchInterval: 30000 } // Refetch every 30 seconds
  )

  const { data: status, isLoading: statusLoading } = useQuery(
    'systemStatus',
    UAgentAPI.getSystemStatus,
    { refetchInterval: 30000 }
  )

  const { data: engineInfo, isLoading: engineLoading } = useQuery(
    'engineInfo',
    UAgentAPI.getEngineInfo
  )

  const isLoading = healthLoading || statusLoading || engineLoading

  const getStatusIcon = (available: boolean) => {
    return available ? CheckCircle : XCircle
  }

  const getStatusColor = (available: boolean) => {
    return available ? 'text-green-500' : 'text-red-500'
  }

  const getStatusBadge = (available: boolean) => {
    return available ? 'badge-success' : 'badge-error'
  }

  const getOverallStatus = () => {
    if (!health || !status) return 'Unknown'

    const healthOk = health.status === 'healthy'
    const systemOk = status.status === 'operational'
    const enginesOk = Object.values(status.engines).every(engine => engine.available)

    if (healthOk && systemOk && enginesOk) return 'Healthy'
    if (!healthOk || !systemOk) return 'Error'
    return 'Warning'
  }

  const overallStatus = getOverallStatus()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">System Status</h1>
          <p className="text-gray-600">Real-time monitoring of UAgent system components</p>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            {overallStatus === 'Healthy' && <CheckCircle className="h-5 w-5 text-green-500" />}
            {overallStatus === 'Warning' && <AlertCircle className="h-5 w-5 text-yellow-500" />}
            {overallStatus === 'Error' && <XCircle className="h-5 w-5 text-red-500" />}
            <span className={clsx('font-medium', {
              'text-green-700': overallStatus === 'Healthy',
              'text-yellow-700': overallStatus === 'Warning',
              'text-red-700': overallStatus === 'Error',
            })}>
              {overallStatus}
            </span>
          </div>

          {!isLoading && (
            <span className="text-sm text-gray-500">
              Last updated: {new Date().toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin h-8 w-8 border-4 border-primary-500 border-t-transparent rounded-full" />
        </div>
      ) : (
        <div className="grid gap-6">
          {/* Overall Health */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">System Health</h2>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="flex items-center space-x-3 p-4 bg-gray-50 rounded-lg">
                <Cpu className="h-6 w-6 text-blue-500" />
                <div>
                  <div className="font-medium text-gray-900">API Status</div>
                  <div className="text-sm text-gray-600">{health?.status || 'Unknown'}</div>
                </div>
                <div className="ml-auto">
                  <span className={clsx('badge', health?.status === 'healthy' ? 'badge-success' : 'badge-error')}>
                    {health?.status === 'healthy' ? 'Online' : 'Offline'}
                  </span>
                </div>
              </div>

              <div className="flex items-center space-x-3 p-4 bg-gray-50 rounded-lg">
                <Database className="h-6 w-6 text-green-500" />
                <div>
                  <div className="font-medium text-gray-900">Cache System</div>
                  <div className="text-sm text-gray-600">
                    {status?.cache_available ? 'Active' : 'Inactive'}
                  </div>
                </div>
                <div className="ml-auto">
                  <span className={clsx('badge', getStatusBadge(status?.cache_available || false))}>
                    {status?.cache_available ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>

              <div className="flex items-center space-x-3 p-4 bg-gray-50 rounded-lg">
                <Zap className="h-6 w-6 text-purple-500" />
                <div>
                  <div className="font-medium text-gray-900">LLM Client</div>
                  <div className="text-sm text-gray-600">
                    {status?.llm_client_available ? 'Ready' : 'Unavailable'}
                  </div>
                </div>
                <div className="ml-auto">
                  <span className={clsx('badge', getStatusBadge(status?.llm_client_available || false))}>
                    {status?.llm_client_available ? 'Ready' : 'Error'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Smart Router Status */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Smart Router</h2>
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className={clsx('p-2 rounded-lg', status?.smart_router.available ? 'bg-green-100' : 'bg-red-100')}>
                  {React.createElement(getStatusIcon(status?.smart_router.available || false), {
                    className: clsx('h-5 w-5', getStatusColor(status?.smart_router.available || false))
                  })}
                </div>
                <div>
                  <div className="font-medium text-gray-900">Classification Engine</div>
                  <div className="text-sm text-gray-600">
                    {status?.smart_router.type || 'Unknown'}
                  </div>
                </div>
              </div>
              <span className={clsx('badge', getStatusBadge(status?.smart_router.available || false))}>
                {status?.smart_router.available ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>

          {/* Research Engines */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Research Engines</h2>
            <div className="space-y-4">
              {status?.engines && Object.entries(status.engines).map(([engineName, engineStatus]) => {
                const engineDetails = engineInfo?.engines[engineName.toUpperCase()]
                const StatusIcon = getStatusIcon(engineStatus.available)

                return (
                  <div key={engineName} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className={clsx('p-2 rounded-lg', engineStatus.available ? 'bg-green-100' : 'bg-red-100')}>
                        <StatusIcon className={clsx('h-5 w-5', getStatusColor(engineStatus.available))} />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium text-gray-900">
                            {engineDetails?.name || engineName.replace('_', ' ')}
                          </span>
                          <span className="text-sm text-gray-500">
                            ({engineStatus.type})
                          </span>
                        </div>
                        {engineDetails?.description && (
                          <p className="text-sm text-gray-600 mt-1">{engineDetails.description}</p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={clsx('badge', getStatusBadge(engineStatus.available))}>
                        {engineStatus.available ? 'Ready' : 'Error'}
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Engine Capabilities */}
          {engineInfo && (
            <div className="card p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Engine Capabilities</h2>
              <div className="grid md:grid-cols-3 gap-6">
                {Object.entries(engineInfo.engines).map(([engineKey, engine]) => (
                  <div key={engineKey} className="space-y-3">
                    <h3 className="font-medium text-gray-900">{engine.name}</h3>
                    <div className="space-y-2">
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-1">Capabilities:</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {engine.capabilities.map((capability, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <span className="text-primary-500 mt-1">•</span>
                              <span>{capability}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-1">Best For:</h4>
                        <div className="flex flex-wrap gap-1">
                          {engine.best_for.slice(0, 3).map((use, index) => (
                            <span key={index} className="badge badge-secondary text-xs">
                              {use}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Version Information */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Version Information</h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-700">API Version:</span>
                  <span className="text-sm text-gray-600">{health?.version || 'Unknown'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-700">Total Engines:</span>
                  <span className="text-sm text-gray-600">{status?.total_engines || 0}</span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-700">Smart Router:</span>
                  <span className="text-sm text-gray-600">{health?.smart_router || 'Unknown'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-700">Last Check:</span>
                  <span className="text-sm text-gray-600">{new Date().toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default SystemStatus