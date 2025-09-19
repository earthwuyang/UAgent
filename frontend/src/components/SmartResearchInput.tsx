import React, { useState } from 'react'
import { Send, Sparkles, AlertCircle, CheckCircle } from 'lucide-react'
import { useQuery, useMutation } from 'react-query'
import UAgentAPI from '../services/api'
import type { ClassificationRequest, ClassificationResult, RouteAndExecuteAck } from '../types/api'
import { clsx } from 'clsx'

interface SmartResearchInputProps {
  onResultReceived?: (ack: RouteAndExecuteAck) => void
}

const SmartResearchInput: React.FC<SmartResearchInputProps> = ({ onResultReceived }) => {
  const [query, setQuery] = useState('')
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null)
  const [showClassification, setShowClassification] = useState(false)

  // Fetch engine info for display
  const { data: engineInfo } = useQuery('engineInfo', UAgentAPI.getEngineInfo)

  // Classification mutation
  const classifyMutation = useMutation(UAgentAPI.classifyRequest, {
    onSuccess: (result) => {
      setClassificationResult(result)
      setShowClassification(true)
    },
    onError: (error) => {
      console.error('Classification failed:', error)
    },
  })

  // Route and execute mutation
  const executeWithRouting = useMutation(UAgentAPI.routeAndExecute, {
    onSuccess: (result) => {
      onResultReceived?.(result)
      setShowClassification(false)
      setQuery('')
      setClassificationResult(null)
    },
    onError: (error) => {
      console.error('Execution failed:', error)
    },
  })

  const handleClassify = () => {
    if (!query.trim()) return

    const request: ClassificationRequest = {
      user_request: query.trim(),
    }

    classifyMutation.mutate(request)
  }

  const handleExecute = () => {
    if (!query.trim()) return

    const request: ClassificationRequest = {
      user_request: query.trim(),
    }

    executeWithRouting.mutate(request)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!showClassification) {
      handleClassify()
    } else {
      handleExecute()
    }
  }

  const getEngineColor = (engine: string) => {
    switch (engine) {
      case 'DEEP_RESEARCH':
        return 'bg-blue-100 text-blue-800'
      case 'CODE_RESEARCH':
        return 'bg-green-100 text-green-800'
      case 'SCIENTIFIC_RESEARCH':
        return 'bg-purple-100 text-purple-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getEngineDescription = (engine: string) => {
    if (!engineInfo) return ''
    return engineInfo.engines[engine]?.description || ''
  }

  return (
    <div className="space-y-6">
      {/* Main Input */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask me anything... I'll intelligently route your request to the best research engine.

Examples:
• 'What are the latest trends in AI?' → Deep Research
• 'Find Python libraries for machine learning' → Code Research
• 'Design experiments to test transformer attention mechanisms' → Scientific Research"
            className="textarea min-h-[120px] pr-12 resize-none"
            disabled={classifyMutation.isLoading || executeWithRouting.isLoading}
          />
          <div className="absolute bottom-3 right-3">
            <Sparkles className="h-5 w-5 text-primary-400" />
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-500">
            {query.length > 0 && (
              <span>{query.length} characters</span>
            )}
          </div>

          <div className="flex space-x-3">
            {showClassification && (
              <button
                type="button"
                onClick={() => {
                  setShowClassification(false)
                  setClassificationResult(null)
                }}
                className="btn btn-outline"
              >
                Modify Query
              </button>
            )}

            <button
              type="submit"
              disabled={!query.trim() || classifyMutation.isLoading || executeWithRouting.isLoading}
              className="btn btn-primary flex items-center space-x-2"
            >
              {classifyMutation.isLoading ? (
                <>
                  <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                  <span>Analyzing<span className="loading-dots" /></span>
                </>
              ) : executeWithRouting.isLoading ? (
                <>
                  <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                  <span>Researching<span className="loading-dots" /></span>
                </>
              ) : showClassification ? (
                <>
                  <Send className="h-4 w-4" />
                  <span>Execute Research</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  <span>Analyze & Route</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Classification Result */}
      {showClassification && classificationResult && (
        <div className="card p-6 border-l-4 border-l-primary-500">
          <div className="flex items-start space-x-4">
            <CheckCircle className="h-6 w-6 text-green-500 mt-1 flex-shrink-0" />
            <div className="flex-1 space-y-4">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Smart Classification Complete
                </h3>
                <div className="flex items-center space-x-3 mb-3">
                  <span className={clsx('badge', getEngineColor(classificationResult.primary_engine))}>
                    {classificationResult.primary_engine.replace('_', ' ')}
                  </span>
                  <span className="text-sm text-gray-500">
                    Confidence: {(classificationResult.confidence_score * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-sm text-gray-600 mb-4">
                  {getEngineDescription(classificationResult.primary_engine)}
                </p>
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Reasoning:</h4>
                  <p className="text-sm text-gray-700">{classificationResult.reasoning}</p>
                </div>
              </div>

              {/* Sub-components */}
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2">Analysis Components:</h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(classificationResult.sub_components).map(([component, enabled]) => (
                    <span
                      key={component}
                      className={clsx(
                        'badge',
                        enabled ? 'badge-success' : 'badge-secondary'
                      )}
                    >
                      {component.replace('_', ' ')}
                    </span>
                  ))}
                </div>
              </div>

              {/* Workflow Plan */}
              {classificationResult.workflow_plan.sub_workflows.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-900 mb-2">
                    Planned Workflow ({classificationResult.workflow_plan.sub_workflows.length} steps):
                  </h4>
                  <div className="space-y-2">
                    {classificationResult.workflow_plan.sub_workflows.map((workflow, index) => (
                      <div key={index} className="flex items-center space-x-3 text-sm">
                        <span className="flex-shrink-0 w-6 h-6 bg-primary-100 text-primary-700 rounded-full flex items-center justify-center text-xs font-medium">
                          {index + 1}
                        </span>
                        <div>
                          <span className="font-medium">{workflow.phase.replace('_', ' ')}</span>
                          <span className="text-gray-500 ml-2">({workflow.priority} priority)</span>
                          {workflow.includes_openhands && (
                            <span className="badge badge-warning ml-2">Code Execution</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <AlertCircle className="h-4 w-4" />
                <span>
                  Ready to execute with {classificationResult.primary_engine.replace('_', ' ')} engine
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error States */}
      {classifyMutation.error && (
        <div className="card p-4 border-l-4 border-l-red-500 bg-red-50">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            <span className="text-sm text-red-700">
              Classification failed. Please try again.
            </span>
          </div>
        </div>
      )}

      {executeWithRouting.error && (
        <div className="card p-4 border-l-4 border-l-red-500 bg-red-50">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            <span className="text-sm text-red-700">
              Research execution failed. Please try again.
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default SmartResearchInput
