import React from 'react'
import { CheckCircle, Search, Code, FlaskConical, ExternalLink, Copy, Download } from 'lucide-react'
import type { RouteAndExecuteResponse } from '../types/api'
import { clsx } from 'clsx'

interface ResearchResultProps {
  result: RouteAndExecuteResponse
}

const ResearchResult: React.FC<ResearchResultProps> = ({ result }) => {
  const { classification, execution } = result

  const getEngineIcon = (engine: string) => {
    switch (engine.toLowerCase()) {
      case 'deep_research':
        return Search
      case 'code_research':
        return Code
      case 'scientific_research':
        return FlaskConical
      default:
        return Search
    }
  }

  const getEngineColor = (engine: string) => {
    switch (engine.toLowerCase()) {
      case 'deep_research':
        return {
          bg: 'bg-blue-100',
          text: 'text-blue-800',
          accent: 'border-blue-500',
        }
      case 'code_research':
        return {
          bg: 'bg-green-100',
          text: 'text-green-800',
          accent: 'border-green-500',
        }
      case 'scientific_research':
        return {
          bg: 'bg-purple-100',
          text: 'text-purple-800',
          accent: 'border-purple-500',
        }
      default:
        return {
          bg: 'bg-gray-100',
          text: 'text-gray-800',
          accent: 'border-gray-500',
        }
    }
  }

  const colors = getEngineColor(execution.engine_used)
  const EngineIcon = getEngineIcon(execution.engine_used)

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const renderExecutionDetails = () => {
    switch (execution.engine_used) {
      case 'deep_research':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{execution.sources_count}</div>
                <div className="text-sm text-gray-600">Sources Analyzed</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{execution.key_findings?.length || 0}</div>
                <div className="text-sm text-gray-600">Key Findings</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{execution.recommendations?.length || 0}</div>
                <div className="text-sm text-gray-600">Recommendations</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{(execution.confidence_score * 100).toFixed(0)}%</div>
                <div className="text-sm text-gray-600">Confidence</div>
              </div>
            </div>

            {execution.key_findings && execution.key_findings.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">Key Findings</h4>
                <ul className="space-y-2">
                  {execution.key_findings.map((finding: string, index: number) => (
                    <li key={index} className="flex items-start space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-700">{finding}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {execution.summary && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">Research Summary</h4>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-700 leading-relaxed">{execution.summary}</p>
                </div>
              </div>
            )}
          </div>
        )

      case 'code_research':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{execution.repositories_count}</div>
                <div className="text-sm text-gray-600">Repositories</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{execution.languages_found?.length || 0}</div>
                <div className="text-sm text-gray-600">Languages</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{execution.best_practices_count || 0}</div>
                <div className="text-sm text-gray-600">Best Practices</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{(execution.confidence_score * 100).toFixed(0)}%</div>
                <div className="text-sm text-gray-600">Confidence</div>
              </div>
            </div>

            {execution.languages_found && execution.languages_found.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">Programming Languages</h4>
                <div className="flex flex-wrap gap-2">
                  {execution.languages_found.map((language: string, index: number) => (
                    <span key={index} className="badge badge-secondary">
                      {language}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {execution.integration_guide_preview && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">Integration Guide</h4>
                <div className="bg-gray-50 rounded-lg p-4">
                  <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono">
                    {execution.integration_guide_preview}
                  </pre>
                </div>
              </div>
            )}
          </div>
        )

      case 'scientific_research':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{execution.hypotheses_count || 0}</div>
                <div className="text-sm text-gray-600">Hypotheses</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{execution.experiments_count || 0}</div>
                <div className="text-sm text-gray-600">Experiments</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{execution.iterations_completed || 0}</div>
                <div className="text-sm text-gray-600">Iterations</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{(execution.confidence_score * 100).toFixed(0)}%</div>
                <div className="text-sm text-gray-600">Confidence</div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                <CheckCircle className={clsx('h-5 w-5', execution.has_literature_review ? 'text-green-500' : 'text-gray-400')} />
                <span className="text-sm font-medium">Literature Review</span>
                <span className={clsx('badge', execution.has_literature_review ? 'badge-success' : 'badge-secondary')}>
                  {execution.has_literature_review ? 'Included' : 'Not Included'}
                </span>
              </div>
              <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                <CheckCircle className={clsx('h-5 w-5', execution.has_code_analysis ? 'text-green-500' : 'text-gray-400')} />
                <span className="text-sm font-medium">Code Analysis</span>
                <span className={clsx('badge', execution.has_code_analysis ? 'badge-success' : 'badge-secondary')}>
                  {execution.has_code_analysis ? 'Included' : 'Not Included'}
                </span>
              </div>
            </div>

            {execution.summary && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">Research Synthesis</h4>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-700 leading-relaxed">{execution.summary}</p>
                </div>
              </div>
            )}
          </div>
        )

      default:
        return (
          <div className="text-center text-gray-500 py-8">
            <p>Execution details not available for this engine type.</p>
          </div>
        )
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className={`card p-6 border-l-4 ${colors.accent}`}>
        <div className="flex items-start space-x-4">
          <div className={`p-3 rounded-lg ${colors.bg}`}>
            <EngineIcon className={`h-6 w-6 ${colors.text}`} />
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-2">
              <h3 className="text-lg font-semibold text-gray-900">Research Complete</h3>
              <span className={`badge ${colors.bg} ${colors.text}`}>
                {classification.primary_engine.replace('_', ' ')}
              </span>
              <span className="text-sm text-gray-500">
                {(classification.confidence_score * 100).toFixed(0)}% confidence
              </span>
            </div>
            <p className="text-gray-600 mb-4">"{execution.query}"</p>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-sm text-gray-700">{classification.reasoning}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <h4 className="text-lg font-semibold text-gray-900">Research Results</h4>
          <div className="flex space-x-2">
            <button
              onClick={() => copyToClipboard(JSON.stringify(result, null, 2))}
              className="btn btn-outline btn-sm flex items-center space-x-1"
            >
              <Copy className="h-4 w-4" />
              <span>Copy</span>
            </button>
            <button className="btn btn-outline btn-sm flex items-center space-x-1">
              <Download className="h-4 w-4" />
              <span>Export</span>
            </button>
          </div>
        </div>

        {renderExecutionDetails()}
      </div>

      {/* Actions */}
      <div className="flex justify-center space-x-4">
        <button className="btn btn-outline flex items-center space-x-2">
          <ExternalLink className="h-4 w-4" />
          <span>View Detailed Report</span>
        </button>
        <button className="btn btn-primary">
          Start New Research
        </button>
      </div>
    </div>
  )
}

export default ResearchResult