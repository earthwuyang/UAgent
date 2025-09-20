import React, { useEffect, useMemo, useState, useCallback, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { ResearchProgressStream } from './ResearchProgressStream'
import ResearchTreeVisualization from './ResearchTreeVisualization'
import ResearchResult from '../ResearchResult'
import type { RouteAndExecuteResponse, ClassificationResult } from '../../types/api'
import UAgentAPI from '../../services/api'
import LLMConversationLogs from './LLMConversationLogs'

const STORAGE_PREFIX = 'uagent:session:'

const TABS: Array<{ id: 'progress' | 'tree' | 'results' | 'llm'; label: string; emoji: string }> = [
  { id: 'progress', label: 'Live Progress', emoji: '📊' },
  { id: 'tree', label: 'Research Tree', emoji: '🌳' },
  { id: 'results', label: 'Results', emoji: '📈' },
  { id: 'llm', label: 'LLM Stream', emoji: '🤖' }
]

const ResearchSessionPage: React.FC = () => {
  const params = useParams()
  const navigate = useNavigate()
  const sessionId = params.sessionId ?? ''

  const [activeTab, setActiveTab] = useState<'progress' | 'tree' | 'results' | 'llm'>('progress')
  const [isConnected, setIsConnected] = useState(false)
  const [storedResult, setStoredResult] = useState<RouteAndExecuteResponse | null>(null)
  const [requestSummary, setRequestSummary] = useState<string>('')
  const [classification, setClassification] = useState<ClassificationResult | null>(null)
  const [events, setEvents] = useState<any[]>([])
  const [hasFetchedReport, setHasFetchedReport] = useState(false)
  const [isFetchingReport, setIsFetchingReport] = useState(false)
  const [autoStartTriggered, setAutoStartTriggered] = useState(false)
  const pollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const storageKey = useMemo(() => `${STORAGE_PREFIX}${sessionId}`, [sessionId])

  const syncFromStorage = useCallback(() => {
    if (!sessionId) return
    try {
      const raw = window.localStorage.getItem(storageKey)
      if (!raw) return
      const parsed = JSON.parse(raw) as {
        request?: string
        classification?: ClassificationResult | null
        result?: RouteAndExecuteResponse | null
      }
      if (parsed.request) {
        setRequestSummary(parsed.request)
      }
      if (parsed.classification) {
        setClassification(parsed.classification as ClassificationResult)
      }
      if (parsed.result) {
        setStoredResult(parsed.result)
        setHasFetchedReport(true)
      }
    } catch (error) {
      console.error('Failed to parse stored session data', error)
    }
  }, [sessionId, storageKey])

  useEffect(() => {
    if (!sessionId) return
    syncFromStorage()

    const handleStorage = (event: StorageEvent) => {
      if (event.key === storageKey) {
        syncFromStorage()
      }
    }

    window.addEventListener('storage', handleStorage)
    return () => {
      window.removeEventListener('storage', handleStorage)
    }
  }, [sessionId, storageKey, syncFromStorage])

  useEffect(() => {
    if (!sessionId || !sessionId.startsWith('session_')) {
      navigate('/dashboard', { replace: true })
    }
  }, [sessionId, navigate])

  if (!sessionId) {
    return null
  }

  const persistSessionData = useCallback((report: RouteAndExecuteResponse | null, newClassification?: ClassificationResult | null) => {
    try {
      const payload = {
        request: requestSummary || undefined,
        classification: newClassification ?? classification,
        result: report,
        updatedAt: new Date().toISOString()
      }
      window.localStorage.setItem(storageKey, JSON.stringify(payload))
    } catch (error) {
      console.error('Failed to persist session report', error)
    }
  }, [classification, requestSummary, storageKey])

  const fetchReport = useCallback(async () => {
    if (!sessionId || isFetchingReport || hasFetchedReport) return
    setIsFetchingReport(true)
    try {
      const report = await UAgentAPI.getSessionReport(sessionId)
      if (report) {
        setStoredResult(report)
        if (report.classification) {
          setClassification(report.classification as ClassificationResult)
        }
        setHasFetchedReport(true)
        if (pollTimeoutRef.current) {
          window.clearTimeout(pollTimeoutRef.current)
          pollTimeoutRef.current = null
        }
        persistSessionData(report, (report.classification as ClassificationResult | undefined) ?? classification)
      } else {
        setHasFetchedReport(true)
        if (pollTimeoutRef.current) {
          window.clearTimeout(pollTimeoutRef.current)
        }
        pollTimeoutRef.current = window.setTimeout(() => {
          setHasFetchedReport(false)
        }, 2000)
      }
    } catch (error) {
      console.error('Failed to fetch session report', error)
    } finally {
      setIsFetchingReport(false)
    }
  }, [sessionId, isFetchingReport, hasFetchedReport, persistSessionData, classification])

  useEffect(() => {
    return () => {
      if (pollTimeoutRef.current) {
        window.clearTimeout(pollTimeoutRef.current)
        pollTimeoutRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!storedResult && sessionId && !hasFetchedReport) {
      fetchReport()
    }
  }, [storedResult, sessionId, hasFetchedReport, fetchReport])

  useEffect(() => {
    if (!sessionId) return
    if (events.length === 0) return
    const latest = events[events.length - 1]
    if (latest?.event_type === 'research_completed') {
      fetchReport()
    }
  }, [events, sessionId, fetchReport])

  useEffect(() => {
    if (!sessionId || autoStartTriggered) return
    if (!requestSummary?.trim()) return
    if (classification) return

    let cancelled = false

    const triggerAutoStart = async () => {
      try {
        setAutoStartTriggered(true)
        const ack = await UAgentAPI.routeAndExecute({
          user_request: requestSummary,
          session_id: sessionId,
          context: { source: 'session_auto_start' }
        })
        if (cancelled) return
        setClassification(ack.classification as ClassificationResult)
        persistSessionData(null, ack.classification as ClassificationResult)
      } catch (error) {
        console.error('Auto-start research failed', error)
        if (!cancelled) {
          setAutoStartTriggered(false)
        }
      }
    }

    triggerAutoStart()

    return () => {
      cancelled = true
    }
  }, [sessionId, requestSummary, classification, autoStartTriggered, persistSessionData])

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="sticky top-0 z-20 bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-6 py-4 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <span>🔬 Research Session</span>
              <Badge variant={isConnected ? 'default' : 'secondary'}>
                {isConnected ? 'Live' : 'Reconnecting'}
              </Badge>
            </h1>
            <p className="text-sm text-gray-500 mt-1 break-all">
              Session ID: <span className="font-mono">{sessionId}</span>
            </p>
            {requestSummary && (
              <p className="text-sm text-gray-600 mt-2 max-w-2xl">
                <span className="font-semibold text-gray-700">Request:</span> {requestSummary}
              </p>
            )}
            {classification && (
              <div className="mt-2 text-sm text-gray-600 space-y-1">
                <div>
                  <span className="font-semibold text-gray-700">Engine:</span>{' '}
                  {classification.primary_engine.replace('_', ' ')}
                </div>
                <div>
                  <span className="font-semibold text-gray-700">Confidence:</span>{' '}
                  {(classification.confidence_score * 100).toFixed(0)}%
                </div>
                <div>
                  <span className="font-semibold text-gray-700">Reasoning:</span>{' '}
                  {classification.reasoning}
                </div>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            {TABS.map((tab) => (
              <Button
                key={tab.id}
                variant={activeTab === tab.id ? 'default' : 'outline'}
                onClick={() => setActiveTab(tab.id)}
                className="text-sm"
              >
                <span className="mr-1">{tab.emoji}</span>
                {tab.label}
              </Button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        <div className={activeTab === 'progress' ? 'block' : 'hidden'}>
          <ResearchProgressStream
            sessionId={sessionId}
            onConnectionChange={setIsConnected}
            onEventsUpdate={setEvents}
          />
        </div>

        <div className={activeTab === 'tree' ? 'block' : 'hidden'}>
          <ResearchTreeVisualization sessionId={sessionId} />
        </div>

        <div className={activeTab === 'results' ? 'block' : 'hidden'}>
          {storedResult ? (
            <ResearchResult result={storedResult} />
          ) : (
            <div className="bg-white border rounded-lg p-6 text-center">
              <div className="text-4xl mb-2">{isFetchingReport ? '🔄' : '⏳'}</div>
              <p className="text-gray-600">
                {isFetchingReport
                  ? 'Generating detailed report...'
                  : 'Waiting for execution results. They will appear here once the research engine finishes.'}
              </p>
            </div>
          )}
        </div>

        <div className={activeTab === 'llm' ? 'block' : 'hidden'}>
          <div className="bg-white border rounded-lg p-2">
            <LLMConversationLogs sessionId={sessionId} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResearchSessionPage
