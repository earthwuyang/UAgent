import React, { useState } from 'react'
import { apiSearch, githubSearch, runExperiment, startAIScientist, startAgentLab, getJob } from './services/api'
import UnifiedDashboard from './components/UnifiedDashboard'
import HierarchicalResearchDashboard from './components/HierarchicalResearchDashboard'
import LLMMonitor from './components/LLMMonitor'

export default function App() {
  const [mode, setMode] = useState<'hierarchical' | 'unified' | 'legacy'>('hierarchical')
  const [showLLMMonitor, setShowLLMMonitor] = useState(false)

  if (mode === 'hierarchical') {
    return (
      <div>
        <div className="fixed top-4 right-4 z-50 space-x-2">
          <button
            onClick={() => setShowLLMMonitor(true)}
            className="px-3 py-1 bg-purple-200 text-purple-700 rounded text-sm hover:bg-purple-300"
          >
            🤖 LLM Monitor
          </button>
          <button
            onClick={() => setMode('unified')}
            className="px-3 py-1 bg-blue-200 text-blue-700 rounded text-sm hover:bg-blue-300"
          >
            Unified UI
          </button>
          <button
            onClick={() => setMode('legacy')}
            className="px-3 py-1 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300"
          >
            Legacy UI
          </button>
        </div>
        <HierarchicalResearchDashboard />
        {/* LLM Monitor overlay */}
        <LLMMonitor
          isVisible={showLLMMonitor}
          onClose={() => setShowLLMMonitor(false)}
        />
      </div>
    )
  }

  if (mode === 'unified') {
    return (
      <div>
        <div className="fixed top-4 right-4 z-50 space-x-2">
          <button
            onClick={() => setShowLLMMonitor(true)}
            className="px-3 py-1 bg-purple-200 text-purple-700 rounded text-sm hover:bg-purple-300"
          >
            🤖 LLM Monitor
          </button>
          <button
            onClick={() => setMode('hierarchical')}
            className="px-3 py-1 bg-green-200 text-green-700 rounded text-sm hover:bg-green-300"
          >
            Tree Research
          </button>
          <button
            onClick={() => setMode('legacy')}
            className="px-3 py-1 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300"
          >
            Legacy UI
          </button>
        </div>
        <UnifiedDashboard />
        {/* LLM Monitor overlay */}
        <LLMMonitor
          isVisible={showLLMMonitor}
          onClose={() => setShowLLMMonitor(false)}
        />
      </div>
    )
  }

  // Legacy UI below
  const [q, setQ] = useState('site:github.com agent framework python')
  const [results, setResults] = useState<any[]>([])
  const [ghQuery, setGhQuery] = useState('agent framework repo')
  const [ghResults, setGhResults] = useState<any[]>([])
  const [expName, setExpName] = useState('baseline')
  const [expResult, setExpResult] = useState<any | null>(null)
  const [aiJob, setAiJob] = useState<string | null>(null)
  const [labJob, setLabJob] = useState<string | null>(null)
  const [jobInfo, setJobInfo] = useState<any | null>(null)

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <h1>uagent (Legacy)</h1>
          <p>ROMA-style UI + RepoMaster search + AI-Scientist parallel experiments</p>
        </div>
        <div className="space-x-2">
          <button
            onClick={() => setMode('hierarchical')}
            style={{
              padding: '8px 16px',
              backgroundColor: '#22c55e',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Tree Research
          </button>
          <button
            onClick={() => setMode('unified')}
            style={{
              padding: '8px 16px',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Unified UI
          </button>
        </div>
      </div>

      <section style={{ marginTop: 24 }}>
        <h2>GitHub Search</h2>
        <input style={{ width: '60%' }} value={q} onChange={(e) => setQ(e.target.value)} />
        <button onClick={async () => setResults(await apiSearch(q))} style={{ marginLeft: 8 }}>Web Search</button>
        <div style={{ marginTop: 8 }}>
          <input style={{ width: '60%' }} value={ghQuery} onChange={(e) => setGhQuery(e.target.value)} placeholder='GitHub terms (site filter auto)'/>
          <button onClick={async () => setGhResults(await githubSearch(ghQuery))} style={{ marginLeft: 8 }}>GitHub Search</button>
        </div>
        <ul>
          {results.map((r, i) => (
            <li key={i}>
              <a href={r.link} target="_blank" rel="noreferrer">{r.title}</a>
              <div style={{ color: '#666' }}>{r.snippet}</div>
            </li>
          ))}
        </ul>
        {ghResults.length > 0 && (
          <>
          <h3>GitHub Results</h3>
          <ul>
            {ghResults.map((r, i) => (
              <li key={i}>
                <a href={r.link} target="_blank" rel="noreferrer">{r.title}</a>
                <div style={{ color: '#666' }}>{r.snippet}</div>
              </li>
            ))}
          </ul>
          </>
        )}
      </section>

      <section style={{ marginTop: 24 }}>
        <h2>Experiment Runner</h2>
        <input value={expName} onChange={(e) => setExpName(e.target.value)} />
        <button onClick={async () => setExpResult(await runExperiment(expName))} style={{ marginLeft: 8 }}>Run</button>
        {expResult && (
          <pre style={{ background: '#f6f8fa', padding: 12 }}>{JSON.stringify(expResult, null, 2)}</pre>
        )}
      </section>

      <section style={{ marginTop: 24 }}>
        <h2>AI-Scientist & AgentLab Jobs</h2>
        <div>
          <button onClick={async () => { const { job_id } = await startAIScientist(0); setAiJob(job_id); }}>Start AI-Scientist</button>
          {aiJob && <span style={{ marginLeft: 8 }}>Job: {aiJob}</span>}
        </div>
        <div style={{ marginTop: 8 }}>
          <button onClick={async () => { const { job_id } = await startAgentLab(); setLabJob(job_id); }}>Start AgentLab</button>
          {labJob && <span style={{ marginLeft: 8 }}>Job: {labJob}</span>}
        </div>
        <div style={{ marginTop: 8 }}>
          <input placeholder='Job ID' value={jobInfo?.id || aiJob || labJob || ''} onChange={(e) => setJobInfo({ id: e.target.value })} />
          <button onClick={async () => { if (!jobInfo?.id && !aiJob && !labJob) return; const id = jobInfo?.id || aiJob || labJob!; const info = await getJob(id); setJobInfo(info); }}>Check Job</button>
        </div>
        {jobInfo && (
          <pre style={{ background: '#f6f8fa', padding: 12, maxHeight: 300, overflow: 'auto' }}>{JSON.stringify(jobInfo, null, 2)}</pre>
        )}
      </section>

      {/* LLM Monitor overlay */}
      <LLMMonitor
        isVisible={showLLMMonitor}
        onClose={() => setShowLLMMonitor(false)}
      />
    </div>
  )
}
