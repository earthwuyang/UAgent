export async function apiSearch(q: string) {
  const r = await fetch(`/api/search?q=${encodeURIComponent(q)}`)
  if (!r.ok) throw new Error('search failed')
  return r.json()
}

export async function githubSearch(q: string) {
  const r = await fetch(`/api/github/search?q=${encodeURIComponent(q)}`)
  if (!r.ok) throw new Error('github search failed')
  return r.json()
}

export async function runExperiment(rootName: string) {
  const r = await fetch(`/api/experiments/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ root: { name: rootName, params: {} }, max_depth: 2, parallelism: 4 }),
  })
  if (!r.ok) throw new Error('experiments failed')
  return r.json()
}

export async function startAIScientist(ideaIdx = 0) {
  const r = await fetch(`/api/experiments/ai-scientist/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ idea_idx: ideaIdx, skip_writeup: true, skip_review: true })
  })
  if (!r.ok) throw new Error('ai scientist start failed')
  return r.json() as Promise<{ job_id: string }>
}

export async function startAgentLab(configYaml = 'experiment_configs/MATH_agentlab.yaml') {
  const r = await fetch(`/api/experiments/agent-lab/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_yaml: configYaml })
  })
  if (!r.ok) throw new Error('agent lab start failed')
  return r.json() as Promise<{ job_id: string }>
}

export async function getJob(jobId: string) {
  const r = await fetch(`/api/jobs/${jobId}`)
  if (!r.ok) throw new Error('job query failed')
  return r.json()
}

// Unified Orchestrator API functions
export async function executeWorkflow(workflowConfig: any) {
  const r = await fetch('/api/unified/workflows/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(workflowConfig)
  })
  if (!r.ok) throw new Error('workflow execution failed')
  return r.json()
}

export async function getWorkflowStatus(workflowId: string) {
  const r = await fetch(`/api/unified/workflows/${workflowId}/status`)
  if (!r.ok) throw new Error('workflow status failed')
  return r.json()
}

export async function getWorkflowResults(workflowId: string) {
  const r = await fetch(`/api/unified/workflows/${workflowId}/results`)
  if (!r.ok) throw new Error('workflow results failed')
  return r.json()
}

export async function listActiveWorkflows() {
  const r = await fetch('/api/unified/workflows/active')
  if (!r.ok) throw new Error('list workflows failed')
  return r.json()
}

export async function cancelWorkflow(workflowId: string) {
  const r = await fetch(`/api/unified/workflows/${workflowId}`, {
    method: 'DELETE'
  })
  if (!r.ok) throw new Error('cancel workflow failed')
  return r.json()
}

export async function startResearchProject(projectData: any) {
  const r = await fetch('/api/unified/research/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(projectData)
  })
  if (!r.ok) throw new Error('research project failed')
  return r.json()
}

export async function analyzeCodeRepository(repoData: any) {
  const r = await fetch('/api/unified/code/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(repoData)
  })
  if (!r.ok) throw new Error('code analysis failed')
  return r.json()
}

export async function intelligentSearch(searchData: any) {
  const r = await fetch('/api/unified/search/intelligent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(searchData)
  })
  if (!r.ok) throw new Error('intelligent search failed')
  return r.json()
}

export async function unifiedSearch(searchData: any) {
  const r = await fetch('/api/unified/search/unified', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(searchData)
  })
  if (!r.ok) throw new Error('unified search failed')
  return r.json()
}

export async function getWorkflowTemplates() {
  const r = await fetch('/api/unified/templates')
  if (!r.ok) throw new Error('get templates failed')
  return r.json()
}

export async function getSystemStatus() {
  const r = await fetch('/api/unified/system/status')
  if (!r.ok) throw new Error('system status failed')
  return r.json()
}

export async function getSystemHealth() {
  const r = await fetch('/api/unified/system/health')
  if (!r.ok) throw new Error('health check failed')
  return r.json()
}

export async function getNodeWorkflowEvents(goalId: string, nodeId: string) {
  const r = await fetch(`/api/research-tree/goals/${goalId}/nodes/${nodeId}/workflow-events`)
  if (!r.ok) throw new Error('workflow events fetch failed')
  return r.json()
}
