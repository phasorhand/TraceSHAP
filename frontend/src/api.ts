const BASE = '/api'

export interface TraceSummary {
  trace_id: string
  framework: string
  agent_name: string
  agent_version: string | null
  task_type: string | null
  step_count: number
  outcome_success: boolean | null
  outcome_quality: number | null
}

export interface AgentStats {
  agent_name: string
  trajectory_count: number
  avg_quality: number | null
  avg_cost: number | null
  avg_latency_ms: number | null
}

export interface Attribution {
  step_id: string
  step_name: string
  quality_delta: number | null
  cost_delta: number | null
  latency_delta: number | null
  confidence_lower: number | null
  confidence_upper: number | null
  verdict: string
}

export interface PruneCandidate {
  target_type: string
  target_id: string
  decision_status: string
  estimated_savings: {
    token_reduction: number
    cost_reduction: number
    latency_reduction_ms: number
    quality_impact_range: [number, number]
  }
  validation: {
    replay_required: boolean
    replay_mode: string
    min_replay_count: number
  }
}

export async function fetchTraces(params?: {
  agent_name?: string
  limit?: number
}): Promise<TraceSummary[]> {
  const query = new URLSearchParams()
  if (params?.agent_name) query.set('agent_name', params.agent_name)
  if (params?.limit) query.set('limit', String(params.limit))
  const res = await fetch(`${BASE}/traces?${query}`)
  return res.json()
}

export async function fetchTrace(traceId: string): Promise<any> {
  const res = await fetch(`${BASE}/traces/${traceId}`)
  return res.json()
}

export async function fetchAttribution(
  traceId: string,
  layers: string = '0'
): Promise<Attribution[]> {
  const res = await fetch(`${BASE}/traces/${traceId}/attribution?layers=${layers}`)
  return res.json()
}

export async function fetchAgentStats(agentName: string): Promise<AgentStats> {
  const res = await fetch(`${BASE}/agents/${agentName}/stats`)
  return res.json()
}

export async function fetchPruneCandidates(
  agentName: string,
  layers: string = '0'
): Promise<{ agent_name: string; trajectory_count: number; candidates: PruneCandidate[] }> {
  const res = await fetch(`${BASE}/agents/${agentName}/prune-candidates?layers=${layers}`)
  return res.json()
}
