import { useEffect, useState } from 'react'
import { fetchTraces, fetchAgentStats, fetchPruneCandidates, type AgentStats, type PruneCandidate, type TraceSummary } from '../api'

export default function Pruning() {
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [selectedAgent, setSelectedAgent] = useState('')
  const [stats, setStats] = useState<AgentStats | null>(null)
  const [candidates, setCandidates] = useState<PruneCandidate[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchTraces({ limit: 100 }).then(setTraces)
  }, [])

  const agents = [...new Set(traces.map(t => t.agent_name))]

  useEffect(() => {
    if (!selectedAgent) return
    setLoading(true)
    Promise.all([
      fetchAgentStats(selectedAgent),
      fetchPruneCandidates(selectedAgent, '0'),
    ]).then(([s, p]) => {
      setStats(s)
      setCandidates(p.candidates)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [selectedAgent])

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Pruning Dashboard</h1>

      <div className="flex items-center gap-4">
        <label className="text-sm font-medium">Agent:</label>
        <select
          className="border rounded px-3 py-1.5 text-sm"
          value={selectedAgent}
          onChange={e => setSelectedAgent(e.target.value)}
        >
          <option value="">Select an agent...</option>
          {agents.map(a => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>
      </div>

      {loading && <div>Loading...</div>}

      {stats && (
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-500">Trajectories</div>
            <div className="text-xl font-bold">{stats.trajectory_count}</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-500">Avg Quality</div>
            <div className="text-xl font-bold">{stats.avg_quality?.toFixed(2) ?? '—'}</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-500">Avg Cost</div>
            <div className="text-xl font-bold">{stats.avg_cost?.toFixed(0) ?? '—'}</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-500">Candidates</div>
            <div className="text-xl font-bold text-red-600">{candidates.length}</div>
          </div>
        </div>
      )}

      {candidates.length > 0 && (
        <div className="bg-white rounded-lg shadow">
          <div className="px-4 py-3 border-b font-medium">Prune Candidates</div>
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left">Target</th>
                <th className="px-4 py-2 text-left">Status</th>
                <th className="px-4 py-2 text-left">Token Savings</th>
                <th className="px-4 py-2 text-left">Cost Savings</th>
                <th className="px-4 py-2 text-left">Latency Savings</th>
                <th className="px-4 py-2 text-left">Quality Impact</th>
                <th className="px-4 py-2 text-left">Replay Mode</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c, i) => (
                <tr key={i} className="border-t hover:bg-gray-50">
                  <td className="px-4 py-2 font-mono text-xs">{c.target_id}</td>
                  <td className="px-4 py-2">
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                      {c.decision_status}
                    </span>
                  </td>
                  <td className="px-4 py-2">{c.estimated_savings.token_reduction}</td>
                  <td className="px-4 py-2">${c.estimated_savings.cost_reduction.toFixed(4)}</td>
                  <td className="px-4 py-2">{c.estimated_savings.latency_reduction_ms}ms</td>
                  <td className="px-4 py-2 text-xs">
                    [{c.estimated_savings.quality_impact_range[0].toFixed(3)}, {c.estimated_savings.quality_impact_range[1].toFixed(3)}]
                  </td>
                  <td className="px-4 py-2 text-xs">{c.validation.replay_mode}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
