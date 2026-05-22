import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchTraces, type TraceSummary } from '../api'

export default function Traces() {
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [agentFilter, setAgentFilter] = useState('')

  useEffect(() => {
    const params = agentFilter ? { agent_name: agentFilter, limit: 100 } : { limit: 100 }
    setLoading(true)
    fetchTraces(params).then(data => {
      setTraces(data)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [agentFilter])

  const agents = [...new Set(traces.map(t => t.agent_name))]

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Traces</h1>
        <div className="flex items-center gap-2">
          <label className="text-sm">Filter:</label>
          <select
            className="border rounded px-3 py-1.5 text-sm"
            value={agentFilter}
            onChange={e => setAgentFilter(e.target.value)}
          >
            <option value="">All agents</option>
            {agents.map(a => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div>Loading...</div>
      ) : (
        <div className="bg-white rounded-lg shadow">
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left">Trace ID</th>
                <th className="px-4 py-2 text-left">Agent</th>
                <th className="px-4 py-2 text-left">Framework</th>
                <th className="px-4 py-2 text-left">Steps</th>
                <th className="px-4 py-2 text-left">Quality</th>
                <th className="px-4 py-2 text-left">Success</th>
                <th className="px-4 py-2 text-left">Actions</th>
              </tr>
            </thead>
            <tbody>
              {traces.map(t => (
                <tr key={t.trace_id} className="border-t hover:bg-gray-50">
                  <td className="px-4 py-2 font-mono text-xs">
                    <Link to={`/traces/${t.trace_id}`} className="text-blue-600 hover:underline">
                      {t.trace_id.slice(0, 20)}...
                    </Link>
                  </td>
                  <td className="px-4 py-2">{t.agent_name}</td>
                  <td className="px-4 py-2">{t.framework}</td>
                  <td className="px-4 py-2">{t.step_count}</td>
                  <td className="px-4 py-2">{t.outcome_quality?.toFixed(2) ?? '—'}</td>
                  <td className="px-4 py-2">
                    {t.outcome_success === null ? '—' : t.outcome_success ? '✓' : '✗'}
                  </td>
                  <td className="px-4 py-2">
                    <Link to={`/plots?trace_id=${t.trace_id}`} className="text-blue-600 text-xs hover:underline">
                      Plots
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
