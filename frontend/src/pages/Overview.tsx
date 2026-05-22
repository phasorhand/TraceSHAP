import { useEffect, useState } from 'react'
import { fetchTraces, type TraceSummary } from '../api'

export default function Overview() {
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchTraces({ limit: 50 }).then(data => {
      setTraces(data)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const agents = [...new Set(traces.map(t => t.agent_name))]
  const avgQuality = traces
    .filter(t => t.outcome_quality !== null)
    .reduce((sum, t) => sum + (t.outcome_quality ?? 0), 0) /
    (traces.filter(t => t.outcome_quality !== null).length || 1)

  if (loading) return <div className="p-6">Loading...</div>

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Overview</h1>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="text-sm text-gray-500">Total Traces</div>
          <div className="text-2xl font-bold">{traces.length}</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="text-sm text-gray-500">Agents</div>
          <div className="text-2xl font-bold">{agents.length}</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="text-sm text-gray-500">Avg Quality</div>
          <div className="text-2xl font-bold">{avgQuality.toFixed(2)}</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="text-sm text-gray-500">Success Rate</div>
          <div className="text-2xl font-bold">
            {((traces.filter(t => t.outcome_success).length / (traces.length || 1)) * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="px-4 py-3 border-b font-medium">Recent Traces</div>
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left">Trace ID</th>
              <th className="px-4 py-2 text-left">Agent</th>
              <th className="px-4 py-2 text-left">Steps</th>
              <th className="px-4 py-2 text-left">Quality</th>
              <th className="px-4 py-2 text-left">Success</th>
            </tr>
          </thead>
          <tbody>
            {traces.slice(0, 20).map(t => (
              <tr key={t.trace_id} className="border-t hover:bg-gray-50">
                <td className="px-4 py-2 font-mono text-xs">
                  <a href={`/traces/${t.trace_id}`} className="text-blue-600 hover:underline">
                    {t.trace_id.slice(0, 16)}...
                  </a>
                </td>
                <td className="px-4 py-2">{t.agent_name}</td>
                <td className="px-4 py-2">{t.step_count}</td>
                <td className="px-4 py-2">{t.outcome_quality?.toFixed(2) ?? '—'}</td>
                <td className="px-4 py-2">
                  {t.outcome_success === null ? '—' : t.outcome_success ? '✓' : '✗'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
