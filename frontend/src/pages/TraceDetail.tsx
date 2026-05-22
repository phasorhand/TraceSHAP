import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { fetchTrace, fetchAttribution, type Attribution } from '../api'

interface StepData {
  step_id: string
  tool_name: string | null
  step_type: string
  side_effect: string
  attempt_index: number
  cost: number | null
  duration_ms: number
  start_time: string
  end_time: string
}

interface TraceData {
  trace_id: string
  framework: string
  agent_name: string
  outcome: { success: boolean; quality_score: number; token_cost: number; latency_ms: number } | null
  steps: StepData[]
  spans: any[]
}

const VERDICT_COLORS: Record<string, string> = {
  keep: 'bg-green-100 text-green-800',
  review: 'bg-yellow-100 text-yellow-800',
  prune_candidate: 'bg-red-100 text-red-800',
  insufficient_evidence: 'bg-gray-100 text-gray-600',
}

export default function TraceDetail() {
  const { traceId } = useParams<{ traceId: string }>()
  const [trace, setTrace] = useState<TraceData | null>(null)
  const [attributions, setAttributions] = useState<Attribution[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!traceId) return
    Promise.all([
      fetchTrace(traceId),
      fetchAttribution(traceId, '0'),
    ]).then(([t, a]) => {
      setTrace(t)
      setAttributions(a)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [traceId])

  if (loading) return <div className="p-6">Loading...</div>
  if (!trace) return <div className="p-6">Trace not found</div>

  const attrMap = Object.fromEntries(attributions.map(a => [a.step_id, a]))

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold font-mono">{trace.trace_id}</h1>
        <span className="text-sm text-gray-500">{trace.framework} / {trace.agent_name}</span>
      </div>

      {trace.outcome && (
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-500">Quality</div>
            <div className="text-xl font-bold">{trace.outcome.quality_score.toFixed(2)}</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-500">Token Cost</div>
            <div className="text-xl font-bold">{trace.outcome.token_cost}</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-500">Latency</div>
            <div className="text-xl font-bold">{trace.outcome.latency_ms}ms</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-500">Success</div>
            <div className="text-xl font-bold">{trace.outcome.success ? '✓' : '✗'}</div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow">
        <div className="px-4 py-3 border-b font-medium">Steps ({trace.steps.length})</div>
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left">#</th>
              <th className="px-4 py-2 text-left">Name</th>
              <th className="px-4 py-2 text-left">Type</th>
              <th className="px-4 py-2 text-left">Duration</th>
              <th className="px-4 py-2 text-left">Quality Δ</th>
              <th className="px-4 py-2 text-left">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {trace.steps.map((step, i) => {
              const attr = attrMap[step.step_id]
              const verdict = attr?.verdict ?? 'insufficient_evidence'
              return (
                <tr key={step.step_id} className="border-t hover:bg-gray-50">
                  <td className="px-4 py-2">{i + 1}</td>
                  <td className="px-4 py-2 font-mono text-xs">{step.tool_name ?? step.step_type}</td>
                  <td className="px-4 py-2">{step.step_type}</td>
                  <td className="px-4 py-2">{step.duration_ms}ms</td>
                  <td className="px-4 py-2">
                    {attr?.quality_delta !== null && attr?.quality_delta !== undefined
                      ? (attr.quality_delta >= 0 ? '+' : '') + attr.quality_delta.toFixed(3)
                      : '—'}
                  </td>
                  <td className="px-4 py-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${VERDICT_COLORS[verdict] ?? ''}`}>
                      {verdict}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
