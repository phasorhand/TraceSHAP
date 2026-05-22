import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { fetchTraces, fetchAttribution, type Attribution, type TraceSummary } from '../api'
import ForcePlot from '../components/ForcePlot'
import WaterfallPlot from '../components/WaterfallPlot'

export default function Plots() {
  const [searchParams] = useSearchParams()
  const initialTraceId = searchParams.get('trace_id') ?? ''
  const [traceId, setTraceId] = useState(initialTraceId)
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [attributions, setAttributions] = useState<Attribution[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchTraces({ limit: 50 }).then(setTraces)
  }, [])

  useEffect(() => {
    if (!traceId) return
    setLoading(true)
    fetchAttribution(traceId, '0').then(data => {
      setAttributions(data)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [traceId])

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">SHAP Plots</h1>

      <div className="flex items-center gap-4">
        <label className="text-sm font-medium">Trace:</label>
        <select
          className="border rounded px-3 py-1.5 text-sm"
          value={traceId}
          onChange={e => setTraceId(e.target.value)}
        >
          <option value="">Select a trace...</option>
          {traces.map(t => (
            <option key={t.trace_id} value={t.trace_id}>
              {t.trace_id.slice(0, 16)} — {t.agent_name} ({t.step_count} steps)
            </option>
          ))}
        </select>
      </div>

      {loading && <div>Loading attribution...</div>}

      {attributions.length > 0 && (
        <div className="space-y-8">
          <div className="bg-white rounded-lg shadow p-4">
            <ForcePlot attributions={attributions} />
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <WaterfallPlot attributions={attributions} />
          </div>
        </div>
      )}
    </div>
  )
}
