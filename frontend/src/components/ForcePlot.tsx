import Plot from 'react-plotly.js'
import type { Attribution } from '../api'

interface Props {
  attributions: Attribution[]
  baseValue?: number
}

export default function ForcePlot({ attributions, baseValue = 0.5 }: Props) {
  const sorted = [...attributions]
    .filter(a => a.quality_delta !== null)
    .sort((a, b) => Math.abs(b.quality_delta ?? 0) - Math.abs(a.quality_delta ?? 0))

  const names = sorted.map(a => a.step_name)
  const values = sorted.map(a => a.quality_delta ?? 0)
  const colors = values.map(v => v >= 0 ? 'rgba(255, 0, 0, 0.6)' : 'rgba(0, 100, 255, 0.6)')

  return (
    <Plot
      data={[{
        type: 'bar',
        orientation: 'h',
        y: names,
        x: values,
        marker: { color: colors },
        text: values.map(v => (v >= 0 ? '+' : '') + v.toFixed(3)),
        textposition: 'outside',
      }]}
      layout={{
        title: `Force Plot (base = ${baseValue.toFixed(2)})`,
        xaxis: { title: 'SHAP value (quality impact)', zeroline: true },
        yaxis: { automargin: true },
        height: Math.max(300, sorted.length * 40),
        margin: { l: 150, r: 50, t: 50, b: 50 },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
