import Plot from 'react-plotly.js'
import type { Attribution } from '../api'

interface Props {
  attributions: Attribution[]
  baseValue?: number
}

export default function WaterfallPlot({ attributions, baseValue = 0.5 }: Props) {
  const sorted = [...attributions]
    .filter(a => a.quality_delta !== null)
    .sort((a, b) => (b.quality_delta ?? 0) - (a.quality_delta ?? 0))

  const names = ['Base', ...sorted.map(a => a.step_name), 'Final']
  const values = sorted.map(a => a.quality_delta ?? 0)
  const finalValue = baseValue + values.reduce((s, v) => s + v, 0)

  const measures = ['absolute', ...values.map(() => 'relative' as const), 'total']
  const allValues = [baseValue, ...values, finalValue]

  return (
    <Plot
      data={[{
        type: 'waterfall',
        x: names,
        y: allValues,
        measure: measures,
        connector: { line: { color: 'rgb(63, 63, 63)' } },
        increasing: { marker: { color: 'rgba(255, 0, 0, 0.6)' } },
        decreasing: { marker: { color: 'rgba(0, 100, 255, 0.6)' } },
        totals: { marker: { color: 'rgba(100, 100, 100, 0.6)' } },
      }]}
      layout={{
        title: 'Waterfall Plot (cumulative attribution)',
        yaxis: { title: 'Score' },
        xaxis: { automargin: true },
        height: 400,
        margin: { l: 60, r: 30, t: 50, b: 100 },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
