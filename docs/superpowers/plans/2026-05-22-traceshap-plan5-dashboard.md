# TraceSHAP Plan 5: Web Dashboard (React + Vite)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight React SPA dashboard served via FastAPI that provides overview, trajectory detail, SHAP plots, and pruning views — all backed by the existing REST API.

**Architecture:** React + Vite SPA built to `frontend/dist/`, served as static files by FastAPI. Uses React-Plotly.js for interactive SHAP charts. Communicates with existing `/api/` endpoints. No auth required for v0.1.

**Tech Stack:** React 18, Vite, TypeScript, React-Plotly.js, React Router, TailwindCSS

---

### Task 1: Frontend Scaffold + Build Integration

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/index.css`
- Modify: `traceshap/api/app.py` (serve static files)

- [ ] **Step 1: Create frontend scaffold**

`frontend/package.json`:
```json
{
  "name": "traceshap-dashboard",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.23.0",
    "react-plotly.js": "^2.6.0",
    "plotly.js": "^2.32.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.4",
    "typescript": "^5.4.5",
    "vite": "^5.3.0"
  }
}
```

`frontend/vite.config.ts`:
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
    },
  },
})
```

`frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

`frontend/postcss.config.js`:
```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

`frontend/tailwind.config.js`:
```javascript
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: { extend: {} },
  plugins: [],
}
```

`frontend/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TraceSHAP Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`frontend/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

`frontend/src/main.tsx`:
```typescript
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
```

`frontend/src/App.tsx`:
```typescript
import { Routes, Route, Link } from 'react-router-dom'

function Overview() {
  return <div className="p-6"><h1 className="text-2xl font-bold">Overview</h1><p className="mt-2 text-gray-600">Loading...</p></div>
}

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b px-6 py-3 flex items-center gap-6">
        <span className="font-bold text-lg">TraceSHAP</span>
        <Link to="/" className="text-blue-600 hover:underline">Overview</Link>
        <Link to="/traces" className="text-blue-600 hover:underline">Traces</Link>
        <Link to="/pruning" className="text-blue-600 hover:underline">Pruning</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Overview />} />
      </Routes>
    </div>
  )
}
```

- [ ] **Step 2: Install dependencies and build**

Run:
```bash
cd frontend && npm install && npm run build
```
Expected: `frontend/dist/` created with `index.html` and JS assets.

- [ ] **Step 3: Update FastAPI to serve static files**

Modify `traceshap/api/app.py` to mount the frontend build:
```python
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from traceshap.api.deps import set_backend
from traceshap.api.routes_traces import router as traces_router
from traceshap.api.routes_attribution import router as attribution_router
from traceshap.api.routes_pruning import router as pruning_router
from traceshap.storage.sqlite import SQLiteBackend

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend" / "dist"


def create_app(backend: SQLiteBackend | None = None) -> FastAPI:
    app = FastAPI(title="TraceSHAP", version="0.1.0")

    if backend is not None:
        set_backend(backend)

    app.include_router(traces_router)
    app.include_router(attribution_router)
    app.include_router(pruning_router)

    if FRONTEND_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

        @app.get("/{path:path}")
        async def serve_spa(path: str):
            file_path = FRONTEND_DIR / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    return app
```

- [ ] **Step 4: Verify existing API tests still pass**

Run: `pytest tests/test_api.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/ traceshap/api/app.py
git commit -m "feat: frontend scaffold (React + Vite + Tailwind) with FastAPI static serving"
```

---

### Task 2: API Client + Overview Page

**Files:**
- Create: `frontend/src/api.ts`
- Create: `frontend/src/pages/Overview.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create API client**

`frontend/src/api.ts`:
```typescript
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
```

- [ ] **Step 2: Create Overview page**

`frontend/src/pages/Overview.tsx`:
```typescript
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
```

- [ ] **Step 3: Update App.tsx with routing**

`frontend/src/App.tsx`:
```typescript
import { Routes, Route, Link } from 'react-router-dom'
import Overview from './pages/Overview'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b px-6 py-3 flex items-center gap-6">
        <span className="font-bold text-lg">TraceSHAP</span>
        <Link to="/" className="text-blue-600 hover:underline">Overview</Link>
        <Link to="/traces" className="text-blue-600 hover:underline">Traces</Link>
        <Link to="/pruning" className="text-blue-600 hover:underline">Pruning</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Overview />} />
      </Routes>
    </div>
  )
}
```

- [ ] **Step 4: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat: API client and Overview page with agent stats and trace list"
```

---

### Task 3: Trajectory Detail Page

**Files:**
- Create: `frontend/src/pages/TraceDetail.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create TraceDetail page**

`frontend/src/pages/TraceDetail.tsx`:
```typescript
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
```

- [ ] **Step 2: Update App.tsx with route**

Add import and route:
```typescript
import TraceDetail from './pages/TraceDetail'
// in Routes:
<Route path="/traces/:traceId" element={<TraceDetail />} />
```

- [ ] **Step 3: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: trajectory detail page with step table and attribution verdicts"
```

---

### Task 4: SHAP Plots Page (Force + Waterfall)

**Files:**
- Create: `frontend/src/pages/Plots.tsx`
- Create: `frontend/src/components/ForcePlot.tsx`
- Create: `frontend/src/components/WaterfallPlot.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create ForcePlot component**

`frontend/src/components/ForcePlot.tsx`:
```typescript
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
```

- [ ] **Step 2: Create WaterfallPlot component**

`frontend/src/components/WaterfallPlot.tsx`:
```typescript
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
```

- [ ] **Step 3: Create Plots page**

`frontend/src/pages/Plots.tsx`:
```typescript
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
```

- [ ] **Step 4: Update App.tsx**

Add import and route:
```typescript
import Plots from './pages/Plots'
// in Routes:
<Route path="/plots" element={<Plots />} />
```

Add nav link: `<Link to="/plots" className="text-blue-600 hover:underline">Plots</Link>`

- [ ] **Step 5: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat: SHAP plots page with interactive force and waterfall charts (Plotly)"
```

---

### Task 5: Pruning Dashboard Page

**Files:**
- Create: `frontend/src/pages/Pruning.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create Pruning page**

`frontend/src/pages/Pruning.tsx`:
```typescript
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
```

- [ ] **Step 2: Update App.tsx with route**

Add import and route:
```typescript
import Pruning from './pages/Pruning'
// in Routes:
<Route path="/pruning" element={<Pruning />} />
```

- [ ] **Step 3: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: pruning dashboard page with candidate table and agent stats"
```

---

### Task 6: Trace List Page + Final Build

**Files:**
- Create: `frontend/src/pages/Traces.tsx`
- Modify: `frontend/src/App.tsx`
- Create: `frontend/.gitignore`

- [ ] **Step 1: Create Traces list page**

`frontend/src/pages/Traces.tsx`:
```typescript
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
```

- [ ] **Step 2: Update App.tsx with all routes**

Final `frontend/src/App.tsx`:
```typescript
import { Routes, Route, Link } from 'react-router-dom'
import Overview from './pages/Overview'
import Traces from './pages/Traces'
import TraceDetail from './pages/TraceDetail'
import Plots from './pages/Plots'
import Pruning from './pages/Pruning'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b px-6 py-3 flex items-center gap-6">
        <span className="font-bold text-lg">TraceSHAP</span>
        <Link to="/" className="text-blue-600 hover:underline">Overview</Link>
        <Link to="/traces" className="text-blue-600 hover:underline">Traces</Link>
        <Link to="/plots" className="text-blue-600 hover:underline">Plots</Link>
        <Link to="/pruning" className="text-blue-600 hover:underline">Pruning</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/traces" element={<Traces />} />
        <Route path="/traces/:traceId" element={<TraceDetail />} />
        <Route path="/plots" element={<Plots />} />
        <Route path="/pruning" element={<Pruning />} />
      </Routes>
    </div>
  )
}
```

- [ ] **Step 3: Create frontend .gitignore**

`frontend/.gitignore`:
```
node_modules
dist
```

- [ ] **Step 4: Final build**

Run: `cd frontend && npm run build`
Expected: Build succeeds, `dist/` created

- [ ] **Step 5: Run backend tests to ensure no regressions**

Run: `pytest -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "feat: traces list page, complete routing, and final dashboard build"
```

---

## Summary

After completing all 6 tasks, you will have:

1. **Frontend Scaffold** — React + Vite + Tailwind + TypeScript, built to `frontend/dist/`, served by FastAPI
2. **Overview Page** — Agent stats cards, recent traces table
3. **Trajectory Detail Page** — Step table with attribution verdicts (color-coded)
4. **SHAP Plots Page** — Interactive force plot and waterfall chart (React-Plotly)
5. **Pruning Dashboard** — Agent selector, stats, candidate table with savings and validation info
6. **Traces List Page** — Filterable trace list with links to detail and plots

**Access:** `traceshap serve` → http://localhost:8080

**Next plans:**
- Plan 6: Documentation + packaging (PyPI-ready)
