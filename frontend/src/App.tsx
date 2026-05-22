import { Routes, Route, Link } from 'react-router-dom'
import Overview from './pages/Overview'
import TraceDetail from './pages/TraceDetail'
import Plots from './pages/Plots'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b px-6 py-3 flex items-center gap-6">
        <span className="font-bold text-lg">TraceSHAP</span>
        <Link to="/" className="text-blue-600 hover:underline">Overview</Link>
        <Link to="/traces" className="text-blue-600 hover:underline">Traces</Link>
        <Link to="/pruning" className="text-blue-600 hover:underline">Pruning</Link>
        <Link to="/plots" className="text-blue-600 hover:underline">Plots</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/traces/:traceId" element={<TraceDetail />} />
        <Route path="/plots" element={<Plots />} />
      </Routes>
    </div>
  )
}
