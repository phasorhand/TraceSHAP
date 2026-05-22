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
