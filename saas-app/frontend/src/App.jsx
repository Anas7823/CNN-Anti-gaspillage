import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import LandingPage from './pages/LandingPage'
import AnalyzePage from './pages/AnalyzePage'
import PerformancePage from './pages/PerformancePage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-forest-950">
        <Navbar />
        <Routes>
          <Route path="/"            element={<LandingPage />} />
          <Route path="/analyze"     element={<AnalyzePage />} />
          <Route path="/performance" element={<PerformancePage />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
