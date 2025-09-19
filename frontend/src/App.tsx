import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './components/HomePage'
import { ResearchDashboard } from './components/research/ResearchDashboard'
import ResearchSessionPage from './components/research/ResearchSessionPage'
import SystemStatus from './components/SystemStatus'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/dashboard" element={<ResearchDashboard />} />
        <Route path="/status" element={<SystemStatus />} />
        <Route path="/session/:sessionId" element={<ResearchSessionPage />} />
        <Route path="/:sessionId" element={<ResearchSessionPage />} />
      </Routes>
    </Layout>
  )
}

export default App
