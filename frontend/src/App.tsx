import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './components/HomePage'
import { ResearchDashboard } from './components/research/ResearchDashboard'
import SystemStatus from './components/SystemStatus'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/dashboard" element={<ResearchDashboard />} />
        <Route path="/status" element={<SystemStatus />} />
      </Routes>
    </Layout>
  )
}

export default App