import React, { useMemo, useState } from 'react'
import HierarchicalResearchDashboard from './components/HierarchicalResearchDashboard'
import LLMMonitor from './components/LLMMonitor'

interface ChatMessage {
  id: string
  role: 'system' | 'user' | 'assistant'
  content: string
  timestamp: Date
}

type SidePanel = 'chat' | null

export default function App() {
  const [sidePanel, setSidePanel] = useState<SidePanel>('chat')
  const [showLLMMonitor, setShowLLMMonitor] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(() => [
    {
      id: 'welcome',
      role: 'system',
      content:
        'Welcome to uagent. Launch research goals in the tree and watch AI-Scientist, AgentLab, and RepoMaster collaborate directly inside the research tree.',
      timestamp: new Date()
    }
  ])
  const [chatInput, setChatInput] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)

  const handleSendChat = async () => {
    const trimmed = chatInput.trim()
    if (!trimmed) return

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmed,
      timestamp: new Date()
    }

    setChatMessages((prev) => [...prev, userMsg])
    setChatInput('')

    // Placeholder assistant response tying back to unified orchestration
    setIsProcessing(true)
    const assistantMsg: ChatMessage = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content:
        'Got it. Track the node in the research tree to see AgentLab collaborators, AI-Scientist runs, and RepoMaster insights as the unified workflow progresses.',
      timestamp: new Date()
    }
    setTimeout(() => {
      setChatMessages((prev) => [...prev, assistantMsg])
      setIsProcessing(false)
    }, 400)
  }

  const sidePanelHeader = useMemo(() => (sidePanel === 'chat' ? 'Research Chat' : ''), [sidePanel])

  return (
    <div className="h-screen bg-gray-50 flex flex-col">
      <header className="bg-white border-b px-6 py-4 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">uagent Research Tree</h1>
            <p className="text-sm text-gray-600">
              Hierarchical research orchestrated across AI-Scientist, AgentLab, RepoMaster, and multi-modal search – all inside the tree
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setSidePanel(sidePanel === 'chat' ? null : 'chat')}
              className={`px-3 py-1 rounded text-sm border ${
                sidePanel === 'chat' ? 'bg-blue-600 text-white border-blue-600' : 'bg-blue-50 text-blue-700 border-blue-200'
              }`}
            >
              💬 Chat
            </button>
            <button
              onClick={() => setShowLLMMonitor(true)}
              className="px-3 py-1 bg-purple-100 text-purple-700 rounded text-sm border border-purple-200 hover:bg-purple-200"
            >
              🤖 LLM Monitor
            </button>
          </div>
        </div>
      </header>

      <div className="flex-1 flex min-h-0 overflow-hidden">
        <div className="flex-1 min-h-0 flex flex-col">
          <HierarchicalResearchDashboard />
        </div>

        {sidePanel && (
          <aside className="w-[420px] bg-white border-l flex flex-col">
            <div className="p-3 border-b bg-gray-50 flex items-center justify-between">
              <h3 className="font-semibold text-gray-900">{sidePanelHeader}</h3>
              <button
                onClick={() => setSidePanel(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>

            <div className="flex-1 overflow-y-auto">
              <div className="flex flex-col h-full">
                <div className="flex-1 overflow-y-auto p-3 space-y-3">
                  {chatMessages.map((message) => (
                    <div
                      key={message.id}
                      className={`text-sm rounded px-3 py-2 ${
                        message.role === 'system'
                          ? 'bg-gray-100 text-gray-700'
                          : message.role === 'assistant'
                          ? 'bg-green-50 text-green-800 border border-green-200'
                          : 'bg-blue-50 text-blue-800 border-blue-200 self-end'
                      }`}
                    >
                      <div>{message.content}</div>
                      <div className="text-[10px] text-gray-500 mt-1">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="p-3 border-t bg-gray-50">
                  <div className="flex space-x-2">
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder="Describe a research goal or code task..."
                      className="flex-1 p-2 border rounded text-sm"
                      disabled={isProcessing}
                    />
                    <button
                      onClick={handleSendChat}
                      disabled={isProcessing || !chatInput.trim()}
                      className="px-3 py-2 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50"
                    >
                      ▶
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Hints: “Draft experiments for X” • “Analyze repo Y” • “Search papers on Z”
                  </p>
                </div>
              </div>
            </div>
          </aside>
        )}

      </div>

      <LLMMonitor isVisible={showLLMMonitor} onClose={() => setShowLLMMonitor(false)} />
    </div>
  )
}
