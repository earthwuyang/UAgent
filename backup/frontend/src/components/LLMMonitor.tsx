import React, { useState, useEffect, useRef } from 'react';

interface LLMMessage {
  type: 'llm_event' | 'connection' | 'heartbeat' | 'pong' | 'status';
  event_type?: 'request' | 'response' | 'error';
  data?: {
    timestamp: string;
    model?: string;
    messages?: Array<{
      role: string;
      content: string;
    }>;
    temperature?: number;
    max_tokens?: number;
    content?: string;
    usage?: {
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
    };
    success?: boolean;
    error?: string;
  };
  message?: string;
  timestamp?: string;
  broadcast_timestamp?: string;
  active_connections?: number;
}

interface LLMMonitorProps {
  isVisible: boolean;
  onClose: () => void;
}

const LLMMonitor: React.FC<LLMMonitorProps> = ({ isVisible, onClose }) => {
  const [messages, setMessages] = useState<LLMMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [activeConnections, setActiveConnections] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const wsUrl = `ws://127.0.0.1:8012/api/ws/llm-monitor`;
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      setIsConnected(true);
      console.log('Connected to LLM Monitor WebSocket');
    };

    wsRef.current.onmessage = (event) => {
      try {
        const message: LLMMessage = JSON.parse(event.data);
        setMessages(prev => [...prev.slice(-49), message]); // Keep last 50 messages

        // Update connection count
        if (message.active_connections !== undefined) {
          setActiveConnections(message.active_connections);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    wsRef.current.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from LLM Monitor WebSocket');
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
  };

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsConnected(false);
    }
  };

  const clearMessages = () => {
    setMessages([]);
  };

  const sendTestMessage = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8012/api/llm-monitor/test', {
        method: 'GET',
      });
      const result = await response.json();
      console.log('Test message sent:', result);
    } catch (error) {
      console.error('Failed to send test message:', error);
    }
  };

  useEffect(() => {
    if (isVisible) {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }

    return () => {
      disconnectWebSocket();
    };
  }, [isVisible]);

  if (!isVisible) {
    return null;
  }

  const formatTimestamp = (timestamp: string | undefined) => {
    if (!timestamp) return '';
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return timestamp;
    }
  };

  const getMessageIcon = (message: LLMMessage) => {
    if (message.type === 'llm_event') {
      switch (message.event_type) {
        case 'request': return '📤';
        case 'response': return message.data?.success ? '✅' : '❌';
        case 'error': return '🚨';
        default: return '🔄';
      }
    }
    switch (message.type) {
      case 'connection': return '🔗';
      case 'heartbeat': return '💓';
      case 'status': return '📊';
      default: return '📝';
    }
  };

  const renderMessageContent = (message: LLMMessage) => {
    if (message.type === 'llm_event' && message.data) {
      const data = message.data;

      if (message.event_type === 'request') {
        return (
          <div className="space-y-1">
            <div className="text-sm text-gray-600">
              Model: {data.model} | Temp: {data.temperature} | Max: {data.max_tokens}
            </div>
            {data.messages?.map((msg, idx) => (
              <div key={idx} className="bg-blue-50 p-2 rounded text-sm">
                <div className="font-semibold text-blue-700">{msg.role.toUpperCase()}:</div>
                <div className="text-gray-800 whitespace-pre-wrap">{msg.content}</div>
              </div>
            ))}
          </div>
        );
      }

      if (message.event_type === 'response') {
        return (
          <div className="space-y-1">
            <div className="text-sm text-gray-600">
              {data.success ?
                `Response from ${data.model}${data.usage ? ` | Tokens: ${data.usage.total_tokens}` : ''}` :
                'Error occurred'
              }
            </div>
            {data.success ? (
              <div className="bg-green-50 p-2 rounded text-sm">
                <div className="font-semibold text-green-700">RESPONSE:</div>
                <div className="text-gray-800 whitespace-pre-wrap">{data.content}</div>
              </div>
            ) : (
              <div className="bg-red-50 p-2 rounded text-sm">
                <div className="font-semibold text-red-700">ERROR:</div>
                <div className="text-red-800">{data.error}</div>
              </div>
            )}
          </div>
        );
      }
    }

    return (
      <div className="text-gray-600 text-sm">
        {message.message || JSON.stringify(message, null, 2)}
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl h-5/6 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center space-x-4">
            <h2 className="text-xl font-bold">🤖 LLM Communication Monitor</h2>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm text-gray-600">
                {isConnected ? 'Connected' : 'Disconnected'} ({activeConnections} active)
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={sendTestMessage}
              className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
              disabled={!isConnected}
            >
              Send Test
            </button>
            <button
              onClick={clearMessages}
              className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600"
            >
              Clear
            </button>
            <button
              onClick={onClose}
              className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
            >
              Close
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-8">
              <div className="text-6xl mb-4">👂</div>
              <p>Listening for LLM communications...</p>
              <p className="text-sm mt-2">
                Try running a MongoDB deployment or sending a test message to see real-time LLM interactions.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg border-l-4 ${
                    message.type === 'llm_event'
                      ? message.event_type === 'request'
                        ? 'bg-blue-100 border-blue-400'
                        : message.data?.success
                        ? 'bg-green-100 border-green-400'
                        : 'bg-red-100 border-red-400'
                      : 'bg-gray-100 border-gray-400'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getMessageIcon(message)}</span>
                      <span className="font-semibold capitalize">
                        {message.type === 'llm_event'
                          ? `LLM ${message.event_type}`
                          : message.type.replace('_', ' ')
                        }
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {formatTimestamp(message.broadcast_timestamp || message.timestamp || message.data?.timestamp)}
                    </span>
                  </div>
                  {renderMessageContent(message)}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LLMMonitor;