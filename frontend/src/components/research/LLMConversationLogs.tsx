/**
 * LLMConversationLogs Component
 * Displays real-time LLM interaction logs with streaming support
 */

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Badge } from '../ui/badge';
import { ScrollArea } from '../ui/scroll-area';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { WS_BASE_URL } from '../../config';

// LLM message interfaces
interface LLMMessage {
  id: string;
  session_id: string;
  type: 'prompt' | 'response' | 'token' | 'error' | 'system';
  timestamp: string;
  content: string;
  engine: string;
  metadata?: {
    prompt_length?: number;
    response_length?: number;
    processing_time?: number;
    token_count?: number;
    model?: string;
  };
}

interface LLMStreamEvent {
  type: 'llm_prompt_start' | 'llm_token' | 'llm_prompt_complete' | 'llm_error' | 'conversation_history';
  session_id: string;
  timestamp: string;
  prompt?: string;
  token?: string;
  error?: string;
  engine?: string;
  response?: string;
  conversation?: LLMMessage[];
}

interface LLMConversationLogsProps {
  sessionId: string;
  nodeId?: string;
  engine?: string;
  onPromptRequest?: (prompt: string) => void;
}

const LLMConversationLogs: React.FC<LLMConversationLogsProps> = ({
  sessionId,
  nodeId,
  engine = 'qwen',
  onPromptRequest
}) => {
  const [messages, setMessages] = useState<LLMMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [currentPrompt, setCurrentPrompt] = useState('');
  const [currentResponse, setCurrentResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [expandedMessageId, setExpandedMessageId] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const currentResponseRef = useRef('');
  const msgCounterRef = useRef(0);

  const transformEventToMessage = useCallback(
    (event: any): LLMMessage | null => {
      const baseTimestamp = event.timestamp || new Date().toISOString();
      switch (event.type) {
        case 'llm_prompt_start':
          return {
            id: event.id ?? `prompt-${baseTimestamp}-${Math.random().toString(36).slice(2,7)}`,
            session_id: event.session_id,
            type: 'prompt',
            timestamp: baseTimestamp,
            content: event.prompt || '',
            engine: event.engine || engine,
          };
        case 'llm_token':
          return null; // tokens are aggregated into the final response
        case 'llm_prompt_complete':
          return {
            id: event.id ?? `response-${baseTimestamp}-${Math.random().toString(36).slice(2,7)}`,
            session_id: event.session_id,
            type: 'response',
            timestamp: baseTimestamp,
            content: event.response || '',
            engine,
          };
        case 'llm_error':
          return {
            id: event.id ?? `error-${baseTimestamp}-${Math.random().toString(36).slice(2,7)}`,
            session_id: event.session_id,
            type: 'error',
            timestamp: baseTimestamp,
            content: event.error || 'Unknown error',
            engine,
          };
        default:
          return null;
      }
    },
    [engine]
  );

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    // Find the scroll area viewport and scroll to bottom
    const scrollElement = document.querySelector('[data-radix-scroll-area-viewport]');
    if (scrollElement) {
      scrollElement.scrollTop = scrollElement.scrollHeight;
    }
  }, []);

  // Connect to LLM stream WebSocket
  useEffect(() => {
    let isCleanup = false;
    let reconnectTimeout: number | null = null;

    const connectWS = () => {
      if (isCleanup) return;

      const wsUrl = `${WS_BASE_URL}/ws/llm/${sessionId}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('LLM stream WebSocket connected');
        setIsConnected(true);

        // Request conversation history
        ws.send(JSON.stringify({
          action: 'get_conversation',
          session_id: sessionId
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data: LLMStreamEvent = JSON.parse(event.data);

          switch (data.type) {
            case 'llm_prompt_start': {
              setCurrentPrompt(data.prompt || '');
              setCurrentResponse('');
              currentResponseRef.current = '';
              setIsStreaming(true);

              const promptMessage: LLMMessage = {
                id: `prompt-${Date.now()}-${msgCounterRef.current++}`,
                session_id: data.session_id,
                type: 'prompt',
                timestamp: data.timestamp,
                content: data.prompt || '',
                engine: data.engine || engine
              };
              setMessages(prev => [...prev, promptMessage]);
              break;
            }

            case 'llm_token': {
              setCurrentResponse(prev => {
                const updated = prev + (data.token || '');
                currentResponseRef.current = updated;
                return updated;
              });
              break;
            }

            case 'llm_prompt_complete': {
              const finalResponse = currentResponseRef.current || data.response || '';
              if (finalResponse) {
                const responseMessage: LLMMessage = {
                  id: `response-${Date.now()}-${msgCounterRef.current++}`,
                  session_id: data.session_id,
                  type: 'response',
                  timestamp: data.timestamp,
                  content: finalResponse,
                  engine: engine
                };
                setMessages(prev => [...prev, responseMessage]);
              }
              setIsStreaming(false);
              setCurrentResponse('');
              currentResponseRef.current = '';
              break;
            }

            case 'llm_error': {
              const errorMessage: LLMMessage = {
                id: `error-${Date.now()}-${msgCounterRef.current++}`,
                session_id: data.session_id,
                type: 'error',
                timestamp: data.timestamp,
                content: data.error || 'Unknown error',
                engine: engine
              };
              setMessages(prev => [...prev, errorMessage]);
              setIsStreaming(false);
              setCurrentResponse('');
              currentResponseRef.current = '';
              break;
            }

            case 'conversation_history': {
              const historyPayload = (data as unknown as { conversation?: any[] }).conversation;
              if (Array.isArray(historyPayload)) {
                const transformed = historyPayload
                  .map(event => transformEventToMessage(event))
                  .filter((msg): msg is LLMMessage => Boolean(msg && msg.content));
                // Ensure unique keys: fix duplicate ids by appending an index suffix
                const seen = new Set<string>();
                const fixed = transformed.map((m, i) => {
                  let id = m.id;
                  if (seen.has(id)) {
                    id = `${id}-${i}`;
                  }
                  seen.add(id);
                  return { ...m, id };
                });
                setMessages(fixed);
              }
              break;
            }
          }
        } catch (error) {
          console.error('Error parsing LLM stream message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('LLM stream WebSocket disconnected');
        setIsConnected(false);

        // Only reconnect if not a clean close and component hasn't unmounted
        if (!isCleanup && event.code !== 1000) {
          reconnectTimeout = setTimeout(connectWS, 3000);
        }
      };

      ws.onerror = (error) => {
        console.error('LLM stream WebSocket error:', error);
        setIsConnected(false);
      };

      wsRef.current = ws;
      return ws;
    };

    const ws = connectWS();
    return () => {
      isCleanup = true;

      // Clear any pending reconnection timeout
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }

      // Close WebSocket connection with clean close code
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close(1000, 'Component unmounting');
      }
      wsRef.current = null;
    };
  }, [sessionId, engine]);

  // Auto-scroll when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, currentResponse, scrollToBottom]);

  // Send prompt to LLM
  const sendPrompt = useCallback((prompt: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'stream_prompt',
        prompt: prompt,
        engine: engine
      }));
    }
  }, [engine]);

  // Handle external prompt request
  useEffect(() => {
    if (onPromptRequest) {
      // onPromptRequest expects a string, but we need to provide sendPrompt function
      // This seems to be a design issue - commenting out for now
      // onPromptRequest(sendPrompt);
    }
  }, [sendPrompt, onPromptRequest]);

  const getMessageIcon = (type: string) => {
    switch (type) {
      case 'prompt': return '🤔';
      case 'response': return '🤖';
      case 'error': return '❌';
      case 'system': return '⚙️';
      default: return '💬';
    }
  };

  const getMessageColor = (type: string) => {
    switch (type) {
      case 'prompt': return 'bg-blue-50 border-blue-200';
      case 'response': return 'bg-green-50 border-green-200';
      case 'error': return 'bg-red-50 border-red-200';
      case 'system': return 'bg-gray-50 border-gray-200';
      default: return 'bg-white border-gray-200';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b">
        <div className="flex items-center space-x-2">
          <h4 className="font-semibold">🤖 LLM Conversation</h4>
          <Badge variant={isConnected ? 'default' : 'secondary'}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </Badge>
        </div>
        <div className="text-xs text-gray-500">
          {engine.toUpperCase()}
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-3">
        <div className="space-y-3">
          {messages.length === 0 && !isStreaming && (
            <div className="text-center text-gray-500 py-8">
              <div className="text-2xl mb-2">💭</div>
              <p>No LLM conversations yet</p>
              <p className="text-xs">Messages will appear here when LLM interactions occur</p>
            </div>
          )}

          {messages.map((message) => (
            <Card
              key={message.id}
              className={`p-3 ${getMessageColor(message.type)} cursor-pointer`}
            >
              <div
                onDoubleClick={() =>
                  setExpandedMessageId(prev => (prev === message.id ? null : message.id))
                }
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">{getMessageIcon(message.type)}</span>
                    <Badge variant="outline" className="text-xs">
                      {message.type.toUpperCase()}
                    </Badge>
                    <span className="text-xs text-gray-500">
                      {formatTimestamp(message.timestamp)}
                    </span>
                  </div>
                  <div className="text-xs text-gray-400">
                    {message.engine}
                  </div>
                </div>

                <div className="text-sm whitespace-pre-wrap break-words">
                  {expandedMessageId === message.id
                    ? message.content
                    : message.content.length > 280
                    ? `${message.content.slice(0, 280)}…`
                    : message.content}
                </div>

                {message.metadata && (
                  <div className="mt-2 pt-2 border-t border-gray-200">
                    <div className="text-xs text-gray-500 space-x-3">
                      {message.metadata.prompt_length && (
                        <span>Prompt: {message.metadata.prompt_length} chars</span>
                      )}
                      {message.metadata.response_length && (
                        <span>Response: {message.metadata.response_length} chars</span>
                      )}
                      {message.metadata.processing_time && (
                        <span>Time: {message.metadata.processing_time}ms</span>
                      )}
                      {message.metadata.token_count && (
                        <span>Tokens: {message.metadata.token_count}</span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </Card>
          ))}

          {/* Current streaming response */}
          {isStreaming && (
            <Card className="p-3 bg-green-50 border-green-200 border-dashed">
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-lg">🤖</span>
                <Badge variant="outline" className="text-xs">
                  STREAMING
                </Badge>
                <div className="flex space-x-1">
                  <div className="w-1 h-1 bg-green-500 rounded-full animate-bounce"></div>
                  <div className="w-1 h-1 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-1 h-1 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
              <div className="text-sm whitespace-pre-wrap break-words">
                {currentResponse}
                <span className="inline-block w-2 h-4 bg-green-500 animate-pulse ml-1"></span>
              </div>
            </Card>
          )}
        </div>
      </ScrollArea>

      {/* Quick Actions */}
      <div className="p-3 border-t bg-gray-50">
        <div className="text-xs text-gray-500 mb-2">Quick Actions</div>
        <div className="flex space-x-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => sendPrompt("Explain what you are currently doing")}
            disabled={!isConnected || isStreaming}
            className="text-xs"
          >
            🤔 Explain
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => sendPrompt("What is the current progress?")}
            disabled={!isConnected || isStreaming}
            className="text-xs"
          >
            📊 Progress
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => sendPrompt("Show me the detailed results so far")}
            disabled={!isConnected || isStreaming}
            className="text-xs"
          >
            📋 Details
          </Button>
        </div>
      </div>
    </div>
  );
};

export default LLMConversationLogs;
