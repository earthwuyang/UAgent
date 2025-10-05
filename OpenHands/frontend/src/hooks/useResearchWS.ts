/**
 * Research WebSocket Hook
 *
 * Manages WebSocket connection for real-time research tree updates.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useResearchTreeStore } from '#/state/research-tree-store';

interface UseResearchWSOptions {
  experimentId: string;
  autoConnect?: boolean;
  onError?: (error: Event) => void;
  onClose?: (event: CloseEvent) => void;
}

export function useResearchWS({
  experimentId,
  autoConnect = true,
  onError,
  onClose,
}: UseResearchWSOptions) {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const store = useResearchTreeStore();

  const connect = useCallback(() => {
    if (!experimentId) {
      console.warn('[Research WS] No experimentId provided, skipping connection');
      return;
    }

    if (ws.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    // Clear any pending reconnect
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }

    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/research/ws/experiment/${experimentId}`;

    console.log(`[Research WS] Connecting to ${wsUrl}`);

    try {
      ws.current = new WebSocket(wsUrl);
    } catch (error) {
      console.error('[Research WS] Failed to create WebSocket connection:', error);
      if (onError) {
        onError(error as Event);
      }
      return;
    }

    ws.current.onopen = () => {
      console.log('[Research WS] Connected');
      store.setConnected(true);
    };

    ws.current.onclose = (event) => {
      console.log('[Research WS] Closed:', event.code, event.reason);
      store.setConnected(false);

      if (onClose) {
        onClose(event);
      }

      // Attempt reconnect if not a normal closure
      if (event.code !== 1000 && event.code !== 1001) {
        console.log('[Research WS] Reconnecting in 3s...');
        reconnectTimeout.current = setTimeout(() => {
          connect();
        }, 3000);
      }
    };

    ws.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        console.log('[Research WS] Message:', message.type, message.version);

        // Route message to store based on type
        switch (message.type) {
          case 'tree_snapshot':
            store.setSnapshot(message);
            break;

          case 'node_added':
            store.applyNodeAdded(message);
            break;

          case 'node_updated':
            store.applyNodeUpdated(message);
            break;

          case 'edge_added':
            store.applyEdgeAdded(message);
            break;

          case 'stats_updated':
            store.applyStatsUpdated(message);
            break;

          case 'event_log':
            store.applyEventLog(message);
            break;

          case 'error':
            console.error('[Research WS] Error message:', message.data.error);
            break;

          case 'complete':
            console.log('[Research WS] Complete:', message.data.summary);
            break;

          default:
            console.warn('[Research WS] Unknown message type:', message.type);
        }
      } catch (err) {
        console.error('[Research WS] Failed to parse message:', err);
      }
    };

    ws.current.onerror = (error) => {
      console.error('[Research WS] Error:', error);
      store.setConnected(false);
      if (onError) {
        onError(error);
      }
    };

    ws.current.onclose = (event) => {
      console.log('[Research WS] Closed:', event.code, event.reason);
      store.setConnected(false);

      if (onClose) {
        onClose(event);
      }

      // Attempt reconnect if not a normal closure
      if (event.code !== 1000 && event.code !== 1001) {
        console.log('[Research WS] Reconnecting in 3s...');
        reconnectTimeout.current = setTimeout(() => {
          connect();
        }, 3000);
      }
    };
  }, [experimentId, store, onError, onClose]);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }

    if (ws.current) {
      ws.current.close(1000, 'User disconnected');
      ws.current = null;
    }

    store.setConnected(false);
  }, [store]);

  const send = useCallback((data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data));
    } else {
      console.warn('[Research WS] Cannot send, not connected');
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [experimentId, autoConnect]); // Only reconnect if experimentId changes

  return {
    isConnected: store.isConnected,
    connect,
    disconnect,
    send,
  };
}
