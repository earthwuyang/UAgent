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

const sharedConnection: {
  socket: WebSocket | null;
  reconnectTimeout: ReturnType<typeof setTimeout> | null;
  experimentId: string | null;
  retryCount: number;
  lastRetryTime: number;
} = {
  socket: null,
  reconnectTimeout: null,
  experimentId: null,
  retryCount: 0,
  lastRetryTime: 0,
};

const BASE_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;
const MAX_RETRY_ATTEMPTS = 10;

export function useResearchWS({
  experimentId,
  autoConnect = true,
  onError,
  onClose,
}: UseResearchWSOptions) {
  const ws = useRef<WebSocket | null>(null);
  const connectRef = useRef<(options?: { skipAcquire?: boolean }) => void>();
  const isConnected = useResearchTreeStore((state) => state.isConnected);

  const cleanupReconnect = useCallback(() => {
    if (sharedConnection.reconnectTimeout) {
      clearTimeout(sharedConnection.reconnectTimeout);
      sharedConnection.reconnectTimeout = null;
    }
  }, []);

const scheduleReconnect = useCallback(() => {
  cleanupReconnect();

  const { activeConnectionRefs } = useResearchTreeStore.getState();
  if (activeConnectionRefs === 0) {
    return;
  }

  // Check retry limit
  if (sharedConnection.retryCount >= MAX_RETRY_ATTEMPTS) {
    console.error(
      `[Research WS] Max retry attempts (${MAX_RETRY_ATTEMPTS}) reached for experiment ${experimentId}. Giving up.`
    );
    const state = useResearchTreeStore.getState();
    state.setError(
      `Failed to connect after ${MAX_RETRY_ATTEMPTS} attempts. Please refresh the page.`
    );
    return;
  }

  // Calculate exponential backoff delay
  const exponentialDelay = Math.min(
    BASE_RECONNECT_DELAY_MS * Math.pow(2, sharedConnection.retryCount),
    MAX_RECONNECT_DELAY_MS
  );
  // Add jitter to prevent thundering herd
  const jitter = Math.random() * 1000;
  const delay = exponentialDelay + jitter;

  sharedConnection.retryCount += 1;
  console.log(
    `[Research WS] Scheduling reconnect attempt ${sharedConnection.retryCount}/${MAX_RETRY_ATTEMPTS} in ${Math.round(delay)}ms`
  );

  sharedConnection.reconnectTimeout = setTimeout(() => {
    sharedConnection.reconnectTimeout = null;
    sharedConnection.lastRetryTime = Date.now();
    
    // Only attempt reconnect if there are still listeners for this experiment
    const state = useResearchTreeStore.getState();
    if (state.activeConnectionRefs > 0 && state.activeConnectionId === experimentId) {
      state.setConnected(false);
      const reconnectFn = connectRef.current;
      if (reconnectFn) {
        reconnectFn({ skipAcquire: true });
      }
    }
  }, delay);
}, [cleanupReconnect, experimentId]);

  const handleClose = useCallback(
    (event: CloseEvent) => {
      console.log('[Research WS] Closed:', event.code, event.reason);
      const store = useResearchTreeStore.getState();
      store.setConnected(false);

      sharedConnection.socket = null;

      if (onClose) {
        onClose(event);
      }

      const shouldRetry = event.code !== 1000 && event.code !== 1001;
      if (shouldRetry) {
        scheduleReconnect();
      }
    },
    [onClose, scheduleReconnect]
  );

  const connect = useCallback(
    (options?: { skipAcquire?: boolean }) => {
      if (!experimentId) {
        console.warn('[Research WS] No experimentId provided, skipping connection');
        return;
      }

      const store = useResearchTreeStore.getState();

      let shouldOpen = options?.skipAcquire ? true : store.acquireConnection(experimentId);
      const existingSocket = sharedConnection.socket;

      if (!shouldOpen) {
        if (!existingSocket || existingSocket.readyState === WebSocket.CLOSED) {
          shouldOpen = true;
        } else if (sharedConnection.experimentId === experimentId) {
          ws.current = existingSocket;
          if (existingSocket.readyState === WebSocket.OPEN) {
            store.setConnected(true);
          }
          return;
        }
      }

      cleanupReconnect();

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/api/research/ws/experiment/${experimentId}`;

      console.log(`[Research WS] Connecting to ${wsUrl}`);

      let socket: WebSocket;
      try {
        socket = new WebSocket(wsUrl);
      } catch (error) {
        console.error('[Research WS] Failed to create WebSocket connection:', error);
        if (!options?.skipAcquire) {
          store.releaseConnection(experimentId);
        }
        if (onError) {
          onError(error as Event);
        }
        return;
      }

      sharedConnection.socket = socket;
      sharedConnection.experimentId = experimentId;
      ws.current = socket;

      socket.onopen = () => {
        console.log('[Research WS] Connected successfully');
        // Reset retry count on successful connection
        sharedConnection.retryCount = 0;
        const latestStore = useResearchTreeStore.getState();
        latestStore.setConnected(true);
        latestStore.setLoading(false);
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          const latestStore = useResearchTreeStore.getState();

          switch (message.type) {
            case 'connected':
              console.log('[Research WS] Connection confirmed:', message.connection_id);
              // Connection confirmation received, no additional action needed
              // The onopen handler already sets connected state
              break;
            case 'tree_snapshot':
              latestStore.setSnapshot(message);
              break;
            case 'node_added':
              latestStore.applyNodeAdded(message);
              break;
            case 'node_updated':
              latestStore.applyNodeUpdated(message);
              break;
            case 'edge_added':
              latestStore.applyEdgeAdded(message);
              break;
            case 'stats_updated':
              latestStore.applyStatsUpdated(message);
              break;
            case 'event_log':
              latestStore.applyEventLog(message);
              break;
            case 'error':
              console.error('[Research WS] Error message:', message.data?.error);
              latestStore.setError(message.data?.error ?? 'Unknown websocket error');
              break;
            case 'complete':
              console.log('[Research WS] Complete:', message.data?.summary);
              break;
            default:
              console.warn('[Research WS] Unknown message type:', message.type);
          }
        } catch (err) {
          console.error('[Research WS] Failed to parse message:', err);
        }
      };

      socket.onerror = (error) => {
        console.error('[Research WS] Error:', error);
        useResearchTreeStore.getState().setConnected(false);
        if (onError) {
          onError(error);
        }
      };

      socket.onclose = handleClose;
    },
    [cleanupReconnect, experimentId, handleClose, onError]
  );

  const disconnect = useCallback(() => {
    if (!experimentId) {
      return;
    }

    cleanupReconnect();

    const store = useResearchTreeStore.getState();
    const shouldClose = store.releaseConnection(experimentId);

    if (shouldClose && sharedConnection.socket) {
      sharedConnection.socket.close(1000, 'User disconnected');
      sharedConnection.socket = null;
      sharedConnection.experimentId = null;
    }

    ws.current = null;
    store.setConnected(false);
  }, [cleanupReconnect, experimentId]);

  const send = useCallback((data: unknown) => {
    const socket = sharedConnection.socket ?? ws.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(data));
    } else {
      console.warn('[Research WS] Cannot send, not connected');
    }
  }, []);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [connect, disconnect, autoConnect, experimentId]);

  return {
    isConnected,
    connect,
    disconnect,
    send,
  };
}
