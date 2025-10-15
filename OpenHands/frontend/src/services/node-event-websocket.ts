/**
 * Node Event WebSocket Client
 *
 * Manages WebSocket connection for real-time node event streaming.
 * Listens for custom events to manage node subscriptions and routes
 * incoming messages to the NodeEventStore.
 */

import { useNodeEventStore, NodeEvent } from '#/state/node-event-store';

interface SubscriptionDetail {
  nodeId: string;
  experimentId: string;
}

interface NodeEventMessage {
  type: 'node_event' | 'subscription_confirmed' | 'node_complete' | 'error';
  data: NodeEvent | { node_id: string; message?: string; error?: string };
}

export class NodeEventWebSocket {
  private ws: WebSocket | null = null;
  private experimentId: string;
  private nodeId: string | null = null; // Optional: for single-node connections
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private isManualDisconnect = false;
  private subscribedNodes = new Set<string>();
  
  private subscribeNodeListener: ((event: Event) => void) | null = null;
  private unsubscribeNodeListener: ((event: Event) => void) | null = null;

  constructor(experimentId: string, nodeId?: string) {
    this.experimentId = experimentId;
    this.nodeId = nodeId || null;
    this.setupEventListeners();
  }

  /**
   * Setup window event listeners for subscription management
   */
  private setupEventListeners(): void {
    this.subscribeNodeListener = ((event: CustomEvent<SubscriptionDetail>) => {
      const { nodeId, experimentId } = event.detail;
      
      // Only handle subscriptions for our experiment
      if (experimentId !== this.experimentId) {
        return;
      }

      console.log('[NodeEventWebSocket] Received subscribe request:', { nodeId, experimentId });
      this.subscribeToNode(nodeId);
    }) as EventListener;

    this.unsubscribeNodeListener = ((event: CustomEvent<{ nodeId: string }>) => {
      const { nodeId } = event.detail;
      console.log('[NodeEventWebSocket] Received unsubscribe request:', { nodeId });
      this.unsubscribeFromNode(nodeId);
    }) as EventListener;

    if (typeof window !== 'undefined') {
      window.addEventListener('subscribe-node', this.subscribeNodeListener);
      window.addEventListener('unsubscribe-node', this.unsubscribeNodeListener);
    }
  }

  /**
   * Remove window event listeners
   */
  private removeEventListeners(): void {
    if (typeof window !== 'undefined' && this.subscribeNodeListener && this.unsubscribeNodeListener) {
      window.removeEventListener('subscribe-node', this.subscribeNodeListener);
      window.removeEventListener('unsubscribe-node', this.unsubscribeNodeListener);
    }
    this.subscribeNodeListener = null;
    this.unsubscribeNodeListener = null;
  }

  /**
   * Calculate reconnection delay with exponential backoff
   */
  private getReconnectDelay(): number {
    const baseDelay = 1000;
    const maxDelay = 10000;
    const delay = Math.min(baseDelay * Math.pow(2, this.reconnectAttempts), maxDelay);
    return delay;
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) {
      console.log('[NodeEventWebSocket] Already connected or connecting');
      return;
    }

    this.isManualDisconnect = false;

    // Build WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const backendHost = import.meta.env.VITE_BACKEND_BASE_URL || window.location.host;
    
    // Use node-specific endpoint if nodeId provided, otherwise use experiment endpoint
    const wsUrl = this.nodeId
      ? `${protocol}//${backendHost}/api/research/ws/experiment/${this.experimentId}/node/${this.nodeId}`
      : `${protocol}//${backendHost}/api/research/ws/experiment/${this.experimentId}`;

    console.log(`[NodeEventWebSocket] Connecting to ${wsUrl} (attempt ${this.reconnectAttempts + 1})`);

    try {
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => this.handleOpen();
      this.ws.onmessage = (event) => this.handleMessage(event);
      this.ws.onerror = (error) => this.handleError(error);
      this.ws.onclose = (event) => this.handleClose(event);
    } catch (error) {
      console.error('[NodeEventWebSocket] Failed to create WebSocket:', error);
      this.scheduleReconnect();
    }
  }

  /**
   * Handle WebSocket open event
   */
  private handleOpen(): void {
    console.log('[NodeEventWebSocket] Connected');
    this.reconnectAttempts = 0;

    // Resubscribe to all previously subscribed nodes
    if (this.subscribedNodes.size > 0) {
      console.log('[NodeEventWebSocket] Resubscribing to nodes:', Array.from(this.subscribedNodes));
      this.subscribedNodes.forEach(nodeId => {
        this.sendSubscribeMessage(nodeId);
      });
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: NodeEventMessage = JSON.parse(event.data);
      
      switch (message.type) {
        case 'node_event':
          this.handleNodeEvent(message.data as NodeEvent);
          break;

        case 'subscription_confirmed':
          this.handleSubscriptionConfirmed(message.data as { node_id: string });
          break;

        case 'node_complete':
          this.handleNodeComplete(message.data as { node_id: string; message?: string });
          break;

        case 'error':
          this.handleErrorMessage(message.data as { error: string; node_id?: string });
          break;

        default:
          console.warn('[NodeEventWebSocket] Unknown message type:', message);
      }
    } catch (error) {
      console.error('[NodeEventWebSocket] Failed to parse message:', error);
    }
  }

  /**
   * Handle node_event message - append to store
   */
  private handleNodeEvent(event: NodeEvent): void {
    console.log('[NodeEventWebSocket] Received node event:', { 
      nodeId: event.node_id, 
      type: event.event_type,
      eventId: event.id 
    });

    // Append event to store
    useNodeEventStore.getState().appendNodeEvent(event.node_id, event);
  }

  /**
   * Handle subscription_confirmed message
   */
  private handleSubscriptionConfirmed(data: { node_id: string }): void {
    console.log('[NodeEventWebSocket] Subscription confirmed for node:', data.node_id);
    
    // Mark node as subscribed in store
    const store = useNodeEventStore.getState();
    const currentSubscribed = new Set(store.subscribedNodes);
    currentSubscribed.add(data.node_id);
  }

  /**
   * Handle node_complete message
   */
  private handleNodeComplete(data: { node_id: string; message?: string }): void {
    console.log('[NodeEventWebSocket] Node completed:', data.node_id, data.message);
  }

  /**
   * Handle error message from server
   */
  private handleErrorMessage(data: { error: string; node_id?: string }): void {
    console.error('[NodeEventWebSocket] Server error:', data.error, 
      data.node_id ? `(node: ${data.node_id})` : '');
    
    if (data.node_id) {
      useNodeEventStore.getState().setNodeError(data.node_id, data.error);
    }
  }

  /**
   * Handle WebSocket error event
   */
  private handleError(error: Event): void {
    console.error('[NodeEventWebSocket] Error:', error);
  }

  /**
   * Handle WebSocket close event
   */
  private handleClose(event: CloseEvent): void {
    console.log('[NodeEventWebSocket] Closed:', {
      code: event.code,
      reason: event.reason,
      wasClean: event.wasClean,
    });

    this.ws = null;

    // Don't reconnect if this was a manual disconnect or clean close
    if (this.isManualDisconnect || event.code === 1000 || event.code === 1001) {
      console.log('[NodeEventWebSocket] Connection closed cleanly, not reconnecting');
      return;
    }

    // Schedule reconnect for unexpected disconnections
    this.scheduleReconnect();
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[NodeEventWebSocket] Max reconnection attempts reached');
      return;
    }

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    const delay = this.getReconnectDelay();
    console.log(`[NodeEventWebSocket] Scheduling reconnect in ${delay}ms (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  /**
   * Subscribe to a node
   */
  private subscribeToNode(nodeId: string): void {
    if (!this.subscribedNodes.has(nodeId)) {
      this.subscribedNodes.add(nodeId);
      console.log('[NodeEventWebSocket] Added node to subscriptions:', nodeId);
    }

    // Send subscription message if connected
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.sendSubscribeMessage(nodeId);
    } else {
      console.log('[NodeEventWebSocket] Not connected, will subscribe when connected');
    }
  }

  /**
   * Send subscribe message to server
   */
  private sendSubscribeMessage(nodeId: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[NodeEventWebSocket] Cannot send subscribe message, not connected');
      return;
    }

    const message = {
      type: 'subscribe_node',
      node_id: nodeId,
    };

    console.log('[NodeEventWebSocket] Sending subscribe message:', message);
    this.ws.send(JSON.stringify(message));
  }

  /**
   * Unsubscribe from a node
   */
  private unsubscribeFromNode(nodeId: string): void {
    this.subscribedNodes.delete(nodeId);
    console.log('[NodeEventWebSocket] Removed node from subscriptions:', nodeId);

    // Send unsubscribe message if connected
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.sendUnsubscribeMessage(nodeId);
    }
  }

  /**
   * Send unsubscribe message to server
   */
  private sendUnsubscribeMessage(nodeId: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[NodeEventWebSocket] Cannot send unsubscribe message, not connected');
      return;
    }

    const message = {
      type: 'unsubscribe_node',
      node_id: nodeId,
    };

    console.log('[NodeEventWebSocket] Sending unsubscribe message:', message);
    this.ws.send(JSON.stringify(message));
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    console.log('[NodeEventWebSocket] Disconnecting');
    
    this.isManualDisconnect = true;

    // Clear reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Close WebSocket connection
    if (this.ws) {
      if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
        this.ws.close(1000, 'Client disconnect');
      }
      this.ws = null;
    }

    // Clear subscribed nodes
    this.subscribedNodes.clear();

    // Remove event listeners
    this.removeEventListeners();
  }

  /**
   * Get connection status
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get current reconnection attempts
   */
  get currentReconnectAttempts(): number {
    return this.reconnectAttempts;
  }

  /**
   * Get subscribed node IDs
   */
  get subscribedNodeIds(): string[] {
    return Array.from(this.subscribedNodes);
  }
}
