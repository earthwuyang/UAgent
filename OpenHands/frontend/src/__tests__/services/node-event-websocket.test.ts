/**
 * Tests for NodeEventWebSocket
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { NodeEventWebSocket } from '#/services/node-event-websocket';
import { useNodeEventStore, NodeEvent } from '#/state/node-event-store';

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  
  sentMessages: string[] = [];

  constructor(public url: string) {
    // Simulate connection opening after a short delay
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 10);
  }

  send(data: string): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.sentMessages.push(data);
  }

  close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      const event = new CloseEvent('close', { code, reason, wasClean: true });
      this.onclose(event);
    }
  }

  // Helper to simulate receiving a message
  simulateMessage(data: unknown): void {
    if (this.onmessage) {
      const event = new MessageEvent('message', { 
        data: JSON.stringify(data) 
      });
      this.onmessage(event);
    }
  }

  // Helper to simulate an error
  simulateError(): void {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }

  // Helper to simulate closing
  simulateClose(code = 1000, reason = '', wasClean = true): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      const event = new CloseEvent('close', { code, reason, wasClean });
      this.onclose(event);
    }
  }
}

// Store last created WebSocket instance for testing
let lastMockWebSocket: MockWebSocket | null = null;

describe('NodeEventWebSocket', () => {
  const mockExperimentId = 'exp_123';
  let client: NodeEventWebSocket;

  beforeEach(() => {
    // Reset Zustand store
    useNodeEventStore.getState().reset();

    // Mock WebSocket
    lastMockWebSocket = null;
    const WebSocketConstructor = function(this: MockWebSocket, url: string) {
      lastMockWebSocket = new MockWebSocket(url);
      Object.assign(this, lastMockWebSocket);
      return this;
    } as unknown as typeof WebSocket;
    
    // Add WebSocket constants
    Object.defineProperty(WebSocketConstructor, 'CONNECTING', { value: 0, writable: false });
    Object.defineProperty(WebSocketConstructor, 'OPEN', { value: 1, writable: false });
    Object.defineProperty(WebSocketConstructor, 'CLOSING', { value: 2, writable: false });
    Object.defineProperty(WebSocketConstructor, 'CLOSED', { value: 3, writable: false });
    
    vi.stubGlobal('WebSocket', WebSocketConstructor);

    // Mock environment
    import.meta.env.VITE_BACKEND_BASE_URL = 'localhost:3000';
    
    // Mock window location
    Object.defineProperty(window, 'location', {
      value: { protocol: 'http:', host: 'localhost:3000' },
      writable: true,
    });

    // Clear timers
    vi.clearAllTimers();
    vi.useFakeTimers();
  });

  afterEach(() => {
    if (client) {
      client.disconnect();
    }
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  describe('Connection Management', () => {
    it('should connect to WebSocket server', async () => {
      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();

      expect(global.WebSocket).toHaveBeenCalledWith(
        `ws://localhost:3000/ws/research/${mockExperimentId}`
      );
      expect(lastMockWebSocket).toBeTruthy();
    });

    it('should set correct WebSocket URL based on protocol', () => {
      // Test HTTPS protocol
      Object.defineProperty(window, 'location', {
        value: { protocol: 'https:', host: 'example.com' },
        writable: true,
      });

      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();

      expect(global.WebSocket).toHaveBeenCalledWith(
        `wss://example.com/ws/research/${mockExperimentId}`
      );
    });

    it('should handle connection open event', async () => {
      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();

      // Fast-forward timers to trigger onopen
      await vi.advanceTimersByTimeAsync(20);

      expect(client.isConnected).toBe(true);
      expect(client.currentReconnectAttempts).toBe(0);
    });

    it('should not reconnect if already connected', () => {
      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();

      const firstWs = lastMockWebSocket;
      
      // Try to connect again
      client.connect();

      // Should not create a new WebSocket
      expect(lastMockWebSocket).toBe(firstWs);
    });

    it('should disconnect cleanly', async () => {
      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();

      await vi.advanceTimersByTimeAsync(20);
      expect(client.isConnected).toBe(true);

      client.disconnect();

      expect(client.isConnected).toBe(false);
      expect(lastMockWebSocket?.readyState).toBe(MockWebSocket.CLOSED);
    });
  });

  describe('Message Routing', () => {
    const mockNodeEvent: NodeEvent = {
      id: 'evt_1',
      node_id: 'node_1',
      experiment_id: mockExperimentId,
      event_type: 'message',
      message: 'Test message',
      timestamp: '2024-01-01T00:00:00Z',
      content: null,
    };

    beforeEach(async () => {
      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();
      await vi.advanceTimersByTimeAsync(20);
    });

    it('should route node_event to store', () => {
      const appendSpy = vi.spyOn(useNodeEventStore.getState(), 'appendNodeEvent');

      lastMockWebSocket?.simulateMessage({
        type: 'node_event',
        data: mockNodeEvent,
      });

      expect(appendSpy).toHaveBeenCalledWith('node_1', mockNodeEvent);
    });

    it('should handle subscription_confirmed message', () => {
      const consoleSpy = vi.spyOn(console, 'log');

      lastMockWebSocket?.simulateMessage({
        type: 'subscription_confirmed',
        data: { node_id: 'node_1' },
      });

      expect(consoleSpy).toHaveBeenCalledWith(
        '[NodeEventWebSocket] Subscription confirmed for node:',
        'node_1'
      );
    });

    it('should handle node_complete message', () => {
      const consoleSpy = vi.spyOn(console, 'log');

      lastMockWebSocket?.simulateMessage({
        type: 'node_complete',
        data: { node_id: 'node_1', message: 'Node execution completed' },
      });

      expect(consoleSpy).toHaveBeenCalledWith(
        '[NodeEventWebSocket] Node completed:',
        'node_1',
        'Node execution completed'
      );
    });

    it('should handle error message from server', () => {
      const setErrorSpy = vi.spyOn(useNodeEventStore.getState(), 'setNodeError');

      lastMockWebSocket?.simulateMessage({
        type: 'error',
        data: { error: 'Node execution failed', node_id: 'node_1' },
      });

      expect(setErrorSpy).toHaveBeenCalledWith('node_1', 'Node execution failed');
    });

    it('should handle malformed messages gracefully', () => {
      const consoleErrorSpy = vi.spyOn(console, 'error');

      // Send invalid JSON through MessageEvent
      if (lastMockWebSocket?.onmessage) {
        const event = new MessageEvent('message', { data: 'invalid json' });
        lastMockWebSocket.onmessage(event);
      }

      expect(consoleErrorSpy).toHaveBeenCalled();
    });

    it('should warn on unknown message types', () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn');

      lastMockWebSocket?.simulateMessage({
        type: 'unknown_type',
        data: { some: 'data' },
      });

      expect(consoleWarnSpy).toHaveBeenCalled();
    });
  });

  describe('Reconnection Logic', () => {
    beforeEach(() => {
      client = new NodeEventWebSocket(mockExperimentId);
    });

    it('should reconnect with exponential backoff', async () => {
      client.connect();
      await vi.advanceTimersByTimeAsync(20);

      // Simulate unexpected disconnect
      lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);

      // First reconnect attempt - 1000ms delay (1000 * 2^0)
      expect(client.currentReconnectAttempts).toBe(0);
      await vi.advanceTimersByTimeAsync(1000);
      expect(client.currentReconnectAttempts).toBe(1);

      // Simulate disconnect again
      lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);

      // Second reconnect attempt - 2000ms delay (1000 * 2^1)
      await vi.advanceTimersByTimeAsync(2000);
      expect(client.currentReconnectAttempts).toBe(2);

      // Simulate disconnect again
      lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);

      // Third reconnect attempt - 4000ms delay (1000 * 2^2)
      await vi.advanceTimersByTimeAsync(4000);
      expect(client.currentReconnectAttempts).toBe(3);
    });

    it('should cap reconnection delay at 10 seconds', async () => {
      client.connect();
      await vi.advanceTimersByTimeAsync(20);

      // Simulate multiple disconnects to reach max delay
      for (let i = 0; i < 5; i++) {
        lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);
        
        const delay = Math.min(1000 * Math.pow(2, i), 10000);
        await vi.advanceTimersByTimeAsync(delay);
      }

      // 6th attempt should still use 10s max delay (not 32000ms)
      lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);
      
      // Should reconnect after 10s, not 32s
      await vi.advanceTimersByTimeAsync(10000);
      expect(client.currentReconnectAttempts).toBeLessThanOrEqual(5);
    });

    it('should stop reconnecting after max attempts', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error');
      
      client.connect();
      await vi.advanceTimersByTimeAsync(20);

      // Simulate 5 disconnects (max attempts)
      for (let i = 0; i < 5; i++) {
        lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);
        const delay = Math.min(1000 * Math.pow(2, i), 10000);
        await vi.advanceTimersByTimeAsync(delay);
      }

      // Simulate one more disconnect
      lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);
      
      // Should log max attempts reached
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        '[NodeEventWebSocket] Max reconnection attempts reached'
      );
    });

    it('should not reconnect on clean disconnect', async () => {
      client.connect();
      await vi.advanceTimersByTimeAsync(20);

      const initialAttempts = client.currentReconnectAttempts;

      // Simulate clean close (code 1000)
      lastMockWebSocket?.simulateClose(1000, 'Normal closure', true);

      // Wait and verify no reconnect attempt
      await vi.advanceTimersByTimeAsync(5000);
      expect(client.currentReconnectAttempts).toBe(initialAttempts);
    });

    it('should not reconnect on manual disconnect', async () => {
      client.connect();
      await vi.advanceTimersByTimeAsync(20);

      client.disconnect();

      // Wait and verify no reconnect attempt
      await vi.advanceTimersByTimeAsync(5000);
      expect(client.isConnected).toBe(false);
    });

    it('should reset reconnection attempts on successful connection', async () => {
      client.connect();
      await vi.advanceTimersByTimeAsync(20);

      // Simulate a disconnect
      lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);
      expect(client.currentReconnectAttempts).toBe(0);

      // Wait for first reconnect
      await vi.advanceTimersByTimeAsync(1000);
      expect(client.currentReconnectAttempts).toBe(1);

      // Connection succeeds
      await vi.advanceTimersByTimeAsync(20);
      expect(client.currentReconnectAttempts).toBe(0);
    });
  });

  describe('Subscription Management', () => {
    beforeEach(async () => {
      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();
      await vi.advanceTimersByTimeAsync(20);
    });

    it('should subscribe to node on custom event', () => {
      const subscribeEvent = new CustomEvent('subscribe-node', {
        detail: { nodeId: 'node_1', experimentId: mockExperimentId },
      });

      window.dispatchEvent(subscribeEvent);

      expect(lastMockWebSocket?.sentMessages).toContainEqual(
        JSON.stringify({ type: 'subscribe_node', node_id: 'node_1' })
      );
      expect(client.subscribedNodeIds).toContain('node_1');
    });

    it('should unsubscribe from node on custom event', () => {
      // First subscribe
      const subscribeEvent = new CustomEvent('subscribe-node', {
        detail: { nodeId: 'node_1', experimentId: mockExperimentId },
      });
      window.dispatchEvent(subscribeEvent);

      // Clear sent messages
      lastMockWebSocket!.sentMessages = [];

      // Then unsubscribe
      const unsubscribeEvent = new CustomEvent('unsubscribe-node', {
        detail: { nodeId: 'node_1' },
      });
      window.dispatchEvent(unsubscribeEvent);

      expect(lastMockWebSocket?.sentMessages).toContainEqual(
        JSON.stringify({ type: 'unsubscribe_node', node_id: 'node_1' })
      );
      expect(client.subscribedNodeIds).not.toContain('node_1');
    });

    it('should ignore subscriptions for different experiments', () => {
      const subscribeEvent = new CustomEvent('subscribe-node', {
        detail: { nodeId: 'node_1', experimentId: 'different_exp' },
      });

      window.dispatchEvent(subscribeEvent);

      expect(lastMockWebSocket?.sentMessages).toHaveLength(0);
      expect(client.subscribedNodeIds).not.toContain('node_1');
    });

    it('should resubscribe to nodes on reconnection', async () => {
      // Subscribe to multiple nodes
      ['node_1', 'node_2', 'node_3'].forEach(nodeId => {
        window.dispatchEvent(new CustomEvent('subscribe-node', {
          detail: { nodeId, experimentId: mockExperimentId },
        }));
      });

      expect(client.subscribedNodeIds).toHaveLength(3);

      // Clear sent messages
      lastMockWebSocket!.sentMessages = [];

      // Simulate disconnect and reconnect
      lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);
      await vi.advanceTimersByTimeAsync(1000);
      await vi.advanceTimersByTimeAsync(20);

      // Should resubscribe to all nodes
      expect(lastMockWebSocket?.sentMessages).toHaveLength(3);
      expect(lastMockWebSocket?.sentMessages).toContainEqual(
        JSON.stringify({ type: 'subscribe_node', node_id: 'node_1' })
      );
      expect(lastMockWebSocket?.sentMessages).toContainEqual(
        JSON.stringify({ type: 'subscribe_node', node_id: 'node_2' })
      );
      expect(lastMockWebSocket?.sentMessages).toContainEqual(
        JSON.stringify({ type: 'subscribe_node', node_id: 'node_3' })
      );
    });

    it('should not send duplicate subscriptions', () => {
      // Subscribe to same node twice
      const subscribeEvent = new CustomEvent('subscribe-node', {
        detail: { nodeId: 'node_1', experimentId: mockExperimentId },
      });

      window.dispatchEvent(subscribeEvent);
      window.dispatchEvent(subscribeEvent);

      // Should only have one subscription message
      const subscribeMessages = lastMockWebSocket?.sentMessages.filter(
        msg => msg === JSON.stringify({ type: 'subscribe_node', node_id: 'node_1' })
      );
      expect(subscribeMessages).toHaveLength(1);
    });

    it('should clear subscriptions on disconnect', () => {
      // Subscribe to nodes
      ['node_1', 'node_2'].forEach(nodeId => {
        window.dispatchEvent(new CustomEvent('subscribe-node', {
          detail: { nodeId, experimentId: mockExperimentId },
        }));
      });

      expect(client.subscribedNodeIds).toHaveLength(2);

      // Disconnect
      client.disconnect();

      expect(client.subscribedNodeIds).toHaveLength(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle WebSocket creation errors', () => {
      const ErrorWebSocket = vi.fn(() => {
        throw new Error('WebSocket creation failed');
      }) as unknown as typeof WebSocket;
      vi.stubGlobal('WebSocket', ErrorWebSocket);

      const consoleErrorSpy = vi.spyOn(console, 'error');

      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();

      expect(consoleErrorSpy).toHaveBeenCalled();
    });

    it('should handle WebSocket error events', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error');

      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();
      await vi.advanceTimersByTimeAsync(20);

      lastMockWebSocket?.simulateError();

      expect(consoleErrorSpy).toHaveBeenCalledWith('[NodeEventWebSocket] Error:', expect.any(Event));
    });

    it('should not crash on send when disconnected', () => {
      client = new NodeEventWebSocket(mockExperimentId);
      
      // Try to subscribe before connecting
      const subscribeEvent = new CustomEvent('subscribe-node', {
        detail: { nodeId: 'node_1', experimentId: mockExperimentId },
      });
      
      // Should not throw
      expect(() => {
        window.dispatchEvent(subscribeEvent);
      }).not.toThrow();
    });
  });

  describe('Cleanup', () => {
    it('should remove event listeners on disconnect', () => {
      const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');

      client = new NodeEventWebSocket(mockExperimentId);
      client.disconnect();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('subscribe-node', expect.any(Function));
      expect(removeEventListenerSpy).toHaveBeenCalledWith('unsubscribe-node', expect.any(Function));
    });

    it('should clear reconnection timeout on disconnect', async () => {
      client = new NodeEventWebSocket(mockExperimentId);
      client.connect();
      await vi.advanceTimersByTimeAsync(20);

      // Trigger reconnection
      lastMockWebSocket?.simulateClose(1006, 'Connection lost', false);
      
      // Disconnect before reconnect happens
      client.disconnect();

      // Wait for what would be the reconnect time
      await vi.advanceTimersByTimeAsync(5000);

      // Should not have reconnected
      expect(client.isConnected).toBe(false);
    });
  });
});
