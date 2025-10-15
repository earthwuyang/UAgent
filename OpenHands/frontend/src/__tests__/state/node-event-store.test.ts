/**
 * Node Event Store Tests
 * 
 * Unit tests for the NodeEventStore with >80% coverage
 */

import { renderHook, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { 
  useNodeEventStore,
  useNodeEvents,
  useNodeEventCount,
  useNodeHasMoreEvents,
  useNodeIsLoading,
  useNodeIsSubscribed,
  useNodeError,
  loadNodeEvents,
  appendNodeEvent,
  subscribeToNode,
  unsubscribeFromNode,
  clearNodeEvents,
  clearAllNodeEvents,
  resetNodeEventStore,
  type NodeEvent,
  type NodeEventsResponse,
} from '#/state/node-event-store';

// Mock fetch API
global.fetch = vi.fn();

// Mock window custom events
const mockDispatchEvent = vi.fn();
Object.defineProperty(window, 'dispatchEvent', {
  value: mockDispatchEvent,
  writable: true,
});

// Test data
const mockNodeEvent: NodeEvent = {
  id: 'event-1',
  node_id: 'node-1',
  experiment_id: 'exp-1',
  event_type: 'message',
  message: 'Test message',
  timestamp: '2024-01-01T00:00:00Z',
  agent_id: 'agent-1',
};

const mockEventsResponse: NodeEventsResponse = {
  events: [mockNodeEvent],
  total_count: 1,
  offset: 0,
  limit: 50,
  has_more: false,
};

describe('NodeEventStore', () => {
  beforeEach(() => {
    // Reset store before each test
    resetNodeEventStore();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Initial State', () => {
    it('should have correct initial state', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      expect(result.current.nodeEvents.size).toBe(0);
      expect(result.current.loadingNodes.size).toBe(0);
      expect(result.current.subscribedNodes.size).toBe(0);
      expect(result.current.eventCounts.size).toBe(0);
      expect(result.current.hasMore.size).toBe(0);
      expect(result.current.offsets.size).toBe(0);
      expect(result.current.errors.size).toBe(0);
    });
  });

  describe('Event Management', () => {
    it('should append event and enforce LRU cache limit', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      // Add 150 events (more than MAX_EVENTS_PER_NODE=100)
      act(() => {
        for (let i = 0; i < 150; i++) {
          result.current.appendNodeEvent('node-1', {
            ...mockNodeEvent,
            id: `event-${i}`,
          });
        }
      });
      
      // Should only keep last 100 events (LRU cache)
      const events = result.current.getNodeEvents('node-1');
      expect(events.length).toBe(100);
      expect(events[0].id).toBe('event-50'); // First kept event
      expect(events[99].id).toBe('event-149'); // Last event
      
      // Event count should track all appended events
      expect(result.current.getEventCount('node-1')).toBe(150);
    });
  });

  describe('Event Loading', () => {
    beforeEach(() => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => mockEventsResponse,
      } as Response);
    });

    it('should load node events successfully', async () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      await act(async () => {
        await result.current.loadNodeEvents('node-1', 'exp-1');
      });
      
      expect(result.current.loadingNodes.has('node-1')).toBe(false);
      expect(result.current.nodeEvents.get('node-1')).toEqual([mockNodeEvent]);
      expect(result.current.eventCounts.get('node-1')).toBe(1);
      expect(result.current.hasMore.get('node-1')).toBe(false);
      expect(result.current.offsets.get('node-1')).toBe(1);
      expect(result.current.errors.get('node-1')).toBeNull();
      
      expect(fetch).toHaveBeenCalledWith(
        '/api/research/experiments/exp-1/nodes/node-1/events?offset=0&limit=50',
        expect.objectContaining({
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });

    it('should handle loading errors', async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      } as Response);
      
      const { result } = renderHook(() => useNodeEventStore());
      
      await act(async () => {
        await result.current.loadNodeEvents('node-1', 'exp-1');
      });
      
      expect(result.current.loadingNodes.has('node-1')).toBe(false);
      expect(result.current.errors.get('node-1')).toBe('Failed to load events: 404 Not Found');
    });

    it('should load events with custom offset and limit', async () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      await act(async () => {
        await result.current.loadNodeEvents('node-1', 'exp-1', 10, 25);
      });
      
      expect(fetch).toHaveBeenCalledWith(
        '/api/research/experiments/exp-1/nodes/node-1/events?offset=10&limit=25',
        expect.any(Object)
      );
    });

    it('should handle network errors', async () => {
      vi.mocked(fetch).mockRejectedValue(new Error('Network error'));
      
      const { result } = renderHook(() => useNodeEventStore());
      
      await act(async () => {
        await result.current.loadNodeEvents('node-1', 'exp-1');
      });
      
      expect(result.current.errors.get('node-1')).toBe('Network error');
      expect(result.current.loadingNodes.has('node-1')).toBe(false);
    });
  });

  describe('Event Management', () => {
    it('should append node event', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.appendNodeEvent('node-1', mockNodeEvent);
      });
      
      expect(result.current.nodeEvents.get('node-1')).toEqual([mockNodeEvent]);
      expect(result.current.eventCounts.get('node-1')).toBe(1);
    });

    it('should limit events per node to max size', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      // Create more events than the max limit
      const events: NodeEvent[] = Array.from({ length: 1010 }, (_, i) => ({
        ...mockNodeEvent,
        id: `event-${i}`,
      }));
      
      act(() => {
        events.forEach(event => {
          result.current.appendNodeEvent('node-1', event);
        });
      });
      
      const storedEvents = result.current.nodeEvents.get('node-1');
      expect(storedEvents?.length).toBeLessThanOrEqual(1000);
      expect(storedEvents?.length).toBe(1000); // Should be exactly max
    });

    it('should clear node events', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.appendNodeEvent('node-1', mockNodeEvent);
        result.current.appendNodeEvent('node-1', { ...mockNodeEvent, id: 'event-2' });
        result.current.clearNodeEvents('node-1');
      });
      
      expect(result.current.nodeEvents.get('node-1')).toBeUndefined();
      expect(result.current.eventCounts.get('node-1')).toBeUndefined();
      expect(result.current.hasMore.get('node-1')).toBeUndefined();
      expect(result.current.offsets.get('node-1')).toBeUndefined();
      expect(result.current.errors.get('node-1')).toBeUndefined();
    });

    it('should clear all node events', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.appendNodeEvent('node-1', mockNodeEvent);
        result.current.appendNodeEvent('node-2', mockNodeEvent);
        result.current.clearAllNodeEvents();
      });
      
      expect(result.current.nodeEvents.size).toBe(0);
      expect(result.current.eventCounts.size).toBe(0);
      expect(result.current.hasMore.size).toBe(0);
      expect(result.current.offsets.size).toBe(0);
      expect(result.current.errors.size).toBe(0);
    });
  });

  describe('Subscription Management', () => {
    it('should subscribe to node', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.subscribeToNode('node-1', 'exp-1');
      });
      
      expect(result.current.subscribedNodes.has('node-1')).toBe(true);
      expect(mockDispatchEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'subscribe-node',
          detail: { nodeId: 'node-1', experimentId: 'exp-1' }
        })
      );
    });

    it('should unsubscribe from node', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.subscribeToNode('node-1', 'exp-1');
        result.current.unsubscribeFromNode('node-1');
      });
      
      expect(result.current.subscribedNodes.has('node-1')).toBe(false);
      expect(mockDispatchEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'unsubscribe-node',
          detail: { nodeId: 'node-1' }
        })
      );
    });
  });

  describe('Pagination', () => {
    beforeEach(() => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => mockEventsResponse,
      } as Response);
    });

    it('should load more events', async () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      // Set up initial state
      act(() => {
        result.current.switchToNodeContext('node-1', 'exp-1');
        result.current.appendNodeEvent('node-1', mockNodeEvent);
      });
      
      await act(async () => {
        await result.current.loadMoreEvents('node-1');
      });
      
      expect(fetch).toHaveBeenCalledWith(
        '/api/research/experiments/exp-1/nodes/node-1/events?offset=0&limit=50',
        expect.any(Object)
      );
    });

    it('should not load more events if no more available', async () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.switchToNodeContext('node-1', 'exp-1');
        result.current.hasMore.set('node-1', false); // No more events
      });
      
      await act(async () => {
        await result.current.loadMoreEvents('node-1');
      });
      
      expect(fetch).not.toHaveBeenCalled();
    });

    it('should reset pagination', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.appendNodeEvent('node-1', mockNodeEvent);
        result.current.resetPagination('node-1');
      });
      
      expect(result.current.offsets.get('node-1')).toBe(0);
      expect(result.current.hasMore.get('node-1')).toBe(true);
    });
  });

  describe('Error Handling', () => {
    it('should set node error', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.setNodeError('node-1', 'Test error');
      });
      
      expect(result.current.errors.get('node-1')).toBe('Test error');
    });

    it('should clear node error', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.setNodeError('node-1', 'Test error');
        result.current.clearNodeError('node-1');
      });
      
      expect(result.current.errors.get('node-1')).toBeUndefined();
    });
  });

  describe('Utility Methods', () => {
    it('should get node events correctly', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.appendNodeEvent('node-1', mockNodeEvent);
      });
      
      const events = result.current.getNodeEvents('node-1');
      expect(events).toEqual([mockNodeEvent]);
      
      const emptyEvents = result.current.getNodeEvents('node-2');
      expect(emptyEvents).toEqual([]);
    });

    it('should get event count correctly', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      expect(result.current.getEventCount('node-1')).toBe(0);
      
      act(() => {
        result.current.appendNodeEvent('node-1', mockNodeEvent);
      });
      
      expect(result.current.getEventCount('node-1')).toBe(1);
    });

    it('should check if has more correctly', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      expect(result.current.getHasMore('node-1')).toBe(true); // Default is true
      
      act(() => {
        result.current.hasMore.set('node-1', false);
      });
      
      expect(result.current.getHasMore('node-1')).toBe(false);
    });

    it('should get offset correctly', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      expect(result.current.getOffset('node-1')).toBe(0); // Default is 0
      
      act(() => {
        result.current.offsets.set('node-1', 10);
      });
      
      expect(result.current.getOffset('node-1')).toBe(10);
    });

    it('should check loading state correctly', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      expect(result.current.isLoading('node-1')).toBe(false);
      
      act(() => {
        result.current.loadingNodes.add('node-1');
      });
      
      expect(result.current.isLoading('node-1')).toBe(true);
    });

    it('should check subscription state correctly', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      expect(result.current.isSubscribed('node-1')).toBe(false);
      
      act(() => {
        result.current.subscribedNodes.add('node-1');
      });
      
      expect(result.current.isSubscribed('node-1')).toBe(true);
    });
  });

  describe('Reset Store', () => {
    it('should reset store to initial state', () => {
      const { result } = renderHook(() => useNodeEventStore());
      
      act(() => {
        result.current.switchToNodeContext('node-1', 'exp-1');
        result.current.appendNodeEvent('node-1', mockNodeEvent);
        result.current.subscribeToNode('node-1', 'exp-1');
        result.current.setNodeError('node-1', 'Error');
      });
      
      expect(result.current.contextMode).toBe('node');
      expect(result.current.nodeEvents.size).toBe(1);
      expect(result.current.subscribedNodes.size).toBe(1);
      expect(result.current.errors.size).toBe(1);
      
      act(() => {
        result.current.reset();
      });
      
      expect(result.current.contextMode).toBe('root');
      expect(result.current.nodeEvents.size).toBe(0);
      expect(result.current.subscribedNodes.size).toBe(0);
      expect(result.current.errors.size).toBe(0);
    });
  });
});

describe('Convenience Selectors', () => {
  beforeEach(() => {
    resetNodeEventStore();
  });

  describe('useActiveNodeEvents', () => {
    it('should return empty array when not in node context', () => {
      const { result } = renderHook(() => useActiveNodeEvents());
      expect(result.current).toEqual([]);
    });

    it('should return node events when in node context', () => {
      const { result: storeResult } = renderHook(() => useNodeEventStore());
      const { result: selectorResult } = renderHook(() => useActiveNodeEvents());
      
      act(() => {
        storeResult.current.switchToNodeContext('node-1', 'exp-1');
        storeResult.current.appendNodeEvent('node-1', mockNodeEvent);
      });
      
      expect(selectorResult.current).toEqual([mockNodeEvent]);
    });
  });

  describe('useIsNodeContext', () => {
    it('should return false when in root context', () => {
      const { result } = renderHook(() => useIsNodeContext());
      expect(result.current).toBe(false);
    });

    it('should return true when in node context', () => {
      const { result: storeResult } = renderHook(() => useNodeEventStore());
      const { result: selectorResult } = renderHook(() => useIsNodeContext());
      
      act(() => {
        storeResult.current.switchToNodeContext('node-1', 'exp-1');
      });
      
      expect(selectorResult.current).toBe(true);
    });
  });

  describe('useActiveNodeId', () => {
    it('should return current active node ID', () => {
      const { result: storeResult } = renderHook(() => useNodeEventStore());
      const { result: selectorResult } = renderHook(() => useActiveNodeId());
      
      expect(selectorResult.current).toBeNull();
      
      act(() => {
        storeResult.current.switchToNodeContext('node-1', 'exp-1');
      });
      
      expect(selectorResult.current).toBe('node-1');
    });
  });

  describe('useActiveExperimentId', () => {
    it('should return current active experiment ID', () => {
      const { result: storeResult } = renderHook(() => useNodeEventStore());
      const { result: selectorResult } = renderHook(() => useActiveExperimentId());
      
      expect(selectorResult.current).toBeNull();
      
      act(() => {
        storeResult.current.switchToNodeContext('node-1', 'exp-1');
      });
      
      expect(selectorResult.current).toBe('exp-1');
    });
  });

  describe('useNodeEvents', () => {
    it('should return events for specific node', () => {
      const { result: storeResult } = renderHook(() => useNodeEventStore());
      const { result: selectorResult } = renderHook(() => useNodeEvents('node-1'));
      
      act(() => {
        storeResult.current.appendNodeEvent('node-1', mockNodeEvent);
      });
      
      expect(selectorResult.current).toEqual([mockNodeEvent]);
    });
  });

  describe('useNodeEventCount', () => {
    it('should return event count for specific node', () => {
      const { result: storeResult } = renderHook(() => useNodeEventStore());
      const { result: selectorResult } = renderHook(() => useNodeEventCount('node-1'));
      
      act(() => {
        storeResult.current.appendNodeEvent('node-1', mockNodeEvent);
      });
      
      expect(selectorResult.current).toBe(1);
    });
  });

  describe('useNodeHasMoreEvents', () => {
    it('should return has_more state for specific node', () => {
      const { result } = renderHook(() => useNodeHasMoreEvents('node-1'));
      expect(result.current).toBe(true);
    });
  });

  describe('useNodeIsLoading', () => {
    it('should return loading state for specific node', () => {
      const { result } = renderHook(() => useNodeIsLoading('node-1'));
      expect(result.current).toBe(false);
    });
  });

  describe('useNodeIsSubscribed', () => {
    it('should return subscription state for specific node', () => {
      const { result } = renderHook(() => useNodeIsSubscribed('node-1'));
      expect(result.current).toBe(false);
    });
  });

  describe('useNodeError', () => {
    it('should return error state for specific node', () => {
      const { result: storeResult } = renderHook(() => useNodeEventStore());
      const { result: selectorResult } = renderHook(() => useNodeError('node-1'));
      
      act(() => {
        storeResult.current.setNodeError('node-1', 'Test error');
      });
      
      expect(selectorResult.current).toBe('Test error');
    });
  });
});

describe('Compatibility Exports', () => {
  beforeEach(() => {
    resetNodeEventStore();
    vi.clearAllMocks();
  });

  it('should export switchToNodeContext function', () => {
    expect(typeof switchToNodeContext).toBe('function');
    
    act(() => {
      switchToNodeContext('node-1', 'exp-1');
    });
    
    const state = useNodeEventStore.getState();
    expect(state.contextMode).toBe('node');
    expect(state.activeNodeId).toBe('node-1');
    expect(state.activeExperimentId).toBe('exp-1');
  });

  it('should export switchToRootContext function', () => {
    expect(typeof switchToRootContext).toBe('function');
    
    // First set node context
    act(() => {
      switchToNodeContext('node-1', 'exp-1');
    });
    
    // Then switch back to root
    act(() => {
      switchToRootContext();
    });
    
    const state = useNodeEventStore.getState();
    expect(state.contextMode).toBe('root');
    expect(state.activeNodeId).toBeNull();
    expect(state.activeExperimentId).toBeNull();
  });

  it('should export loadNodeEvents function', async () => {
    expect(typeof loadNodeEvents).toBe('function');
    
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => mockEventsResponse,
    } as Response);
    
    await act(async () => {
      await loadNodeEvents('node-1', 'exp-1');
    });
    
    const state = useNodeEventStore.getState();
    expect(state.nodeEvents.get('node-1')?.length).toBe(1);
  });

  it('should export appendNodeEvent function', () => {
    expect(typeof appendNodeEvent).toBe('function');
    
    act(() => {
      appendNodeEvent('node-1', mockNodeEvent);
    });
    
    const state = useNodeEventStore.getState();
    expect(state.nodeEvents.get('node-1')).toEqual([mockNodeEvent]);
  });

  it('should export subscribeToNode function', () => {
    expect(typeof subscribeToNode).toBe('function');
    
    act(() => {
      subscribeToNode('node-1', 'exp-1');
    });
    
    const state = useNodeEventStore.getState();
    expect(state.subscribedNodes.has('node-1')).toBe(true);
  });

  it('should export unsubscribeFromNode function', () => {
    expect(typeof unsubscribeFromNode).toBe('function');
    
    act(() => {
      subscribeToNode('node-1', 'exp-1');
      unsubscribeFromNode('node-1');
    });
    
    const state = useNodeEventStore.getState();
    expect(state.subscribedNodes.has('node-1')).toBe(false);
  });

  it('should export clearNodeEvents function', () => {
    expect(typeof clearNodeEvents).toBe('function');
    
    act(() => {
      appendNodeEvent('node-1', mockNodeEvent);
      clearNodeEvents('node-1');
    });
    
    const state = useNodeEventStore.getState();
    expect(state.nodeEvents.get('node-1')).toBeUndefined();
  });

  it('should export clearAllNodeEvents function', () => {
    expect(typeof clearAllNodeEvents).toBe('function');
    
    act(() => {
      appendNodeEvent('node-1', mockNodeEvent);
      appendNodeEvent('node-2', mockNodeEvent);
      clearAllNodeEvents();
    });
    
    const state = useNodeEventStore.getState();
    expect(state.nodeEvents.size).toBe(0);
  });

  it('should export resetNodeEventStore function', () => {
    expect(typeof resetNodeEventStore).toBe('function');
    
    act(() => {
      switchToNodeContext('node-1', 'exp-1');
      appendNodeEvent('node-1', mockNodeEvent);
      resetNodeEventStore();
    });
    
    const state = useNodeEventStore.getState();
    expect(state.contextMode).toBe('root');
    expect(state.nodeEvents.size).toBe(0);
  });
});
