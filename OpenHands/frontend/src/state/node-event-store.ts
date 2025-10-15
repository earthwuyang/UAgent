/**
 * Node Event Store
 *
 * Manages node-level events for research tree nodes.
 * Stores events per node with LRU cache to prevent memory leaks.
 * Separate from main conversation store to avoid coupling.
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// Event types matching backend ResearchEvent
export type NodeEventType = 
  | 'task_start'
  | 'task_complete'
  | 'observation'
  | 'action'
  | 'message'
  | 'error'
  | 'warning'
  | 'info'
  | 'step_start'
  | 'step_complete'
  | 'tool_use'
  | 'tool_result';

export interface NodeEvent {
  id: string;
  node_id: string;
  experiment_id: string;
  event_type: NodeEventType;
  message?: string | null;
  content?: Record<string, unknown> | null;
  timestamp: string;
  agent_id?: string | null;
  step_id?: string | null;
  tool_name?: string | null;
  tool_result?: Record<string, unknown> | null;
  metadata?: Record<string, unknown>;
}

// API response types
export interface NodeEventsResponse {
  events: NodeEvent[];
  total_count: number;
  offset: number;
  limit: number;
  has_more: boolean;
}

// State interface
export interface NodeEventState {
  // Events storage
  nodeEvents: Map<string, NodeEvent[]>;
  
  // Loading state
  loadingNodes: Set<string>;
  subscribedNodes: Set<string>;
  
  // Pagination state
  eventCounts: Map<string, number>;
  hasMore: Map<string, boolean>;
  offsets: Map<string, number>;
  
  // Error state
  errors: Map<string, string | null>;

  // Event management
  loadNodeEvents: (nodeId: string, experimentId: string, offset?: number, limit?: number) => Promise<void>;
  appendNodeEvent: (nodeId: string, event: NodeEvent) => void;
  clearNodeEvents: (nodeId: string) => void;
  clearAllNodeEvents: () => void;
  
  // Subscription management
  subscribeToNode: (nodeId: string, experimentId: string) => void;
  unsubscribeFromNode: (nodeId: string) => void;
  
  // Pagination helpers
  loadMoreEvents: (nodeId: string) => Promise<void>;
  resetPagination: (nodeId: string) => void;
  
  // Error handling
  setNodeError: (nodeId: string, error: string | null) => void;
  clearNodeError: (nodeId: string) => void;
  
  // Utility
  getNodeEvents: (nodeId: string) => NodeEvent[];
  getEventCount: (nodeId: string) => number;
  getHasMore: (nodeId: string) => boolean;
  getOffset: (nodeId: string) => number;
  isLoading: (nodeId: string) => boolean;
  isSubscribed: (nodeId: string) => boolean;
  
  // Reset
  reset: () => void;
}

// Constants
const DEFAULT_LIMIT = 50;
const MAX_EVENTS_PER_NODE = 100; // LRU cache: keep last 100 events per node to prevent memory leaks
const EMPTY_EVENTS: NodeEvent[] = []; // Cached empty array to prevent infinite re-renders in selectors

// Helper functions
const createInitialState = (): Omit<NodeEventState, 'loadNodeEvents' | 'appendNodeEvent' | 'clearNodeEvents' | 'clearAllNodeEvents' | 'subscribeToNode' | 'unsubscribeFromNode' | 'loadMoreEvents' | 'resetPagination' | 'setNodeError' | 'clearNodeError' | 'getNodeEvents' | 'getEventCount' | 'getHasMore' | 'getOffset' | 'isLoading' | 'isSubscribed' | 'reset'> => ({
  nodeEvents: new Map(),
  loadingNodes: new Set(),
  subscribedNodes: new Set(),
  eventCounts: new Map(),
  hasMore: new Map(),
  offsets: new Map(),
  errors: new Map(),
});

// Store creation
export const useNodeEventStore = create<NodeEventState>()(
  devtools(
    (set, get) => ({
      ...createInitialState(),

      // Event management
      loadNodeEvents: async (nodeId: string, experimentId: string, offset = 0, limit = DEFAULT_LIMIT) => {
        console.log('[NodeEventStore] Loading node events:', { nodeId, experimentId, offset, limit });
        
        // Set loading state
        set((state) => ({
          loadingNodes: new Set(state.loadingNodes).add(nodeId),
          errors: new Map(state.errors).set(nodeId, null),
        }));

        try {
          // Call the API endpoint
          const response = await fetch(
            `/api/research/experiments/${experimentId}/nodes/${nodeId}/events?offset=${offset}&limit=${limit}`,
            {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
              },
            }
          );

          if (!response.ok) {
            throw new Error(`Failed to load events: ${response.status} ${response.statusText}`);
          }

          const data: NodeEventsResponse = await response.json();
          
          console.log('[NodeEventStore] Loaded events:', { nodeId, count: data.events.length, total: data.total_count });

          set((state) => {
            // Update events map
            const existingEvents = state.nodeEvents.get(nodeId) || [];
            const allEvents = offset === 0 ? data.events : [...existingEvents, ...data.events];
            
            // Trim to max events if needed
            const trimmedEvents = allEvents.slice(-MAX_EVENTS_PER_NODE);
            
            return {
              nodeEvents: new Map(state.nodeEvents).set(nodeId, trimmedEvents),
              eventCounts: new Map(state.eventCounts).set(nodeId, data.total_count),
              hasMore: new Map(state.hasMore).set(nodeId, data.has_more),
              offsets: new Map(state.offsets).set(nodeId, offset + data.events.length),
              loadingNodes: (() => {
                const newLoading = new Set(state.loadingNodes);
                newLoading.delete(nodeId);
                return newLoading;
              })(),
            };
          });

        } catch (error) {
          console.error('[NodeEventStore] Failed to load node events:', error);
          
          set((state) => ({
            loadingNodes: (() => {
                const newLoading = new Set(state.loadingNodes);
                newLoading.delete(nodeId);
                return newLoading;
              })(),
            errors: new Map(state.errors).set(nodeId, error instanceof Error ? error.message : 'Unknown error'),
          }));
        }
      },

      appendNodeEvent: (nodeId: string, event: NodeEvent) => {
        console.log('[NodeEventStore] Appending node event:', { nodeId, eventId: event.id, type: event.event_type });
        
        set((state) => {
          const existingEvents = state.nodeEvents.get(nodeId) || [];
          // LRU cache: Keep only last MAX_EVENTS_PER_NODE events to prevent memory leaks
          const updatedEvents = [...existingEvents, event].slice(-MAX_EVENTS_PER_NODE);
          
          // Update event count
          const currentCount = state.eventCounts.get(nodeId) || 0;
          
          return {
            nodeEvents: new Map(state.nodeEvents).set(nodeId, updatedEvents),
            eventCounts: new Map(state.eventCounts).set(nodeId, currentCount + 1),
          };
        });
      },

      clearNodeEvents: (nodeId: string) => {
        console.log('[NodeEventStore] Clearing node events:', { nodeId });
        
        set((state) => ({
          nodeEvents: (() => {
              const newEvents = new Map(state.nodeEvents);
              newEvents.delete(nodeId);
              return newEvents;
            })(),
          eventCounts: (() => {
              const newCounts = new Map(state.eventCounts);
              newCounts.delete(nodeId);
              return newCounts;
            })(),
          hasMore: (() => {
              const newHasMore = new Map(state.hasMore);
              newHasMore.delete(nodeId);
              return newHasMore;
            })(),
          offsets: (() => {
              const newOffsets = new Map(state.offsets);
              newOffsets.delete(nodeId);
              return newOffsets;
            })(),
          errors: (() => {
              const newErrors = new Map(state.errors);
              newErrors.delete(nodeId);
              return newErrors;
            })(),
        }));
      },

      clearAllNodeEvents: () => {
        console.log('[NodeEventStore] Clearing all node events');
        set({
          nodeEvents: new Map(),
          eventCounts: new Map(),
          hasMore: new Map(),
          offsets: new Map(),
          errors: new Map(),
        });
      },

      // Subscription management
      subscribeToNode: (nodeId: string, experimentId: string) => {
        console.log('[NodeEventStore] Subscribing to node:', { nodeId, experimentId });
        
        set((state) => ({
          subscribedNodes: new Set(state.subscribedNodes).add(nodeId),
        }));

        // Dispatch custom event for WebSocket client
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('subscribe-node', {
            detail: { nodeId, experimentId }
          }));
        }
      },

      unsubscribeFromNode: (nodeId: string) => {
        console.log('[NodeEventStore] Unsubscribing from node:', { nodeId });
        
        set((state) => ({
          subscribedNodes: (() => {
              const newSubscribed = new Set(state.subscribedNodes);
              newSubscribed.delete(nodeId);
              return newSubscribed;
            })(),
        }));

        // Dispatch custom event for WebSocket client
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('unsubscribe-node', {
            detail: { nodeId }
          }));
        }
      },

      // Pagination helpers
      loadMoreEvents: async (nodeId: string) => {
        console.warn('[NodeEventStore] loadMoreEvents requires experimentId parameter');
      },

      resetPagination: (nodeId: string) => {
        set((state) => ({
          offsets: new Map(state.offsets).set(nodeId, 0),
          hasMore: new Map(state.hasMore).set(nodeId, true),
        }));
      },

      // Error handling
      setNodeError: (nodeId: string, error: string | null) => {
        set((state) => ({
          errors: new Map(state.errors).set(nodeId, error),
        }));
      },

      clearNodeError: (nodeId: string) => {
        set((state) => ({
          errors: (() => {
              const newErrors = new Map(state.errors);
              newErrors.delete(nodeId);
              return newErrors;
            })(),
        }));
      },

      // Utility getters
      getNodeEvents: (nodeId: string) => {
        const events = get().nodeEvents.get(nodeId);
        return events ? [...events] : [];
      },

      getEventCount: (nodeId: string) => {
        return get().eventCounts.get(nodeId) || 0;
      },

      getHasMore: (nodeId: string) => {
        return get().hasMore.get(nodeId) ?? true;
      },

      getOffset: (nodeId: string) => {
        return get().offsets.get(nodeId) || 0;
      },

      isLoading: (nodeId: string) => {
        return get().loadingNodes.has(nodeId);
      },

      isSubscribed: (nodeId: string) => {
        return get().subscribedNodes.has(nodeId);
      },

      // Reset store
      reset: () => {
        console.log('[NodeEventStore] Resetting store');
        set(createInitialState());
      },
    }),
    {
      name: 'node-event-store',
    }
  )
);

// Convenience selectors
export const useNodeEvents = (nodeId: string) => {
  return useNodeEventStore((state) => state.nodeEvents.get(nodeId) || EMPTY_EVENTS);
};

export const useNodeEventCount = (nodeId: string) => {
  return useNodeEventStore((state) => state.getEventCount(nodeId));
};

export const useNodeHasMoreEvents = (nodeId: string) => {
  return useNodeEventStore((state) => state.getHasMore(nodeId));
};

export const useNodeIsLoading = (nodeId: string) => {
  return useNodeEventStore((state) => state.isLoading(nodeId));
};

export const useNodeIsSubscribed = (nodeId: string) => {
  return useNodeEventStore((state) => state.isSubscribed(nodeId));
};

export const useNodeError = (nodeId: string) => {
  return useNodeEventStore((state) => state.errors.get(nodeId) || null);
};

// Compatibility exports for direct calling (matching existing store patterns)
export const loadNodeEvents = (nodeId: string, experimentId: string, offset?: number, limit?: number) => {
  return useNodeEventStore.getState().loadNodeEvents(nodeId, experimentId, offset, limit);
};

export const appendNodeEvent = (nodeId: string, event: NodeEvent) => {
  useNodeEventStore.getState().appendNodeEvent(nodeId, event);
};

export const subscribeToNode = (nodeId: string, experimentId: string) => {
  useNodeEventStore.getState().subscribeToNode(nodeId, experimentId);
};

export const unsubscribeFromNode = (nodeId: string) => {
  useNodeEventStore.getState().unsubscribeFromNode(nodeId);
};

export const clearNodeEvents = (nodeId: string) => {
  useNodeEventStore.getState().clearNodeEvents(nodeId);
};

export const clearAllNodeEvents = () => {
  useNodeEventStore.getState().clearAllNodeEvents();
};

export const resetNodeEventStore = () => {
  useNodeEventStore.getState().reset();
};
