/**
 * Research Tree Store
 *
 * Manages research tree state with incremental updates from WebSocket.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

// Types matching backend models
export interface ResearchNode {
  id: string;
  type: string; // root, idea, hypothesis, experiment, etc.
  title: string;
  content: string;
  status: string; // pending, running, complete, failed
  visits: number;
  prior: number; // PUCT prior probability
  avg_value: number; // PUCT Q value
  cost: number;
  tokens_used: number;
  created_at?: string;
  completed_at?: string;
  isFilteredMatch?: boolean;
  isFilteredOut?: boolean;
}

export interface ResearchEdge {
  parent_id: string;
  child_id: string;
}

export interface TreeStats {
  total_nodes?: number;
  total_edges?: number;
  max_depth?: number;
  total_cost?: number;
  total_tokens?: number;
  completed_nodes?: number;
  failed_nodes?: number;
}

export interface TreeSnapshot {
  version: number;
  timestamp: string;
  experiment_id: string;
  data: {
    nodes: ResearchNode[];
    edges: ResearchEdge[];
    stats: TreeStats;
  };
}

// WebSocket message types
export interface WSMessage<T = unknown> {
  type: string;
  version: number;
  timestamp: string;
  data: T;
}

export interface ResearchTreeState {
  // Tree data
  version: number;
  lastUpdate: string | null;
  nodes: Map<string, ResearchNode>;
  edges: ResearchEdge[];
  stats: TreeStats;
  experimentId: string | null;

  // UI state
  selectedNodeId: string | null;
  selectedNodeIds: Set<string>;
  expandedNodeIds: Set<string>;
  filterType: string | null;
  filterStatus: string | null;
  searchQuery: string;
  error: string | null;
  isConnected: boolean;
  isLoading: boolean;

  // Connection coordination
  activeConnectionId: string | null;
  activeConnectionRefs: number;

  // Actions
  setSnapshot: (snapshot: TreeSnapshot) => void;
  setExperimentId: (id: string | null) => void;

  // Incremental updates (from WebSocket)
  applyNodeAdded: (message: WSMessage<{ node_id: string; node: ResearchNode }>) => void;
  applyNodeUpdated: (message: WSMessage<{ node_id: string; updates: Partial<ResearchNode> }>) => void;
  applyEdgeAdded: (message: WSMessage<{ parent_id: string; child_id: string }>) => void;
  applyStatsUpdated: (message: WSMessage<{ stats: TreeStats }>) => void;
  applyEventLog: (message: WSMessage) => void;

  // UI actions
  selectNode: (nodeId: string | null) => void;
  toggleNodeSelection: (nodeId: string) => void;
  clearSelection: () => void;
  expandNode: (nodeId: string) => void;
  collapseNode: (nodeId: string) => void;
  toggleExpanded: (nodeId: string) => void;
  expandAll: () => void;
  collapseAll: () => void;

  setFilterType: (type: string | null) => void;
  setFilterStatus: (status: string | null) => void;
  setSearchQuery: (query: string) => void;
  clearFilters: () => void;
  resetFilters: () => void;
  resetSelection: () => void;

  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  getFilteredNodes: () => ResearchNode[];

  acquireConnection: (experimentId: string) => boolean;
  releaseConnection: (experimentId: string) => boolean;

  // Reset
  reset: () => void;
}

interface ResearchTreePersistedState {
  filterType: string | null;
  filterStatus: string | null;
  searchQuery: string;
  expandedNodeIds: string[];
}

const memoryStorage: Storage = {
  get length() {
    return 0;
  },
  clear: () => undefined,
  getItem: () => null,
  key: () => null,
  removeItem: () => undefined,
  setItem: () => undefined,
};

const isBrowser = typeof window !== 'undefined';

const storage = createJSONStorage<ResearchTreePersistedState>(() =>
  isBrowser ? window.localStorage : memoryStorage
);

const createInitialState = () => ({
  version: 0,
  lastUpdate: null as string | null,
  nodes: new Map<string, ResearchNode>(),
  edges: [] as ResearchEdge[],
  stats: {} as TreeStats,
  experimentId: null as string | null,
  selectedNodeId: null as string | null,
  selectedNodeIds: new Set<string>(),
  expandedNodeIds: new Set<string>(),
  filterType: null as string | null,
  filterStatus: null as string | null,
  searchQuery: '',
  error: null as string | null,
  isConnected: false,
  isLoading: false,
  activeConnectionId: null as string | null,
  activeConnectionRefs: 0,
});

const initialState = createInitialState();

export const useResearchTreeStore = create<ResearchTreeState>()(
  persist(
    (set, get) => {
      const parseTimestamp = (value?: string | null) => {
        if (!value) {
          return null;
        }
        const parsed = Date.parse(value);
        return Number.isNaN(parsed) ? null : parsed;
      };

      const shouldApply = (incomingVersion?: number, timestamp?: string | null) => {
        const state = get();
        const currentVersion = state.version ?? 0;

        if (typeof incomingVersion !== 'number') {
          return true;
        }

        if (currentVersion === 0 && state.lastUpdate === null) {
          return true;
        }

        if (incomingVersion > currentVersion) {
          return true;
        }

        if (incomingVersion < currentVersion) {
          return false;
        }

        const incomingTime = parseTimestamp(timestamp);
        const currentTime = parseTimestamp(state.lastUpdate);

        if (currentTime === null) {
          return true;
        }

        if (incomingTime === null) {
          return false;
        }

        return incomingTime >= currentTime;
      };

      return {
        ...initialState,

        setSnapshot: (snapshot: TreeSnapshot) => {
          if (!snapshot || typeof snapshot !== 'object') {
            console.warn('[ResearchTree] Invalid snapshot received');
            set({
              ...createInitialState(),
              expandedNodeIds: new Set(get().expandedNodeIds),
              filterType: get().filterType,
              filterStatus: get().filterStatus,
              searchQuery: get().searchQuery,
            });
            return;
          }

          if (!shouldApply(snapshot.version, snapshot.timestamp)) {
            set({ isLoading: false });
            return;
          }

          const nodesMap = new Map<string, ResearchNode>();
          const nodes = Array.isArray(snapshot.data?.nodes) ? snapshot.data.nodes : [];
          nodes.forEach((node) => {
            if (node && typeof node === 'object' && node.id) {
              nodesMap.set(node.id, node);
            }
          });

          set({
            version: snapshot.version ?? 0,
            lastUpdate: snapshot.timestamp ?? new Date().toISOString(),
            nodes: nodesMap,
            edges: Array.isArray(snapshot.data?.edges) ? snapshot.data.edges : [],
            stats: snapshot.data?.stats ?? {},
            experimentId: snapshot.experiment_id ?? null,
            isLoading: false,
            error: null,
          });
        },

        setExperimentId: (id: string | null) => {
          set({ experimentId: id });
        },

        applyNodeAdded: (message) => {
          if (!shouldApply(message.version, message.timestamp)) {
            return;
          }

          const { node_id, node } = message.data;
          set((state) => {
            const nodes = new Map(state.nodes);
            nodes.set(node_id, node);
            return {
              nodes,
              version: message.version,
              lastUpdate: message.timestamp,
            };
          });
        },

        applyNodeUpdated: (message) => {
          if (!shouldApply(message.version, message.timestamp)) {
            return;
          }

          const { node_id, updates } = message.data;
          set((state) => {
            const existing = state.nodes.get(node_id);
            if (!existing) {
              return {};
            }
            const nodes = new Map(state.nodes);
            nodes.set(node_id, {
              ...existing,
              ...updates,
            });
            return {
              nodes,
              version: message.version,
              lastUpdate: message.timestamp,
            };
          });
        },

        applyEdgeAdded: (message) => {
          if (!shouldApply(message.version, message.timestamp)) {
            return;
          }

          const { parent_id, child_id } = message.data;
          set((state) => ({
            edges: [...state.edges, { parent_id, child_id }],
            version: message.version,
            lastUpdate: message.timestamp,
          }));
        },

        applyStatsUpdated: (message) => {
          if (!shouldApply(message.version, message.timestamp)) {
            return;
          }

          const { stats } = message.data;
          set({ stats, version: message.version, lastUpdate: message.timestamp });
        },

        applyEventLog: (message) => {
          console.log('[Research Event]', message.data);
          set({ lastUpdate: message.timestamp });
        },

        selectNode: (nodeId) => {
          set({ selectedNodeId: nodeId });
        },

        toggleNodeSelection: (nodeId) => {
          set((state) => {
            const next = new Set(state.selectedNodeIds);
            if (next.has(nodeId)) {
              next.delete(nodeId);
            } else {
              next.add(nodeId);
            }
            return { selectedNodeIds: next };
          });
        },

        clearSelection: () => {
          set({ selectedNodeId: null, selectedNodeIds: new Set() });
        },

        expandNode: (nodeId) => {
          set((state) => {
            const expanded = new Set(state.expandedNodeIds);
            expanded.add(nodeId);
            return { expandedNodeIds: expanded };
          });
        },

        collapseNode: (nodeId) => {
          set((state) => {
            const expanded = new Set(state.expandedNodeIds);
            expanded.delete(nodeId);
            return { expandedNodeIds: expanded };
          });
        },

        toggleExpanded: (nodeId) => {
          set((state) => {
            const expanded = new Set(state.expandedNodeIds);
            if (expanded.has(nodeId)) {
              expanded.delete(nodeId);
            } else {
              expanded.add(nodeId);
            }
            return { expandedNodeIds: expanded };
          });
        },

        expandAll: () => {
          set((state) => {
            const expanded = new Set<string>();
            state.nodes.forEach((_value, key) => expanded.add(key));
            return { expandedNodeIds: expanded };
          });
        },

        collapseAll: () => {
          set({ expandedNodeIds: new Set() });
        },

        setFilterType: (type) => {
          set({ filterType: type });
        },

        setFilterStatus: (status) => {
          set({ filterStatus: status });
        },

        setSearchQuery: (query) => {
          set({ searchQuery: query });
        },

        clearFilters: () => {
          set({ filterType: null, filterStatus: null, searchQuery: '' });
        },

        resetFilters: () => {
          set({ filterType: null, filterStatus: null, searchQuery: '' });
        },

        resetSelection: () => {
          set({ selectedNodeId: null, selectedNodeIds: new Set() });
        },

        setConnected: (connected) => {
          set({ isConnected: connected });
        },

        setLoading: (loading) => {
          set({ isLoading: loading });
        },

        setError: (error) => {
          set({ error });
        },

        getFilteredNodes: () => {
          const state = get();
          const query = state.searchQuery.trim().toLowerCase();
          const hasTypeFilter = Boolean(state.filterType);
          const hasStatusFilter = Boolean(state.filterStatus);
          const hasQuery = query.length > 0;

          const nodes = Array.from(state.nodes.values());

          if (!hasTypeFilter && !hasStatusFilter && !hasQuery) {
            return nodes;
          }

          return nodes.filter((node) => {
            if (hasTypeFilter && node.type !== state.filterType) {
              return false;
            }
            if (hasStatusFilter && node.status !== state.filterStatus) {
              return false;
            }
            if (hasQuery) {
              const haystack = `${node.title ?? ''} ${node.content ?? ''}`.toLowerCase();
              if (!haystack.includes(query)) {
                return false;
              }
            }
            return true;
          });
        },

        acquireConnection: (experimentId) => {
          if (!experimentId) {
            return false;
          }

          let shouldOpen = false;
          set((state) => {
            if (state.activeConnectionId && state.activeConnectionId !== experimentId) {
              shouldOpen = true;
              return {
                activeConnectionId: experimentId,
                activeConnectionRefs: 1,
              } as Partial<ResearchTreeState>;
            }

            const nextRefs = state.activeConnectionRefs + 1;
            shouldOpen = nextRefs === 1;
            return {
              activeConnectionId: experimentId,
              activeConnectionRefs: nextRefs,
            } as Partial<ResearchTreeState>;
          });

          return shouldOpen;
        },

        releaseConnection: (experimentId) => {
          if (!experimentId) {
            return false;
          }

          let shouldClose = false;
          set((state) => {
            if (state.activeConnectionId !== experimentId) {
              return {};
            }

            const nextRefs = Math.max(0, state.activeConnectionRefs - 1);
            shouldClose = nextRefs === 0;
            return {
              activeConnectionRefs: nextRefs,
              activeConnectionId: nextRefs === 0 ? null : state.activeConnectionId,
            } as Partial<ResearchTreeState>;
          });

          return shouldClose;
        },

        reset: () => {
          const fresh = createInitialState();
          set({ ...fresh });
        },
      };
    },
    {
      name: 'research-tree-ui-state',
      storage,
      partialize: (state) => ({
        filterType: state.filterType,
        filterStatus: state.filterStatus,
        searchQuery: state.searchQuery,
        expandedNodeIds: Array.from(state.expandedNodeIds),
      }),
      merge: (persistedState, currentState) => {
        if (!persistedState) {
          return currentState;
        }

        const { filterType, filterStatus, searchQuery, expandedNodeIds } =
          persistedState;

        return {
          ...currentState,
          filterType: filterType ?? currentState.filterType,
          filterStatus: filterStatus ?? currentState.filterStatus,
          searchQuery: searchQuery ?? currentState.searchQuery,
          expandedNodeIds: Array.isArray(expandedNodeIds)
            ? new Set(expandedNodeIds)
            : currentState.expandedNodeIds,
        } as ResearchTreeState;
      },
    }
  )
);
