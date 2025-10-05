/**
 * Research Tree Store
 *
 * Manages research tree state with incremental updates from WebSocket.
 * Based on ROMA's graph visualization approach.
 */

import { create } from 'zustand';

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
export interface WSMessage {
  type: string;
  version: number;
  timestamp: string;
  data: any;
}

interface ResearchTreeState {
  // Tree data
  version: number;
  nodes: Map<string, ResearchNode>;
  edges: ResearchEdge[];
  stats: TreeStats;
  experimentId: string | null;

  // UI state
  selectedNodeId: string | null;
  selectedNodeIds: Set<string>;
  expandedNodeIds: Set<string>;
  isConnected: boolean;
  isLoading: boolean;

  // Actions
  setSnapshot: (snapshot: TreeSnapshot) => void;
  setExperimentId: (id: string) => void;

  // Incremental updates (from WebSocket)
  applyNodeAdded: (message: WSMessage) => void;
  applyNodeUpdated: (message: WSMessage) => void;
  applyEdgeAdded: (message: WSMessage) => void;
  applyStatsUpdated: (message: WSMessage) => void;
  applyEventLog: (message: WSMessage) => void;

  // UI actions
  selectNode: (nodeId: string | null) => void;
  toggleNodeSelection: (nodeId: string) => void;
  clearSelection: () => void;
  expandNode: (nodeId: string) => void;
  collapseNode: (nodeId: string) => void;

  // Connection state
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;

  // Reset
  reset: () => void;
}

const initialState = {
  version: 0,
  nodes: new Map<string, ResearchNode>(),
  edges: [],
  stats: {},
  experimentId: null,
  selectedNodeId: null,
  selectedNodeIds: new Set<string>(),
  expandedNodeIds: new Set<string>(),
  isConnected: false,
  isLoading: false,
};

export const useResearchTreeStore = create<ResearchTreeState>((set, get) => ({
  ...initialState,

  setSnapshot: (snapshot: TreeSnapshot) => {
    // Validate snapshot structure
    if (!snapshot || typeof snapshot !== 'object') {
      console.warn('Invalid snapshot received, initializing with empty tree');
      set({
        version: 0,
        nodes: new Map<string, ResearchNode>(),
        edges: [],
        stats: {},
        experimentId: null,
        isLoading: false,
      });
      return;
    }

    const nodesMap = new Map<string, ResearchNode>();
    // Handle case where snapshot.data.nodes might be undefined or not an array
    const nodes = Array.isArray(snapshot.data?.nodes) ? snapshot.data.nodes : [];
    nodes.forEach((node) => {
      // Validate node structure before adding
      if (node && typeof node === 'object' && node.id) {
        nodesMap.set(node.id, node);
      }
    });

    set({
      version: snapshot.version || 0,
      nodes: nodesMap,
      edges: Array.isArray(snapshot.data?.edges) ? snapshot.data.edges : [],
      stats: snapshot.data?.stats || {},
      experimentId: snapshot.experiment_id || null,
      isLoading: false,
    });
  },

  setExperimentId: (id: string) => {
    set({ experimentId: id });
  },

  applyNodeAdded: (message: WSMessage) => {
    const { node_id, node } = message.data;

    set((state) => {
      const newNodes = new Map(state.nodes);
      newNodes.set(node_id, node);

      return {
        nodes: newNodes,
        version: message.version,
      };
    });
  },

  applyNodeUpdated: (message: WSMessage) => {
    const { node_id, updates } = message.data;

    set((state) => {
      const node = state.nodes.get(node_id);
      if (!node) return state;

      const newNodes = new Map(state.nodes);
      newNodes.set(node_id, {
        ...node,
        ...updates,
      });

      return {
        nodes: newNodes,
        version: message.version,
      };
    });
  },

  applyEdgeAdded: (message: WSMessage) => {
    const { parent_id, child_id } = message.data;

    set((state) => ({
      edges: [
        ...state.edges,
        { parent_id, child_id },
      ],
      version: message.version,
    }));
  },

  applyStatsUpdated: (message: WSMessage) => {
    const { stats } = message.data;

    set({
      stats,
      version: message.version,
    });
  },

  applyEventLog: (message: WSMessage) => {
    // Event logs are displayed but don't update tree structure
    // Could be used for logging panel
    console.log('[Research Event]', message.data);
  },

  selectNode: (nodeId: string | null) => {
    set({ selectedNodeId: nodeId });
  },

  toggleNodeSelection: (nodeId: string) => {
    set((state) => {
      const newSelection = new Set(state.selectedNodeIds);
      if (newSelection.has(nodeId)) {
        newSelection.delete(nodeId);
      } else {
        newSelection.add(nodeId);
      }
      return { selectedNodeIds: newSelection };
    });
  },

  clearSelection: () => {
    set({
      selectedNodeId: null,
      selectedNodeIds: new Set(),
    });
  },

  expandNode: (nodeId: string) => {
    set((state) => {
      const newExpanded = new Set(state.expandedNodeIds);
      newExpanded.add(nodeId);
      return { expandedNodeIds: newExpanded };
    });
  },

  collapseNode: (nodeId: string) => {
    set((state) => {
      const newExpanded = new Set(state.expandedNodeIds);
      newExpanded.delete(nodeId);
      return { expandedNodeIds: newExpanded };
    });
  },

  setConnected: (connected: boolean) => {
    set({ isConnected: connected });
  },

  setLoading: (loading: boolean) => {
    set({ isLoading: loading });
  },

  reset: () => {
    set(initialState);
  },
}));
