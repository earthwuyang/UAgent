/**
 * Research Tree Panel Component
 *
 * Floating panel that displays the research tree visualization with detail view.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity,
  DollarSign,
  Info,
  TrendingUp,
  X,
} from 'lucide-react';
import { ResearchTreeView } from './ResearchTreeView';
import { useResearchWS } from '#/hooks/useResearchWS';
import { Loader } from '#/components/shared/loader';
import { useResearchTreeStore } from '#/state/research-tree-store';
import type { ResearchNode } from '#/state/research-tree-store';
import { ResearchErrorBoundary } from './ResearchErrorBoundary';
import { cn } from '#/utils/utils';
import './research-tree.css';

export interface ResearchTreePanelProps {
  experimentId: string;
  onClose: () => void;
  size?: { width: number; height: number };
  onSizeChange?: (size: { width: number; height: number }) => void;
  onDragHandleMouseDown?: (e: React.MouseEvent) => void;
}

const detailPanelTransition = {
  type: 'spring',
  damping: 25,
  stiffness: 240,
};

export function ResearchTreePanel({ 
  experimentId, 
  onClose,
  size,
  onSizeChange,
  onDragHandleMouseDown,
}: ResearchTreePanelProps) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const {
    nodes,
    edges,
    stats,
    isLoading,
    setLoading,
    selectedNodeId,
    selectNode,
    setExperimentId,
    error,
    setError,
    lastUpdate,
  } = useResearchTreeStore((state) => ({
    nodes: state.nodes,
    edges: state.edges,
    stats: state.stats,
    isLoading: state.isLoading,
    setLoading: state.setLoading,
    selectedNodeId: state.selectedNodeId,
    selectNode: state.selectNode,
    setExperimentId: state.setExperimentId,
    error: state.error,
    setError: state.setError,
    lastUpdate: state.lastUpdate,
  }));

  const selectedNode = useMemo(() => {
    if (!selectedNodeId) {
      return null;
    }
    return nodes.get(selectedNodeId) ?? null;
  }, [nodes, selectedNodeId]);

  const childNodes = useMemo(() => {
    if (!selectedNodeId) {
      return [];
    }
    return edges
      .filter((edge) => edge.parent_id === selectedNodeId)
      .map((edge) => nodes.get(edge.child_id))
      .filter(Boolean) as ResearchNode[];
  }, [edges, nodes, selectedNodeId]);

  const parentNode = useMemo(() => {
    if (!selectedNodeId) {
      return null;
    }
    const parentEdge = edges.find((edge) => edge.child_id === selectedNodeId);
    if (!parentEdge) {
      return null;
    }
    return nodes.get(parentEdge.parent_id) ?? null;
  }, [edges, nodes, selectedNodeId]);

  const { isConnected, connect } = useResearchWS({
    experimentId,
    autoConnect: true,
    onError: () => {
      setError('Real-time updates temporarily unavailable');
    },
  });

  const fetchTree = useCallback(
    async (showSpinner = false) => {
      if (!experimentId) {
        return;
      }

      try {
        if (showSpinner) {
          setLoading(true);
        }

        setError(null);
        setFetchError(null);

        const response = await fetch(`/api/research/experiments/${experimentId}/tree`);

        if (!response.ok) {
          throw new Error(`Failed to fetch tree: ${response.statusText}`);
        }

        const snapshot = await response.json();
        useResearchTreeStore.getState().setSnapshot(snapshot);
      } catch (err) {
        console.error('[Research Tree] Failed to fetch snapshot', err);
        const message = err instanceof Error ? err.message : 'Failed to fetch tree';
        setFetchError(message);
        setError(message);
      } finally {
        if (showSpinner) {
          setLoading(false);
        }
      }
    },
    [experimentId, setError, setLoading]
  );

  const handleReset = useCallback(() => {
    useResearchTreeStore.getState().reset();
    connect();
    fetchTree(true);
  }, [connect, fetchTree]);

  useEffect(() => {
    setExperimentId(experimentId);
    fetchTree(true);
  }, [experimentId, fetchTree, setExperimentId]);

  const statsEntries = useMemo(
    () => [
      { label: 'Nodes', value: nodes.size },
      { label: 'Edges', value: edges.length },
      { label: 'Depth', value: stats.max_depth ?? 0 },
      { label: 'Cost', value: stats.total_cost ?? 0, formatter: (value: number) => `$${value.toFixed(3)}` },
      {
        label: 'Tokens',
        value: stats.total_tokens ?? 0,
        formatter: (value: number) => value.toLocaleString(),
      },
    ],
    [edges.length, nodes.size, stats.max_depth, stats.total_cost, stats.total_tokens]
  );

  const panelStyle = size ? {
    width: `${size.width}px`,
    height: `${size.height}px`,
  } : undefined;

  return (
    <div 
      className={cn('research-tree-panel', 'research-tree-panel-floating', isMinimized && 'minimized')}
      style={panelStyle}
    >
      <div 
        className="research-tree-header"
        onMouseDown={onDragHandleMouseDown}
        style={{ cursor: onDragHandleMouseDown ? 'move' : 'default' }}
      >
        <div>
          <h2>🔬 Research Tree</h2>
          <div
            className={cn(
              'connection-indicator',
              isConnected ? 'connected' : fetchError ? 'error' : 'disconnected'
            )}
          >
            <span className="indicator-dot" />
            {isConnected ? 'Connected' : fetchError ? 'Error' : 'Disconnected'}
            {isLoading && <Loader size="small" className="ml-2" />}
          </div>
        </div>

        <div className="header-controls">
          <button
            type="button"
            className="header-button"
            onClick={() => setIsMinimized((prev) => !prev)}
            title={isMinimized ? 'Expand panel' : 'Minimize panel'}
          >
            {isMinimized ? 'Expand' : 'Minimize'}
          </button>
          <button
            type="button"
            className="header-button close-button"
            onClick={onClose}
            title="Close"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {!isMinimized && (
        <>
          <div className="research-tree-stats">
            {statsEntries.map((entry) => (
              <div key={entry.label} className="stat-item" title={entry.label}>
                <span>{entry.label}</span>
                {isLoading ? (
                  <span className="stat-skeleton" />
                ) : (
                  <span className="stat-value">
                    {entry.formatter ? entry.formatter(entry.value) : entry.value}
                  </span>
                )}
              </div>
            ))}
          </div>

          {fetchError && (
            <div className="px-5 py-3 text-sm text-red-500">
              <div className="flex items-center justify-between rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2">
                <span>{fetchError}</span>
                <button
                  type="button"
                  className="clear-button"
                  onClick={() => fetchTree(true)}
                >
                  Retry
                </button>
              </div>
            </div>
          )}

          <div className="research-tree-content">
            <ResearchErrorBoundary onReset={handleReset}>
              <ReactFlowProvider>
                <ResearchTreeView />
              </ReactFlowProvider>
            </ResearchErrorBoundary>

            <AnimatePresence>
              {selectedNode && (
                <motion.aside
                  key={selectedNode.id}
                  className="research-detail-panel custom-scrollbar"
                  initial={{ x: '100%' }}
                  animate={{ x: 0 }}
                  exit={{ x: '100%' }}
                  transition={detailPanelTransition}
                >
                  <header>
                    <div>
                      <h3 className="text-base font-semibold text-slate-800 dark:text-slate-100">
                        {selectedNode.title}
                      </h3>
                      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                        {selectedNode.type} — {selectedNode.status}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="header-button"
                      onClick={() => selectNode(null)}
                      title="Close details"
                    >
                      <X size={16} />
                    </button>
                  </header>

                  <div className="detail-body custom-scrollbar">
                    <section className="research-detail-section">
                      <h5>Summary</h5>
                      <p className="text-sm text-slate-600 dark:text-slate-300">
                        {selectedNode.content || 'No summary available yet.'}
                      </p>
                    </section>

                    <section className="research-detail-section">
                      <h5>Metrics</h5>
                      <div className="research-detail-grid">
                        <div className="research-detail-metric">
                          <span className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300">
                            <Activity size={16} /> Visits
                          </span>
                          <strong>{selectedNode.visits}</strong>
                        </div>
                        <div className="research-detail-metric">
                          <span className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300">
                            <TrendingUp size={16} /> Q Value
                          </span>
                          <strong>{selectedNode.avg_value.toFixed(3)}</strong>
                        </div>
                        <div className="research-detail-metric">
                          <span className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300">
                            <Info size={16} /> Prior
                          </span>
                          <strong>{selectedNode.prior.toFixed(3)}</strong>
                        </div>
                        <div className="research-detail-metric">
                          <span className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300">
                            <DollarSign size={16} /> Cost
                          </span>
                          <strong>${selectedNode.cost.toFixed(3)}</strong>
                        </div>
                      </div>
                    </section>

                    <section className="research-detail-section">
                      <h5>Tokens</h5>
                      <p className="text-sm text-slate-600 dark:text-slate-300">
                        {selectedNode.tokens_used.toLocaleString()} tokens consumed.
                      </p>
                    </section>

                    <section className="research-detail-section">
                      <h5>Relationships</h5>
                      <div className="research-detail-links">
                        {parentNode && (
                          <button
                            type="button"
                            className="research-detail-link"
                            onClick={() => selectNode(parentNode.id)}
                          >
                            Parent: {parentNode.title}
                          </button>
                        )}
                        {childNodes.length > 0 ? (
                          childNodes.map((child) => (
                            <button
                              key={child.id}
                              type="button"
                              className="research-detail-link"
                              onClick={() => selectNode(child.id)}
                            >
                              Child: {child.title}
                            </button>
                          ))
                        ) : (
                          <p className="text-xs text-slate-500 dark:text-slate-400">
                            No child nodes yet.
                          </p>
                        )}
                      </div>
                    </section>
                  </div>
                </motion.aside>
              )}
            </AnimatePresence>

            {isLoading && nodes.size === 0 && (
              <div className="research-tree-loading">
                <Loader size="large" />
                <p>Initializing research tree...</p>
              </div>
            )}
          </div>

          {error && !fetchError && (
            <div className="px-5 pb-4 text-sm text-amber-400">
              <div className="rounded-md border border-amber-500/20 bg-amber-500/10 px-3 py-2">
                {error}
              </div>
            </div>
          )}

          {lastUpdate && (
            <div className="px-5 pb-4 text-right text-xs text-slate-500 dark:text-slate-400">
              Last update {new Date(lastUpdate).toLocaleTimeString()}
            </div>
          )}
        </>
      )}
    </div>
  );
}
