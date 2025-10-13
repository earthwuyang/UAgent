/**
 * Research Tree Panel Component
 *
 * Floating panel that displays the research tree visualization with detail view.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { X } from 'lucide-react';
import { ResearchTreeView } from './ResearchTreeView';
import { useResearchWS } from '#/hooks/useResearchWS';
import { Loader } from '#/components/shared/loader';
import { useResearchTreeStore } from '#/state/research-tree-store';
import { useResearchEventStream } from '#/hooks/useResearchEventStream';
import { ResearchErrorBoundary } from './ResearchErrorBoundary';
import { ResearchNodeDetailPanel } from './ResearchNodeDetailPanel';
import { cn } from '#/utils/utils';
import './research-tree.css';

interface ResearchTreePanelProps {
  experimentId: string;
  onClose: () => void;
}

export function ResearchTreePanel({ experimentId, onClose }: ResearchTreePanelProps) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Use separate store selectors to avoid creating new objects on every render
  const nodes = useResearchTreeStore((state) => state.nodes);
  const edges = useResearchTreeStore((state) => state.edges);
  const stats = useResearchTreeStore((state) => state.stats);
  const isLoading = useResearchTreeStore((state) => state.isLoading);
  const setLoading = useResearchTreeStore((state) => state.setLoading);
  const setExperimentId = useResearchTreeStore((state) => state.setExperimentId);
  const error = useResearchTreeStore((state) => state.error);
  const setError = useResearchTreeStore((state) => state.setError);
  const lastUpdate = useResearchTreeStore((state) => state.lastUpdate);

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

  useResearchEventStream(experimentId);

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

  return (
    <div className={cn('research-tree-panel', isMinimized && 'minimized')}>
      <div className="research-tree-header">
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
            <div
              style={{
                flex: 1,
                display: 'flex',
                position: 'relative',
                minWidth: 0,
                minHeight: 0,
                height: '100%',
              }}
            >
              <ResearchErrorBoundary onReset={handleReset}>
                <ReactFlowProvider>
                  <ResearchTreeView />
                </ReactFlowProvider>
              </ResearchErrorBoundary>
            </div>

            <ResearchNodeDetailPanel />
          </div>

          {isLoading && nodes.size === 0 && (
            <div className="research-tree-loading">
              <Loader size="large" />
              <p>Initializing research tree...</p>
            </div>
          )}

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
