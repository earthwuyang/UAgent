/**
 * Research Tree Panel Component
 *
 * Floating panel that displays the research tree visualization.
 * Includes header with stats, connection status, and controls.
 */

import React, { useEffect, useState } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { ResearchTreeView } from './ResearchTreeView';
import { useResearchWS } from '#/hooks/useResearchWS';
import { useResearchTreeStore } from '#/state/research-tree-store';
import { X, Minimize2, Maximize2 } from 'lucide-react';

interface ResearchTreePanelProps {
  experimentId: string;
  onClose: () => void;
}

export function ResearchTreePanel({ experimentId, onClose }: ResearchTreePanelProps) {
  const [isMinimized, setIsMinimized] = useState(false);
  const { isConnected } = useResearchWS({ experimentId });
  const { stats, nodes, edges, setLoading } = useResearchTreeStore();

  // Fetch initial tree snapshot
  useEffect(() => {
    const fetchTree = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/research/experiments/${experimentId}/tree`);

        if (!response.ok) {
          throw new Error(`Failed to fetch tree: ${response.statusText}`);
        }

        const snapshot = await response.json();
        useResearchTreeStore.getState().setSnapshot(snapshot);
      } catch (error) {
        console.error('[Research Tree] Failed to fetch initial snapshot:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTree();
  }, [experimentId, setLoading]);

  const connectionIndicator = isConnected ? (
    <span className="connection-indicator connected">
      <span className="indicator-dot" />
      Connected
    </span>
  ) : (
    <span className="connection-indicator disconnected">
      <span className="indicator-dot" />
      Disconnected
    </span>
  );

  return (
    <div className={`research-tree-panel ${isMinimized ? 'minimized' : ''}`}>
      <div className="research-tree-header">
        <div className="header-title">
          <h2>🔬 Research Tree</h2>
          {connectionIndicator}
        </div>

        <div className="header-controls">
          <button
            className="header-button"
            onClick={() => setIsMinimized(!isMinimized)}
            title={isMinimized ? 'Maximize' : 'Minimize'}
          >
            {isMinimized ? <Maximize2 size={16} /> : <Minimize2 size={16} />}
          </button>
          <button
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
            <div className="stat-item">
              <span className="stat-label">Nodes:</span>
              <span className="stat-value">{nodes.size}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Edges:</span>
              <span className="stat-value">{edges.length}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Depth:</span>
              <span className="stat-value">{stats.max_depth || 0}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Cost:</span>
              <span className="stat-value">${(stats.total_cost || 0).toFixed(3)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Tokens:</span>
              <span className="stat-value">
                {(stats.total_tokens || 0).toLocaleString()}
              </span>
            </div>
          </div>

          <div className="research-tree-content">
            <ReactFlowProvider>
              <ResearchTreeView />
            </ReactFlowProvider>
          </div>
        </>
      )}
    </div>
  );
}
