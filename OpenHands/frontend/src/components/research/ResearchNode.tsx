/**
 * Research Node Component
 *
 * Custom node for ReactFlow representing a research tree node.
 * Shows node status, type, title, and PUCT metrics.
 */

import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { ResearchNode as ResearchNodeType } from '#/state/research-tree-store';

const STATUS_COLORS = {
  pending: '#9ca3af', // gray
  running: '#3b82f6', // blue
  complete: '#10b981', // green
  failed: '#ef4444', // red
  cancelled: '#f59e0b', // amber
};

const TYPE_LABELS = {
  root: '🌱 Root',
  idea: '💡 Idea',
  hypothesis: '🔬 Hypothesis',
  plan: '📋 Plan',
  web_search: '🔍 Web Search',
  code_search: '💻 Code Search',
  experiment: '⚗️ Experiment',
  result: '📊 Result',
};

export const ResearchNode = memo(({ data, selected }: NodeProps<ResearchNodeType>) => {
  const statusColor = STATUS_COLORS[data.status as keyof typeof STATUS_COLORS] || STATUS_COLORS.pending;
  const typeLabel = TYPE_LABELS[data.type as keyof typeof TYPE_LABELS] || data.type;

  return (
    <div
      className={`research-node ${selected ? 'selected' : ''}`}
      style={{
        borderColor: statusColor,
        borderWidth: selected ? '3px' : '2px',
      }}
    >
      <Handle type="target" position={Position.Top} className="research-node-handle" />

      <div className="research-node-header">
        <div className="research-node-type" title={data.type}>
          {typeLabel}
        </div>
        <div
          className="research-node-status"
          style={{ backgroundColor: statusColor }}
          title={data.status}
        >
          {data.status}
        </div>
      </div>

      <div className="research-node-title" title={data.title}>
        {data.title}
      </div>

      {data.content && (
        <div className="research-node-content" title={data.content}>
          {data.content.substring(0, 80)}
          {data.content.length > 80 ? '...' : ''}
        </div>
      )}

      <div className="research-node-stats">
        <div className="research-node-stat" title="Visits (PUCT N)">
          <span className="stat-label">N:</span>
          <span className="stat-value">{data.visits}</span>
        </div>
        <div className="research-node-stat" title="Q value (avg value)">
          <span className="stat-label">Q:</span>
          <span className="stat-value">{data.avg_value.toFixed(2)}</span>
        </div>
        <div className="research-node-stat" title="Prior (PUCT P)">
          <span className="stat-label">P:</span>
          <span className="stat-value">{data.prior.toFixed(2)}</span>
        </div>
      </div>

      <div className="research-node-footer">
        <div className="research-node-cost" title="Execution cost">
          ${data.cost.toFixed(3)}
        </div>
        {data.tokens_used > 0 && (
          <div className="research-node-tokens" title="Tokens used">
            {data.tokens_used.toLocaleString()} tok
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} className="research-node-handle" />
    </div>
  );
});

ResearchNode.displayName = 'ResearchNode';
