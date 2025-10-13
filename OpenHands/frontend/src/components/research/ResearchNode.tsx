/**
 * Research Node Component
 *
 * Enhanced custom node for ReactFlow with expandable details, animations, and rich metrics.
 */

import React, { memo, useCallback, useMemo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  ChevronDown,
  ChevronUp,
  Clock,
  Info,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { cn } from '#/utils/utils';
import {
  ResearchNode as ResearchNodeType,
  useResearchTreeStore,
} from '#/state/research-tree-store';

const STATUS_COLORS: Record<string, string> = {
  pending: '#9ca3af',
  running: '#3b82f6',
  complete: '#10b981',
  failed: '#ef4444',
  cancelled: '#f59e0b',
};

const TYPE_LABELS: Record<string, string> = {
  root: 'Root',
  idea: 'Idea',
  hypothesis: 'Hypothesis',
  plan: 'Plan',
  web_search: 'Web Search',
  code_search: 'Code Search',
  experiment: 'Experiment',
  result: 'Result',
};

const HIGH_VALUE_THRESHOLD = 0.7;

const formatDuration = (start?: string, end?: string) => {
  if (!start || !end) {
    return null;
  }

  const startDate = new Date(start);
  const endDate = new Date(end);
  const diffMs = endDate.getTime() - startDate.getTime();
  if (Number.isNaN(diffMs) || diffMs < 0) {
    return null;
  }

  const seconds = Math.floor(diffMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  if (minutes === 0) {
    return `${remainingSeconds}s`;
  }

  return `${minutes}m ${remainingSeconds}s`;
};

const dateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
});

const statusAnimation = (status: string) => {
  switch (status) {
    case 'running':
      return {
        animate: { scale: [1, 1.05, 1], opacity: [1, 0.8, 1] },
        transition: { duration: 1.2, repeat: Infinity, ease: 'easeInOut' },
      };
    case 'complete':
      return {
        animate: { scale: [1, 1.1, 1] },
        transition: { duration: 0.6, ease: 'easeOut' },
      };
    case 'failed':
      return {
        animate: { x: [0, -3, 3, 0] },
        transition: { duration: 0.6, ease: 'easeInOut' },
      };
    default:
      return {
        animate: { scale: 1, opacity: 1, x: 0 },
        transition: { duration: 0.3 },
      };
  }
};

const springTransition = { type: 'spring', damping: 24, stiffness: 220 };

export const ResearchNode = memo(({ data, selected }: NodeProps<ResearchNodeType>) => {
  // Use separate store selectors to avoid creating new objects on every render
  const isExpanded = useResearchTreeStore((state) => state.expandedNodeIds.has(data.id));
  const toggleExpanded = useResearchTreeStore((state) => state.toggleExpanded);
  const selectNode = useResearchTreeStore((state) => state.selectNode);

  const statusColor = STATUS_COLORS[data.status] ?? STATUS_COLORS.pending;
  const typeLabel = TYPE_LABELS[data.type] ?? data.type;
  const duration = useMemo(
    () => formatDuration(data.created_at, data.completed_at),
    [data.created_at, data.completed_at]
  );
  const isHighValue = data.avg_value >= HIGH_VALUE_THRESHOLD;

  const handleToggle = useCallback(() => {
    toggleExpanded(data.id);
  }, [data.id, toggleExpanded]);

  const handleKeyboardToggle = useCallback(
    (event: React.KeyboardEvent<HTMLButtonElement>) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        event.stopPropagation();
        handleToggle();
      }
    },
    [handleToggle]
  );

  const createdAtLabel = data.created_at
    ? dateFormatter.format(new Date(data.created_at))
    : null;
  const completedAtLabel = data.completed_at
    ? dateFormatter.format(new Date(data.completed_at))
    : null;

  const onSelect = useCallback(() => {
    selectNode(data.id);
  }, [data.id, selectNode]);

  return (
    <motion.div
      layout
      data-node-id={data.id}
      className={cn(
        'research-node',
        selected && 'selected',
        isExpanded && 'expanded',
        data.isFilteredOut && 'filtered-out',
        data.isFilteredMatch && 'filtered-match'
      )}
      style={{ borderColor: statusColor }}
      initial={false}
      whileHover={{ y: -2, boxShadow: '0 12px 32px rgba(15, 23, 42, 0.18)' }}
      transition={springTransition}
      onClick={onSelect}
    >
      <Handle type="target" position={Position.Top} className="research-node-handle" />

      <div className="research-node-header">
        <div className="research-node-type" title={data.type}>
          {isHighValue && <Zap size={14} className="high-value-icon" aria-hidden />}
          <span>{typeLabel}</span>
        </div>
        <motion.span
          key={data.status}
          className="research-node-status"
          style={{ backgroundColor: statusColor }}
          {...statusAnimation(data.status)}
        >
          {data.status}
        </motion.span>
        <button
          type="button"
          className="research-node-toggle"
          onClick={(event) => {
            event.stopPropagation();
            handleToggle();
          }}
          onKeyDown={handleKeyboardToggle}
          aria-expanded={isExpanded}
          aria-controls={`research-node-${data.id}-details`}
        >
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      <div className="research-node-title" title={data.title}>
        {data.title}
      </div>

      <div className="research-node-content" title={data.content}>
        {isExpanded ? data.content : `${data.content?.slice(0, 120) ?? ''}${
          data.content && data.content.length > 120 ? '…' : ''
        }`}
      </div>

      <div className="research-node-stats">
        <div className="research-node-stat" title="Visits (PUCT N)">
          <span className="stat-label">N</span>
          <span className="stat-value">{data.visits}</span>
        </div>
        <div className="research-node-stat" title="Q value (average)">
          <span className="stat-label">Q</span>
          <span className="stat-value">{data.avg_value.toFixed(2)}</span>
        </div>
        <div className="research-node-stat" title="Prior (PUCT P)">
          <span className="stat-label">P</span>
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

      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            id={`research-node-${data.id}-details`}
            className="research-node-details"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={springTransition}
          >
            <div className="research-node-timestamps">
              {createdAtLabel && (
                <div className="timestamp-item" title={`Created at ${createdAtLabel}`}>
                  <Clock size={14} aria-hidden />
                  <span>{createdAtLabel}</span>
                </div>
              )}
              {completedAtLabel && (
                <div className="timestamp-item" title={`Completed at ${completedAtLabel}`}>
                  <Clock size={14} aria-hidden />
                  <span>{completedAtLabel}</span>
                </div>
              )}
              {duration && (
                <div className="timestamp-item" title="Execution duration">
                  <Activity size={14} aria-hidden />
                  <span>{duration}</span>
                </div>
              )}
            </div>

            <div className="research-node-metrics">
              <div className="metric-item" title="PUCT exploration bonus explanation">
                <Info size={14} aria-hidden />
                <div>
                  <span className="metric-label">Exploration</span>
                  <p className="metric-description">
                    Prior probability guiding exploration to promising branches.
                  </p>
                </div>
                <span className="metric-value">{data.prior.toFixed(3)}</span>
              </div>
              <div className="metric-item" title="Value estimate">
                <TrendingValue value={data.avg_value} />
              </div>
            </div>

            {data.status === 'running' && (
              <div className="research-node-progress" aria-label="Node in progress">
                <motion.span
                  className="progress-indicator"
                  animate={{ width: ['10%', '90%'] }}
                  transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
                />
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <Handle type="source" position={Position.Bottom} className="research-node-handle" />
    </motion.div>
  );
});

ResearchNode.displayName = 'ResearchNode';

interface TrendingValueProps {
  value: number;
}

function TrendingValue({ value }: TrendingValueProps) {
  const status = value >= HIGH_VALUE_THRESHOLD ? 'strong' : value >= 0.4 ? 'steady' : 'weak';
  const label = status === 'strong' ? 'High confidence' : status === 'steady' ? 'Moderate confidence' : 'Low confidence';

  return (
    <div className="metric-item-inline" title={`Average value indicates ${label.toLowerCase()}`}>
      <TrendingUp size={14} aria-hidden />
      <div>
        <span className="metric-label">Confidence</span>
        <p className="metric-description">Balanced by visit count for exploitation.</p>
      </div>
      <span className="metric-value">{value.toFixed(2)}</span>
    </div>
  );
}
