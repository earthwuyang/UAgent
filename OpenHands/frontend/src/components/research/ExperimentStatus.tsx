import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, CheckCircle, XCircle, Ban } from 'lucide-react';
import { useResearchTreeStore } from '#/state/research-tree-store';
import type { ExperimentStatus as ExperimentState } from '#/api/research-api';

interface ExperimentStatusProps {
  experimentId: string;
  status: ExperimentState;
  showCost?: boolean;
  showDuration?: boolean;
  showTokens?: boolean;
}

const ExperimentStatusComponent: React.FC<ExperimentStatusProps> = ({
  experimentId,
  status,
  showCost = true,
  showDuration = true,
  showTokens = true,
}) => {
  const { stats } = useResearchTreeStore();
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);

  // Track experiment start time
  useEffect(() => {
    if (status === 'running' && !startTime) {
      setStartTime(Date.now());
    } else if (status === 'idle' || status === 'complete' || status === 'failed' || status === 'cancelled') {
      setStartTime(null);
      setElapsedTime(0);
    }
  }, [status, startTime]);

  // Update elapsed time every second
  useEffect(() => {
    if (!startTime || status === 'paused') return;

    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime, status]);

  // Format duration
  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Get status config
  const getStatusConfig = (status: ExperimentState) => {
    switch (status) {
      case 'idle':
        return { color: 'bg-gray-500', text: 'Idle', icon: Play };
      case 'running':
        return { color: 'bg-blue-500', text: 'Running', icon: Play };
      case 'paused':
        return { color: 'bg-yellow-500', text: 'Paused', icon: Pause };
      case 'complete':
        return { color: 'bg-green-500', text: 'Complete', icon: CheckCircle };
      case 'failed':
        return { color: 'bg-red-500', text: 'Failed', icon: XCircle };
      case 'cancelled':
        return { color: 'bg-orange-500', text: 'Cancelled', icon: Ban };
      default:
        return { color: 'bg-gray-500', text: 'Unknown', icon: Play };
    }
  };

  const statusConfig = getStatusConfig(status);
  const StatusIcon = statusConfig.icon;

  // Format cost
  const formatCost = (cost: number | undefined): string => {
    if (cost === undefined || cost === null) return '--';
    return `$${cost.toFixed(3)}`;
  };

  // Format tokens
  const formatTokens = (tokens: number | undefined): string => {
    if (tokens === undefined || tokens === null) return '--';
    return `${tokens.toLocaleString()} tokens`;
  };

  return (
    <div className="flex items-center gap-4 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
      {/* Status Badge */}
      <motion.div
        className={`flex items-center gap-2 px-3 py-1 rounded-full text-white text-sm font-medium ${statusConfig.color}`}
        animate={status === 'running' ? { scale: [1, 1.05, 1] } : {}}
        transition={{ duration: 1, repeat: status === 'running' ? Infinity : 0 }}
      >
        <StatusIcon size={16} />
        <span>{statusConfig.text}</span>
      </motion.div>

      {/* Divider */}
      {(showDuration || showCost || showTokens) && (
        <div className="h-6 w-px bg-slate-300 dark:bg-slate-600" />
      )}

      {/* Duration */}
      {showDuration && (
        <div className="flex flex-col items-center">
          <span className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">Duration</span>
          <motion.span
            className="text-sm font-mono font-semibold text-slate-700 dark:text-slate-300"
            key={elapsedTime}
            initial={{ opacity: 0.5 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            {formatDuration(elapsedTime)}
          </motion.span>
        </div>
      )}

      {/* Divider */}
      {showDuration && showCost && (
        <div className="h-6 w-px bg-slate-300 dark:bg-slate-600" />
      )}

      {/* Cost */}
      {showCost && (
        <div className="flex flex-col items-center">
          <span className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">Cost</span>
          <span
            className="text-sm font-semibold text-slate-700 dark:text-slate-300"
            title={stats?.total_cost !== undefined && stats.total_cost !== null ? `Total cost: $${stats.total_cost.toFixed(3)}` : undefined}
          >
            {formatCost(stats?.total_cost)}
          </span>
        </div>
      )}

      {/* Divider */}
      {showCost && showTokens && (
        <div className="h-6 w-px bg-slate-300 dark:bg-slate-600" />
      )}

      {/* Tokens */}
      {showTokens && (
        <div className="flex flex-col items-center">
          <span className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">Tokens</span>
          <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
            {formatTokens(stats?.total_tokens)}
          </span>
        </div>
      )}
    </div>
  );
};

export default ExperimentStatusComponent;
