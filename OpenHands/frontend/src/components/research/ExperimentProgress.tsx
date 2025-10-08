import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Globe, Code, Github } from 'lucide-react';
import { useResearchTreeStore, ResearchNode } from '#/state/research-tree-store';

interface ExperimentProgressProps {
  experimentId: string;
  compact?: boolean;
}

const ExperimentProgress: React.FC<ExperimentProgressProps> = ({ experimentId, compact = false }) => {
  const { nodes } = useResearchTreeStore();

  // Calculate progress metrics in a single pass
  const { totalNodes, completedNodes, failedNodes, runningSteps } = Array.from(nodes.values()).reduce(
    (acc, node) => {
      acc.totalNodes += 1;
      if (node.status === 'complete') {
        acc.completedNodes += 1;
      } else if (node.status === 'failed') {
        acc.failedNodes += 1;
      } else if (node.status === 'running') {
        acc.runningSteps.push(node as ResearchNode);
      }
      return acc;
    },
    { totalNodes: 0, completedNodes: 0, failedNodes: 0, runningSteps: [] as ResearchNode[] }
  );

  const progressPercentage = totalNodes > 0 ? ((completedNodes + failedNodes) / totalNodes) * 100 : 0;

  // Helper to get adapter info based on node type
  const getAdapterInfo = (node: ResearchNode) => {
    const type = node.type.toLowerCase();
    if (type.includes('research') || type.includes('deep')) {
      return { name: 'DeepResearch', icon: Globe };
    }
    if (type.includes('code') || type.includes('act')) {
      return { name: 'CodeAct', icon: Code };
    }
    if (type.includes('repo') || type.includes('master')) {
      return { name: 'RepoMaster', icon: Github };
    }
    return { name: node.type, icon: Globe }; // default
  };

  // Helper to calculate elapsed time
  const getElapsedTime = (node: ResearchNode) => {
    const start = new Date(node.created_at || Date.now());
    const now = new Date();
    const diff = now.getTime() - start.getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  // Progress bar segments
  const completedWidth = totalNodes > 0 ? (completedNodes / totalNodes) * 100 : 0;
  const failedWidth = totalNodes > 0 ? (failedNodes / totalNodes) * 100 : 0;
  const pendingWidth = 100 - completedWidth - failedWidth;

  const progressBar = (
    <div className="relative w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden">
      {/* Completed segment */}
      <motion.div
        className="absolute left-0 top-0 h-full bg-green-500"
        initial={{ width: 0 }}
        animate={{ width: `${completedWidth}%` }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      />
      {/* Failed segment */}
      <motion.div
        className="absolute top-0 h-full bg-red-500"
        initial={{ width: 0, left: `${completedWidth}%` }}
        animate={{ width: `${failedWidth}%`, left: `${completedWidth}%` }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      />
      {/* Pending segment */}
      <motion.div
        className="absolute top-0 h-full bg-gray-300 dark:bg-gray-600"
        initial={{ width: 0, left: `${completedWidth + failedWidth}%` }}
        animate={{ width: `${pendingWidth}%`, left: `${completedWidth + failedWidth}%` }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      />
      {/* Percentage overlay */}
      <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-gray-900 dark:text-gray-100">
        {progressPercentage.toFixed(1)}%
      </div>
    </div>
  );

  if (compact) {
    return (
      <div className="flex items-center space-x-2">
        <span className="text-sm font-medium">Progress:</span>
        <div className="flex-1">{progressBar}</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium">Experiment Progress</span>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {completedNodes + failedNodes}/{totalNodes} nodes
          </span>
        </div>
        {progressBar}
      </div>

      <div>
        <h4 className="text-sm font-medium mb-2">Current Steps</h4>
        <AnimatePresence>
          {runningSteps.length === 0 ? (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-sm text-gray-500 dark:text-gray-400"
            >
              Idle
            </motion.div>
          ) : (
            runningSteps.map(node => {
              const adapter = getAdapterInfo(node);
              const Icon = adapter.icon;
              return (
                <motion.div
                  key={node.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                  className="flex items-center space-x-2 p-2 bg-gray-50 dark:bg-gray-800 rounded-md"
                >
                  <Icon className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm font-medium">{adapter.name}</span>
                  <span className="text-sm text-gray-600 dark:text-gray-300 flex-1 truncate">
                    {node.title || 'Processing...'}
                  </span>
                  <span className="text-xs text-gray-400 dark:text-gray-500">
                    {getElapsedTime(node)}
                  </span>
                </motion.div>
              );
            })
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default ExperimentProgress;
