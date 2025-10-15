/**
 * Node Context Indicator Component
 *
 * Displays a banner at the top of the screen when viewing a specific node's context.
 * Shows node information and provides a button to return to the main conversation.
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ArrowLeft, Circle } from 'lucide-react';
import { useNodeEventStore } from '#/state/node-event-store';
import { useResearchTreeStore } from '#/state/research-tree-store';
import { cn } from '#/utils/utils';

export function NodeContextIndicator() {
  const isNodeContext = useNodeEventStore((state) => state.contextMode === 'node');
  const activeNodeId = useNodeEventStore((state) => state.activeNodeId);
  const switchToRootContext = useNodeEventStore((state) => state.switchToRootContext);
  const nodes = useResearchTreeStore((state) => state.nodes);

  // Get node details
  const node = activeNodeId ? nodes.get(activeNodeId) : null;

  if (!isNodeContext || !node) {
    return null;
  }

  // Map node status to color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'text-blue-500';
      case 'complete':
        return 'text-green-500';
      case 'failed':
        return 'text-red-500';
      case 'cancelled':
        return 'text-gray-500';
      default:
        return 'text-yellow-500';
    }
  };

  // Map node type to label
  const getTypeLabel = (type: string) => {
    return type
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const handleReturn = () => {
    switchToRootContext();
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
        className={cn(
          'fixed top-0 left-0 right-0 z-40',
          'bg-gradient-to-r from-blue-600 to-blue-500',
          'shadow-lg backdrop-blur-sm',
          'border-b border-blue-400/30'
        )}
      >
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between gap-4">
            {/* Left: Node Info */}
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <Circle
                  size={8}
                  className={cn('fill-current', getStatusColor(node.status))}
                />
                <span className="text-blue-100 text-sm font-medium">
                  {getTypeLabel(node.type)}
                </span>
              </div>
              
              <div className="h-4 w-px bg-blue-400/30" />
              
              <div className="flex-1 min-w-0">
                <h3 className="text-white font-semibold text-sm truncate">
                  {node.title}
                </h3>
                {node.content && (
                  <p className="text-blue-100 text-xs truncate">
                    {node.content}
                  </p>
                )}
              </div>
            </div>

            {/* Right: Actions */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleReturn}
                className={cn(
                  'flex items-center gap-2 px-3 py-1.5',
                  'bg-white/10 hover:bg-white/20',
                  'text-white text-sm font-medium',
                  'rounded-md transition-colors',
                  'border border-white/20'
                )}
              >
                <ArrowLeft size={14} />
                <span>Return to Main Conversation</span>
              </button>
              
              <button
                type="button"
                onClick={handleReturn}
                className={cn(
                  'p-1.5 rounded-md',
                  'hover:bg-white/10 transition-colors',
                  'text-white'
                )}
                aria-label="Close node context"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
