/**
 * Node Progress Header Component
 *
 * Displays a static header at the top of the node progress page.
 * Shows node information, status, and key metrics.
 */

import React from "react";
import {
  ArrowLeft,
  Circle,
  Activity,
  TrendingUp,
  DollarSign,
} from "lucide-react";
import { useNavigate } from "react-router";
import { cn } from "#/utils/utils";
import type { ResearchNode } from "#/state/research-tree-store";

interface NodeProgressHeaderProps {
  node: ResearchNode;
  conversationId: string;
  experimentId: string;
}

export function NodeProgressHeader({
  node,
  conversationId,
  experimentId,
}: NodeProgressHeaderProps) {
  const navigate = useNavigate();

  const handleBack = () => {
    navigate(`/conversations/${conversationId}`);
  };

  // Map node status to color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "running":
        return "text-blue-500 bg-blue-500/10 border-blue-500/30";
      case "complete":
        return "text-green-500 bg-green-500/10 border-green-500/30";
      case "failed":
        return "text-red-500 bg-red-500/10 border-red-500/30";
      case "cancelled":
        return "text-gray-500 bg-gray-500/10 border-gray-500/30";
      default:
        return "text-yellow-500 bg-yellow-500/10 border-yellow-500/30";
    }
  };

  const getStatusDotColor = (status: string) => {
    switch (status) {
      case "running":
        return "fill-blue-500 text-blue-500";
      case "complete":
        return "fill-green-500 text-green-500";
      case "failed":
        return "fill-red-500 text-red-500";
      case "cancelled":
        return "fill-gray-500 text-gray-500";
      default:
        return "fill-yellow-500 text-yellow-500";
    }
  };

  // Map node type to label
  const getTypeLabel = (type: string) => {
    return type
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  return (
    <header
      className={cn(
        "sticky top-0 z-30",
        "bg-white dark:bg-slate-900",
        "border-b border-slate-200 dark:border-slate-800",
        "shadow-sm",
      )}
    >
      {/* Main Header Bar */}
      <div className="max-w-7xl mx-auto px-4 py-3">
        <div className="flex items-center justify-between gap-4">
          {/* Left: Back Button + Node Info */}
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <button
              type="button"
              onClick={handleBack}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5",
                "bg-slate-100 hover:bg-slate-200",
                "dark:bg-slate-800 dark:hover:bg-slate-700",
                "text-slate-700 dark:text-slate-300",
                "text-sm font-medium rounded-md transition-colors",
                "border border-slate-200 dark:border-slate-700",
              )}
            >
              <ArrowLeft size={14} />
              <span>Back</span>
            </button>

            <div className="h-6 w-px bg-slate-200 dark:bg-slate-700" />

            <div className="flex items-center gap-2">
              <Circle size={8} className={cn(getStatusDotColor(node.status))} />
              <span
                className={cn(
                  "px-2 py-1 text-xs font-medium rounded border",
                  getStatusColor(node.status),
                )}
              >
                {node.status}
              </span>
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {getTypeLabel(node.type)}
                </span>
              </div>
              <h1 className="text-base font-semibold text-slate-900 dark:text-slate-100 truncate">
                {node.title}
              </h1>
            </div>
          </div>

          {/* Right: Key Metrics */}
          <div className="hidden md:flex items-center gap-4">
            <div className="flex items-center gap-1.5 text-sm">
              <Activity
                size={14}
                className="text-slate-500 dark:text-slate-400"
              />
              <span className="text-slate-600 dark:text-slate-300 font-medium">
                {node.visits ?? 0}
              </span>
              <span className="text-slate-400 dark:text-slate-500 text-xs">
                visits
              </span>
            </div>

            <div className="flex items-center gap-1.5 text-sm">
              <TrendingUp
                size={14}
                className="text-slate-500 dark:text-slate-400"
              />
              <span className="text-slate-600 dark:text-slate-300 font-medium">
                {(node.avg_value ?? 0).toFixed(3)}
              </span>
              <span className="text-slate-400 dark:text-slate-500 text-xs">
                Q-value
              </span>
            </div>

            <div className="flex items-center gap-1.5 text-sm">
              <DollarSign
                size={14}
                className="text-slate-500 dark:text-slate-400"
              />
              <span className="text-slate-600 dark:text-slate-300 font-medium">
                {(node.cost ?? 0).toFixed(3)}
              </span>
              <span className="text-slate-400 dark:text-slate-500 text-xs">
                cost
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Secondary Info Bar */}
      <div className="bg-slate-50 dark:bg-slate-800/50 border-t border-slate-200 dark:border-slate-700">
        <div className="max-w-7xl mx-auto px-4 py-2">
          <div className="flex items-center justify-between gap-4 text-xs">
            <div className="flex items-center gap-4">
              <div>
                <span className="text-slate-500 dark:text-slate-400">
                  Node ID:
                </span>{" "}
                <code className="text-slate-700 dark:text-slate-300 font-mono">
                  {node.id.slice(0, 8)}...
                </code>
              </div>
              <div>
                <span className="text-slate-500 dark:text-slate-400">
                  Experiment:
                </span>{" "}
                <code className="text-slate-700 dark:text-slate-300 font-mono">
                  {experimentId.slice(0, 8)}...
                </code>
              </div>
              {node.created_at && (
                <div>
                  <span className="text-slate-500 dark:text-slate-400">
                    Created:
                  </span>{" "}
                  <span className="text-slate-700 dark:text-slate-300">
                    {new Date(node.created_at).toLocaleString()}
                  </span>
                </div>
              )}
            </div>

            {/* Mobile metrics (shown when desktop metrics are hidden) */}
            <div className="flex md:hidden items-center gap-3">
              <div className="flex items-center gap-1">
                <Activity size={12} className="text-slate-400" />
                <span className="text-slate-600 dark:text-slate-300">
                  {node.visits ?? 0}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <TrendingUp size={12} className="text-slate-400" />
                <span className="text-slate-600 dark:text-slate-300">
                  {(node.avg_value ?? 0).toFixed(2)}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <DollarSign size={12} className="text-slate-400" />
                <span className="text-slate-600 dark:text-slate-300">
                  {(node.cost ?? 0).toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
