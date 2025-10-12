import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ReactFlowProvider } from "reactflow";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import {
  ResearchTreeView,
  ResearchErrorBoundary,
} from "#/components/research";
import { Loader } from "#/components/shared/loader";
import {
  getExperimentStatus,
  ExperimentStatus as ExperimentState,
  ResearchAPIError,
} from "#/api/research-api";
import { useResearchWS } from "#/hooks/useResearchWS";
import { useResearchTreeStore } from "#/state/research-tree-store";

const TREE_POLL_INTERVAL = 5000;
const STATUS_POLL_INTERVAL = 5000;

export default function ResearchTab() {
  const { conversationId } = useConversationId();
  const { data: activeConversation } = useActiveConversation();

  const resolvedExperimentId =
    activeConversation?.research_experiment_id ?? conversationId ?? null;
  const researchGoal = activeConversation?.title ?? undefined;

  const [isClient, setIsClient] = useState(false);
  const [experimentStatus, setExperimentStatus] = useState<ExperimentState>("idle");
  const [statusError, setStatusError] = useState<string | null>(null);
  const [treeError, setTreeError] = useState<string | null>(null);
  // Render guard: only mount ReactFlow tree when backend returns a valid snapshot
  const [canRenderTree, setCanRenderTree] = useState(false);

  const statusPollRef = useRef<NodeJS.Timeout | null>(null);
  const lastTreeHashRef = useRef<string>('');
  // Access store state for display (nodes, edges, stats)
  const nodes = useResearchTreeStore((state) => state.nodes);
  const edges = useResearchTreeStore((state) => state.edges);
  const stats = useResearchTreeStore((state) => state.stats);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const experimentId = isClient ? resolvedExperimentId : null;

  const clearStatusPolling = useCallback(() => {
    if (statusPollRef.current) {
      clearInterval(statusPollRef.current);
      statusPollRef.current = null;
    }
  }, []);

  // Fetch tree snapshot - use store methods directly to avoid dependency issues
  const fetchTreeSnapshot = useCallback(
    async (showSpinner = false) => {
      if (!experimentId) return;

      const emptySnapshot = {
        version: 0,
        timestamp: new Date().toISOString(),
        experiment_id: experimentId,
        data: {
          nodes: [],
          edges: [],
          stats: {
            total_nodes: 0,
            total_edges: 0,
            total_cost: 0,
            total_tokens: 0,
            completed_nodes: 0,
            failed_nodes: 0,
          },
        },
      };

      try {
        if (showSpinner) {
          // Access setLoading directly from store to avoid dependency issues
          useResearchTreeStore.getState().setLoading(true);
        }

        const backendHost = import.meta.env.VITE_BACKEND_BASE_URL || window.location.host;
        const response = await fetch(
          `${window.location.protocol}//${backendHost}/api/research/experiments/${experimentId}/tree`,
        );

        if (response.status === 404) {
          useResearchTreeStore.getState().setSnapshot(emptySnapshot);
          setTreeError(null);
          setCanRenderTree(false);
          return;
        }

        if (!response.ok) {
          throw new Error(`Failed to fetch tree: ${response.statusText}`);
        }

        const snapshot = await response.json();
        // Use JSON.stringify as a simple hash to avoid writing identical snapshots
        const hash = JSON.stringify(snapshot?.data ?? snapshot);
        if (hash !== lastTreeHashRef.current) {
          lastTreeHashRef.current = hash;
          useResearchTreeStore.getState().setSnapshot(snapshot);
          setTreeError(null);
          // Allow rendering only when there is at least one node
          setCanRenderTree(Boolean(snapshot?.data?.nodes?.length));
        }
      } catch (error) {
        console.error("Failed to fetch research tree:", error);
        setTreeError(
          error instanceof Error
            ? error.message
            : "Failed to fetch research tree",
        );
        setCanRenderTree(false);
      } finally {
        if (showSpinner) {
          useResearchTreeStore.getState().setLoading(false);
        }
      }
    },
    [experimentId], // Only depend on experimentId
  );

  const fetchStatus = useCallback(async () => {
    if (!experimentId) return;

    try {
      const statusResponse = await getExperimentStatus(experimentId);
      // Only update status if it actually changed (prevents unnecessary re-renders)
      setExperimentStatus(prev => prev === statusResponse.status ? prev : statusResponse.status);
      setStatusError(null);

      if (
        statusResponse.status === "complete" ||
        statusResponse.status === "failed" ||
        statusResponse.status === "cancelled"
      ) {
        clearStatusPolling();
      }
    } catch (error) {
      if (
        error instanceof ResearchAPIError &&
        (error.statusCode === 404 || error.statusCode === 410)
      ) {
        setExperimentStatus("idle");
        setStatusError(null);
        clearStatusPolling();
        setCanRenderTree(false);
      } else {
        const message =
          error instanceof Error ? error.message : "Failed to fetch status";
        setStatusError(message);
      }
    }
  }, [experimentId, clearStatusPolling]);

  useEffect(() => {
    if (!experimentId) {
      setExperimentStatus("idle");
      setStatusError(null);
      setTreeError(null);
      return;
    }

    useResearchTreeStore.getState().setExperimentId(experimentId);
    setCanRenderTree(false);
    fetchTreeSnapshot(true);
    fetchStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experimentId]);

  useEffect(() => {
    if (!experimentId) {
      clearStatusPolling();
      return;
    }

    if (experimentStatus !== "running" && experimentStatus !== "paused") {
      clearStatusPolling();
      return;
    }

    if (statusPollRef.current) {
      return;
    }

    fetchStatus();
    statusPollRef.current = setInterval(() => {
      fetchStatus();
    }, STATUS_POLL_INTERVAL);

    return () => {
      clearStatusPolling();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experimentId, experimentStatus]);

  useEffect(() => {
    if (!experimentId) {
      return;
    }

    if (experimentStatus !== "running" && experimentStatus !== "paused") {
      return;
    }

    fetchTreeSnapshot(false);
    const interval = setInterval(() => {
      fetchTreeSnapshot(false);
    }, TREE_POLL_INTERVAL);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experimentId, experimentStatus]);

  const { isConnected } = useResearchWS({
    experimentId: experimentId ?? "",
    autoConnect: Boolean(experimentId && experimentStatus !== "idle"),
  });

  const headerStats = useMemo(
    () => [
      { label: "Nodes", value: nodes.size.toString() },
      { label: "Edges", value: edges.length.toString() },
      {
        label: "Cost",
        value:
          stats?.total_cost !== undefined
            ? `$${stats.total_cost.toFixed(3)}`
            : "--",
      },
      {
        label: "Tokens",
        value:
          stats?.total_tokens !== undefined
            ? stats.total_tokens.toLocaleString()
            : "--",
      },
    ],
    [nodes.size, edges.length, stats?.total_cost, stats?.total_tokens],
  );

  if (!isClient) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-950 text-slate-100">
        <Loader size="large" />
      </div>
    );
  }

  if (!experimentId) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-950 p-8 text-center text-slate-200">
        <div className="max-w-md space-y-2">
          <h3 className="text-lg font-semibold">No Active Research</h3>
          <p className="text-sm text-slate-400">
            Start a research session to see the research tree visualization here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-hidden bg-slate-950 p-4 text-slate-100">
      {/* Single stats bar - no control buttons */}
      <div className="rounded-lg border border-slate-800 bg-slate-900 px-4 py-3 text-sm text-slate-300">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="font-medium">Research Tree</span>
            <span
              className={`flex items-center gap-2 text-xs ${
                isConnected ? "text-emerald-400" : "text-amber-400"
              }`}
            >
              <span className="h-2 w-2 rounded-full bg-current" />
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs text-slate-400">
            {headerStats.map((item) => (
              <div key={item.label} className="flex items-center gap-1">
                <span>{item.label}:</span>
                <span className="text-slate-200">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {statusError && (
        <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
          {statusError}
        </div>
      )}

      {treeError && (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          {treeError}
        </div>
      )}

      <div className="flex-1 overflow-hidden rounded-lg border border-slate-800 bg-slate-900">
        {canRenderTree ? (
          <ReactFlowProvider>
            <ResearchErrorBoundary>
              <ResearchTreeView />
            </ResearchErrorBoundary>
          </ReactFlowProvider>
        ) : (
          <div className="flex h-full items-center justify-center text-slate-400">
            <div className="text-center space-y-2">
              <div className="text-lg">No research data available</div>
              <div className="text-xs">Start or resume the experiment to see the tree</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
