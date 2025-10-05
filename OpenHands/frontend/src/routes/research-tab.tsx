import React from "react";
import { ReactFlowProvider } from "reactflow";
import { useConversationId } from "#/hooks/use-conversation-id";
import { ResearchTreeView } from "#/components/research/ResearchTreeView";
import { useResearchWS } from "#/hooks/useResearchWS";
import { useResearchTreeStore } from "#/state/research-tree-store";

/**
 * Research Tree Tab Component
 *
 * Displays the research tree visualization for the current conversation.
 * Uses ReactFlow for interactive graph visualization and Zustand for state management.
 * Connects via WebSocket for real-time updates as the research progresses.
 */
export default function ResearchTab() {
  const { conversationId } = useConversationId();
  const { nodes, edges, stats } = useResearchTreeStore();
  const [isConnected, setIsConnected] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  // For now, we'll use conversationId as experimentId
  // In the future, you may want to fetch the actual experimentId from the conversation
  const experimentId = conversationId;
  console.log('[ResearchTab] conversationId:', conversationId);

  // Connect to WebSocket for real-time updates (if available)
  // const { isConnected, error } = useResearchWS({
  //   experimentId: experimentId || "",
  //   autoConnect: !!experimentId,
  // });

  // Set connected status to true since we're polling
  React.useEffect(() => {
    if (experimentId) {
      setIsConnected(true);
    } else {
      setIsConnected(false);
    }
  }, [experimentId]);

  // Fetch initial tree snapshot when component mounts
  React.useEffect(() => {
    if (experimentId) {
      fetch(`/api/research/experiments/${experimentId}/tree`)
        .then(async (res) => {
          if (!res.ok) {
            // If experiment doesn't exist yet, initialize with empty tree
            return {
              version: 0,
              timestamp: new Date().toISOString(),
              experiment_id: experimentId,
              data: {
                nodes: [],
                edges: [],
                stats: {}
              }
            };
          }
          const data = await res.json();
          // Validate response structure
          if (!data || typeof data !== 'object') {
            throw new Error('Invalid response format');
          }
          return data;
        })
        .then((snapshot) => {
          // Ensure snapshot has required structure
          const validSnapshot = {
            version: snapshot.version || 0,
            timestamp: snapshot.timestamp || new Date().toISOString(),
            experiment_id: snapshot.experiment_id || experimentId,
            data: {
              nodes: Array.isArray(snapshot.data?.nodes) ? snapshot.data.nodes : [],
              edges: Array.isArray(snapshot.data?.edges) ? snapshot.data.edges : [],
              stats: snapshot.data?.stats || {}
            }
          };
          useResearchTreeStore.getState().setSnapshot(validSnapshot);
        })
        .catch((err) => {
          console.error("Failed to fetch research tree:", err);
          // Initialize with empty tree on error
          useResearchTreeStore.getState().setSnapshot({
            version: 0,
            timestamp: new Date().toISOString(),
            experiment_id: experimentId,
            data: {
              nodes: [],
              edges: [],
              stats: {}
            }
          });
        });
    }
  }, [experimentId]);

  // Poll for updates as a fallback for WebSocket
  React.useEffect(() => {
    if (!experimentId) return;

    const pollInterval = setInterval(() => {
      fetch(`/api/research/experiments/${experimentId}/tree`)
        .then(async (res) => {
          if (!res.ok) return null;
          return await res.json();
        })
        .then((snapshot) => {
          if (snapshot) {
            // Ensure snapshot has required structure
            const validSnapshot = {
              version: snapshot.version || 0,
              timestamp: snapshot.timestamp || new Date().toISOString(),
              experiment_id: snapshot.experiment_id || experimentId,
              data: {
                nodes: Array.isArray(snapshot.data?.nodes) ? snapshot.data.nodes : [],
                edges: Array.isArray(snapshot.data?.edges) ? snapshot.data.edges : [],
                stats: snapshot.data?.stats || {}
              }
            };
            useResearchTreeStore.getState().setSnapshot(validSnapshot);
          }
        })
        .catch((err) => {
          console.error("Failed to poll research tree:", err);
        });
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(pollInterval);
  }, [experimentId]);

  if (!experimentId) {
    return (
      <div className="flex items-center justify-center h-full p-8 text-center">
        <div className="max-w-md">
          <h3 className="text-lg font-semibold mb-2">No Active Research</h3>
          <p className="text-sm text-gray-500">
            Start a research session to see the research tree visualization here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full flex flex-col bg-gray-900">
      {/* Header with connection status and stats */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? "bg-green-500 animate-pulse" : "bg-yellow-500"
              }`}
            />
            <span className="text-sm text-gray-300">
              {isConnected ? "Polling" : "Disconnected"}
            </span>
          </div>
          <div className="text-sm text-gray-400">
            Nodes: {nodes.size} | Edges: {edges.length}
          </div>
          {stats?.total_cost !== undefined && (
            <div className="text-sm text-gray-400">
              Cost: ${stats.total_cost.toFixed(3)}
            </div>
          )}
        </div>
      </div>

      {/* Tree visualization */}
      <div className="flex-1 relative">
        <ReactFlowProvider>
          <ResearchTreeView />
        </ReactFlowProvider>
      </div>

      {/* Error display */}
      {error && (
        <div className="absolute bottom-4 left-4 right-4 bg-red-500/90 text-white px-4 py-2 rounded-lg shadow-lg">
          <p className="text-sm font-semibold">Connection Error</p>
          <p className="text-xs">{error}</p>
        </div>
      )}
    </div>
  );
}
