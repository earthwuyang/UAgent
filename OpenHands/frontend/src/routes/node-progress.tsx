/**
 * Node Progress Page
 *
 * Displays execution progress for a specific research tree node.
 * Opened in a new tab from the ResearchNodeDetailPanel.
 * Shows real-time event stream via WebSocket connection.
 *
 * URL: /conversations/:conversationId/nodes/:nodeId?experimentId=...
 */

import React, { useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router";
import { AlertCircle, Loader2 } from "lucide-react";
import {
  NodeProgressHeader,
  NodeEventTimeline,
} from "#/components/research/node-progress";
import {
  useNodeEventStore,
  useNodeEvents,
  useNodeIsSubscribed,
  useNodeError,
} from "#/state/node-event-store";
import { NodeEventWebSocket } from "#/services/node-event-websocket";

export default function NodeProgressPage() {
  const { conversationId, nodeId } = useParams<{
    conversationId: string;
    nodeId: string;
  }>();
  const [searchParams] = useSearchParams();
  const experimentId = searchParams.get("experimentId");

  // State for node data (fetch from API since this is a new tab)
  const [node, setNode] = useState<any>(null);
  const [loadingNode, setLoadingNode] = useState(true);
  const [nodeError, setNodeError] = useState<string | null>(null);

  // Get events from node event store
  const events = useNodeEvents(nodeId || "");
  const isSubscribed = useNodeIsSubscribed(nodeId || "");
  const error = useNodeError(nodeId || "");

  // WebSocket client ref
  const wsClientRef = useRef<NodeEventWebSocket | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [wsError, setWsError] = useState<string | null>(null);

  // Fetch node data from API (new tabs don't share Zustand state)
  useEffect(() => {
    if (!conversationId || !nodeId) {
      return;
    }

    const fetchNodeData = async () => {
      try {
        setLoadingNode(true);
        // API endpoint uses conversation ID, not full experiment ID
        const response = await fetch(`/api/research/experiments/${conversationId}/tree`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch tree: ${response.status}`);
        }

        const data = await response.json();
        
        // Find the specific node in the tree
        const targetNode = data.nodes?.find((n: any) => n.id === nodeId);
        
        if (targetNode) {
          setNode(targetNode);
        } else {
          setNodeError(`Node ${nodeId} not found in research tree`);
        }
      } catch (err) {
        console.error('[NodeProgressPage] Failed to fetch node data:', err);
        setNodeError(err instanceof Error ? err.message : 'Failed to load node data');
      } finally {
        setLoadingNode(false);
      }
    };

    fetchNodeData();
  }, [conversationId, nodeId]);

  // Initialize WebSocket connection and subscribe to node
  useEffect(() => {
    if (!experimentId || !nodeId) {
      return;
    }

    console.log("[NodeProgressPage] Initializing WebSocket connection", {
      experimentId,
      nodeId,
    });

    // Create WebSocket client with nodeId for per-node event streaming
    const wsClient = new NodeEventWebSocket(experimentId, nodeId);
    wsClientRef.current = wsClient;

    // Connect and subscribe
    wsClient.connect();

    // Wait a bit for connection to establish, then subscribe
    const subscribeTimeout = setTimeout(() => {
      console.log("[NodeProgressPage] Subscribing to node:", nodeId);
      useNodeEventStore.getState().subscribeToNode(nodeId, experimentId);
      setWsConnected(true);
    }, 500);

    // Cleanup on unmount
    return () => {
      clearTimeout(subscribeTimeout);
      console.log("[NodeProgressPage] Cleaning up WebSocket connection");

      if (nodeId) {
        useNodeEventStore.getState().unsubscribeFromNode(nodeId);
      }

      if (wsClient) {
        wsClient.disconnect();
      }

      wsClientRef.current = null;
      setWsConnected(false);
    };
  }, [experimentId, nodeId]);

  // Validate required parameters
  if (!conversationId || !nodeId || !experimentId) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-slate-50 dark:bg-slate-900">
        <AlertCircle size={48} className="text-red-500 mb-4" />
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
          Missing Required Parameters
        </h1>
        <p className="text-slate-600 dark:text-slate-400 text-center max-w-md">
          This page requires conversationId, nodeId, and experimentId
          parameters. Please navigate from the research tree view.
        </p>
      </div>
    );
  }

  // Show error state if node data failed to load
  if (nodeError) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-slate-50 dark:bg-slate-900">
        <AlertCircle size={48} className="text-red-500 mb-4" />
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
          Failed to Load Node Data
        </h1>
        <p className="text-slate-600 dark:text-slate-400 text-center max-w-md">
          {nodeError}
        </p>
      </div>
    );
  }

  // Show loading state while node data loads
  if (loadingNode || !node) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-slate-50 dark:bg-slate-900">
        <Loader2 size={48} className="text-blue-500 animate-spin mb-4" />
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
          Loading Node Data...
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Fetching node information from the research tree.
        </p>
      </div>
    );
  }

  // Show error state if there's a connection error
  if (wsError) {
    return (
      <div className="flex flex-col h-screen bg-slate-50 dark:bg-slate-900">
        <NodeProgressHeader
          node={node}
          conversationId={conversationId}
          experimentId={experimentId}
        />
        <div className="flex-1 flex flex-col items-center justify-center p-6">
          <AlertCircle size={48} className="text-red-500 mb-4" />
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-2">
            WebSocket Connection Error
          </h2>
          <p className="text-slate-600 dark:text-slate-400 text-center max-w-md">
            {wsError}
          </p>
        </div>
      </div>
    );
  }

  // Main render
  return (
    <div className="flex flex-col h-screen bg-slate-50 dark:bg-slate-900">
      {/* Header */}
      <NodeProgressHeader
        node={node}
        conversationId={conversationId}
        experimentId={experimentId}
      />

      {/* Connection status banner (only show when not connected) */}
      {!wsConnected && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border-b border-yellow-200 dark:border-yellow-800 px-4 py-2">
          <div className="flex items-center gap-2 text-sm text-yellow-800 dark:text-yellow-200">
            <Loader2 size={16} className="animate-spin" />
            <span>Connecting to event stream...</span>
          </div>
        </div>
      )}

      {/* Store error banner */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border-b border-red-200 dark:border-red-800 px-4 py-2">
          <div className="flex items-center gap-2 text-sm text-red-800 dark:text-red-200">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Event Timeline */}
      <NodeEventTimeline events={events} autoScroll={true} />
    </div>
  );
}
