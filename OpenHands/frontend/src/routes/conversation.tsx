import React from "react";
import { useNavigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";

import { useConversationId } from "#/hooks/use-conversation-id";
import { clearTerminal } from "#/state/command-store";
import { useEffectOnce } from "#/hooks/use-effect-once";
import { clearJupyter } from "#/state/jupyter-store";
import { resetConversationState } from "#/state/conversation-store";
import { setCurrentAgentState } from "#/stores/agent-store";
import { AgentState } from "#/types/agent-state";

import { useBatchFeedback } from "#/hooks/query/use-batch-feedback";
import { WsClientProvider } from "#/context/ws-client-provider";
import { EventHandler } from "../wrapper/event-handler";
import { useConversationConfig } from "#/hooks/query/use-conversation-config";

import { useActiveConversation } from "#/hooks/query/use-active-conversation";

import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useDocumentTitleFromState } from "#/hooks/use-document-title-from-state";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { ConversationSubscriptionsProvider } from "#/context/conversation-subscriptions-provider";
import { useUserProviders } from "#/hooks/use-user-providers";

import { ConversationMain } from "#/components/features/conversation/conversation-main/conversation-main";
import { ConversationName } from "#/components/features/conversation/conversation-name";

import { ConversationTabs } from "#/components/features/conversation/conversation-tabs/conversation-tabs";
import { useStartConversation } from "#/hooks/mutation/use-start-conversation";
import { useConversationSubscriptions } from "#/context/conversation-subscriptions-provider";

function AppContent() {
  useConversationConfig();

  const { conversationId } = useConversationId();
  const { data: conversation, isFetched, refetch } = useActiveConversation();
  const { mutate: startConversation } = useStartConversation();
  const { data: isAuthed } = useIsAuthed();
  const { providers } = useUserProviders();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { subscribeToConversation, isSubscribedToConversation } = useConversationSubscriptions();

  // Track if we've already started the conversation to prevent infinite loops
  const startedConversationRef = React.useRef<string | null>(null);

  // Fetch batch feedback data when conversation is loaded
  useBatchFeedback();

  // Set the document title to the conversation title when available
  useDocumentTitleFromState();

  // Force fresh conversation data when navigating to prevent stale cache issues
  React.useEffect(() => {
    queryClient.invalidateQueries({
      queryKey: ["user", "conversation", conversationId],
    });
  }, [conversationId, queryClient]);

  // UAG-37 FIX: Subscribe to WebSocket for existing RUNNING conversations
  // This handles the case when navigating to an existing conversation
  // (not creating a new one), which doesn't go through the create-and-subscribe flow
  React.useEffect(() => {
    console.log("[UAG-37 DEBUG] Subscription effect running", {
      conversation: conversation?.conversation_id,
      status: conversation?.status,
      isSubscribed: conversation ? isSubscribedToConversation(conversation.conversation_id) : 'no conversation',
    });
    
    if (
      conversation &&
      conversation.status === "RUNNING" &&
      !isSubscribedToConversation(conversation.conversation_id)
    ) {
      console.log("[UAG-37] Subscribing to existing RUNNING conversation:", conversation.conversation_id);
      // Determine the correct base URL and socket path
      let baseUrl = "";
      let socketPath = "/socket.io/"; // UAG-38 FIX: Must include trailing slash to match backend
      
      if (conversation.url && !conversation.url.startsWith("/")) {
        const u = new URL(conversation.url);
        // UAG-38 FIX: Socket.IO needs full URL with protocol, not just host
        baseUrl = `${u.protocol}//${u.host}`;
        const pathBeforeApi =
          u.pathname.split("/api/conversations")[0] || "/";
        socketPath = `${pathBeforeApi.replace(/\/$/, "")}/socket.io/`;
      } else {
        // UAG-38 FIX: Socket.IO needs full URL with protocol
        const host = (import.meta.env.VITE_BACKEND_BASE_URL as string | undefined) ||
          window?.location.host;
        // If VITE_BACKEND_BASE_URL already has protocol, use as-is, otherwise add http://
        baseUrl = host.startsWith('http') ? host : `http://${host}`;
      }

      console.log("[UAG-38 DEBUG] Socket.IO connection params:", { baseUrl, socketPath });

      // Subscribe to the conversation's WebSocket
      subscribeToConversation({
        conversationId: conversation.conversation_id,
        sessionApiKey: conversation.session_api_key,
        providersSet: providers,
        baseUrl,
        socketPath,
      });
    }
  }, [
    conversation,
    conversation?.conversation_id,
    conversation?.status,
    conversation?.url,
    conversation?.session_api_key,
    isSubscribedToConversation,
    subscribeToConversation,
    providers,
  ]);

  React.useEffect(() => {
    if (isFetched && !conversation && isAuthed) {
      displayErrorToast(
        "This conversation does not exist, or you do not have permission to access it.",
      );
      navigate("/");
    } else if (
      conversation?.status === "STOPPED" &&
      startedConversationRef.current !== conversation.conversation_id
    ) {
      // If conversation is STOPPED and we haven't started it yet
      startedConversationRef.current = conversation.conversation_id;
      startConversation(
        { conversationId: conversation.conversation_id, providers },
        {
          onError: (error) => {
            displayErrorToast(`Failed to start conversation: ${error.message}`);
            // Refetch the conversation to ensure UI consistency
            refetch();
            // Reset the ref on error so we can retry
            startedConversationRef.current = null;
          },
        },
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    conversation?.conversation_id,
    conversation?.status,
    isFetched,
    isAuthed,
    providers,
    // Intentionally omitting startConversation, navigate, and refetch
    // These are functions that don't have stable references and would cause infinite loops
  ]);



  React.useEffect(() => {
    clearTerminal();
    clearJupyter();
    resetConversationState();
    setCurrentAgentState(AgentState.LOADING);
  }, [conversationId]);

  useEffectOnce(() => {
    clearTerminal();
    clearJupyter();
    resetConversationState();
    setCurrentAgentState(AgentState.LOADING);
  });

  return (
    <div
      data-testid="app-route"
      className="p-3 md:p-0 flex flex-col h-full gap-3"
    >
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4.5 pt-2 lg:pt-0">
        <ConversationName />
        <ConversationTabs />
      </div>

      <div className="flex h-full overflow-auto">
        <ConversationMain />
      </div>
    </div>
  );
}

// Wrapper component that provides ConversationSubscriptionsProvider context
// before AppContent tries to use it
function AppWithProviders() {
  const { conversationId } = useConversationId();
  
  return (
    <WsClientProvider conversationId={conversationId}>
      <ConversationSubscriptionsProvider>
        <EventHandler>
          <AppContent />
        </EventHandler>
      </ConversationSubscriptionsProvider>
    </WsClientProvider>
  );
}

function App() {
  return <AppWithProviders />;
}

export default App;
