import React from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { ChatInterfaceWrapper } from "./chat-interface-wrapper";
import { ConversationTabContent } from "../conversation-tabs/conversation-tab-content/conversation-tab-content";
import { FloatingResearchPanel } from "#/components/research";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useConversationStore } from "#/state/conversation-store";

interface DesktopLayoutProps {
  isRightPanelShown: boolean;
}

export function DesktopLayout({ 
  isRightPanelShown,
}: DesktopLayoutProps) {
  const { data: conversation } = useActiveConversation();
  const conversationId = useConversationId();
  const userHasClosedPanelRef = React.useRef(false);

  // Use Zustand store for research panel visibility
  const isResearchPanelVisible = useConversationStore((state) => state.isResearchPanelVisible);
  const setIsResearchPanelVisible = useConversationStore((state) => state.setIsResearchPanelVisible);

  // Extract research experiment ID from conversation
  const researchExperimentId = conversation?.research_experiment_id ?? null;

  // Reset manual close flag when conversation changes
  React.useEffect(() => {
    userHasClosedPanelRef.current = false;
  }, [conversationId]);

  // Auto-show logic based on research_experiment_id
  React.useEffect(() => {
    if (researchExperimentId && !userHasClosedPanelRef.current) {
      // Auto-show when research starts
      setIsResearchPanelVisible(true);
    } else if (!researchExperimentId) {
      // Auto-hide when research ends
      setIsResearchPanelVisible(false);
      // Reset the manual close flag
      userHasClosedPanelRef.current = false;
    }
  }, [researchExperimentId, setIsResearchPanelVisible]);

  const handleClosePanel = () => {
    setIsResearchPanelVisible(false);
    userHasClosedPanelRef.current = true;
  };

  return (
    <>
      <PanelGroup
        direction="horizontal"
        className="grow h-full min-h-0 min-w-0"
        autoSaveId="react-resizable-panels:layout"
      >
        <Panel minSize={30} maxSize={80} className="overflow-hidden bg-base">
          <ChatInterfaceWrapper isRightPanelShown={isRightPanelShown} />
        </Panel>
        {isRightPanelShown && (
          <>
            <PanelResizeHandle className="cursor-ew-resize" />
            <Panel
              minSize={20}
              maxSize={70}
              className="flex flex-col overflow-hidden"
            >
              <div className="flex flex-col flex-1 gap-3">
                <ConversationTabContent />
              </div>
            </Panel>
          </>
        )}
      </PanelGroup>

      {/* Floating Research Panel */}
      {researchExperimentId && (
        <FloatingResearchPanel
          experimentId={researchExperimentId}
          isVisible={isResearchPanelVisible}
          onClose={handleClosePanel}
          conversationId={conversationId}
        />
      )}
    </>
  );
}
