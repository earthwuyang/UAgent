import { useConversationStore } from "#/state/conversation-store";
import { MobileLayout } from "./mobile-layout";
import { DesktopLayout } from "./desktop-layout";
import React from "react";

export function ConversationMain() {
  // Use Zustand store instead of Redux
  const isRightPanelShown = useConversationStore((state) => state.isRightPanelShown);

  // Default to desktop layout initially to avoid SSR/hydration issues
  const [width, setWidth] = React.useState<number>(1024);
  const [isMounted, setIsMounted] = React.useState(false);

  React.useEffect(() => {
    // Set initial width and mounted state
    setIsMounted(true);
    if (typeof window !== 'undefined') {
      setWidth(window.innerWidth);

      // Add resize listener
      const handleResize = () => setWidth(window.innerWidth);
      window.addEventListener('resize', handleResize);

      // Cleanup
      return () => window.removeEventListener('resize', handleResize);
    }
  }, []);

  if (!isMounted) {
    // Show desktop layout during SSR to avoid hydration mismatch
    return <DesktopLayout isRightPanelShown={isRightPanelShown} />;
  }

  if (width <= 1024) {
    return <MobileLayout isRightPanelShown={isRightPanelShown} />;
  }

  return <DesktopLayout isRightPanelShown={isRightPanelShown} />;
}
