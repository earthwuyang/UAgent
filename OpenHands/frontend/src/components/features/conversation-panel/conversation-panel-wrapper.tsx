import ReactDOM from "react-dom";
import { useLocation } from "react-router";
import { cn } from "#/utils/utils";
import React from "react";

interface ConversationPanelWrapperProps {
  isOpen: boolean;
}

export function ConversationPanelWrapper({
  isOpen,
  children,
}: React.PropsWithChildren<ConversationPanelWrapperProps>) {
  // Use hook at the top level, before any conditions
  const { pathname } = useLocation();

  // SSR-safe: only render portal when mounted on client
  const [isMounted, setIsMounted] = React.useState(false);

  React.useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isOpen) return null;

  // Ensure portal target exists and we're in a client context
  if (!isMounted || typeof document === 'undefined') return null;

  const portalTarget = document.getElementById("root-outlet");
  if (!portalTarget) return null;

  return ReactDOM.createPortal(
    <div
      className={cn(
        "absolute h-full w-full left-0 top-0 z-20 bg-black/80 rounded-xl",
        pathname === "/" && "bottom-0 top-0 md:top-3 md:bottom-3 h-auto",
      )}
    >
      {children}
    </div>,
    portalTarget,
  );
}
