import React, { createContext, useContext, ReactNode, RefObject } from "react";
import { useScrollToBottom } from "#/hooks/use-scroll-to-bottom";

interface ScrollContextType {
  scrollRef: RefObject<HTMLDivElement | null>;
  autoScroll: boolean;
  setAutoScroll: (value: boolean) => void;
  scrollDomToBottom: () => void;
  hitBottom: boolean;
  setHitBottom: (value: boolean) => void;
  onChatBodyScroll: (e: HTMLElement) => void;
}

export const ScrollContext = createContext<ScrollContextType | undefined>(
  undefined,
);

interface ScrollProviderProps {
  children: ReactNode;
  value?: ScrollContextType;
}

export function ScrollProvider({ children, value }: ScrollProviderProps) {
  // If a value is provided, use it directly (client-side with pre-initialized hooks)
  if (value) {
    return (
      <ScrollContext.Provider value={value}>
        {children}
      </ScrollContext.Provider>
    );
  }

  // During SSR, provide a minimal context to avoid hydration issues
  if (typeof window === 'undefined') {
    const emptyRef = React.createRef<HTMLDivElement>();
    const defaultContext: ScrollContextType = {
      scrollRef: emptyRef,
      autoScroll: true,
      setAutoScroll: () => {},
      scrollDomToBottom: () => {},
      hitBottom: true,
      setHitBottom: () => {},
      onChatBodyScroll: () => {},
    };
    return (
      <ScrollContext.Provider value={defaultContext}>
        {children}
      </ScrollContext.Provider>
    );
  }

  // On client side, initialize the scroll hook when no value is provided
  // Create a ref that will be shared between the provider and the hook
  const scrollRef = React.createRef<HTMLDivElement>();

  // Initialize the scroll hook with the shared ref
  const scrollHook = useScrollToBottom(scrollRef);

  const contextValue: ScrollContextType = {
    scrollRef,
    autoScroll: scrollHook.autoScroll,
    setAutoScroll: scrollHook.setAutoScroll,
    scrollDomToBottom: scrollHook.scrollDomToBottom,
    hitBottom: scrollHook.hitBottom,
    setHitBottom: scrollHook.setHitBottom,
    onChatBodyScroll: scrollHook.onChatBodyScroll,
  };

  return (
    <ScrollContext.Provider value={contextValue}>
      {children}
    </ScrollContext.Provider>
  );
}

export function useScrollContext() {
  const context = useContext(ScrollContext);
  if (context === undefined) {
    throw new Error("useScrollContext must be used within a ScrollProvider");
  }
  return context;
}
