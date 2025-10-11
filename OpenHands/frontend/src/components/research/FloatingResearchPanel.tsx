/**
 * Floating Research Panel Component
 *
 * Integration layer combining DraggableResizableWrapper and ResearchTreePanel
 * with visibility management and portal rendering.
 */

import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { DraggableResizableWrapper } from './DraggableResizableWrapper';
import { ResearchTreePanel } from './ResearchTreePanel';

interface FloatingResearchPanelProps {
  experimentId: string;
  isVisible: boolean;
  onClose: () => void;
  conversationId?: string;
}

export function FloatingResearchPanel({
  experimentId,
  isVisible,
  onClose,
  conversationId,
}: FloatingResearchPanelProps) {
  const [dragHandleMouseDown, setDragHandleMouseDown] = useState<((e: React.MouseEvent) => void) | null>(null);
  // Stable callback to avoid triggering infinite update loops
  const handleDragHandleReady = React.useCallback(
    (handler: (e: React.MouseEvent) => void) => {
      setDragHandleMouseDown(() => handler);
    },
    []
  );
  const [portalContainer] = useState(() => {
    if (typeof document === 'undefined') return null;
    const container = document.createElement('div');
    container.id = `research-portal-${experimentId}`;
    document.body.appendChild(container);
    return container;
  });

  // Cleanup portal container on unmount
  React.useEffect(() => {
    return () => {
      if (portalContainer && document.body.contains(portalContainer)) {
        // Wait for animations to complete before removing
        setTimeout(() => {
          if (document.body.contains(portalContainer)) {
            document.body.removeChild(portalContainer);
          }
        }, 300);
      }
    };
  }, [portalContainer]);

  if (!isVisible) {
    return null;
  }

  // Calculate default initial position (top-right with offset)
  const defaultPosition = {
    x: typeof window !== 'undefined' ? window.innerWidth - 800 : 100,
    y: 100,
  };

  // Calculate default initial size (responsive to viewport)
  const defaultSize = {
    width: 700,
    height: typeof window !== 'undefined' ? window.innerHeight * 0.8 : 600,
  };

  const panelContent = (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.2 }}
        >
          <DraggableResizableWrapper
            initialPosition={defaultPosition}
            initialSize={defaultSize}
            minWidth={400}
            minHeight={300}
            maxWidth={typeof window !== 'undefined' ? window.innerWidth * 0.9 : undefined}
            maxHeight={typeof window !== 'undefined' ? window.innerHeight * 0.95 : undefined}
            conversationId={conversationId}
            onDragHandleReady={handleDragHandleReady}
          >
            <ResearchTreePanel
              experimentId={experimentId}
              onClose={onClose}
              onDragHandleMouseDown={dragHandleMouseDown || undefined}
            />
          </DraggableResizableWrapper>
        </motion.div>
      )}
    </AnimatePresence>
  );

  // Render using portal with managed container
  return portalContainer
    ? createPortal(panelContent, portalContainer)
    : null;
}
