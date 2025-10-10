/**
 * Draggable Resizable Wrapper Component
 *
 * Provides drag and resize functionality using framer-motion and custom resize handles.
 * Persists position and size to localStorage per conversation.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, useDragControls } from 'framer-motion';

interface DraggableResizableWrapperProps {
  children: React.ReactNode;
  initialPosition?: { x: number; y: number };
  initialSize?: { width: number; height: number };
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
  onPositionChange?: (position: { x: number; y: number }) => void;
  onSizeChange?: (size: { width: number; height: number }) => void;
  bounds?: { left: number; top: number; right: number; bottom: number };
  conversationId?: string;
  onDragHandleReady?: (onMouseDown: (e: React.MouseEvent) => void) => void;
}

type ResizeDirection = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

export function DraggableResizableWrapper({
  children,
  initialPosition = { x: 100, y: 100 },
  initialSize = { width: 700, height: 600 },
  minWidth = 400,
  minHeight = 300,
  maxWidth,
  maxHeight,
  onPositionChange,
  onSizeChange,
  bounds,
  conversationId,
  onDragHandleReady,
}: DraggableResizableWrapperProps) {
  const [position, setPosition] = useState(initialPosition);
  const [size, setSize] = useState(initialSize);
  const [isResizing, setIsResizing] = useState(false);
  const [dragConstraints, setDragConstraints] = useState({
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
  });
  
  const resizeDirection = useRef<ResizeDirection | null>(null);
  const resizeStartPos = useRef({ x: 0, y: 0 });
  const resizeStartSize = useRef({ width: 0, height: 0 });
  const resizeStartPosition = useRef({ x: 0, y: 0 });
  
  // Drag controls for header-only dragging
  const controls = useDragControls();

  // Load persisted state from localStorage
  useEffect(() => {
    if (conversationId) {
      const positionKey = `research-panel-position-${conversationId}`;
      const sizeKey = `research-panel-size-${conversationId}`;
      
      const savedPosition = localStorage.getItem(positionKey);
      const savedSize = localStorage.getItem(sizeKey);
      
      if (savedPosition) {
        try {
          const parsed = JSON.parse(savedPosition);
          setPosition(parsed);
        } catch (e) {
          console.error('Failed to parse saved position', e);
        }
      }
      
      if (savedSize) {
        try {
          const parsed = JSON.parse(savedSize);
          setSize(parsed);
        } catch (e) {
          console.error('Failed to parse saved size', e);
        }
      }
    }
  }, [conversationId]);

  // Persist state to localStorage
  useEffect(() => {
    if (conversationId) {
      const positionKey = `research-panel-position-${conversationId}`;
      const sizeKey = `research-panel-size-${conversationId}`;
      
      localStorage.setItem(positionKey, JSON.stringify(position));
      localStorage.setItem(sizeKey, JSON.stringify(size));
    }
  }, [position, size, conversationId]);

  // Update drag constraints when size or window changes
  const updateDragConstraints = useCallback(() => {
    const newConstraints = bounds || {
      left: 0,
      top: 0,
      right: window.innerWidth - size.width,
      bottom: window.innerHeight - size.height,
    };
    setDragConstraints(newConstraints);
  }, [bounds, size.width, size.height]);

  useEffect(() => {
    updateDragConstraints();
  }, [updateDragConstraints]);

  // Handle window resize to keep panel within bounds and update constraints
  useEffect(() => {
    const handleWindowResize = () => {
      // Update constraints
      updateDragConstraints();
      
      // Keep panel within bounds
      setPosition(prev => {
        const maxX = window.innerWidth - size.width;
        const maxY = window.innerHeight - size.height;
        return {
          x: Math.max(0, Math.min(prev.x, maxX)),
          y: Math.max(0, Math.min(prev.y, maxY)),
        };
      });
    };

    window.addEventListener('resize', handleWindowResize);
    return () => window.removeEventListener('resize', handleWindowResize);
  }, [size, updateDragConstraints]);

  // Expose drag handle starter to parent
  useEffect(() => {
    if (onDragHandleReady) {
      onDragHandleReady((e: React.MouseEvent) => {
        controls.start(e as any);
      });
    }
  }, [onDragHandleReady, controls]);

  const handleDragEnd = useCallback((_event: any, info: any) => {
    setPosition(prev => {
      const newX = Math.max(0, Math.min(prev.x + info.offset.x, window.innerWidth - size.width));
      const newY = Math.max(0, Math.min(prev.y + info.offset.y, window.innerHeight - size.height));
      const next = { x: newX, y: newY };
      onPositionChange?.(next);
      return next;
    });
  }, [onPositionChange, size.width, size.height]);

  const handleResizeStart = useCallback((direction: ResizeDirection, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);
    resizeDirection.current = direction;
    resizeStartPos.current = { x: e.clientX, y: e.clientY };
    resizeStartSize.current = { ...size };
    resizeStartPosition.current = { ...position };
  }, [size, position]);

  const handleResizeMove = useCallback((e: MouseEvent) => {
    if (!isResizing || !resizeDirection.current) return;

    const deltaX = e.clientX - resizeStartPos.current.x;
    const deltaY = e.clientY - resizeStartPos.current.y;
    const direction = resizeDirection.current;

    let newWidth = resizeStartSize.current.width;
    let newHeight = resizeStartSize.current.height;
    let newX = resizeStartPosition.current.x;
    let newY = resizeStartPosition.current.y;

    // Calculate new dimensions based on resize direction
    if (direction.includes('e')) {
      newWidth = resizeStartSize.current.width + deltaX;
    }
    if (direction.includes('w')) {
      newWidth = resizeStartSize.current.width - deltaX;
      newX = resizeStartPosition.current.x + deltaX;
    }
    if (direction.includes('s')) {
      newHeight = resizeStartSize.current.height + deltaY;
    }
    if (direction.includes('n')) {
      newHeight = resizeStartSize.current.height - deltaY;
      newY = resizeStartPosition.current.y + deltaY;
    }

    // Apply constraints
    const effectiveMaxWidth = maxWidth || window.innerWidth * 0.9;
    const effectiveMaxHeight = maxHeight || window.innerHeight * 0.95;
    
    newWidth = Math.max(minWidth, Math.min(newWidth, effectiveMaxWidth));
    newHeight = Math.max(minHeight, Math.min(newHeight, effectiveMaxHeight));

    // Adjust position if resizing from left or top
    if (direction.includes('w')) {
      newX = resizeStartPosition.current.x + (resizeStartSize.current.width - newWidth);
    }
    if (direction.includes('n')) {
      newY = resizeStartPosition.current.y + (resizeStartSize.current.height - newHeight);
    }

    // Ensure panel stays within viewport
    newX = Math.max(0, Math.min(newX, window.innerWidth - newWidth));
    newY = Math.max(0, Math.min(newY, window.innerHeight - newHeight));

    setSize({ width: newWidth, height: newHeight });
    setPosition({ x: newX, y: newY });
    onSizeChange?.({ width: newWidth, height: newHeight });
    onPositionChange?.({ x: newX, y: newY });
  }, [isResizing, minWidth, minHeight, maxWidth, maxHeight, onSizeChange, onPositionChange]);

  const handleResizeEnd = useCallback(() => {
    setIsResizing(false);
    resizeDirection.current = null;
  }, []);

  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', handleResizeMove);
      window.addEventListener('mouseup', handleResizeEnd);
      return () => {
        window.removeEventListener('mousemove', handleResizeMove);
        window.removeEventListener('mouseup', handleResizeEnd);
      };
    }
  }, [isResizing, handleResizeMove, handleResizeEnd]);

  return (
    <motion.div
      drag
      dragControls={controls}
      dragListener={false}
      dragMomentum={false}
      dragElastic={0}
      dragConstraints={dragConstraints}
      onDragEnd={handleDragEnd}
      className="draggable-wrapper"
      style={{
        position: 'fixed',
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height,
        zIndex: 1000,
      }}
    >
      {children}

      {/* Resize handles */}
      <div
        className="resize-handle resize-handle-n"
        onMouseDown={(e) => handleResizeStart('n', e)}
      />
      <div
        className="resize-handle resize-handle-s"
        onMouseDown={(e) => handleResizeStart('s', e)}
      />
      <div
        className="resize-handle resize-handle-e"
        onMouseDown={(e) => handleResizeStart('e', e)}
      />
      <div
        className="resize-handle resize-handle-w"
        onMouseDown={(e) => handleResizeStart('w', e)}
      />
      <div
        className="resize-handle resize-handle-ne"
        onMouseDown={(e) => handleResizeStart('ne', e)}
      />
      <div
        className="resize-handle resize-handle-nw"
        onMouseDown={(e) => handleResizeStart('nw', e)}
      />
      <div
        className="resize-handle resize-handle-se"
        onMouseDown={(e) => handleResizeStart('se', e)}
      />
      <div
        className="resize-handle resize-handle-sw"
        onMouseDown={(e) => handleResizeStart('sw', e)}
      />
    </motion.div>
  );
}
