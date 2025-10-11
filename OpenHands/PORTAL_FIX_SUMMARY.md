# Portal Cleanup Fix - DOM removeChild Error

## Error
```
Failed to execute 'removeChild' on 'Node': The node to be removed is not a child of this node.
```

## Root Cause
The error occurred due to a race condition in React portal cleanup:

1. **FloatingResearchPanel** uses `createPortal(content, document.body)` to render outside the normal React tree
2. **AnimatePresence** handles exit animations with 200ms duration
3. When the component unmounts (triggered by our new `key={researchExperimentId}` prop), React tries to remove the portal node
4. However, the animation cleanup may have already removed the node, causing the DOM error

## The Fix

### Before:
```typescript
// Direct portal to document.body
return typeof document !== 'undefined' 
  ? createPortal(panelContent, document.body)
  : null;
```

**Problem:** React tries to remove nodes directly from `document.body` during cleanup, but AnimatePresence may have already cleaned them up.

### After:
```typescript
// Create managed portal container
const [portalContainer] = useState(() => {
  if (typeof document === 'undefined') return null;
  const container = document.createElement('div');
  container.id = `research-portal-${experimentId}`;
  document.body.appendChild(container);
  return container;
});

// Cleanup with safety checks and animation delay
React.useEffect(() => {
  return () => {
    if (portalContainer && document.body.contains(portalContainer)) {
      // Wait for animations to complete before removing
      setTimeout(() => {
        if (document.body.contains(portalContainer)) {
          document.body.removeChild(portalContainer);
        }
      }, 300); // 300ms > 200ms animation duration
    }
  };
}, [portalContainer]);

// Use managed container
return portalContainer
  ? createPortal(panelContent, portalContainer)
  : null;
```

## Benefits

1. **Unique Container**: Each component instance gets its own portal container
2. **Safe Cleanup**: Check if node exists in DOM before attempting removal
3. **Animation Delay**: Wait 300ms (> 200ms animation) before cleanup
4. **No Race Conditions**: Cleanup only happens after animations complete
5. **Proper Key Support**: Works correctly with `key={researchExperimentId}` prop changes

## Files Modified
- `frontend/src/components/research/FloatingResearchPanel.tsx`
  - Lines 28-34: Created managed portal container
  - Lines 37-48: Added cleanup effect with delay
  - Lines 96-99: Updated portal rendering

## Testing
✅ Build successful
✅ No DOM errors during component mount/unmount
✅ Animations complete before cleanup
✅ Works with key prop changes

