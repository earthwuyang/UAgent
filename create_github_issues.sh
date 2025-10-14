#!/bin/bash

# Script to create GitHub issues for UAGENT-24 implementation
# Run this from the UAgent directory

set -e

echo "Creating GitHub issues for Issue #24 implementation..."
echo ""

# Create milestone for the project
echo "Creating milestone..."
gh api repos/:owner/:repo/milestones -f title="Node Context Switching (Issue #24)" \
  -f description="Implement node context switching functionality for research tree" \
  -f due_on="2025-11-15T00:00:00Z" \
  -f state="open" || echo "Milestone might already exist"

echo ""

# PHASE 1: Backend Infrastructure

echo "Creating Phase 1 tickets (Backend Infrastructure)..."

# Ticket 1: EventBus modifications
ISSUE_1=$(gh issue create \
  --title "[UAGENT-24-1] Modify EventBus for Per-Node Event Storage" \
  --label "enhancement,backend,phase-1,priority:P0" \
  --body "## Story
**Type**: Story  
**Priority**: P0 (Blocker)  
**Story Points**: 5  
**Assignee**: Backend Engineer  
**Phase**: 1 - Backend Infrastructure

## Description
Modify the EventBus class to support per-node event storage alongside the existing global event stream. This enables efficient retrieval of events for individual research nodes.

## Acceptance Criteria
- [ ] Add \`_node_events: Dict[str, List[ResearchEvent]]\` to EventBus class
- [ ] Add \`_node_subscribers: Dict[str, Dict[str, List[Callable]]]\` to EventBus class
- [ ] Implement \`init_node_stream(node_id: str)\` method
- [ ] Implement \`publish_node_event(node_id: str, event: ResearchEvent)\` method
- [ ] Implement \`get_node_events(node_id, offset, limit, event_types, since)\` method with pagination and filtering
- [ ] Implement \`subscribe_node(node_id, callback)\` method
- [ ] Implement \`unsubscribe_node(node_id, callback)\` method
- [ ] Implement \`clear_node_events(node_id)\` cleanup method
- [ ] All existing EventBus functionality continues to work (backward compatible)
- [ ] Unit tests pass with >90% coverage for new code

## Technical Details
**File**: \`OpenHands/extensions/uagent_research/orchestrator/event_bus.py\`

**Key Changes**:
- Memory optimization: Store events as list, not set (preserve order)
- Thread safety: Consider adding locks if needed for concurrent access
- Cleanup: Implement TTL or cleanup on node completion

## Testing Requirements
- Unit test: \`test_init_node_stream()\`
- Unit test: \`test_publish_node_event()\`
- Unit test: \`test_get_node_events_with_filters()\`
- Unit test: \`test_node_event_pagination()\`
- Unit test: \`test_subscribe_unsubscribe_node()\`
- Unit test: \`test_backward_compatibility()\`

## Definition of Done
- [ ] Code reviewed and approved
- [ ] All unit tests passing
- [ ] No breaking changes to existing functionality
- [ ] Documentation updated (docstrings)

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_1"

# Ticket 2: TreeOrchestrator integration
ISSUE_2=$(gh issue create \
  --title "[UAGENT-24-2] Integrate Per-Node Event Publishing in TreeOrchestrator" \
  --label "enhancement,backend,phase-1,priority:P0" \
  --body "## Story
**Type**: Story  
**Priority**: P0 (Blocker)  
**Story Points**: 5  
**Assignee**: Backend Engineer  
**Phase**: 1 - Backend Infrastructure  
**Depends On**: $ISSUE_1

## Description
Modify the TreeOrchestrator to publish events to both the global event stream (existing) and per-node event streams (new). This enables dual event tracking without breaking existing functionality.

## Acceptance Criteria
- [ ] Call \`init_node_stream(node.id)\` at start of \`_execute_node()\`
- [ ] Publish each event to both global and node-specific streams
- [ ] Ensure \`node_id\` field is set on all events
- [ ] Verify events appear in both streams during execution
- [ ] No performance degradation (measure execution time)
- [ ] Unit tests pass with >85% coverage

## Technical Details
**File**: \`OpenHands/extensions/uagent_research/orchestrator/tree_orchestrator.py\`  
**Method**: \`_execute_node(self, node: ResearchNode)\`

**Add dual publishing**:
\`\`\`python
async for event in adapter.run(task, context):
    event.node_id = node.id
    event.experiment_id = self.tree.research_id
    await self.event_bus.publish(event)  # Global
    await self.event_bus.publish_node_event(node.id, event)  # Node
\`\`\`

## Testing Requirements
- Unit test: \`test_node_event_dual_publishing()\`
- Integration test: \`test_events_in_both_streams()\`
- Integration test: \`test_multiple_nodes_concurrent_execution()\`
- Performance test: Measure overhead of dual publishing (<5% increase)

## Dependencies
Blocked by: $ISSUE_1

## Definition of Done
- [ ] Code reviewed and approved
- [ ] All unit and integration tests passing
- [ ] Performance benchmarks met
- [ ] No regression in existing orchestrator functionality

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_2"

# Ticket 3: REST API endpoints
ISSUE_3=$(gh issue create \
  --title "[UAGENT-24-3] Implement REST API Endpoints for Node Events" \
  --label "enhancement,backend,api,phase-1,priority:P0" \
  --body "## Story
**Type**: Story  
**Priority**: P0 (Blocker)  
**Story Points**: 3  
**Assignee**: Backend Engineer  
**Phase**: 1 - Backend Infrastructure  
**Depends On**: $ISSUE_1, $ISSUE_2

## Description
Create REST API endpoints to retrieve historical events for specific research nodes. Supports pagination, filtering, and summary statistics.

## Acceptance Criteria
- [ ] Implement \`GET /api/research/experiments/{experiment_id}/nodes/{node_id}/events\`
  - Query params: offset (default 0), limit (default 100, max 500), event_types, since
  - Response: JSON with events array, pagination metadata, total count
  - Status codes: 200 (success), 404 (experiment/node not found), 400 (invalid params)
- [ ] Implement \`GET /api/research/experiments/{experiment_id}/nodes/{node_id}/summary\`
  - Response: event_count, first_event_at, last_event_at, event_type_breakdown, status
  - Status codes: 200 (success), 404 (not found)
- [ ] Add request validation (validate UUIDs, numeric ranges)
- [ ] Add authentication check (user has access to experiment)
- [ ] Error handling with appropriate HTTP status codes
- [ ] API tests pass with >85% coverage

## Technical Details
**File**: \`OpenHands/extensions/uagent_research/uagent_research/api/research_routes.py\`

**Implementation**:
- Use FastAPI router
- Get orchestrator from session manager
- Call \`orchestrator.event_bus.get_node_events(...)\`
- Return paginated response

## Testing Requirements
- API test: \`test_get_node_events_success()\`
- API test: \`test_get_node_events_not_found()\`
- API test: \`test_get_node_events_pagination()\`
- API test: \`test_get_node_events_filtering()\`
- API test: \`test_get_node_summary()\`
- API test: \`test_authentication_required()\`

## Dependencies
Blocked by: $ISSUE_1, $ISSUE_2

## Definition of Done
- [ ] Code reviewed and approved
- [ ] All API tests passing
- [ ] Postman collection created for manual testing
- [ ] API documentation updated

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_3"

# Ticket 4: WebSocket endpoint
ISSUE_4=$(gh issue create \
  --title "[UAGENT-24-4] Implement WebSocket Endpoint for Real-Time Node Events" \
  --label "enhancement,backend,websocket,phase-1,priority:P0" \
  --body "## Story
**Type**: Story  
**Priority**: P0 (Blocker)  
**Story Points**: 5  
**Assignee**: Backend Engineer  
**Phase**: 1 - Backend Infrastructure  
**Depends On**: $ISSUE_1, $ISSUE_2

## Description
Create WebSocket endpoint to stream real-time events for specific research nodes. Supports subscription management and handles connection lifecycle.

## Acceptance Criteria
- [ ] Implement \`WS /ws/research/{experiment_id}/nodes/{node_id}\`
- [ ] Handle client messages:
  - \`subscribe_node\` - Subscribe to node events
  - \`unsubscribe_node\` - Unsubscribe from node
  - \`subscribe_nodes\` - Subscribe to multiple nodes (optional)
- [ ] Send server messages:
  - \`node_event\` - Event occurred in node
  - \`subscription_confirmed\` - Subscription successful
  - \`node_complete\` - Node execution finished
- [ ] Handle connection lifecycle (connect, disconnect, error)
- [ ] Cleanup subscriptions on disconnect (no memory leaks)
- [ ] Support multiple subscriptions per connection
- [ ] WebSocket tests pass

## Technical Details
**File**: \`OpenHands/extensions/uagent_research/uagent_research/api/research_routes.py\`

**Implementation**:
- Use FastAPI WebSocket
- Subscribe to EventBus node events via callback
- Send JSON messages over WebSocket
- Handle \`WebSocketDisconnect\` exception for cleanup

## Testing Requirements
- Integration test: \`test_websocket_subscription()\`
- Integration test: \`test_websocket_event_streaming()\`
- Integration test: \`test_websocket_disconnect_cleanup()\`
- Integration test: \`test_multiple_subscriptions()\`
- Load test: 10 concurrent connections with 1000 events each

## Dependencies
Blocked by: $ISSUE_1, $ISSUE_2

## Definition of Done
- [ ] Code reviewed and approved
- [ ] All WebSocket tests passing
- [ ] Load tests pass (no memory leaks)
- [ ] Connection cleanup verified

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_4"

echo ""
echo "Creating Phase 2 tickets (Frontend State Management)..."

# Ticket 5: Node Event Store
ISSUE_5=$(gh issue create \
  --title "[UAGENT-24-5] Create Node Event Store with Zustand" \
  --label "enhancement,frontend,phase-2,priority:P0" \
  --body "## Story
**Type**: Story  
**Priority**: P0 (Blocker)  
**Story Points**: 5  
**Assignee**: Frontend Engineer  
**Phase**: 2 - Frontend State Management  
**Depends On**: $ISSUE_3, $ISSUE_4

## Description
Create a dedicated Zustand store to manage node-level events and context switching state. Separate from main conversation store to avoid coupling.

## Acceptance Criteria
- [ ] Create \`node-event-store.ts\` with TypeScript interfaces
- [ ] Implement state:
  - contextMode: 'root' | 'node'
  - activeNodeId, activeExperimentId
  - nodeEvents: Map<string, NodeEvent[]>
  - loadingNodes, subscribedNodes
  - eventCounts, hasMore (for pagination)
- [ ] Implement actions:
  - \`switchToNodeContext(nodeId, experimentId)\`
  - \`switchToRootContext()\`
  - \`loadNodeEvents(nodeId, experimentId, offset)\`
  - \`appendNodeEvent(nodeId, event)\`
  - \`subscribeToNode(nodeId, experimentId)\`
  - \`unsubscribeFromNode(nodeId)\`
  - \`clearNodeEvents(nodeId)\`
  - \`clearAllNodeEvents()\`
- [ ] Add convenience selectors (\`useActiveNodeEvents\`, \`useIsNodeContext\`)
- [ ] Store tests pass with >80% coverage

## Technical Details
**File**: \`OpenHands/frontend/src/state/node-event-store.ts\`

**Implementation**:
- Use Zustand with devtools middleware
- Use Map for efficient lookups
- Call REST API in \`loadNodeEvents\`
- Dispatch custom events for WebSocket client

## Testing Requirements
- Unit test: \`test_switch_to_node_context()\`
- Unit test: \`test_switch_to_root_context()\`
- Unit test: \`test_append_node_event()\`
- Unit test: \`test_load_node_events()\`
- Unit test: \`test_pagination_state()\`

## Dependencies
Blocked by: $ISSUE_3, $ISSUE_4

## Definition of Done
- [ ] Code reviewed and approved
- [ ] All unit tests passing
- [ ] TypeScript compilation without errors
- [ ] Zustand devtools working

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_5"

# Ticket 6: WebSocket Client
ISSUE_6=$(gh issue create \
  --title "[UAGENT-24-6] Implement WebSocket Client for Node Events" \
  --label "enhancement,frontend,websocket,phase-2,priority:P0" \
  --body "## Story
**Type**: Story  
**Priority**: P0 (Blocker)  
**Story Points**: 3  
**Assignee**: Frontend Engineer  
**Phase**: 2 - Frontend State Management  
**Depends On**: $ISSUE_4, $ISSUE_5

## Description
Create WebSocket client to handle real-time event streaming for research nodes. Includes reconnection logic and integration with NodeEventStore.

## Acceptance Criteria
- [ ] Create \`node-event-websocket.ts\` class
- [ ] Connect to \`ws://host/ws/research/{experimentId}\`
- [ ] Handle connection lifecycle (open, close, error)
- [ ] Implement reconnection with exponential backoff (max 5 attempts)
- [ ] Listen for custom events: 'subscribe-node', 'unsubscribe-node'
- [ ] Route messages to NodeEventStore:
  - \`node_event\` → \`appendNodeEvent()\`
  - \`subscription_confirmed\` → log confirmation
  - \`node_complete\` → log completion
- [ ] Cleanup on disconnect
- [ ] Unit tests pass

## Technical Details
**File**: \`OpenHands/frontend/src/services/node-event-websocket.ts\`

**Implementation**:
- Use browser WebSocket API
- Listen to \`window\` custom events for sub/unsub
- Call \`useNodeEventStore.getState().appendNodeEvent()\` to update store
- Reconnection delay: \`Math.min(1000 * 2^attempts, 10000)\`

## Testing Requirements
- Unit test: \`test_websocket_connection()\`
- Unit test: \`test_message_routing()\`
- Unit test: \`test_reconnection_logic()\`
- Unit test: \`test_subscription_management()\`
- Integration test: Connect to real backend

## Dependencies
Blocked by: $ISSUE_4, $ISSUE_5

## Definition of Done
- [ ] Code reviewed and approved
- [ ] All unit tests passing
- [ ] Reconnection tested manually
- [ ] No console errors

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_6"

# Ticket 7: Research Tree Integration
ISSUE_7=$(gh issue create \
  --title "[UAGENT-24-7] Integrate Node Context Switching in Research Tree View" \
  --label "enhancement,frontend,phase-2,priority:P0" \
  --body "## Story
**Type**: Story  
**Priority**: P0 (Blocker)  
**Story Points**: 2  
**Assignee**: Frontend Engineer  
**Phase**: 2 - Frontend State Management  
**Depends On**: $ISSUE_5, $ISSUE_6

## Description
Modify the ResearchTreeView component to trigger node context switching when users double-click on nodes.

## Acceptance Criteria
- [ ] Import \`useNodeEventStore\` in \`ResearchTreeView.tsx\`
- [ ] Modify \`onNodeDoubleClick\` callback to call \`switchToNodeContext()\`
- [ ] Extract \`conversationId\` from node metadata
- [ ] Call store action before navigation
- [ ] Verify context switch occurs (manual testing)
- [ ] Verify WebSocket subscribes to node
- [ ] No TypeScript errors

## Technical Details
**File**: \`OpenHands/frontend/src/components/research/ResearchTreeView.tsx\`

**Changes**:
\`\`\`typescript
const conversationId = extractConversationId(metadata);
if (conversationId) {
  useNodeEventStore.getState().switchToNodeContext(node.id, conversationId);
  navigate(\`/conversations/\${conversationId}\`);
}
\`\`\`

## Testing Requirements
- Manual test: Double-click node, verify store state changes
- Manual test: Verify WebSocket connection established
- Manual test: Verify navigation occurs

## Dependencies
Blocked by: $ISSUE_5, $ISSUE_6

## Definition of Done
- [ ] Code reviewed and approved
- [ ] Manual testing completed
- [ ] No console errors
- [ ] Context switching works end-to-end

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_7"

echo ""
echo "Creating Phase 3 tickets (UI Components)..."

# Ticket 8: Context Indicator
ISSUE_8=$(gh issue create \
  --title "[UAGENT-24-8] Create Node Context Indicator Component" \
  --label "enhancement,frontend,ui,phase-3,priority:P1" \
  --body "## Story
**Type**: Story  
**Priority**: P1  
**Story Points**: 3  
**Assignee**: Frontend Engineer  
**Phase**: 3 - UI Components  
**Depends On**: $ISSUE_5

## Description
Create an animated banner component that appears at the top of the screen to indicate when the user is viewing a specific node's context.

## Acceptance Criteria
- [ ] Create \`NodeContextIndicator.tsx\` component
- [ ] Show only when \`isNodeContext === true\`
- [ ] Display node information: title, type, ID
- [ ] Show \"Return to Main Conversation\" button
- [ ] Show close button (X)
- [ ] Use Framer Motion for enter/exit animations
- [ ] Styled with blue gradient background, backdrop blur
- [ ] Call \`switchToRootContext()\` on button click
- [ ] Responsive design (mobile, tablet, desktop)
- [ ] Component tests pass

## Technical Details
**File**: \`OpenHands/frontend/src/components/research/NodeContextIndicator.tsx\`

**Implementation**:
- Use \`useNodeEventStore\` hooks
- Use \`useResearchTreeStore\` to get node details
- Fixed position at top (z-index: 40)
- AnimatePresence for smooth transitions

## Testing Requirements
- Component test: \`test_not_visible_in_root_context()\`
- Component test: \`test_visible_in_node_context()\`
- Component test: \`test_return_button_calls_switchToRoot()\`
- Visual test: Screenshot comparison

## Dependencies
Blocked by: $ISSUE_5

## Definition of Done
- [ ] Code reviewed and approved
- [ ] Component tests passing
- [ ] Visual review completed
- [ ] Responsive on all screen sizes

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_8"

# Ticket 9: Chat Interface modifications
ISSUE_9=$(gh issue create \
  --title "[UAGENT-24-9] Modify Chat Interface for Context Switching" \
  --label "enhancement,frontend,ui,phase-3,priority:P1" \
  --body "## Story
**Type**: Story  
**Priority**: P1  
**Story Points**: 5  
**Assignee**: Frontend Engineer  
**Phase**: 3 - UI Components  
**Depends On**: $ISSUE_5, $ISSUE_8

## Description
Modify the ChatInterface component to display node events when in node context and root conversation when in root context. Add visual feedback and disable input in node context.

## Acceptance Criteria
- [ ] Import \`useNodeEventStore\`, \`useActiveNodeEvents\`, \`useIsNodeContext\` hooks
- [ ] Render \`NodeContextIndicator\` component
- [ ] Switch displayed events based on \`isNodeContext\`:
  - Root context: Show main conversation messages
  - Node context: Show node events (converted to message format)
- [ ] Implement \`formatNodeEvent(event)\` helper function
- [ ] Add top margin to message list when indicator shown
- [ ] Show warning banner when in node context (yellow, informational)
- [ ] Disable chat input when in node context
- [ ] Component tests pass

## Technical Details
**File**: \`OpenHands/frontend/src/components/chat/ChatInterface.tsx\`

**Implementation**:
- Use \`useMemo\` to compute \`displayEvents\`
- Map node events to message-compatible format
- Warning text: \"You're viewing a specific node's context. Messages sent here will go to the main conversation.\"

## Testing Requirements
- Component test: \`test_shows_root_messages_in_root_context()\`
- Component test: \`test_shows_node_events_in_node_context()\`
- Component test: \`test_input_disabled_in_node_context()\`
- Component test: \`test_warning_banner_shown()\`
- Integration test: End-to-end context switching

## Dependencies
Blocked by: $ISSUE_5, $ISSUE_8

## Definition of Done
- [ ] Code reviewed and approved
- [ ] Component tests passing
- [ ] Manual testing completed
- [ ] UX reviewed by designer

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_9"

# Ticket 10: Research Node enhancements
ISSUE_10=$(gh issue create \
  --title "[UAGENT-24-10] Add Context Switch Button to Research Nodes" \
  --label "enhancement,frontend,ui,phase-3,priority:P2" \
  --body "## Story
**Type**: Story  
**Priority**: P2  
**Story Points**: 2  
**Assignee**: Frontend Engineer  
**Phase**: 3 - UI Components  
**Depends On**: $ISSUE_5

## Description
Enhance ResearchNode component with a quick context switch button and visual indicator for the active node.

## Acceptance Criteria
- [ ] Add Eye/MessageSquare icon button to node header
- [ ] Button calls \`switchToNodeContext()\` on click
- [ ] Show Eye icon when node is active, MessageSquare when inactive
- [ ] Add blue ring highlight when node is active
- [ ] Add \"Active\" badge (top-right corner) when node is active
- [ ] Button has hover state and tooltip
- [ ] Responsive and accessible (keyboard navigation)
- [ ] Component tests pass

## Technical Details
**File**: \`OpenHands/frontend/src/components/research/ResearchNode.tsx\`

**Implementation**:
- Use \`useNodeEventStore\` to check if \`activeNodeId === node.id\`
- Extract \`conversationId\` from node metadata
- Apply conditional styling with Tailwind classes

## Testing Requirements
- Component test: \`test_button_switches_context()\`
- Component test: \`test_active_badge_shown()\`
- Component test: \`test_ring_highlight_when_active()\`
- Visual test: Screenshot comparison

## Dependencies
Blocked by: $ISSUE_5

## Definition of Done
- [ ] Code reviewed and approved
- [ ] Component tests passing
- [ ] Keyboard navigation works
- [ ] Tooltip text clear and helpful

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_10"

# Ticket 11: Events Loader
ISSUE_11=$(gh issue create \
  --title "[UAGENT-24-11] Create Node Events Loader Component" \
  --label "enhancement,frontend,ui,phase-3,priority:P2" \
  --body "## Story
**Type**: Story  
**Priority**: P2  
**Story Points**: 2  
**Assignee**: Frontend Engineer  
**Phase**: 3 - UI Components  
**Depends On**: $ISSUE_5

## Description
Create a component to handle pagination of node events with loading states and \"Load More\" functionality.

## Acceptance Criteria
- [ ] Create \`NodeEventsLoader.tsx\` component
- [ ] Show loading spinner when fetching events
- [ ] Show \"Load More Events (X loaded)\" button when \`hasMore === true\`
- [ ] Call \`loadNodeEvents(nodeId, experimentId, offset)\` on button click
- [ ] Hide when all events loaded
- [ ] Component tests pass

## Technical Details
**File**: \`OpenHands/frontend/src/components/research/NodeEventsLoader.tsx\`

**Implementation**:
- Use \`useNodeEventStore\` to get loading state, hasMore, eventCount
- Render at bottom of event list in ChatInterface

## Testing Requirements
- Component test: \`test_shows_loading_spinner()\`
- Component test: \`test_shows_load_more_button()\`
- Component test: \`test_calls_loadNodeEvents_on_click()\`
- Component test: \`test_hides_when_no_more()\`

## Dependencies
Blocked by: $ISSUE_5

## Definition of Done
- [ ] Code reviewed and approved
- [ ] Component tests passing
- [ ] Loading states work correctly

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_11"

# Ticket 12: Layout integration
ISSUE_12=$(gh issue create \
  --title "[UAGENT-24-12] Integrate WebSocket in Conversation Layout" \
  --label "enhancement,frontend,phase-3,priority:P1" \
  --body "## Story
**Type**: Story  
**Priority**: P1  
**Story Points**: 2  
**Assignee**: Frontend Engineer  
**Phase**: 3 - UI Components  
**Depends On**: $ISSUE_6, $ISSUE_9

## Description
Initialize WebSocket connection for node events in the ConversationLayout and handle cleanup on unmount.

## Acceptance Criteria
- [ ] Import \`NodeEventWebSocket\` in \`ConversationLayout.tsx\`
- [ ] Initialize WebSocket on mount (useEffect)
- [ ] Store WebSocket instance in state
- [ ] Disconnect WebSocket on unmount
- [ ] Clear node context on unmount (\`switchToRootContext()\`)
- [ ] Clear node events on conversation change
- [ ] No memory leaks (verify with React DevTools)

## Technical Details
**File**: \`OpenHands/frontend/src/layouts/ConversationLayout.tsx\`

**Implementation**:
- Create WebSocket with \`conversationId\` from route params
- Cleanup in useEffect return function

## Testing Requirements
- Integration test: \`test_websocket_initialized_on_mount()\`
- Integration test: \`test_websocket_disconnected_on_unmount()\`
- Integration test: \`test_context_cleared_on_unmount()\`
- Manual test: Check Network tab for WebSocket connection

## Dependencies
Blocked by: $ISSUE_6, $ISSUE_9

## Definition of Done
- [ ] Code reviewed and approved
- [ ] Integration tests passing
- [ ] No memory leaks detected
- [ ] WebSocket connection visible in DevTools

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_12"

echo ""
echo "Creating Phase 4 tickets (Testing & QA)..."

# Ticket 13: E2E tests
ISSUE_13=$(gh issue create \
  --title "[UAGENT-24-13] Create E2E Tests for Context Switching" \
  --label "testing,phase-4,priority:P1" \
  --body "## Task
**Type**: Task  
**Priority**: P1  
**Story Points**: 3  
**Assignee**: QA Engineer  
**Phase**: 4 - Testing & QA  
**Depends On**: $ISSUE_12

## Description
Create end-to-end tests using Playwright to verify the complete context switching flow works correctly.

## Acceptance Criteria
- [ ] Test: Load research tree and double-click node
- [ ] Test: Verify context indicator appears
- [ ] Test: Verify chat shows node events
- [ ] Test: Click \"Return to Main Conversation\" button
- [ ] Test: Verify context indicator disappears
- [ ] Test: Verify chat shows root messages
- [ ] Test: Switch between multiple nodes
- [ ] Test: Verify WebSocket connection established
- [ ] All E2E tests pass

## Technical Details
**Framework**: Playwright  
**Test file**: \`e2e/node-context-switching.spec.ts\`

**Implementation**:
- Use data-testid attributes for element selection
- Run against local backend + frontend

## Testing Requirements
- E2E test: \`test_double_click_switches_context()\`
- E2E test: \`test_return_to_main_conversation()\`
- E2E test: \`test_multiple_node_switching()\`
- E2E test: \`test_events_stream_in_realtime()\`

## Dependencies
Blocked by: $ISSUE_12

## Definition of Done
- [ ] All E2E tests written and passing
- [ ] Tests run in CI pipeline
- [ ] Tests documented with comments

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_13"

# Ticket 14: Performance testing
ISSUE_14=$(gh issue create \
  --title "[UAGENT-24-14] Performance Testing and Benchmarking" \
  --label "testing,performance,phase-4,priority:P1" \
  --body "## Task
**Type**: Task  
**Priority**: P1  
**Story Points**: 3  
**Assignee**: Backend Engineer + QA Engineer  
**Phase**: 4 - Testing & QA  
**Depends On**: $ISSUE_13

## Description
Conduct performance testing to ensure context switching meets latency requirements and memory usage stays within acceptable limits.

## Acceptance Criteria
- [ ] Test context switch latency (target: <200ms)
- [ ] Test memory usage per node (target: <10MB)
- [ ] Test WebSocket throughput (1000 events/sec)
- [ ] Test with 10 concurrent nodes
- [ ] Test 30-minute session for memory leaks
- [ ] Document performance metrics
- [ ] All benchmarks meet targets

## Technical Details
**Tools**:
- Use Python \`psutil\` for memory measurement
- Use Playwright for UI latency measurement
- Use \`locust\` or \`k6\` for load testing
- Profile with Chrome DevTools Performance tab

## Testing Requirements
- Performance test: \`test_context_switch_latency()\`
- Performance test: \`test_memory_usage_per_node()\`
- Performance test: \`test_websocket_throughput()\`
- Performance test: \`test_no_memory_leaks()\`

## Dependencies
Blocked by: $ISSUE_13

## Definition of Done
- [ ] All performance tests pass
- [ ] Metrics documented in report
- [ ] No performance regressions detected

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_14"

# Ticket 15: Bug fixes
ISSUE_15=$(gh issue create \
  --title "[UAGENT-24-15] Bug Fixes and Polish" \
  --label "bug,polish,phase-4,priority:P1" \
  --body "## Task
**Type**: Task  
**Priority**: P1  
**Story Points**: 3  
**Assignee**: All Engineers  
**Phase**: 4 - Testing & QA  
**Depends On**: $ISSUE_13, $ISSUE_14

## Description
Fix bugs discovered during E2E and performance testing. Polish UI/UX based on manual testing feedback.

## Acceptance Criteria
- [ ] All P0/P1 bugs from testing are fixed
- [ ] All tests passing after bug fixes
- [ ] UX polish items addressed:
  - Animation smoothness
  - Loading states
  - Error messages
  - Responsive design issues
- [ ] Code cleaned up (remove console.logs, commented code)
- [ ] No TypeScript errors
- [ ] No console warnings

## Technical Details
- Create sub-tickets for each bug found
- Prioritize P0 (blocking) and P1 (high) bugs
- P2/P3 bugs can be deferred to follow-up

## Testing Requirements
- Regression test: Verify bug fixes don't break other features
- Manual test: Walk through entire flow multiple times

## Dependencies
Blocked by: $ISSUE_13, $ISSUE_14

## Definition of Done
- [ ] All P0/P1 bugs resolved
- [ ] All tests passing
- [ ] Code quality checks pass
- [ ] No known blocking issues

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_15"

# Ticket 16: Documentation
ISSUE_16=$(gh issue create \
  --title "[UAGENT-24-16] Documentation and Code Review" \
  --label "documentation,phase-4,priority:P1" \
  --body "## Task
**Type**: Task  
**Priority**: P1  
**Story Points**: 2  
**Assignee**: All Engineers  
**Phase**: 4 - Testing & QA  
**Depends On**: $ISSUE_15

## Description
Create comprehensive documentation for the new feature and conduct thorough code review before deployment.

## Acceptance Criteria
- [ ] Update user documentation (how to use node context switching)
- [ ] Update developer documentation (architecture, API contracts)
- [ ] Add inline code comments for complex logic
- [ ] Create Postman collection for API endpoints
- [ ] Update README with new dependencies/setup steps
- [ ] Code review completed by 2+ engineers
- [ ] All review feedback addressed

## Technical Details
**Deliverables**:
- USER_GUIDE.md updated
- DEVELOPER_GUIDE.md updated
- API_REFERENCE.md created
- Postman collection exported
- Code review approved

**Implementation**:
- User docs: Add screenshots/GIFs of feature in action
- Dev docs: Include sequence diagrams for event flow
- API docs: OpenAPI/Swagger spec updated

## Dependencies
Blocked by: $ISSUE_15

## Definition of Done
- [ ] All documentation complete
- [ ] Code review approved by team
- [ ] PR ready to merge

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_16"

echo ""
echo "Creating Phase 5 tickets (Rollout)..."

# Ticket 17: Staging deployment
ISSUE_17=$(gh issue create \
  --title "[UAGENT-24-17] Deploy to Staging Environment" \
  --label "deployment,phase-5,priority:P0" \
  --body "## Task
**Type**: Task  
**Priority**: P0 (Blocker)  
**Story Points**: 2  
**Assignee**: DevOps Engineer + Backend Engineer  
**Phase**: 5 - Rollout  
**Depends On**: $ISSUE_16

## Description
Deploy the complete feature to staging environment for final smoke testing before production rollout.

## Acceptance Criteria
- [ ] Deploy backend changes to staging
- [ ] Deploy frontend build to staging
- [ ] Run database migrations (if any)
- [ ] Verify WebSocket endpoint accessible
- [ ] Run smoke tests on staging
- [ ] Verify feature flag works (disabled by default)
- [ ] No errors in staging logs

## Technical Details
- Use existing CI/CD pipeline
- Backend deployment: Update Docker image, restart services
- Frontend deployment: Build production bundle, upload to CDN
- Test URL: \`https://staging.uagent.example.com\`

## Testing Requirements
- Smoke test: Load research tree
- Smoke test: Double-click node and switch context
- Smoke test: Return to root context
- Smoke test: Check WebSocket connection in Network tab
- Smoke test: Verify events stream in real-time

## Dependencies
Blocked by: $ISSUE_16

## Definition of Done
- [ ] Staging deployment successful
- [ ] All smoke tests pass
- [ ] No errors in logs
- [ ] Feature flag verified

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_17"

# Ticket 18: Production rollout
ISSUE_18=$(gh issue create \
  --title "[UAGENT-24-18] Production Rollout with Gradual Enablement" \
  --label "deployment,phase-5,priority:P0" \
  --body "## Task
**Type**: Task  
**Priority**: P0 (Blocker)  
**Story Points**: 3  
**Assignee**: DevOps Engineer + Product Manager  
**Phase**: 5 - Rollout  
**Depends On**: $ISSUE_17

## Description
Deploy feature to production with feature flag control. Gradually enable for internal users, then small percentage, then all users. Monitor metrics throughout.

## Acceptance Criteria
- [ ] Deploy to production (feature flag OFF)
- [ ] Enable for internal team (10 users)
- [ ] Monitor for 24 hours (no issues)
- [ ] Enable for 25% of users
- [ ] Monitor for 48 hours (no issues)
- [ ] Enable for 100% of users
- [ ] Set up monitoring dashboard
- [ ] Configure alerts for key metrics

## Technical Details
**Feature flag**: \`ENABLE_NODE_CONTEXT_SWITCHING\`  
**Tool**: LaunchDarkly or similar for gradual rollout

**Monitor**:
- API error rate
- WebSocket connection failures
- Context switch latency (p50, p95, p99)
- Memory usage per experiment
- User adoption rate

## Rollout Schedule
- Day 1 Morning: Deploy (flag OFF)
- Day 1 Afternoon: Enable for internal (10 users)
- Day 2: Monitor internal usage
- Day 3: Enable for 25% users
- Day 4-5: Monitor 25% rollout
- Day 6: Enable for 100% users

## Dependencies
Blocked by: $ISSUE_17

## Definition of Done
- [ ] Feature deployed to production
- [ ] Gradual rollout completed
- [ ] Monitoring dashboard active
- [ ] Alerts configured
- [ ] No P0/P1 incidents
- [ ] User feedback collected

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_18"

# Ticket 19: Post-launch monitoring
ISSUE_19=$(gh issue create \
  --title "[UAGENT-24-19] Post-Launch Monitoring and Metrics" \
  --label "monitoring,phase-5,priority:P2" \
  --body "## Task
**Type**: Task  
**Priority**: P2  
**Story Points**: 1  
**Assignee**: Product Manager + Backend Engineer  
**Phase**: 5 - Rollout  
**Depends On**: $ISSUE_18

## Description
Monitor feature usage and performance metrics for 1 week post-launch. Collect user feedback and plan follow-up improvements.

## Acceptance Criteria
- [ ] Track adoption rate (% users who use feature)
- [ ] Track average context switch latency
- [ ] Track WebSocket connection success rate
- [ ] Track memory usage per experiment
- [ ] Collect user feedback via survey
- [ ] Analyze support tickets related to feature
- [ ] Document findings in report
- [ ] Create follow-up tickets for improvements

## Technical Details
**Analytics**: Use Mixpanel, Amplitude, etc.

**Custom events**:
- \`node_context_switched\`
- \`returned_to_root_context\`
- \`load_more_events_clicked\`

**User survey**: NPS score + open feedback

## Metrics to Track
1. Adoption: % of users who switch to node context at least once
2. Engagement: Average number of context switches per session
3. Performance: Average latency (target <200ms)
4. Reliability: WebSocket connection success rate (target >95%)
5. Satisfaction: User survey results (target >80% positive)

## Dependencies
Blocked by: $ISSUE_18

## Definition of Done
- [ ] All metrics tracked for 1 week
- [ ] User feedback collected
- [ ] Findings documented
- [ ] Follow-up tickets created

---
Part of #24" \
  --milestone "Node Context Switching (Issue #24)")

echo "✓ Created: $ISSUE_19"

echo ""
echo "=========================================="
echo "✓ Successfully created 19 GitHub issues!"
echo "=========================================="
echo ""
echo "Issue Numbers Created:"
echo "  Phase 1 (Backend): $ISSUE_1, $ISSUE_2, $ISSUE_3, $ISSUE_4"
echo "  Phase 2 (Frontend State): $ISSUE_5, $ISSUE_6, $ISSUE_7"
echo "  Phase 3 (UI Components): $ISSUE_8, $ISSUE_9, $ISSUE_10, $ISSUE_11, $ISSUE_12"
echo "  Phase 4 (Testing): $ISSUE_13, $ISSUE_14, $ISSUE_15, $ISSUE_16"
echo "  Phase 5 (Rollout): $ISSUE_17, $ISSUE_18, $ISSUE_19"
echo ""
echo "View all issues:"
echo "  gh issue list --milestone 'Node Context Switching (Issue #24)'"
echo ""
echo "Next steps:"
echo "  1. Review created issues: gh issue list"
echo "  2. Assign team members to specific issues"
echo "  3. Start with $ISSUE_1 (EventBus modifications)"
echo ""
