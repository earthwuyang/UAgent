#!/bin/bash

# Continue creating remaining issues
# Issues 28 and 29 already created

cd /Users/wuy/Desktop/code/UAgent

echo "Creating remaining issues for Issue #24..."
echo ""

# Issue 3
echo "Creating Issue 3..."
gh issue create \
  --title "[UAGENT-24-3] Implement REST API Endpoints for Node Events" \
  --label "enhancement,backend,phase-1,priority:P0" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Implement REST API endpoints to retrieve historical events for specific research nodes. Supports pagination, filtering, and summary statistics.

**Depends on**: #28, #29

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 3"

# Issue 4
echo "Creating Issue 4..."
gh issue create \
  --title "[UAGENT-24-4] Implement WebSocket Endpoint for Real-Time Node Events" \
  --label "enhancement,backend,websocket,phase-1,priority:P0" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Create WebSocket endpoint to stream real-time events for specific research nodes. Supports subscription management and handles connection lifecycle.

**Depends on**: #28, #29

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 4"

# Issue 5
echo "Creating Issue 5..."
gh issue create \
  --title "[UAGENT-24-5] Create Node Event Store with Zustand" \
  --label "enhancement,frontend,phase-2,priority:P0" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Create a dedicated Zustand store to manage node-level events and context switching state. Separate from main conversation store to avoid coupling.

**Story Points**: 5

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 5"

# Issue 6
echo "Creating Issue 6..."
gh issue create \
  --title "[UAGENT-24-6] Implement WebSocket Client for Node Events" \
  --label "enhancement,frontend,websocket,phase-2,priority:P0" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Create WebSocket client to handle real-time event streaming for research nodes. Includes reconnection logic and integration with NodeEventStore.

**Story Points**: 3

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 6"

# Issue 7
echo "Creating Issue 7..."
gh issue create \
  --title "[UAGENT-24-7] Integrate Node Context Switching in Research Tree View" \
  --label "enhancement,frontend,phase-2,priority:P0" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Modify the ResearchTreeView component to trigger node context switching when users double-click on nodes.

**Story Points**: 2

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 7"

# Issue 8
echo "Creating Issue 8..."
gh issue create \
  --title "[UAGENT-24-8] Create Node Context Indicator Component" \
  --label "enhancement,frontend,ui,phase-3,priority:P1" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Create an animated banner component that appears at the top of the screen to indicate when the user is viewing a specific node's context.

**Story Points**: 3

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 8"

# Issue 9
echo "Creating Issue 9..."
gh issue create \
  --title "[UAGENT-24-9] Modify Chat Interface for Context Switching" \
  --label "enhancement,frontend,ui,phase-3,priority:P1" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Modify the ChatInterface component to display node events when in node context and root conversation when in root context. Add visual feedback and disable input in node context.

**Story Points**: 5

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 9"

# Issue 10
echo "Creating Issue 10..."
gh issue create \
  --title "[UAGENT-24-10] Add Context Switch Button to Research Nodes" \
  --label "enhancement,frontend,ui,phase-3,priority:P2" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Enhance ResearchNode component with a quick context switch button and visual indicator for the active node.

**Story Points**: 2

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 10"

# Issue 11
echo "Creating Issue 11..."
gh issue create \
  --title "[UAGENT-24-11] Create Node Events Loader Component" \
  --label "enhancement,frontend,ui,phase-3,priority:P2" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Create a component to handle pagination of node events with loading states and Load More functionality.

**Story Points**: 2

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 11"

# Issue 12
echo "Creating Issue 12..."
gh issue create \
  --title "[UAGENT-24-12] Integrate WebSocket in Conversation Layout" \
  --label "enhancement,frontend,phase-3,priority:P1" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Initialize WebSocket connection for node events in the ConversationLayout and handle cleanup on unmount.

**Story Points**: 2

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 12"

# Issue 13
echo "Creating Issue 13..."
gh issue create \
  --title "[UAGENT-24-13] Create E2E Tests for Context Switching" \
  --label "testing,phase-4,priority:P1" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Create end-to-end tests using Playwright to verify the complete context switching flow works correctly.

**Story Points**: 3

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 13"

# Issue 14
echo "Creating Issue 14..."
gh issue create \
  --title "[UAGENT-24-14] Performance Testing and Benchmarking" \
  --label "testing,phase-4,priority:P1" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Conduct performance testing to ensure context switching meets latency requirements and memory usage stays within acceptable limits.

**Story Points**: 3
**Target**: <200ms latency, <10MB memory per node

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 14"

# Issue 15
echo "Creating Issue 15..."
gh issue create \
  --title "[UAGENT-24-15] Bug Fixes and Polish" \
  --label "bug,polish,phase-4,priority:P1" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Fix bugs discovered during E2E and performance testing. Polish UI/UX based on manual testing feedback.

**Story Points**: 3

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 15"

# Issue 16
echo "Creating Issue 16..."
gh issue create \
  --title "[UAGENT-24-16] Documentation and Code Review" \
  --label "documentation,phase-4,priority:P1" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Create comprehensive documentation for the new feature and conduct thorough code review before deployment.

**Story Points**: 2

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 16"

# Issue 17
echo "Creating Issue 17..."
gh issue create \
  --title "[UAGENT-24-17] Deploy to Staging Environment" \
  --label "deployment,phase-5,priority:P0" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Deploy the complete feature to staging environment for final smoke testing before production rollout.

**Story Points**: 2

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 17"

# Issue 18
echo "Creating Issue 18..."
gh issue create \
  --title "[UAGENT-24-18] Production Rollout with Gradual Enablement" \
  --label "deployment,phase-5,priority:P0" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Deploy feature to production with feature flag control. Gradually enable for internal users, then small percentage, then all users. Monitor metrics throughout.

**Story Points**: 3

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 18"

# Issue 19
echo "Creating Issue 19..."
gh issue create \
  --title "[UAGENT-24-19] Post-Launch Monitoring and Metrics" \
  --label "monitoring,phase-5,priority:P2" \
  --milestone "Node Context Switching (Issue #24)" \
  --body "Monitor feature usage and performance metrics for 1 week post-launch. Collect user feedback and plan follow-up improvements.

**Story Points**: 1

See PROJECT_TICKETS_ISSUE_24.md for full details.

Part of #24" || echo "Failed to create issue 19"

echo ""
echo "Done! Check created issues with:"
echo "  gh issue list --milestone 'Node Context Switching (Issue #24)'"
