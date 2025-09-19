# Real-Time Research Streaming System Specification

## Overview

The Real-Time Research Streaming System provides live visualization and monitoring of research processes across all research engines (Deep Research, Code Research, Scientific Research) in the UAgent system. It enables users to observe research progress, LLM interactions, and completion status in real-time through WebSocket-based streaming.

## Requirements

### Functional Requirements

#### FR1: Research Progress Streaming
- **FR1.1**: The system SHALL broadcast real-time progress events for all research activities
- **FR1.2**: Progress events SHALL include event type, session ID, timestamp, progress percentage, and message
- **FR1.3**: The system SHALL support multiple concurrent research sessions with isolated streaming
- **FR1.4**: Progress events SHALL be delivered to all connected WebSocket clients for the specific session

#### FR2: Research Completion Events
- **FR2.1**: The system SHALL broadcast completion events when research activities finish
- **FR2.2**: Completion events SHALL include result summaries and metadata
- **FR2.3**: Tree visualization nodes SHALL transition from "running" to "completed" status upon receiving completion events
- **FR2.4**: Completion events SHALL set progress percentage to 100%

#### FR3: LLM Interaction Streaming
- **FR3.1**: The system SHALL capture and broadcast all LLM interactions during research
- **FR3.2**: LLM streams SHALL include prompt start, token streaming, completion, and error events
- **FR3.3**: LLM interactions SHALL be attributed to the correct research session
- **FR3.4**: Token-level streaming SHALL provide real-time conversation visibility

#### FR4: WebSocket Connection Management
- **FR4.1**: The system SHALL support multiple WebSocket connection types (research, LLM, engines, metrics)
- **FR4.2**: Connections SHALL automatically reconnect on non-clean disconnections
- **FR4.3**: Connection cleanup SHALL prevent memory leaks on component unmounting
- **FR4.4**: The system SHALL handle concurrent connections efficiently

### Non-Functional Requirements

#### NFR1: Performance
- **NFR1.1**: WebSocket message broadcasting SHALL complete within 100ms
- **NFR1.2**: The system SHALL support up to 100 concurrent WebSocket connections
- **NFR1.3**: Memory usage for event history SHALL not exceed 50MB per session
- **NFR1.4**: Event history SHALL be limited to prevent unbounded growth

#### NFR2: Reliability
- **NFR2.1**: WebSocket connections SHALL reconnect automatically within 3 seconds
- **NFR2.2**: Event broadcasting SHALL not fail due to individual connection errors
- **NFR2.3**: The system SHALL maintain event delivery even during temporary disconnections
- **NFR2.4**: Connection errors SHALL not affect research execution

#### NFR3: Scalability
- **NFR3.1**: The system SHALL support horizontal scaling of WebSocket connections
- **NFR3.2**: Event broadcasting SHALL be efficient for large numbers of connections
- **NFR3.3**: Session isolation SHALL prevent cross-session data leakage

## Interface

### WebSocket Endpoints

#### Research Progress Streaming
```
ws://localhost:8000/ws/research/{session_id}
```

**Input Messages:**
- `{"action": "pause"}` - Pause research execution
- `{"action": "resume"}` - Resume research execution
- `{"action": "cancel"}` - Cancel research execution

**Output Messages:**
```typescript
interface ProgressEvent {
  type: "research_event";
  event: {
    event_type: "research_started" | "research_progress" | "research_completed" | "research_error";
    session_id: string;
    timestamp: string;
    data: {
      engine: string;
      phase?: string;
      metadata?: any;
    };
    source: string;
    progress_percentage?: number;
    message?: string;
  };
}
```

#### LLM Interaction Streaming
```
ws://localhost:8000/ws/llm/{session_id}
```

**Input Messages:**
- `{"action": "stream_prompt", "prompt": string, "engine": string}` - Send manual prompt
- `{"action": "get_conversation"}` - Request conversation history

**Output Messages:**
```typescript
interface LLMStreamEvent {
  type: "llm_prompt_start" | "llm_token" | "llm_prompt_complete" | "llm_error";
  session_id: string;
  timestamp: string;
  prompt?: string;
  token?: string;
  error?: string;
  engine?: string;
}
```

### Backend API Components

#### StreamingLLMClient
```python
class StreamingLLMClient:
    def __init__(self, llm_client: LLMClient, session_id: Optional[str] = None)

    async def classify(self, request: str, prompt: str) -> Dict[str, Any]
    async def generate(self, prompt: str, **kwargs) -> str
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]

    def with_session(self, session_id: str) -> 'StreamingLLMClient'
```

#### ResearchProgressTracker
```python
class ResearchProgressTracker:
    async def log_research_started(self, session_id: str, request: str, engine: str)
    async def log_research_completed(self, session_id: str, engine: str, result_summary: str, metadata: Dict[str, Any] = None)
    async def log_research_progress(self, session_id: str, engine: str, phase: str, progress: float, message: str, metadata: Dict[str, Any] = None)
    async def log_error(self, session_id: str, engine: str, error: str, phase: str)
```

### Frontend Components

#### LLMConversationLogs
```typescript
interface LLMConversationLogsProps {
  sessionId: string;
  nodeId?: string;
  engine?: string;
  onPromptRequest?: (prompt: string) => void;
}
```

#### ResearchProgressStream
```typescript
interface ResearchProgressStreamProps {
  sessionId: string;
  onConnectionChange?: (connected: boolean) => void;
  onEventsUpdate?: (events: ProgressEvent[]) => void;
}
```

## Behavior

### Normal Operation Flow

#### Research Execution with Streaming
1. **Initialization**: User initiates research request via smart router
2. **Session Creation**: System generates unique session ID for tracking
3. **WebSocket Connections**: Frontend establishes WebSocket connections for research progress and LLM streaming
4. **Research Start**: Progress tracker broadcasts `research_started` event
5. **LLM Client Replacement**: Smart router temporarily replaces engine LLM clients with StreamingLLMClient
6. **Research Execution**: Engine executes research with real-time streaming of:
   - Progress events (phase transitions, percentage updates)
   - LLM interactions (prompts, tokens, completions)
   - Error events (if any)
7. **Research Completion**: Progress tracker broadcasts `research_completed` event with results
8. **Client Restoration**: Original LLM clients are restored
9. **Frontend Updates**: Tree visualization and chat components update in real-time

#### WebSocket Connection Lifecycle
1. **Connection Establishment**: Frontend initiates WebSocket connection
2. **Authentication**: Connection accepted and added to connection pool
3. **Event History Replay**: Existing events for session are sent to new connections
4. **Live Updates**: Real-time events are broadcast to all connected clients
5. **Disconnection Handling**: Clean disconnections remove connection from pool
6. **Reconnection**: Non-clean disconnections trigger automatic reconnection after 3 seconds

### Edge Cases

#### Connection Failures
- **Scenario**: WebSocket connection fails during research
- **Behavior**: Frontend attempts reconnection every 3 seconds
- **Recovery**: Successfully reconnected clients receive event history replay
- **Impact**: Research execution continues uninterrupted

#### Research Engine Errors
- **Scenario**: Research engine encounters an error
- **Behavior**: Error event is broadcast with error details
- **Recovery**: Frontend displays error message and maintains connection
- **Impact**: Other concurrent research sessions are unaffected

#### Multiple Concurrent Sessions
- **Scenario**: Multiple research sessions running simultaneously
- **Behavior**: Each session maintains isolated WebSocket connections and event streams
- **Recovery**: Session isolation prevents cross-contamination
- **Impact**: Performance scales linearly with number of sessions

## Error Handling

### WebSocket Errors
- **Connection timeout**: Automatic reconnection with exponential backoff
- **Message parsing errors**: Log error and continue processing other messages
- **Broadcasting failures**: Skip failed connections and continue with others
- **Memory overflow**: Implement event history limits and cleanup

### LLM Streaming Errors
- **LLM API errors**: Broadcast error event and continue research
- **Token streaming interruption**: Complete current response and mark as partial
- **Session mismatch**: Validate session ID and drop invalid messages

### Research Progress Errors
- **Missing completion events**: Implement timeout-based completion detection
- **Progress percentage errors**: Validate and clamp progress values to 0-100%
- **Event ordering issues**: Use timestamp-based ordering for event display

## Testing

### Test Scenarios

#### Unit Tests
1. **StreamingLLMClient Functionality**
   - Test LLM interaction broadcasting
   - Test session ID isolation
   - Test error handling and recovery

2. **WebSocket Manager**
   - Test connection management
   - Test event broadcasting
   - Test disconnection handling

3. **Progress Tracker**
   - Test event creation and formatting
   - Test completion event generation
   - Test error event handling

#### Integration Tests
1. **End-to-End Research Streaming**
   - Test complete research workflow with streaming
   - Verify real-time updates in frontend
   - Test completion event propagation

2. **WebSocket Communication**
   - Test real-time message delivery
   - Test connection recovery scenarios
   - Test concurrent session isolation

3. **LLM Interaction Capture**
   - Test automatic LLM interaction broadcasting
   - Test token-level streaming
   - Test engine-specific attribution

#### Performance Tests
1. **Concurrent Connections**
   - Test system behavior with 100+ concurrent connections
   - Measure memory usage and CPU impact
   - Test event broadcasting latency

2. **Event History Management**
   - Test event history limits and cleanup
   - Measure memory usage over time
   - Test history replay performance

### Success Criteria

#### Functional Success
- ✅ Tree nodes transition from "running" to "completed" status
- ✅ LLM interactions appear in real-time in sidebar
- ✅ Research progress updates are visible immediately
- ✅ Multiple concurrent sessions work independently
- ✅ WebSocket reconnection works reliably

#### Performance Success
- Response time for event broadcasting < 100ms
- Memory usage < 50MB per session
- Support for 100+ concurrent connections
- Event history cleanup prevents memory leaks

#### Reliability Success
- WebSocket reconnection within 3 seconds
- No research execution failures due to streaming errors
- Event delivery success rate > 99%
- Session isolation maintains data integrity

## Performance Benchmarks

### Latency Targets
- Event broadcasting: < 100ms
- WebSocket connection establishment: < 500ms
- Event history replay: < 1 second
- Frontend UI updates: < 50ms

### Throughput Targets
- Events per second per session: 100+
- Concurrent sessions supported: 50+
- WebSocket messages per second: 1000+
- LLM tokens per second: 100+

### Resource Usage Targets
- Memory per session: < 50MB
- CPU usage for streaming: < 10%
- Network bandwidth per session: < 1MB/s
- Database operations per event: < 5

## Implementation Notes

### Security Considerations
- Session ID validation prevents unauthorized access
- WebSocket connections are origin-restricted
- Error messages do not expose sensitive information
- Event history is session-isolated

### Monitoring and Observability
- WebSocket connection metrics
- Event broadcasting success rates
- Session lifecycle tracking
- Performance metrics collection

### Deployment Considerations
- WebSocket scaling requires sticky sessions
- Event history requires persistent storage for production
- Connection limits should be configured per environment
- Health checks should include WebSocket functionality