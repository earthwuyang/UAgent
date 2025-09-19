# API Endpoints Specification

## Overview

The UAgent API provides RESTful endpoints for managing research workflows, engine coordination, and real-time monitoring. The API is built with FastAPI and follows OpenAPI 3.0 standards. It supports both synchronous operations and asynchronous workflows with WebSocket integration for real-time updates.

## Requirements

### Functional Requirements

#### Smart Router API
- **FR-SR-1**: Classify user requests into research engine types
- **FR-SR-2**: Route requests to appropriate engines with context
- **FR-SR-3**: Provide engine capabilities and status information
- **FR-SR-4**: Support manual routing overrides
- **FR-SR-5**: Cache classification results for performance

#### Research Management API
- **FR-RM-1**: Create and manage research sessions
- **FR-RM-2**: Execute research workflows with progress tracking
- **FR-RM-3**: Retrieve research results and reports
- **FR-RM-4**: Support research session cancellation and cleanup
- **FR-RM-5**: Provide research history and analytics

#### Engine-Specific APIs
- **FR-ES-1**: Deep Research endpoints for comprehensive search
- **FR-ES-2**: Code Research endpoints for repository analysis
- **FR-ES-3**: Scientific Research endpoints for experimental workflows
- **FR-ES-4**: OpenHands integration endpoints for code execution

#### Real-time Communication
- **FR-RT-1**: WebSocket endpoints for live progress updates
- **FR-RT-2**: Stream research execution logs
- **FR-RT-3**: Real-time system status monitoring
- **FR-RT-4**: Notification system for completed research

### Non-Functional Requirements

- **NFR-1**: API response time <2 seconds for simple operations
- **NFR-2**: Support 1000+ concurrent API requests
- **NFR-3**: 99.9% API availability and reliability
- **NFR-4**: Comprehensive error handling and user feedback
- **NFR-5**: Rate limiting and abuse prevention
- **NFR-6**: API versioning and backward compatibility

### Performance Requirements

- **PR-1**: Classification endpoints: <500ms response time
- **PR-2**: Research initiation: <2 seconds
- **PR-3**: Status queries: <100ms
- **PR-4**: WebSocket message latency: <50ms
- **PR-5**: Concurrent WebSocket connections: 100+

## Interface

### Base Configuration

```python
# FastAPI application configuration
app = FastAPI(
    title="Universal Agent (UAgent) API",
    description="Intelligent research system with multi-engine orchestration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Smart Router API Endpoints

#### POST /api/router/classify
Classify user request into research engine type.

**Request Body:**
```json
{
  "request": "string",
  "context": {
    "user_id": "string",
    "previous_research": "string"
  },
  "confidence_threshold": 0.8
}
```

**Response:**
```json
{
  "engine": "DEEP_RESEARCH|CODE_RESEARCH|SCIENTIFIC_RESEARCH",
  "confidence": 0.95,
  "reasoning": "string",
  "sub_components": {
    "requires_deep_research": true,
    "requires_code_research": false,
    "requires_experimentation": true
  },
  "estimated_duration": 1800,
  "complexity_score": 0.85
}
```

#### POST /api/router/route-and-execute
Classify and immediately execute research request.

**Request Body:**
```json
{
  "request": "string",
  "execute_immediately": true,
  "priority": "normal|high|urgent",
  "context": {}
}
```

**Response:**
```json
{
  "session_id": "string",
  "classification": {
    "engine": "string",
    "confidence": 0.95
  },
  "status": "started",
  "estimated_completion": "2024-01-01T12:00:00Z"
}
```

#### GET /api/router/engines
List available research engines and their capabilities.

**Response:**
```json
{
  "engines": [
    {
      "name": "DEEP_RESEARCH",
      "description": "Comprehensive web and academic research",
      "capabilities": ["web_search", "academic_search", "fact_checking"],
      "status": "active",
      "average_duration": 300
    },
    {
      "name": "CODE_RESEARCH",
      "description": "Repository analysis and code understanding",
      "capabilities": ["github_search", "code_analysis", "pattern_extraction"],
      "status": "active",
      "average_duration": 180
    },
    {
      "name": "SCIENTIFIC_RESEARCH",
      "description": "Experimental research with hypothesis testing",
      "capabilities": ["hypothesis_generation", "experiment_design", "code_execution"],
      "status": "active",
      "average_duration": 1800
    }
  ]
}
```

#### GET /api/router/status
Get router system status and performance metrics.

**Response:**
```json
{
  "status": "healthy",
  "cache_hit_rate": 0.85,
  "average_classification_time": 245,
  "requests_processed": 1542,
  "accuracy_score": 0.96,
  "last_updated": "2024-01-01T12:00:00Z"
}
```

### Research Management API Endpoints

#### POST /api/research/sessions
Create new research session.

**Request Body:**
```json
{
  "request": "string",
  "engine_type": "DEEP_RESEARCH|CODE_RESEARCH|SCIENTIFIC_RESEARCH",
  "config": {
    "max_iterations": 3,
    "timeout": 1800,
    "priority": "normal"
  },
  "auto_start": true
}
```

**Response:**
```json
{
  "session_id": "string",
  "status": "created",
  "engine_type": "string",
  "estimated_duration": 1800,
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### GET /api/research/sessions
List research sessions with filtering.

**Query Parameters:**
- `status`: Filter by session status
- `engine_type`: Filter by research engine
- `limit`: Number of sessions to return
- `offset`: Pagination offset

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "string",
      "request": "string",
      "engine_type": "string",
      "status": "completed",
      "created_at": "2024-01-01T12:00:00Z",
      "duration": 1245,
      "progress": 100
    }
  ],
  "total": 42,
  "has_more": false
}
```

#### GET /api/research/sessions/{session_id}
Get detailed session information.

**Response:**
```json
{
  "session_id": "string",
  "request": "string",
  "engine_type": "string",
  "status": "in_progress",
  "progress": 65,
  "current_step": "analysis",
  "steps_completed": ["classification", "search", "analysis"],
  "steps_remaining": ["synthesis", "reporting"],
  "created_at": "2024-01-01T12:00:00Z",
  "estimated_completion": "2024-01-01T12:30:00Z",
  "intermediate_results": {},
  "error_message": null
}
```

#### POST /api/research/sessions/{session_id}/execute
Execute or resume research session.

**Request Body:**
```json
{
  "resume_from_step": "string",
  "override_config": {}
}
```

**Response:**
```json
{
  "session_id": "string",
  "status": "executing",
  "execution_id": "string",
  "started_at": "2024-01-01T12:00:00Z"
}
```

#### GET /api/research/sessions/{session_id}/results
Get research results.

**Response:**
```json
{
  "session_id": "string",
  "status": "completed",
  "results": {
    "summary": "string",
    "key_findings": ["string"],
    "sources": ["string"],
    "confidence_score": 0.92,
    "methodology": "string"
  },
  "metadata": {
    "execution_time": 1245,
    "steps_executed": ["string"],
    "resources_used": {}
  },
  "artifacts": [
    {
      "type": "report",
      "format": "markdown",
      "url": "/api/research/sessions/{session_id}/artifacts/report.md"
    }
  ]
}
```

#### DELETE /api/research/sessions/{session_id}
Cancel and cleanup research session.

**Response:**
```json
{
  "session_id": "string",
  "status": "cancelled",
  "cleanup_completed": true,
  "resources_released": true
}
```

### Engine-Specific API Endpoints

#### Deep Research API (/api/engines/deep-research/)

##### POST /api/engines/deep-research/search
Execute comprehensive search.

**Request Body:**
```json
{
  "query": "string",
  "sources": ["web", "academic", "news"],
  "depth": "comprehensive",
  "fact_check": true,
  "max_results": 50
}
```

##### POST /api/engines/deep-research/synthesize
Synthesize information from multiple sources.

**Request Body:**
```json
{
  "sources": [
    {
      "content": "string",
      "url": "string",
      "credibility_score": 0.85
    }
  ],
  "synthesis_type": "comprehensive|summary|analysis"
}
```

#### Code Research API (/api/engines/code-research/)

##### POST /api/engines/code-research/repositories
Search and analyze repositories.

**Request Body:**
```json
{
  "query": "string",
  "languages": ["python", "javascript"],
  "frameworks": ["fastapi", "react"],
  "include_examples": true,
  "analyze_quality": true
}
```

##### POST /api/engines/code-research/patterns
Extract code patterns and best practices.

**Request Body:**
```json
{
  "repositories": ["string"],
  "pattern_types": ["design_patterns", "best_practices", "anti_patterns"],
  "language": "python"
}
```

#### Scientific Research API (/api/engines/scientific-research/)

##### POST /api/engines/scientific-research/hypothesis
Generate research hypotheses.

**Request Body:**
```json
{
  "research_question": "string",
  "background_knowledge": "string",
  "constraints": {}
}
```

##### POST /api/engines/scientific-research/experiment
Design and execute experiments.

**Request Body:**
```json
{
  "hypothesis": "string",
  "methodology": "experimental|observational|meta_analysis",
  "resources": {},
  "execute_immediately": true
}
```

### WebSocket API Endpoints

#### /ws/research/{session_id}
Real-time research progress updates.

**Message Types:**
- `progress`: Progress percentage and current step
- `log`: Execution logs and debug information
- `result`: Intermediate and final results
- `error`: Error messages and recovery suggestions
- `status`: Status changes and notifications

**Example Messages:**
```json
{
  "type": "progress",
  "data": {
    "session_id": "string",
    "progress": 65,
    "current_step": "analysis",
    "step_progress": 80,
    "estimated_remaining": 450
  }
}

{
  "type": "log",
  "data": {
    "level": "info",
    "message": "Starting hypothesis generation",
    "timestamp": "2024-01-01T12:00:00Z",
    "component": "scientific_research"
  }
}

{
  "type": "result",
  "data": {
    "step": "analysis",
    "result": {},
    "confidence": 0.85,
    "next_steps": ["synthesis"]
  }
}
```

#### /ws/system/status
System-wide status monitoring.

**Message Types:**
- `health`: System health and performance metrics
- `engines`: Engine status and availability
- `resources`: Resource usage and capacity
- `alerts`: System alerts and warnings

### Error Responses

All API endpoints follow consistent error response format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Human-readable error message",
    "details": {
      "field": "request",
      "reason": "Request text is required"
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "string"
  }
}
```

**Common Error Codes:**
- `INVALID_REQUEST`: Malformed request data
- `AUTHENTICATION_REQUIRED`: Missing authentication
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `ENGINE_UNAVAILABLE`: Research engine not available
- `SESSION_NOT_FOUND`: Invalid session ID
- `EXECUTION_FAILED`: Research execution failed
- `TIMEOUT_EXCEEDED`: Operation timed out
- `RESOURCE_EXHAUSTED`: System resources exceeded

## Behavior

### Request Processing Flow

1. **Authentication**: Validate API key and user permissions
2. **Rate Limiting**: Check request rate limits
3. **Validation**: Validate request format and parameters
4. **Processing**: Execute requested operation
5. **Response**: Return structured response with appropriate status
6. **Logging**: Log request and response for monitoring

### Asynchronous Operations

For long-running operations:
1. **Immediate Response**: Return session ID and estimated duration
2. **Background Processing**: Execute operation asynchronously
3. **Progress Updates**: Send real-time updates via WebSocket
4. **Completion Notification**: Notify when operation completes
5. **Result Retrieval**: Provide endpoints to retrieve results

### Error Handling

- **Validation Errors**: Return 400 with detailed field errors
- **Authentication Errors**: Return 401 with authentication requirements
- **Authorization Errors**: Return 403 with permission details
- **Not Found Errors**: Return 404 with resource information
- **Rate Limiting**: Return 429 with retry information
- **Server Errors**: Return 500 with error tracking ID

### Caching Strategy

- **Classification Results**: Cache for identical requests (24 hours)
- **Engine Status**: Cache for 5 minutes with invalidation
- **Research Results**: Cache completed results indefinitely
- **API Responses**: Cache GET responses for 1 minute

## Testing

### Test Scenarios

#### Unit Tests

1. **Endpoint Functionality**:
   - Request validation and parsing
   - Response formatting and serialization
   - Error handling and status codes
   - Authentication and authorization

2. **Business Logic**:
   - Research workflow orchestration
   - Engine coordination and routing
   - Progress tracking and notifications
   - Resource management and cleanup

#### Integration Tests

1. **End-to-End Workflows**:
   - Complete research session lifecycle
   - Multi-engine coordination
   - Real-time progress tracking
   - Error recovery and retries

2. **External Service Integration**:
   - Engine communication and coordination
   - WebSocket connection management
   - Database persistence and retrieval
   - Cache invalidation and updates

#### Performance Tests

1. **Load Testing**:
   - Concurrent request handling
   - WebSocket connection scaling
   - Response time under load
   - Resource usage optimization

2. **Stress Testing**:
   - Maximum concurrent sessions
   - Memory and CPU usage limits
   - Database connection pooling
   - Error handling under stress

### Success Criteria

- **Functionality**: All endpoints work as specified
- **Performance**: Meet response time requirements under load
- **Reliability**: <0.1% error rate under normal conditions
- **Compatibility**: OpenAPI 3.0 compliance and client generation
- **Security**: Proper authentication, authorization, and input validation

### Performance Benchmarks

#### Response Times (95th percentile)
- Classification: <500ms
- Session creation: <2s
- Status queries: <100ms
- Result retrieval: <1s
- WebSocket messages: <50ms

#### Throughput
- API requests: 1000+ per second
- Concurrent sessions: 100+
- WebSocket connections: 100+
- Database operations: 500+ per second

## Implementation Notes

### Technology Stack

- **Framework**: FastAPI with async/await
- **Serialization**: Pydantic models for request/response
- **WebSockets**: FastAPI WebSocket support
- **Authentication**: JWT tokens with API keys
- **Documentation**: Automatic OpenAPI generation

### Database Integration

- **Session Storage**: PostgreSQL for research sessions
- **Results Storage**: PostgreSQL with JSON fields
- **Cache Layer**: Redis for API response caching
- **File Storage**: S3-compatible storage for artifacts

### Security Implementation

- **Authentication**: API key and JWT token validation
- **Authorization**: Role-based access control (RBAC)
- **Rate Limiting**: Redis-based rate limiting
- **Input Validation**: Pydantic model validation
- **CORS**: Configured for specific frontend origins

### Monitoring and Observability

- **Request Logging**: Structured logging with correlation IDs
- **Performance Metrics**: Response times, throughput, error rates
- **Health Checks**: Endpoint for load balancer health monitoring
- **Alerting**: Integration with monitoring systems for alerts

### API Versioning

- **URL Versioning**: `/api/v1/` prefix for version 1
- **Backward Compatibility**: Support for previous versions
- **Deprecation**: Clear deprecation warnings and migration guides
- **Documentation**: Version-specific API documentation