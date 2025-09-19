# OpenHands Integration Specification

## Overview

The OpenHands Integration provides a secure, isolated code execution environment for the UAgent system. It serves as the foundation for the Scientific Research Engine's ability to generate, execute, and test code as part of experimental research workflows. This integration is based on the OpenHands CLI architecture and provides workspace management, code execution, and real-time monitoring capabilities.

## Requirements

### Functional Requirements

#### Workspace Management
- **FR-WM-1**: Create isolated workspaces for each research session
- **FR-WM-2**: Manage workspace lifecycle (create, configure, cleanup)
- **FR-WM-3**: Support concurrent workspaces with resource isolation
- **FR-WM-4**: Workspace persistence for long-running experiments
- **FR-WM-5**: File system isolation and security boundaries
- **FR-WM-6**: Workspace templates for different research types

#### Code Execution
- **FR-CE-1**: Execute Python code in isolated environments
- **FR-CE-2**: Support multiple programming languages (Python, JavaScript, etc.)
- **FR-CE-3**: Real-time output streaming during execution
- **FR-CE-4**: Error capture and detailed debugging information
- **FR-CE-5**: Execution timeout and resource limits
- **FR-CE-6**: Interactive code execution with state preservation

#### Session Management
- **FR-SM-1**: Create and manage research sessions
- **FR-SM-2**: Session state persistence and restoration
- **FR-SM-3**: Session configuration and customization
- **FR-SM-4**: Multi-step execution workflows
- **FR-SM-5**: Session monitoring and health checks
- **FR-SM-6**: Graceful session termination and cleanup

#### Code Generation
- **FR-CG-1**: LLM-powered code generation for experiments
- **FR-CG-2**: Code template management and customization
- **FR-CG-3**: Automatic dependency management and installation
- **FR-CG-4**: Code quality validation and testing
- **FR-CG-5**: Integration with existing codebases
- **FR-CG-6**: Code documentation generation

### Non-Functional Requirements

- **NFR-1**: Support 50+ concurrent isolated workspaces
- **NFR-2**: Code execution startup time <5 seconds
- **NFR-3**: Memory isolation between workspaces
- **NFR-4**: Security: prevent access to host system
- **NFR-5**: Reliability: >99.9% uptime for workspace management
- **NFR-6**: Resource efficiency: <100MB base memory per workspace

### Performance Requirements

- **PR-1**: Workspace creation time <10 seconds
- **PR-2**: Code execution latency <1 second for simple scripts
- **PR-3**: File I/O operations <100ms for typical files
- **PR-4**: Session state save/restore <2 seconds
- **PR-5**: Concurrent execution support: 20+ parallel code executions

## Interface

### Workspace Management Interface

```python
@dataclass
class WorkspaceConfig:
    workspace_id: str
    base_path: str
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_total_size: int = 100 * 1024 * 1024  # 100MB
    timeout: int = 300  # 5 minutes
    python_path: str = "/usr/bin/python3"
    allowed_commands: List[str] = None

@dataclass
class WorkspaceStatus:
    workspace_id: str
    status: str  # created, active, stopped, error
    created_at: str
    last_activity: str
    file_count: int
    total_size: int
    processes: List[Dict[str, Any]]

class WorkspaceManager:
    async def create_workspace(self, research_id: str = None, config: Dict[str, Any] = None) -> WorkspaceConfig
    async def get_workspace_status(self, workspace_id: str) -> Optional[WorkspaceStatus]
    async def write_file(self, workspace_id: str, file_path: str, content: str) -> bool
    async def read_file(self, workspace_id: str, file_path: str) -> Optional[str]
    async def list_files(self, workspace_id: str, directory: str = ".") -> List[Dict[str, Any]]
    async def cleanup_workspace(self, workspace_id: str) -> bool
```

### Code Execution Interface

```python
@dataclass
class ExecutionResult:
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    files_created: List[str]
    files_modified: List[str]

@dataclass
class ExecutionCommand:
    command: str
    working_directory: str = "."
    timeout: int = 300
    capture_output: bool = True
    shell: bool = True
    env_vars: Dict[str, str] = None

class CodeExecutor:
    async def execute_python_code(self, workspace_id: str, code: str, file_name: str = None, timeout: int = 300) -> ExecutionResult
    async def execute_bash_command(self, workspace_id: str, command: str, timeout: int = 300) -> ExecutionResult
    async def execute_command(self, workspace_id: str, command: ExecutionCommand) -> ExecutionResult
    async def execute_jupyter_notebook(self, workspace_id: str, notebook_content: str, timeout: int = 600) -> List[ExecutionResult]
    async def stream_execution(self, workspace_id: str, command: ExecutionCommand) -> AsyncGenerator[Dict[str, Any], None]
```

### Session Management Interface

```python
@dataclass
class SessionConfig:
    session_id: str
    research_type: str  # deep_research, code_research, scientific_research
    max_iterations: int = 10
    timeout_per_step: int = 300
    enable_code_execution: bool = True
    workspace_config: Dict[str, Any] = None

@dataclass
class SessionState:
    session_id: str
    status: str  # created, running, paused, completed, error
    current_step: int
    total_steps: int
    workspace_id: str
    created_at: float
    last_activity: float
    error_message: Optional[str] = None

class OpenHandsClient:
    async def create_session(self, research_type: str, session_id: str = None, config: Dict[str, Any] = None) -> SessionConfig
    async def get_session_state(self, session_id: str) -> Optional[SessionState]
    async def get_workspace_status(self, session_id: str) -> Optional[WorkspaceStatus]
    async def generate_and_execute_code(self, request: CodeGenerationRequest, llm_client=None) -> CodeGenerationResult
    async def execute_code_streaming(self, session_id: str, code: str, language: str = "python") -> AsyncGenerator[Dict[str, Any], None]
    async def close_session(self, session_id: str) -> bool
```

### Code Generation Interface

```python
@dataclass
class CodeGenerationRequest:
    session_id: str
    task_description: str
    context: Dict[str, Any] = None
    execute_immediately: bool = True
    language: str = "python"
    timeout: int = 300

@dataclass
class CodeGenerationResult:
    session_id: str
    task_description: str
    generated_code: str
    execution_result: Optional[ExecutionResult] = None
    analysis: str = ""
    success: bool = False
    next_steps: List[str] = None
```

### Error Conditions

- **EC-1**: `WorkspaceCreationError` - Failed to create workspace
- **EC-2**: `ExecutionTimeoutError` - Code execution exceeded timeout
- **EC-3**: `ResourceLimitError` - Workspace exceeded resource limits
- **EC-4**: `SecurityViolationError` - Attempted unauthorized access
- **EC-5**: `SessionNotFoundError` - Invalid session ID
- **EC-6**: `CodeGenerationError` - Failed to generate valid code
- **EC-7**: `WorkspaceCleanupError` - Failed to cleanup workspace resources

## Behavior

### Workspace Lifecycle

1. **Creation**: Initialize isolated workspace with standard structure
2. **Configuration**: Set up environment, dependencies, and security boundaries
3. **Usage**: Execute code, manage files, monitor resource usage
4. **Persistence**: Save state and intermediate results
5. **Cleanup**: Remove temporary files and release resources

### Code Execution Flow

1. **Validation**: Validate code and execution parameters
2. **Environment Setup**: Prepare execution environment with dependencies
3. **Execution**: Run code with monitoring and resource limits
4. **Output Capture**: Collect stdout, stderr, and file system changes
5. **Result Analysis**: Analyze execution results and extract insights
6. **State Management**: Update session state and preserve context

### Session Management Flow

1. **Session Creation**: Initialize session with workspace and configuration
2. **Context Setup**: Establish research context and objectives
3. **Execution Loop**: Iterative code generation and execution
4. **Progress Tracking**: Monitor progress and update session state
5. **Error Handling**: Handle failures and implement recovery strategies
6. **Session Termination**: Clean up resources and save final results

### Security Model

1. **Process Isolation**: Each workspace runs in isolated processes
2. **File System Boundaries**: Restrict access to workspace directories only
3. **Network Restrictions**: Limited network access with allowlist
4. **Resource Limits**: CPU, memory, and disk usage limits
5. **Command Filtering**: Whitelist of allowed commands and operations
6. **Audit Logging**: Comprehensive logging of all operations

### Error Handling and Recovery

- **Timeout Management**: Graceful handling of execution timeouts
- **Resource Exhaustion**: Automatic cleanup when limits exceeded
- **Process Failures**: Recovery mechanisms for crashed processes
- **Network Issues**: Retry logic for network-dependent operations
- **Corruption Detection**: Validation of workspace integrity
- **Rollback Mechanisms**: Restore previous known good states

## Testing

### Test Scenarios

#### Unit Tests

1. **Workspace Management Tests**:
   - Workspace creation and configuration
   - File operations (read, write, list)
   - Resource limit enforcement
   - Cleanup and garbage collection

2. **Code Execution Tests**:
   - Python code execution with various outputs
   - Command execution with different parameters
   - Timeout handling and resource limits
   - Error condition testing

3. **Session Management Tests**:
   - Session creation and configuration
   - State persistence and restoration
   - Multi-step execution workflows
   - Session cleanup and resource release

#### Integration Tests

1. **End-to-End Workflow Tests**:
   - Complete research session lifecycle
   - Code generation and execution pipeline
   - Multi-workspace coordination
   - Long-running experiment scenarios

2. **Performance Tests**:
   - Concurrent workspace creation
   - Parallel code execution
   - Memory usage under load
   - Resource cleanup efficiency

3. **Security Tests**:
   - Isolation boundary testing
   - Privilege escalation attempts
   - File system access restrictions
   - Network access limitations

#### Stress Tests

1. **Resource Stress Tests**:
   - Maximum concurrent workspaces
   - High-memory code execution
   - CPU-intensive operations
   - Disk space exhaustion scenarios

2. **Failure Recovery Tests**:
   - Process crash recovery
   - Workspace corruption handling
   - Network failure scenarios
   - System resource exhaustion

### Success Criteria

- **Reliability**: >99.9% successful workspace operations
- **Performance**: Meet all specified timing requirements
- **Security**: Zero security violations in testing
- **Resource Management**: Proper cleanup in >99% of cases
- **Concurrency**: Support target concurrent load without failures

### Performance Benchmarks

#### Workspace Operations
- Creation: <10 seconds
- File operations: <100ms for typical files
- Cleanup: <5 seconds
- Status queries: <50ms

#### Code Execution
- Simple Python scripts: <1 second
- Complex computations: <60 seconds (with proper progress)
- Jupyter notebook cells: <30 seconds per cell
- Interactive sessions: <200ms response time

#### Session Management
- Session creation: <5 seconds
- State save/restore: <2 seconds
- Multi-step workflows: <10% overhead
- Concurrent sessions: Linear scaling up to 50 sessions

## Implementation Notes

### Architecture Components

#### WorkspaceManager
- Manages isolated workspace creation and lifecycle
- Handles file system operations with security boundaries
- Monitors resource usage and enforces limits
- Provides cleanup and garbage collection

#### CodeExecutor
- Executes code in controlled environments
- Captures output and monitors execution
- Handles timeouts and resource limits
- Supports multiple execution modes (script, interactive, notebook)

#### OpenHandsClient
- Main integration point for the UAgent system
- Coordinates workspace and execution operations
- Manages session lifecycle and state
- Provides high-level API for research workflows

### Security Implementation

#### Process Isolation
- Use containerization or sandboxing for process isolation
- Implement proper user/group permissions
- Network namespace isolation where possible
- Resource cgroups for CPU/memory limits

#### File System Security
- Chroot or similar file system isolation
- Read-only base system with writable workspace areas
- File type restrictions and size limits
- Automatic cleanup of temporary files

#### Command Filtering
- Whitelist of allowed commands and operations
- Parameter validation for dangerous operations
- Monitoring and logging of all command executions
- Automatic termination of suspicious activities

### Integration Points

#### Scientific Research Engine
- Primary consumer of OpenHands integration
- Uses for experimental code generation and execution
- Requires long-running session support
- Needs advanced error handling and recovery

#### Code Research Engine
- Uses for code analysis and testing
- Requires repository cloning and analysis capabilities
- Needs support for multiple programming languages
- Benefits from caching and reuse mechanisms

#### Deep Research Engine
- May use for data processing and analysis
- Requires support for web scraping and data collection
- Needs document processing capabilities
- Benefits from parallel execution support

### Monitoring and Observability

#### Performance Metrics
- Workspace creation/cleanup times
- Code execution performance
- Resource utilization tracking
- Error rates and failure modes

#### Security Monitoring
- Failed access attempts
- Resource limit violations
- Suspicious command executions
- Security policy violations

#### Business Metrics
- Research session success rates
- Code generation effectiveness
- User satisfaction with execution environment
- System utilization and capacity planning

### Configuration Management

#### Environment Configuration
- Python version and package management
- Language runtime configuration
- Tool and utility availability
- Resource limit settings

#### Security Configuration
- Allowed commands and operations
- Network access policies
- File system restrictions
- Execution timeout settings

#### Integration Configuration
- LLM client settings for code generation
- Research engine coordination settings
- Session management parameters
- Error handling and retry policies