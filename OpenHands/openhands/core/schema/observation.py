from enum import Enum


class ObservationType(str, Enum):
    READ = 'read'
    """The content of a file
    """

    WRITE = 'write'

    EDIT = 'edit'

    BROWSE = 'browse'
    """The HTML content of a URL
    """

    RUN = 'run'
    """The output of a command
    """

    RUN_IPYTHON = 'run_ipython'
    """Runs a IPython cell.
    """

    CHAT = 'chat'
    """A message from the user
    """

    DELEGATE = 'delegate'
    """The result of a task delegated to another agent
    """

    MESSAGE = 'message'

    ERROR = 'error'

    SUCCESS = 'success'

    NULL = 'null'

    THINK = 'think'

    AGENT_STATE_CHANGED = 'agent_state_changed'

    USER_REJECTED = 'user_rejected'

    CONDENSE = 'condense'
    """Result of a condensation operation."""

    RECALL = 'recall'
    """Result of a recall operation. This can be the workspace context, a microagent, or other types of information."""

    MCP = 'mcp'
    """Result of a MCP Server operation"""

    DOWNLOAD = 'download'
    """Result of downloading/opening a file via the browser"""

    TASK_TRACKING = 'task_tracking'
    """Result of a task tracking operation"""
    SUB_AGENT_SPAWNED = 'sub_agent_spawned'
    """A sub-agent was spawned"""

    SUB_AGENT_PROGRESS = 'sub_agent_progress'
    """Progress update from a sub-agent"""

    SUB_AGENT_COMPLETED = 'sub_agent_completed'
    DEPENDENCY_ANALYSIS = 'dependency_analysis'
    """Dependency analysis event"""

    DEPENDENCY_GRAPH = 'dependency_graph'
    """Dependency graph event"""
    """A sub-agent completed its task"""