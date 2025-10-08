"""
Multi-Agent Coordinator

Manages the lifecycle of research sub-agents (TreeSearchOrchestrator instances)
and provides a clean API for spawning, tracking, and cleanup.

This coordinator acts as a facade over the existing research infrastructure,
making it easier to integrate with the session and providing proper resource management.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from logging import LoggerAdapter

from extensions.uagent_research.orchestrator.tree_orchestrator import TreeSearchOrchestrator
from extensions.uagent_research.uagent_research.models.research_tree import Budget
from extensions.uagent_research.orchestrator.event_bus import EventBus, get_event_bus
from extensions.uagent_research.control.control_bus import ControlBus
from openhands.events.observation.sub_agent import (
    SubAgentSpawnedObservation,
    SubAgentProgressObservation,
    SubAgentCompletedObservation,
)
from openhands.events.event import EventSource
from openhands.server.session.message_bus import MessageBus
from openhands.events.agent_event import AgentSpawnedEvent, NodeCompleteEvent, ProgressUpdateEvent, CommandEvent


logger = logging.getLogger(__name__)


# Import centralized adapter registration
from extensions.uagent_research.adapters.ensure_adapters import ensure_research_adapters_registered as register_research_adapters




@dataclass
class SubAgent:
    """
    Represents a sub-agent (research orchestrator) with lifecycle tracking.
    """
    id: str  # Unique sub-agent identifier (e.g., "research_abc123")
    type: str  # Sub-agent type ("research", "code", "analysis")
    orchestrator: TreeSearchOrchestrator  # The orchestrator instance
    status: str  # Current status ("running", "paused", "complete", "failed")
    task: Optional[asyncio.Task]  # Background task running the orchestrator
    goal: str  # Research goal/objective
    session_id: str  # Parent session ID
    created_at: float  # Unix timestamp of creation
    max_iterations: int  # Max iterations for this sub-agent
    completed_at: Optional[float] = None  # Unix timestamp of completion


class MultiAgentCoordinator:
    """
    Coordinates multiple research sub-agents for a single session.
    
    Provides lifecycle management, status tracking, command routing,
    and proper resource cleanup for research orchestrators.
    
    Features:
    - Spawn research sub-agents with clean API
    - Track sub-agent lifecycle and status
    - Send control commands (pause, resume, cancel)
    - Query sub-agent status for UI/API
    - Cleanup resources properly
    - MessageBus integration (placeholder for future phases)
    
    Example:
        coordinator = MultiAgentCoordinator(session_id="sess_123")
        
        # Spawn a research sub-agent
        sub_agent_id = await coordinator.spawn_research_agent(
            goal="Research neural architecture search",
            config={"max_iterations": 10, "max_cost": 1.0}
        )
        
        # Get status
        status = coordinator.get_sub_agent_status(sub_agent_id)
        
        # Send command
        await coordinator.send_command(sub_agent_id, "pause")
        
        # Cleanup
        await coordinator.cleanup_sub_agent(sub_agent_id)
    """
    

    def _emit_sub_agent_event(self, observation):
        """
        Emit a sub-agent observation to the session event stream.
        
        Args:
            observation: The observation to emit (SubAgentSpawnedObservation, etc.)
        """
        if self.session and hasattr(self.session, 'agent_session'):
            try:
                self.session.agent_session.event_stream.add_event(observation, EventSource.AGENT)
            except Exception as e:
                self.logger.warning(f"Failed to emit sub-agent event: {e}")
    
    def _should_emit_progress(self, sub_agent_id: str) -> bool:
        """
        Check if enough time has passed to emit progress (throttle to max once per 10s).
        
        Args:
            sub_agent_id: Sub-agent ID
            
        Returns:
            True if progress should be emitted
        """
        now = time.time()
        last_emit = self._last_progress_emit.get(sub_agent_id, 0)
        if now - last_emit >= 10.0:  # 10 second throttle
            self._last_progress_emit[sub_agent_id] = now
            return True
        return False

    def __init__(
        self,
        session_id: str,
        logger: Optional[LoggerAdapter] = None,
        event_bus: Optional[EventBus] = None,
        control_bus: Optional[ControlBus] = None,
        session: Optional[Any] = None  # Session reference for event emission
    ):
        """
        Initialize the multi-agent coordinator.
        
        Args:
            session_id: Parent session ID for tracking
            logger: Optional LoggerAdapter for logging (creates default if not provided)
            event_bus: Optional EventBus instance (uses global if not provided)
            control_bus: Optional ControlBus instance (creates new if not provided)
        """
        self.session_id = session_id
        self.session = session  # Reference to WebSession for event emission
        self.main_agent: Optional[Any] = None  # Placeholder for main conversation agent
        self.sub_agents: Dict[str, SubAgent] = {}  # Track active sub-agents by ID
        
        # Initialize event and control buses
        self.event_bus = event_bus if event_bus else get_event_bus()
        self.control_bus = control_bus if control_bus else ControlBus()
        
        # MessageBus placeholder - integration deferred to subsequent phases
        # TODO: Implement MessageBus for inter-agent communication patterns beyond
        # the current EventBus (event streaming) and ControlBus (command routing).
        # MessageBus will support request/response, pub/sub between agents, and
        # message routing based on agent capabilities/roles.
        # Initialize MessageBus for inter-agent communication
        self.message_bus = MessageBus()
        self.message_bus.bridge_event_bus(self.event_bus)
        self.message_bus.bridge_control_bus(self.control_bus)
        self.message_bus.register_agent(self.session_id, "coordinator", ["orchestration", "lifecycle"])
        
        # Setup logger with session context
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f"{__name__}.{session_id}")

        # Track last progress emit time for throttling (max once per 10s)
        self._last_progress_emit: Dict[str, float] = {}  # sub_agent_id -> timestamp
        self.logger.info(f"MultiAgentCoordinator initialized for session {session_id}")
        
        # Register research adapters to ensure they're available for orchestrator routing
        register_research_adapters()
    
    async def spawn_research_agent(
        self,
        goal: str,
        config: Optional[Dict[str, Any]] = None,
        sub_agent_id: Optional[str] = None
    ) -> str:
        """
        Spawn a new research sub-agent using middleware.
        
        This ensures existing progress/control flows keep working by routing
        through the middleware infrastructure.
        
        Args:
            goal: Research goal/objective
            config: Optional configuration dict
            sub_agent_id: Optional custom sub-agent ID (passed to middleware as session_id)
        
        Returns:
            experiment_id: Unique identifier for the spawned research
        """
        
        logger.info(f"[COORDINATOR] spawn_research_agent() called: goal={goal[:100] if goal else 'N/A'}")
        logger.info(f"[COORDINATOR] Config: {config}")
        try:
            from extensions.uagent_research.middleware.research_middleware import research_middleware
            
            # Extract session_id from config or use coordinator's session_id
            config = config or {}
            session_id = config.get('session_id', self.session_id)
            
            self.logger.info(f"Spawning research via middleware for session {session_id}")
            self.logger.info(f"Goal: {goal[:100]}...")
            
            # Use middleware to start research (this creates orchestrator and background task)
            experiment_id = await research_middleware.start_research(
                goal=goal,
                session_id=session_id,
                research_type='scientific',
                config=config
            )
            logger.info(f"[COORDINATOR] Middleware returned experiment_id: {experiment_id}")
            
            # Track the experiment in coordinator
            if self.track_existing_experiment(experiment_id):
                logger.info(f"[COORDINATOR] Experiment {experiment_id} tracked successfully")
            else:
                logger.error(f"[COORDINATOR] Failed to track experiment {experiment_id}")
        
        if
                self.logger.info(f"Research experiment {experiment_id} spawned and tracked")
            else:
                self.logger.warning(f"Research experiment {experiment_id} spawned but tracking failed")
            
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Failed to spawn research agent: {e}", exc_info=True)
            raise


    async def _run_sub_agent(self, sub_agent_id: str, goal: str):
        """
        Background task that runs the orchestrator.
        
        Args:
            sub_agent_id: Sub-agent ID
            goal: Research goal
        """
        self.logger.info(f"Starting background execution for sub-agent {sub_agent_id}")
        
        sub_agent = self.sub_agents.get(sub_agent_id)
        if not sub_agent:
            self.logger.error(f"Sub-agent {sub_agent_id} not found in tracking dict")
            return
        
        orchestrator = sub_agent.orchestrator
        max_iterations = sub_agent.max_iterations
        
        try:
            self.logger.info(
                f"Running orchestrator for {sub_agent_id}: "
                f"goal={goal[:50]}..., max_iterations={max_iterations}"
            )
            
            # Run the orchestrator
            tree = await orchestrator.run(
                goal=goal,
                max_iterations=max_iterations,
                research_id=sub_agent_id
            )
            
            # Update status to complete
            sub_agent.status = "complete"
            sub_agent.completed_at = time.time()
            
            # Log completion with tree stats
            tree_stats = tree.stats if hasattr(tree, 'stats') else {}
            
            # Emit completion event
            self._emit_sub_agent_event(
                SubAgentCompletedObservation(
                    content=f"Sub-agent {sub_agent_id} completed successfully",
                    sub_agent_id=sub_agent_id,
                    status="complete",
                    result="Success",
                    tree_stats=tree_stats
                )
            )
            self.logger.info(
                f"Sub-agent {sub_agent_id} completed successfully. "
                f"Tree stats: {tree_stats}"
            )
            
        except asyncio.CancelledError:
            self.logger.info(f"Sub-agent {sub_agent_id} was cancelled")
            sub_agent.status = "cancelled"
            sub_agent.completed_at = time.time()
            
            # Emit cancellation event
            self._emit_sub_agent_event(
                SubAgentCompletedObservation(
                    content=f"Sub-agent {sub_agent_id} was cancelled",
                    sub_agent_id=sub_agent_id,
                    status="cancelled",
                    result="Cancelled by user"
                )
            )
            raise
            
        except Exception as e:
            self.logger.error(
                f"Sub-agent {sub_agent_id} failed with error: {e}",
                exc_info=True
            )
            sub_agent.status = "failed"
            sub_agent.completed_at = time.time()
            
            # Emit failure event
            self._emit_sub_agent_event(
                SubAgentCompletedObservation(
                    content=f"Sub-agent {sub_agent_id} failed",
                    sub_agent_id=sub_agent_id,
                    status="failed",
                    result=str(e)
                )
            )
            
        finally:
            # Keep sub-agent in dict for status queries
            self.logger.debug(
                f"Background task for {sub_agent_id} finished. "
                f"Final status: {sub_agent.status}"
            )
    

    def emit_progress_if_needed(self, sub_agent_id: str):
        """
        Emit progress update for a sub-agent if throttle period has passed.
        
        Args:
            sub_agent_id: Sub-agent ID to check and emit progress for
        """
        if not self._should_emit_progress(sub_agent_id):
            return  # Throttled
        
        status = self.get_sub_agent_status(sub_agent_id)
        if not status:
            return
        
        # Emit progress observation
        self._emit_sub_agent_event(
            SubAgentProgressObservation(
                content=f"Sub-agent {sub_agent_id} progress update",
                sub_agent_id=sub_agent_id,
                status=status['status'],
                progress=status['progress'],
                current_task=status['current_task'],
                tree_stats=status.get('tree_stats')
            )
        )

    def get_sub_agent(self, sub_agent_id: str) -> Optional[SubAgent]:
        """
        Get sub-agent by ID.
        
        Args:
            sub_agent_id: Sub-agent ID
        
        Returns:
            SubAgent instance or None if not found
        """
        return self.sub_agents.get(sub_agent_id)
    
    def list_sub_agents(self) -> List[SubAgent]:
        """
        List all sub-agents.
        
        Returns:
            List of all SubAgent instances
        """
        return list(self.sub_agents.values())
    
    def get_all_sub_agents(self) -> List[Dict[str, Any]]:
        """
        Get status for all sub-agents.
        
        Returns:
            List of status dicts for all sub-agents
        """
        return [
            self.get_sub_agent_status(sub_agent_id)
            for sub_agent_id in self.sub_agents.keys()
            if self.get_sub_agent_status(sub_agent_id) is not None
        ]
    
    def track_existing_experiment(self, experiment_id: str) -> bool:
        """
        Track an existing experiment started by middleware.
        
        Pulls orchestrator metadata from middleware and creates a SubAgent
        to track it in the coordinator.
        
        Args:
            experiment_id: Experiment ID from middleware
            
        Returns:
            True if tracking successful, False if experiment not found
        """
        
        logger.info(f"[COORDINATOR] track_existing_experiment() called for {experiment_id}")
        try:
            from extensions.uagent_research.middleware.research_middleware import research_middleware
            
            # Get orchestrator metadata from middleware
            exp_data = research_middleware.get_orchestrator_for_tracking(experiment_id)
        if exp_data:
            logger.info(f"[COORDINATOR] Found experiment data: goal={exp_data.get('goal', 'N/A')[:50] if exp_data.get('goal') else 'N/A'}")
        else:
            logger.error(f"[COORDINATOR] Experiment {experiment_id} not found in middleware")
            return False
            if not exp_data:
                self.logger.warning(f"Cannot track experiment {experiment_id}: not found in middleware")
                return False
            
            # Extract metadata
            orchestrator = exp_data.get('orchestrator')
            goal = exp_data.get('goal', 'Unknown goal')
            session_id = exp_data.get('session_id', self.session_id)
            max_iterations = exp_data.get('max_iterations', 50)
            
            if not orchestrator:
                self.logger.warning(f"Experiment {experiment_id} has no orchestrator")
                return False
            
            # Create SubAgent for tracking
        logger.info(f"[COORDINATOR] Creating SubAgent for tracking")
            sub_agent = SubAgent(
                id=experiment_id,
                type="research",
                orchestrator=orchestrator,
                status="running",
                task=None,  # Middleware manages the task
                goal=goal,
                session_id=session_id,
                created_at=time.time(),
                max_iterations=max_iterations,
                completed_at=None
            )
            
            # Store in tracking dict
            self.sub_agents[experiment_id] = sub_agent
        logger.info(f"[COORDINATOR] SubAgent created and stored. Total sub-agents: {len(self.sub_agents)}")
            
            # Register with MessageBus
            self.message_bus.register_agent(
                agent_id=experiment_id,
                agent_type="research",
                capabilities=["tree_search", "code_execution", "analysis"]
            )
            
            # Emit spawned event (SubAgentSpawnedObservation)
            self._emit_sub_agent_event(
                SubAgentSpawnedObservation(
                    content=f"Research sub-agent {experiment_id} spawned",
                    sub_agent_id=experiment_id,
                    sub_agent_type="research",
                    goal=goal,
                    session_id=session_id
                )
            )
            
            # Emit AgentSpawnedEvent via MessageBus (broadcast)
            import asyncio
            asyncio.create_task(
                self.message_bus.send_message(
                    from_agent_id=self.session_id,
                    to_agent_id=None,  # Broadcast
                    message=AgentSpawnedEvent(
                        from_agent_id=self.session_id,
                        sub_agent_id=experiment_id,
                        agent_type="research",
                        goal=goal,
                        parent_agent_id=self.session_id,
                        capabilities=["tree_search", "code_execution", "analysis"]
                    )
                )
            )
            
            self.logger.info(f"Now tracking experiment {experiment_id} via coordinator")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to track experiment {experiment_id}: {e}", exc_info=True)
            return False
    
    def get_sub_agent_status(self, sub_agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status for a sub-agent.
        
        Args:
            sub_agent_id: Sub-agent ID
        
        Returns:
            Status dict with:
                - id: Sub-agent ID
                - type: Sub-agent type
                - status: Current status
                - goal: Research goal
                - session_id: Parent session ID
                - created_at: Creation timestamp
                - max_iterations: Max iterations
                - tree_stats: Tree statistics (nodes, edges, cost, etc.)
                - is_running: Whether task is still running
            
            Returns None if sub-agent not found
        """
        sub_agent = self.sub_agents.get(sub_agent_id)
        if not sub_agent:
            return None
        
        # Get orchestrator tree and stats
        tree_stats = {}
        if sub_agent.orchestrator and sub_agent.orchestrator.tree:
            tree = sub_agent.orchestrator.tree
            tree_stats = {
                'total_nodes': len(tree.nodes) if hasattr(tree, 'nodes') else 0,
                'total_edges': len(tree.edges) if hasattr(tree, 'edges') else 0,
                'max_depth': tree.calculate_max_depth() if hasattr(tree, 'calculate_max_depth') else 0,
                'total_cost': sub_agent.orchestrator.stats.get('total_cost', 0.0),
                'total_tokens': sub_agent.orchestrator.stats.get('total_tokens', 0),
                'completed_nodes': sub_agent.orchestrator.stats.get('completed_nodes', 0),
                'failed_nodes': sub_agent.orchestrator.stats.get('failed_nodes', 0),
                'iterations': sub_agent.orchestrator.stats.get('iterations', 0)
            }
        
        # Calculate progress from tree stats
        total_nodes = tree_stats.get('total_nodes', 0)
        completed_nodes = tree_stats.get('completed_nodes', 0)
        iterations = tree_stats.get('iterations', 0)
        
        # Progress: ratio of completed nodes to total nodes, or iterations/max_iterations
        if total_nodes > 0:
            progress = min(completed_nodes / max(total_nodes, 1), 1.0)
        else:
            progress = min(iterations / max(sub_agent.max_iterations, 1), 1.0)
        
        # Current task description
        current_task = f"Iteration {iterations}/{sub_agent.max_iterations}"
        if iterations >= sub_agent.max_iterations:
            current_task = "Completed all iterations"
        elif sub_agent.status in ('complete', 'failed', 'cancelled'):
            current_task = f"Task {sub_agent.status}"
        
        # Fix is_running: derive from status, not just task
        is_running = (
            sub_agent.status == 'running' and 
            sub_agent.task is not None and 
            not sub_agent.task.done()
        )
        
        return {
            'id': sub_agent.id,
            'type': sub_agent.type,
            'status': sub_agent.status,
            'goal': sub_agent.goal,
            'session_id': sub_agent.session_id,
            'created_at': sub_agent.created_at,
            'completed_at': sub_agent.completed_at,
            'max_iterations': sub_agent.max_iterations,
            'progress': progress,
            'current_task': current_task,
            'tree_stats': tree_stats,
            'is_running': is_running
        }
    
    async def cleanup_sub_agent(self, sub_agent_id: str) -> bool:
        """
        Cleanup a specific sub-agent and release all resources.
        
        Args:
            sub_agent_id: Sub-agent ID to cleanup
        
        Returns:
            True if cleanup successful, False if sub-agent not found
        """
        sub_agent = self.sub_agents.get(sub_agent_id)
        if not sub_agent:
            self.logger.warning(f"Cannot cleanup sub-agent {sub_agent_id}: not found")
            return False
        
        self.logger.info(f"Cleaning up sub-agent {sub_agent_id}")
        
        try:
            # Cancel and await task with timeout (complete before orchestrator cancellation)
            task_cancelled = False
            if sub_agent.task and not sub_agent.task.done():
                self.logger.debug(f"Cancelling task for sub-agent {sub_agent_id}")
                sub_agent.task.cancel()
                
                try:
                    await asyncio.wait_for(sub_agent.task, timeout=5.0)
                    task_cancelled = True
                    self.logger.debug(f"Task for {sub_agent_id} cancelled successfully")
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Task cancellation for {sub_agent_id} timed out after 5s"
                    )
                    task_cancelled = True  # Consider it done even if timeout
                except asyncio.CancelledError:
                    self.logger.debug(f"Task for {sub_agent_id} cancelled successfully")
                    task_cancelled = True
                except Exception as e:
                    self.logger.error(f"Error during task cancellation: {e}")
                    task_cancelled = True
            
            # Only cancel orchestrator if status is still running or paused
            # (avoid race condition with already-completed orchestrator)
            if sub_agent.orchestrator and sub_agent.status in ("running", "paused"):
                self.logger.debug(f"Cancelling orchestrator for sub-agent {sub_agent_id}")
                try:
                    await sub_agent.orchestrator.cancel()
                except Exception as e:
                    self.logger.error(
                        f"Error cancelling orchestrator for {sub_agent_id}: {e}"
                    )
            
            # Cleanup control bus subscriptions
            try:
                self.control_bus.unsubscribe_all(sub_agent_id)
                self.logger.debug(f"Unsubscribed control bus for {sub_agent_id}")
            except Exception as e:
                self.logger.error(
                    f"Error unsubscribing control bus for {sub_agent_id}: {e}"
                )
            
            # Cleanup event logs
            try:
                self.event_bus.clear_event_log(sub_agent_id)
                self.logger.debug(f"Cleared event log for {sub_agent_id}")
            except Exception as e:
                self.logger.error(
                    f"Error clearing event log for {sub_agent_id}: {e}"
                )
            
            # Set completed_at if not already set
            if sub_agent.completed_at is None:
                sub_agent.completed_at = time.time()
            
            # Remove from tracking dict
            del self.sub_agents[sub_agent_id]
            
            self.logger.info(
                f"Sub-agent {sub_agent_id} cleaned up successfully. "
                f"Remaining sub-agents: {len(self.sub_agents)}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Unexpected error during cleanup of {sub_agent_id}: {e}",
                exc_info=True
            )
            return False
    
    async def cleanup_all_sub_agents(self):
        """
        Cleanup all sub-agents and release resources.
        
        This is idempotent and safe to call multiple times.
        """
        sub_agent_ids = list(self.sub_agents.keys())  # Copy to avoid modification during iteration
        
        if not sub_agent_ids:
            self.logger.debug("No sub-agents to cleanup")
            return
        
        self.logger.info(f"Cleaning up all {len(sub_agent_ids)} sub-agents")
        
        cleanup_count = 0
        for sub_agent_id in sub_agent_ids:
            try:
                success = await self.cleanup_sub_agent(sub_agent_id)
                if success:
                    cleanup_count += 1
            except Exception as e:
                self.logger.error(
                    f"Error cleaning up sub-agent {sub_agent_id}: {e}",
                    exc_info=True
                )
        
        self.logger.info(
            f"Cleanup complete: {cleanup_count}/{len(sub_agent_ids)} sub-agents cleaned up"
        )
    
    async def send_command(
        self,
        sub_agent_id: str,
        command: str,
        payload: Optional[Dict[str, Any]] = None
    ):
        """
        Send control command to a sub-agent.
        
        Args:
            sub_agent_id: Target sub-agent ID
            command: Command action (pause, resume, cancel, steer, etc.)
            payload: Optional command payload
        
        Example:
            # Pause research
            await coordinator.send_command("research_abc123", "pause")
            
            # Steer research with guidance
            await coordinator.send_command(
                "research_abc123",
                "steer",
                payload={"text": "Focus on recent papers"}
            )
        """
        from extensions.uagent_research.control.control_bus import ControlMessage
        
        # Verify sub-agent exists
        if sub_agent_id not in self.sub_agents:
            self.logger.warning(
                f"Cannot send command to {sub_agent_id}: sub-agent not found"
            )
            return
        
        # Create control message
        message = ControlMessage(
            action=command,
            payload=payload or {},
            target={},
            sender="coordinator"
        )
        
        # Send via control bus (for backward compatibility)
        try:
            await self.control_bus.publish(sub_agent_id, message)
            self.logger.info(
                f"Command sent to sub-agent {sub_agent_id} via ControlBus: "
                f"action={command}, payload={payload}"
            )
        except Exception as e:
            self.logger.error(
                f"Error sending command via ControlBus to {sub_agent_id}: {e}",
                exc_info=True
            )
        
        # Also send via MessageBus
        try:
            await self.message_bus.send_message(
                from_agent_id=self.session_id,
                to_agent_id=sub_agent_id,
                message=CommandEvent(
                    from_agent_id=self.session_id,
                    to_agent_id=sub_agent_id,
                    command_type=command,
                    target_agent_id=sub_agent_id,
                    payload=payload or {}
                )
            )
            self.logger.info(
                f"Command sent to sub-agent {sub_agent_id} via MessageBus: "
                f"action={command}"
            )
        except Exception as e:
            self.logger.error(
                f"Error sending command via MessageBus to {sub_agent_id}: {e}",
                exc_info=True
            )
