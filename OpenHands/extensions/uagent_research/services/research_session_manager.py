"""
ResearchSessionManager - Lifecycle and state management for research experiments

Provides:
- Experiment registration and lifecycle tracking
- Real-time status aggregation from EventBus
- Control command routing to ControlBus
- Status snapshots for progress queries
"""

import asyncio
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Global singleton instance for ResearchSessionManager
_global_session_manager_instance: Optional['ResearchSessionManager'] = None
_lock = threading.Lock()


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AdapterStatus:
    """Status snapshot for a single adapter"""
    adapter_name: str
    status: str = "idle"
    current_step: str = ""
    last_event_time: str = ""
    total_cost: float = 0.0
    total_tokens: int = 0
    events_count: int = 0


@dataclass
class BranchStatus:
    """Status snapshot for an active branch"""
    branch_id: str
    title: str
    adapter: str
    status: str
    progress: str = ""
    cost: float = 0.0


@dataclass
class ExperimentState:
    """Complete state snapshot for an experiment"""
    experiment_id: str
    status: ExperimentStatus = ExperimentStatus.INITIALIZING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # References to runtime objects
    orchestrator: Any = None
    ws_publisher: Any = None

    # Aggregated statistics
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    running_nodes: int = 0
    pending_nodes: int = 0

    total_cost: float = 0.0
    total_tokens: int = 0

    # Adapter states
    adapter_states: Dict[str, AdapterStatus] = field(default_factory=dict)

    # Active branches
    active_branches: List[BranchStatus] = field(default_factory=list)

    # Last update time
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ResearchSessionManager:
    """
    Manages active research sessions and aggregates status.

    Responsibilities:
    - Register new experiments
    - Track experiment lifecycle
    - Aggregate status from EventBus
    - Route control commands to ControlBus
    - Provide status snapshots for progress queries

    Example:
        manager = ResearchSessionManager(event_bus, control_bus)

        # Register new experiment
        manager.register(experiment_id, orchestrator, ws_publisher)

        # Query status
        status = manager.get_status(experiment_id)

        # Send control command
        await manager.send_control(
            experiment_id,
            ControlMessage(action="pause")
        )
    """

    def __init__(self, event_bus=None, control_bus=None):
        """
        Initialize session manager.

        Args:
            event_bus: EventBus instance for subscribing to research events
            control_bus: ControlBus instance for sending control commands
        """
        self.experiments: Dict[str, ExperimentState] = {}
        self.event_bus = event_bus
        self.control_bus = control_bus

        # Start background event subscription if event_bus provided
        self._event_subscription_task: Optional[asyncio.Task] = None
        if event_bus:
            self._event_subscription_task = asyncio.create_task(
                self._subscribe_events()
            )

        logger.info(f"ResearchSessionManager initialized (instance ID: {id(self)})")

    def register(
        self,
        experiment_id: str,
        orchestrator: Any,
        ws_publisher: Any = None
    ):
        """
        Register new experiment for tracking.

        Args:
            experiment_id: Unique experiment identifier
            orchestrator: TreeSearchOrchestrator instance
            ws_publisher: Optional WebSocket publisher

        Example:
            manager.register("exp_123", orchestrator, ws_publisher)
        """
        if experiment_id in self.experiments:
            logger.warning(f"Experiment {experiment_id} already registered")
            return

        self.experiments[experiment_id] = ExperimentState(
            experiment_id=experiment_id,
            orchestrator=orchestrator,
            ws_publisher=ws_publisher,
            status=ExperimentStatus.RUNNING
        )

        logger.info(f"✅ Registered experiment: {experiment_id}")
        logger.info(f"   Instance ID: {id(self)}")
        logger.info(f"   Total experiments: {len(self.experiments)}")
        logger.info(f"   Active experiments: {list(self.experiments.keys())}")

    def unregister(self, experiment_id: str):
        """
        Unregister and cleanup experiment.

        Args:
            experiment_id: Experiment to remove
        """
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            logger.info(f"Unregistered experiment: {experiment_id}")

    def get_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get aggregated status summary for an experiment.

        Args:
            experiment_id: Experiment to query

        Returns:
            Status dictionary with structure:
            {
                "experiment_id": "exp_...",
                "status": "running|paused|complete|failed",
                "stats": {
                    "total_nodes": 10,
                    "completed": 5,
                    "failed": 1,
                    "running": 4,
                    "pending": 0,
                    "total_cost": 0.25,
                    "total_tokens": 5000
                },
                "adapters": {
                    "deepresearch": {
                        "status": "running",
                        "current_step": "Browsing https://...",
                        "last_event": "2025-10-06T10:30:00",
                        "cost": 0.05,
                        "tokens": 1200
                    },
                    ...
                },
                "active_branches": [
                    {
                        "branch_id": "idea-0-hyp-0",
                        "title": "Test using pgvector",
                        "adapter": "codeact",
                        "status": "running",
                        "progress": "Executing benchmark...",
                        "cost": 0.10
                    }
                ],
                "created_at": "2025-10-06T10:00:00",
                "last_update": "2025-10-06T10:30:00"
            }

        Raises:
            KeyError: If experiment_id not found
        """
        if experiment_id not in self.experiments:
            raise KeyError(f"Experiment {experiment_id} not found")

        state = self.experiments[experiment_id]

        # Build status response
        status = {
            "experiment_id": experiment_id,
            "status": state.status.value,
            "stats": {
                "total_nodes": state.total_nodes,
                "completed": state.completed_nodes,
                "failed": state.failed_nodes,
                "running": state.running_nodes,
                "pending": state.pending_nodes,
                "total_cost": state.total_cost,
                "total_tokens": state.total_tokens
            },
            "adapters": {
                name: {
                    "status": adapter.status,
                    "current_step": adapter.current_step,
                    "last_event": adapter.last_event_time,
                    "cost": adapter.total_cost,
                    "tokens": adapter.total_tokens
                }
                for name, adapter in state.adapter_states.items()
            },
            "active_branches": [
                {
                    "branch_id": branch.branch_id,
                    "title": branch.title,
                    "adapter": branch.adapter,
                    "status": branch.status,
                    "progress": branch.progress,
                    "cost": branch.cost
                }
                for branch in state.active_branches
            ],
            "created_at": state.created_at,
            "last_update": state.last_update
        }

        return status

    def list_active(self) -> List[str]:
        """
        List all active experiment IDs.

        Returns:
            List of experiment IDs
        """
        return [
            exp_id for exp_id, state in self.experiments.items()
            if state.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]
        ]

    async def send_control(self, experiment_id: str, command):
        """
        Send control command to experiment via ControlBus.

        Args:
            experiment_id: Target experiment
            command: ControlMessage instance

        Example:
            from ..control.control_bus import ControlMessage

            await manager.send_control(
                "exp_123",
                ControlMessage(action="pause")
            )
        """
        if not self.control_bus:
            logger.error("ControlBus not configured")
            return

        if experiment_id not in self.experiments:
            logger.warning(f"Experiment {experiment_id} not found for control command")
            return

        await self.control_bus.publish(experiment_id, command)
        logger.info(f"Sent control command to {experiment_id}: {command.action}")

    async def _subscribe_events(self):
        """
        Subscribe to EventBus and update status snapshots in real-time.

        This runs in background and updates experiment states based on
        incoming research events.
        """
        if not self.event_bus:
            return

        logger.info("Starting EventBus subscription for status updates")

        try:
            # Subscribe to all events using a stable subscriber id
            async for event in self.event_bus.subscribe(
                subscriber_id="research-session-manager",
                event_types=None,
                branch_ids=None,
            ):
                await self._handle_event(event)
        except asyncio.CancelledError:
            logger.info("EventBus subscription cancelled")
        except Exception as e:
            logger.error(f"Error in event subscription: {e}", exc_info=True)

    async def _handle_event(self, event):
        """
        Handle incoming research event and update state.

        Args:
            event: ResearchEvent instance
        """
        # Extract experiment_id from event (may need to map from branch_id)
        experiment_id = self._get_experiment_id_from_event(event)

        if not experiment_id or experiment_id not in self.experiments:
            return

        state = self.experiments[experiment_id]

        # Update state based on event type
        event_type = getattr(event, 'type', None) or type(event).__name__

        # Update adapter status
        if hasattr(event, 'adapter_name'):
            adapter_name = event.adapter_name
            if adapter_name not in state.adapter_states:
                state.adapter_states[adapter_name] = AdapterStatus(adapter_name)

            adapter = state.adapter_states[adapter_name]
            adapter.status = "running"
            adapter.last_event_time = datetime.utcnow().isoformat()
            adapter.events_count += 1

            if hasattr(event, 'cost'):
                adapter.total_cost += event.cost
                state.total_cost += event.cost

            if hasattr(event, 'tokens'):
                adapter.total_tokens += event.tokens
                state.total_tokens += event.tokens

            # Update current step from event
            if event_type == 'StepEvent' and hasattr(event, 'action'):
                adapter.current_step = event.action

        # Update active branches tracking
        if hasattr(event, 'branch_id') and event.branch_id:
            branch_id = event.branch_id
            branch = next((b for b in state.active_branches if b.branch_id == branch_id), None)

            if not branch:
                branch = BranchStatus(
                    branch_id=branch_id,
                    title=getattr(event, 'title', branch_id),
                    adapter=getattr(event, 'adapter_name', ''),
                    status='running',
                )
                state.active_branches.append(branch)

            branch.adapter = getattr(event, 'adapter_name', branch.adapter)
            branch.status = 'running'
            branch.progress = getattr(event, 'action', branch.progress)
            if hasattr(event, 'cost') and event.cost:
                branch.cost += event.cost

            if event_type == 'CompleteEvent':
                branch.status = 'complete'
                branch.progress = getattr(event, 'summary', branch.progress)
            elif event_type == 'ErrorEvent':
                branch.status = 'failed'
                branch.progress = getattr(event, 'message', branch.progress)

        # Update node statistics
        if hasattr(event, 'node_id'):
            # Would need to query orchestrator's tree for accurate counts
            # For now, approximate based on event types
            if event_type == 'CompleteEvent':
                state.completed_nodes += 1
                state.running_nodes = max(0, state.running_nodes - 1)
            elif event_type == 'ErrorEvent':
                state.failed_nodes += 1
                state.running_nodes = max(0, state.running_nodes - 1)
            elif event_type == 'NodeCreatedEvent':
                state.total_nodes += 1
                state.pending_nodes += 1
            elif event_type == 'NodeStartedEvent':
                state.running_nodes += 1
                state.pending_nodes = max(0, state.pending_nodes - 1)

        # Update timestamp
        state.last_update = datetime.utcnow().isoformat()

    def _get_experiment_id_from_event(self, event) -> Optional[str]:
        """
        Extract experiment_id from event.

        Args:
            event: ResearchEvent instance

        Returns:
            Experiment ID or None
        """
        # Try direct experiment_id field
        if hasattr(event, 'experiment_id'):
            return event.experiment_id

        # Try extracting from branch_id (format: "exp_{id}_branch_...")
        if hasattr(event, 'branch_id'):
            branch_id = event.branch_id
            # Simple heuristic: find matching experiment by checking if
            # orchestrator has this branch
            for exp_id, state in self.experiments.items():
                # Would need to check orchestrator's tree
                # For now, return first match
                return exp_id

        return None

    def update_experiment_status(
        self,
        experiment_id: str,
        status: ExperimentStatus
    ):
        """
        Manually update experiment status.

        Args:
            experiment_id: Experiment to update
            status: New status
        """
        if experiment_id in self.experiments:
            self.experiments[experiment_id].status = status
            self.experiments[experiment_id].last_update = datetime.utcnow().isoformat()
            logger.info(f"Updated {experiment_id} status to {status.value}")

    async def close(self):
        """Cleanup and stop background tasks"""
        if self._event_subscription_task:
            self._event_subscription_task.cancel()
            try:
                await self._event_subscription_task
            except asyncio.CancelledError:
                pass

        logger.info("ResearchSessionManager closed")


def get_global_session_manager(
    event_bus=None,
    control_bus=None,
    llm=None
) -> 'ResearchSessionManager':
    """
    Get or create the global singleton ResearchSessionManager instance.
    
    Thread-safe singleton implementation using double-check locking pattern.
    
    Args:
        event_bus: Optional EventBus instance (created if not provided)
        control_bus: Optional ControlBus instance (created if not provided)
        llm: Optional LLM instance for research agents
    
    Returns:
        The global ResearchSessionManager singleton instance
    
    Example:
        >>> session_mgr = get_global_session_manager()
        >>> session_mgr.register(experiment_id, orchestrator)
    """
    global _global_session_manager_instance
    
    # Fast path: instance already exists
    if _global_session_manager_instance is not None:
        logger.debug(
            f"Returning existing global session manager singleton (instance ID: {id(_global_session_manager_instance)})"
        )
        return _global_session_manager_instance
    
    # Slow path: need to create instance
    with _lock:
        # Double-check after acquiring lock
        if _global_session_manager_instance is None:
            try:
                # Get event bus if not provided
                if event_bus is None:
                    try:
                        from ..orchestrator.event_bus import get_event_bus
                        event_bus = get_event_bus()
                        logger.info("Using global EventBus for session manager")
                    except Exception as e:
                        logger.warning(f"Could not get EventBus: {e}")
                        event_bus = None
                
                # Get control bus if not provided
                if control_bus is None:
                    try:
                        from ..control.control_bus import ControlBus
                        control_bus = ControlBus()
                        logger.info("Created ControlBus for session manager")
                    except Exception as e:
                        logger.warning(f"Could not create ControlBus: {e}")
                        control_bus = None
                
                # Create the singleton instance
                _global_session_manager_instance = ResearchSessionManager(
                    event_bus=event_bus,
                    control_bus=control_bus
                )
                
                # Store LLM if provided
                if llm:
                    _global_session_manager_instance.llm = llm
                
                logger.info(f"✅ Global ResearchSessionManager singleton created (instance ID: {id(_global_session_manager_instance)})")
                logger.info(f"   Event bus: {'available' if event_bus else 'unavailable'}")
                logger.info(f"   Control bus: {'available' if control_bus else 'unavailable'}")
                logger.info(f"   LLM: {'available' if llm else 'unavailable'}")
                
            except Exception as e:
                logger.error(f"Failed to create global session manager: {e}")
                raise
        
        return _global_session_manager_instance


def reset_global_session_manager():
    """
    Reset the global session manager singleton.
    
    This function is primarily for testing purposes to ensure a clean state
    between test runs.
    
    Example:
        >>> reset_global_session_manager()
        >>> # Now get_global_session_manager() will create a new instance
    """
    global _global_session_manager_instance
    with _lock:
        if _global_session_manager_instance is not None:
            logger.info("Resetting global ResearchSessionManager singleton")
            _global_session_manager_instance = None


# Backward compatibility: keep old function name but use new implementation
def get_session_manager(event_bus=None, control_bus=None) -> ResearchSessionManager:
    """Get global session manager instance (deprecated: use get_global_session_manager)"""
    logger.warning(
        "get_session_manager() is deprecated, use get_global_session_manager() instead"
    )
    return get_global_session_manager(event_bus, control_bus)


# Example usage and tests
async def test_session_manager():
    """Test ResearchSessionManager functionality"""
    from ..control.control_bus import ControlBus, ControlMessage

    control_bus = ControlBus()
    manager = ResearchSessionManager(event_bus=None, control_bus=control_bus)

    # Register experiment
    class MockOrchestrator:
        pass

    orchestrator = MockOrchestrator()
    manager.register("test-exp", orchestrator)

    # Check registration
    assert "test-exp" in manager.experiments
    assert manager.list_active() == ["test-exp"]

    # Get status
    status = manager.get_status("test-exp")
    assert status["experiment_id"] == "test-exp"
    assert status["status"] == "running"
    assert status["stats"]["total_nodes"] == 0

    # Update status manually
    manager.update_experiment_status("test-exp", ExperimentStatus.PAUSED)
    status = manager.get_status("test-exp")
    assert status["status"] == "paused"

    # Send control command
    await manager.send_control(
        "test-exp",
        ControlMessage(action="resume")
    )

    # Unregister
    manager.unregister("test-exp")
    assert "test-exp" not in manager.experiments

    await manager.close()

    logger.info(" All session manager tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_session_manager())
