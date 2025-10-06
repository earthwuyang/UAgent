"""
Research Middleware

Automatically triggers research mode for complex tasks.
Intercepts user messages and routes to appropriate execution mode.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from ..classifier.task_classifier import task_classifier, TaskType
from ..orchestrator.tree_orchestrator import TreeSearchOrchestrator
from ..uagent_research.models.research_tree import Budget

try:
    from ..config import ENABLE_AUTO_RESEARCH_TRIGGER, RESEARCH_CONFIDENCE_THRESHOLD
except ImportError:
    # Fallback if config not available
    ENABLE_AUTO_RESEARCH_TRIGGER = False
    RESEARCH_CONFIDENCE_THRESHOLD = 0.7

logger = logging.getLogger(__name__)


class ResearchMiddleware:
    """
    Middleware that intercepts user messages and triggers research mode when needed.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        enable_auto_trigger: bool = True,
    ):
        """
        Initialize research middleware.

        Args:
            confidence_threshold: Minimum confidence to auto-trigger research (0-1)
            enable_auto_trigger: Enable automatic research triggering
        """
        self.confidence_threshold = confidence_threshold
        self.enable_auto_trigger = enable_auto_trigger
        self.active_orchestrators: Dict[str, TreeSearchOrchestrator] = {}

    async def process_message(
        self,
        user_message: str,
        session_id: str,
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process user message and determine execution mode.

        Args:
            user_message: User's input message
            session_id: Conversation/session ID
            conversation_metadata: Optional conversation metadata

        Returns:
            Dict with:
            - mode: "research" or "normal"
            - should_trigger_research: bool
            - task_type: TaskType
            - confidence: float
            - reasoning: Dict
            - experiment_id: Optional[str] (if research triggered)
        """
        if not self.enable_auto_trigger:
            return {
                'mode': 'normal',
                'should_trigger_research': False,
                'task_type': TaskType.SIMPLE,
                'confidence': 1.0,
                'reasoning': {'decision': 'Auto-trigger disabled'},
            }

        # Classify the task
        should_trigger, task_type, confidence, reasoning = task_classifier.should_trigger_research(
            user_message,
            confidence_threshold=self.confidence_threshold
        )

        result = {
            'mode': 'research' if should_trigger else 'normal',
            'should_trigger_research': should_trigger,
            'task_type': task_type,
            'confidence': confidence,
            'reasoning': reasoning,
        }

        # If research should be triggered, start it (non-blocking)
        if should_trigger:
            logger.info(f"Triggering research mode for session {session_id}")
            logger.info(f"Task type: {task_type.value}, confidence: {confidence:.2f}")
            logger.info(f"User message: {user_message[:100]}...")

            try:
                # Start research asynchronously without blocking
                experiment_id = await self.start_research(
                    goal=user_message,
                    session_id=session_id,
                    research_type='scientific',  # Can be made configurable
                    config=conversation_metadata or {}
                )
                result['experiment_id'] = experiment_id
                result['status'] = 'research_started'

                logger.info(f"Research started successfully: experiment_id={experiment_id}")

            except Exception as e:
                # Log error but don't fail - research is optional
                logger.error(f"Failed to start research: {str(e)}", exc_info=True)
                # Keep research mode indicator but mark as failed
                result['status'] = 'research_failed'
                result['error'] = str(e)
                # Don't change mode to 'normal' - let user know research was attempted

        return result

    async def start_research(
        self,
        goal: str,
        session_id: str,
        research_type: str = 'scientific',
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a research experiment.

        Args:
            goal: Research goal
            session_id: Session ID
            research_type: Type of research (scientific, code, roma)
            config: Optional configuration

        Returns:
            experiment_id: ID of the created experiment
        """
        import time
        import uuid

        # Generate experiment ID
        experiment_id = f"exp_{session_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Create orchestrator configuration
        config = config or {}
        max_iterations = config.get('max_iterations', 50)
        max_cost = config.get('max_cost', 10.0)
        max_parallel = config.get('max_parallel', 3)

        budget = Budget(
            max_iterations=max_iterations,
            max_cost=max_cost,
            max_tokens=config.get('max_tokens', 100000),
            deadline=None,
        )

        # Create orchestrator
        orchestrator = TreeSearchOrchestrator(
            max_parallel=max_parallel,
            budget=budget,
        )

        # Store orchestrator and goal
        self.active_orchestrators[experiment_id] = {
            'orchestrator': orchestrator,
            'goal': goal,
            'session_id': session_id,
            'max_iterations': max_iterations,
        }

        # Start research in background
        asyncio.create_task(self._run_research(experiment_id))

        return experiment_id

    async def _run_research(
        self,
        experiment_id: str,
    ):
        """
        Run research in background.

        Args:
            experiment_id: Experiment ID
        """
        try:
            logger.info(f"Starting research execution: {experiment_id}")

            exp_data = self.active_orchestrators.get(experiment_id)
            if not exp_data:
                logger.error(f"Experiment data not found: {experiment_id}")
                return

            orchestrator = exp_data['orchestrator']
            goal = exp_data['goal']
            max_iterations = exp_data['max_iterations']

            # Run orchestrator
            tree = await orchestrator.run(
                goal=goal,
                max_iterations=max_iterations
            )

            logger.info(f"Research completed: {experiment_id}")
            logger.info(f"Tree stats: {tree.stats if hasattr(tree, 'stats') else 'N/A'}")

        except Exception as e:
            logger.error(f"Research failed: {experiment_id}, error: {str(e)}", exc_info=True)

        finally:
            # Cleanup
            if experiment_id in self.active_orchestrators:
                del self.active_orchestrators[experiment_id]

    def get_orchestrator(self, experiment_id: str) -> Optional[TreeSearchOrchestrator]:
        """Get active orchestrator by experiment ID"""
        exp_data = self.active_orchestrators.get(experiment_id)
        return exp_data['orchestrator'] if exp_data else None

    def cancel_research(self, experiment_id: str) -> bool:
        """
        Cancel active research.

        Args:
            experiment_id: Experiment ID to cancel

        Returns:
            True if cancelled, False if not found
        """
        exp_data = self.active_orchestrators.get(experiment_id)
        if exp_data:
            orchestrator = exp_data['orchestrator']
            orchestrator.cancel()
            del self.active_orchestrators[experiment_id]
            logger.info(f"Cancelled research: {experiment_id}")
            return True
        return False


# Global middleware instance
# Configured via config.py or environment variables
# To enable: export ENABLE_AUTO_RESEARCH_TRIGGER=true
research_middleware = ResearchMiddleware(
    confidence_threshold=RESEARCH_CONFIDENCE_THRESHOLD,
    enable_auto_trigger=ENABLE_AUTO_RESEARCH_TRIGGER,
)
