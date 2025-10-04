"""
Scientific Research Agent - Extends CodeActAgent with research capabilities
"""

import logging
from typing import List

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.events.action import Action, MessageAction, AgentFinishAction
from openhands.llm.llm_registry import LLMRegistry

from ..engines.scientific_research import ScientificResearchEngine

logger = logging.getLogger(__name__)


class ScientificResearchAgent(CodeActAgent):
    """
    Agent specialized for scientific research experiments.

    Extends CodeActAgent with research planning and execution capabilities.
    Can conduct multi-step experiments with hypothesis testing.
    """

    VERSION = "1.0"

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """
        Initialize scientific research agent.

        Args:
            config: Agent configuration
            llm_registry: LLM registry for accessing models
        """
        super().__init__(config, llm_registry)

        # Initialize research engine
        self.research_engine = ScientificResearchEngine(
            llm=self.llm,
            config=getattr(config, 'research_config', {})
        )

        self.current_experiment = None
        self.research_mode = False

        logger.info("ScientificResearchAgent initialized")

    def step(self, state: State) -> Action:
        """
        Execute one agent step.

        Determines if this is a research task and routes appropriately.

        Args:
            state: Current agent state

        Returns:
            Next action to take
        """
        # Check if this is a research task
        if self._is_research_task(state):
            logger.info("Detected research task, using research mode")
            self.research_mode = True
            return self._research_step(state)
        else:
            # Fall back to normal CodeAct behavior
            logger.info("Normal coding task, using CodeAct mode")
            self.research_mode = False
            return super().step(state)

    def _is_research_task(self, state: State) -> bool:
        """
        Detect if current task requires research capabilities.

        Args:
            state: Current agent state

        Returns:
            True if task is research-related
        """
        research_keywords = [
            'experiment', 'hypothesis', 'research', 'investigate',
            'analyze performance', 'compare approaches', 'benchmark',
            'scientific method', 'test hypothesis', 'validate',
            'ml model', 'train model', 'evaluate model'
        ]

        # Check latest user message
        if state.history:
            latest_events = state.history.get_events_as_list()
            for event in reversed(latest_events[-10:]):  # Check last 10 events
                if hasattr(event, 'message') and event.message:
                    message_lower = event.message.lower()
                    if any(keyword in message_lower for keyword in research_keywords):
                        return True

        return False

    def _research_step(self, state: State) -> Action:
        """
        Execute research-specific step.

        Args:
            state: Current agent state

        Returns:
            Research action
        """
        # If no experiment in progress, start new one
        if not self.current_experiment:
            return self._start_experiment(state)

        # Continue ongoing experiment
        return self._continue_experiment(state)

    def _start_experiment(self, state: State) -> Action:
        """
        Start new research experiment.

        Args:
            state: Current agent state

        Returns:
            Action to start experiment
        """
        logger.info("Starting new research experiment")

        # Get research goal from state
        goal = self._extract_research_goal(state)

        # Create a MessageAction that will trigger experiment execution
        # In a real implementation, this would use the research engine asynchronously
        return MessageAction(
            content=f"""Starting scientific research experiment:
Goal: {goal}

I will:
1. Generate testable hypotheses
2. Design experiments to test each hypothesis
3. Execute experiments using available tools
4. Analyze results
5. Draw conclusions

This may take some time. I'll provide updates as I progress through each phase.
""",
            thought="Initiating research experiment"
        )

    def _continue_experiment(self, state: State) -> Action:
        """
        Continue ongoing experiment.

        Args:
            state: Current agent state

        Returns:
            Next research action
        """
        # In production, this would check experiment status and continue
        # For now, use CodeAct behavior
        return super().step(state)

    def _extract_research_goal(self, state: State) -> str:
        """
        Extract research goal from state.

        Args:
            state: Current agent state

        Returns:
            Research goal string
        """
        if state.history:
            latest_events = state.history.get_events_as_list()
            for event in reversed(latest_events):
                if hasattr(event, 'message') and event.message:
                    # Return the latest user message as goal
                    return event.message

        return "Conduct scientific research experiment"

    def reset(self) -> None:
        """Reset agent state"""
        super().reset()
        self.current_experiment = None
        self.research_mode = False
        logger.info("ScientificResearchAgent reset")


# Register agent type
AGENT_CLS = ScientificResearchAgent
