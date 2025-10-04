"""
Code Research Agent - Extends CodeActAgent with code analysis capabilities
"""

import logging

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.events.action import Action, MessageAction
from openhands.llm.llm_registry import LLMRegistry

from ..engines.code_research import CodeResearchEngine

logger = logging.getLogger(__name__)


class CodeResearchAgent(CodeActAgent):
    """
    Agent specialized for code repository analysis.

    Integrates RepoMaster-style functionality for understanding codebases.
    """

    VERSION = "1.0"

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """
        Initialize code research agent.

        Args:
            config: Agent configuration
            llm_registry: LLM registry for accessing models
        """
        super().__init__(config, llm_registry)

        # Initialize code research engine
        self.code_engine = CodeResearchEngine(
            llm=self.llm,
            config=getattr(config, 'code_research_config', {})
        )

        self.current_analysis = None
        self.analysis_mode = False

        logger.info("CodeResearchAgent initialized")

    def step(self, state: State) -> Action:
        """
        Execute one agent step.

        Args:
            state: Current agent state

        Returns:
            Next action to take
        """
        if self._is_code_analysis_task(state):
            logger.info("Detected code analysis task")
            self.analysis_mode = True
            return self._analyze_code(state)
        else:
            self.analysis_mode = False
            return super().step(state)

    def _is_code_analysis_task(self, state: State) -> bool:
        """
        Detect if task requires code analysis.

        Args:
            state: Current agent state

        Returns:
            True if task is code analysis related
        """
        code_keywords = [
            'analyze code', 'understand repository', 'find implementation',
            'code structure', 'architecture', 'how does', 'where is',
            'explain code', 'trace', 'dependency', 'what does this code',
            'code review', 'code quality'
        ]

        if state.history:
            latest_events = state.history.get_events_as_list()
            for event in reversed(latest_events[-10:]):
                if hasattr(event, 'message') and event.message:
                    message_lower = event.message.lower()
                    if any(keyword in message_lower for keyword in code_keywords):
                        return True

        return False

    def _analyze_code(self, state: State) -> Action:
        """
        Perform code analysis.

        Args:
            state: Current agent state

        Returns:
            Analysis action
        """
        logger.info("Analyzing code for user query")

        # Extract query
        query = self._extract_query(state)

        # Create message action indicating analysis
        return MessageAction(
            content=f"""Analyzing codebase to answer: {query}

I will:
1. Examine the repository structure
2. Find relevant files
3. Analyze the code
4. Provide a comprehensive answer

Starting analysis...
""",
            thought="Initiating code analysis"
        )

    def _extract_query(self, state: State) -> str:
        """
        Extract analysis query from state.

        Args:
            state: Current agent state

        Returns:
            Query string
        """
        if state.history:
            latest_events = state.history.get_events_as_list()
            for event in reversed(latest_events):
                if hasattr(event, 'message') and event.message:
                    return event.message

        return "Analyze code repository"

    def reset(self) -> None:
        """Reset agent state"""
        super().reset()
        self.current_analysis = None
        self.analysis_mode = False
        logger.info("CodeResearchAgent reset")


# Register agent type
AGENT_CLS = CodeResearchAgent
