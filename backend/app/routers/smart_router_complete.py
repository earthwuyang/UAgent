"""Production smart router using OpenHands complete goal delegation.

This router solves the repetitive file creation issue by delegating
complete research goals to OpenHands' internal agent controller.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..services.openhands_complete_runner import delegate_to_openhands_complete
from ..core.websocket_manager import websocket_manager
from ..core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SmartRouterComplete:
    """Smart router that delegates complete goals to OpenHands."""

    def __init__(self):
        self.llm_client = LLMClient()
        self.logger = logging.getLogger(__name__)

    async def classify_query(self, query: str) -> str:
        """Classify the query to determine which engine to use."""

        classification_prompt = f"""
        Classify this query into one of three categories:
        1. "scientific" - for experimental research, hypothesis testing, data analysis
        2. "code" - for code analysis, repository understanding, code generation
        3. "deep" - for general web research, literature review, information gathering

        Query: {query}

        Respond with only the category name.
        """

        response = await self.llm_client.generate(
            classification_prompt,
            max_tokens=50,
            temperature=0.1,
        )

        category = response.strip().lower()
        if category not in ["scientific", "code", "deep"]:
            # Fallback classification based on keywords
            query_lower = query.lower()
            if any(word in query_lower for word in ["hypothesis", "experiment", "test", "data", "analysis"]):
                category = "scientific"
            elif any(word in query_lower for word in ["code", "repository", "function", "class", "api"]):
                category = "code"
            else:
                category = "deep"

        return category

    async def route_to_scientific_research(
        self,
        query: str,
        session_id: str,
        workspace_base: Path,
    ) -> Dict[str, Any]:
        """Route to scientific research with complete goal delegation."""

        workspace_path = workspace_base / f"scientific_{session_id}"

        # Format as a comprehensive scientific research goal
        research_goal = f"""
        Scientific Research Task:
        {query}

        Requirements:
        1. Formulate a clear hypothesis based on the research question
        2. Design experiments to test the hypothesis
        3. Implement the experimental code in the code/ directory
        4. Run experiments and collect data
        5. Analyze results and draw conclusions
        6. Generate a research report with findings

        Guidelines:
        - Create well-organized code files with descriptive names
        - Avoid creating duplicate or similar files
        - Use code/ directory for all implementation
        - Document your methodology and findings
        - Provide reproducible results
        """

        # Progress callback for WebSocket updates
        async def progress_callback(event_type: str, data: Dict[str, Any]):
            await websocket_manager.broadcast_progress(
                session_id=session_id,
                message={
                    "type": event_type,
                    "engine": "scientific_research",
                    "data": data,
                }
            )

        # Delegate complete goal to OpenHands
        result = await delegate_to_openhands_complete(
            goal=research_goal,
            workspace_path=workspace_path,
            model="gpt-4o-mini",  # or use config
            max_iterations=50,
            progress_cb=progress_callback,
        )

        return {
            "engine": "scientific_research",
            "session_id": session_id,
            **result
        }

    async def route_to_code_research(
        self,
        query: str,
        session_id: str,
        workspace_base: Path,
        repository_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route to code research with complete goal delegation."""

        workspace_path = workspace_base / f"code_{session_id}"

        # Format as a comprehensive code research goal
        code_goal = f"""
        Code Research Task:
        {query}

        {"Repository: " + repository_url if repository_url else ""}

        Requirements:
        1. Analyze the codebase structure and architecture
        2. Identify key components related to the research question
        3. Document code patterns and relationships
        4. Create analysis scripts if needed
        5. Generate a comprehensive report

        Guidelines:
        - Create analysis files in the code/ directory
        - Use clear, descriptive file names
        - Avoid creating duplicate files
        - Provide code examples and explanations
        """

        async def progress_callback(event_type: str, data: Dict[str, Any]):
            await websocket_manager.broadcast_progress(
                session_id=session_id,
                message={
                    "type": event_type,
                    "engine": "code_research",
                    "data": data,
                }
            )

        result = await delegate_to_openhands_complete(
            goal=code_goal,
            workspace_path=workspace_path,
            model="gpt-4o-mini",
            max_iterations=40,
            progress_cb=progress_callback,
        )

        return {
            "engine": "code_research",
            "session_id": session_id,
            **result
        }

    async def route_to_deep_research(
        self,
        query: str,
        session_id: str,
        workspace_base: Path,
    ) -> Dict[str, Any]:
        """Route to deep research with complete goal delegation."""

        workspace_path = workspace_base / f"deep_{session_id}"

        deep_goal = f"""
        Deep Research Task:
        {query}

        Requirements:
        1. Gather comprehensive information on the topic
        2. Synthesize findings from multiple sources
        3. Create summary documents
        4. Provide actionable insights

        Guidelines:
        - Organize findings in the code/ directory
        - Create structured documentation
        - Avoid duplicate content files
        """

        async def progress_callback(event_type: str, data: Dict[str, Any]):
            await websocket_manager.broadcast_progress(
                session_id=session_id,
                message={
                    "type": event_type,
                    "engine": "deep_research",
                    "data": data,
                }
            )

        result = await delegate_to_openhands_complete(
            goal=deep_goal,
            workspace_path=workspace_path,
            model="gpt-4o-mini",
            max_iterations=30,
            progress_cb=progress_callback,
        )

        return {
            "engine": "deep_research",
            "session_id": session_id,
            **result
        }

    async def route(
        self,
        query: str,
        session_id: str,
        workspace_base: Path = Path("/tmp/uagent_workspace"),
        engine_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main routing method with complete goal delegation.

        This method classifies the query and routes it to the appropriate
        research engine, using OpenHands' complete goal delegation to avoid
        repetitive file creation issues.

        Args:
            query: User's research query
            session_id: Session identifier
            workspace_base: Base directory for workspaces
            engine_override: Optional engine type override

        Returns:
            Research results dictionary
        """
        workspace_base = Path(workspace_base)
        workspace_base.mkdir(parents=True, exist_ok=True)

        # Determine engine type
        if engine_override:
            engine_type = engine_override
        else:
            engine_type = await self.classify_query(query)

        self.logger.info(f"Routing query to {engine_type} engine with complete delegation")

        # Route to appropriate engine
        if engine_type == "scientific":
            return await self.route_to_scientific_research(query, session_id, workspace_base)
        elif engine_type == "code":
            return await self.route_to_code_research(query, session_id, workspace_base)
        else:
            return await self.route_to_deep_research(query, session_id, workspace_base)


# Global router instance
smart_router_complete = SmartRouterComplete()