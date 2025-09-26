"""Modified smart router that delegates complete goals to OpenHands CLI.

This version avoids step-by-step CodeAct interactions and instead passes
the entire research goal to OpenHands for autonomous execution.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from ..services.openhands_goal_runner import OpenHandsGoalRunner
from ..core.websocket_manager import websocket_manager


async def route_to_scientific_research_openhands(
    query: str,
    session_id: str,
    workspace_base: Path = Path("/home/wuy/AI/uagent-workspace"),
) -> Dict[str, Any]:
    """Route scientific research queries to OpenHands with full goal delegation.

    Instead of step-by-step CodeAct interactions, this delegates the entire
    research goal to OpenHands CLI for autonomous execution.

    Args:
        query: The research goal/query
        session_id: Session identifier
        workspace_base: Base directory for workspaces

    Returns:
        Research results from OpenHands execution
    """
    # Create workspace for this research session
    workspace_path = workspace_base / f"session_{session_id}"
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Progress callback for WebSocket updates
    async def progress_callback(event_type: str, data: Dict[str, Any]):
        """Send progress updates via WebSocket."""
        await websocket_manager.broadcast_progress(
            session_id=session_id,
            message={
                "type": event_type,
                "engine": "scientific_research_openhands",
                "data": data,
            }
        )

    # Initialize OpenHands goal runner
    runner = OpenHandsGoalRunner(
        openhands_binary="uvx --python 3.12 --from openhands-ai openhands",
        max_iterations=50,  # Allow sufficient iterations for complex research
        model="claude-3-5-sonnet-20241022",  # Or configure based on settings
    )

    # Format the research goal with clear instructions
    research_goal = f"""
    Scientific Research Task:
    {query}

    Instructions:
    1. Create a hypothesis based on the research question
    2. Design experiments to test the hypothesis
    3. Implement the experimental code in the code/ directory
    4. Run the experiments and collect data
    5. Analyze results and draw conclusions
    6. Generate a final research report

    Important:
    - Create all code files in the code/ directory
    - Use meaningful file names (not repetitive variations)
    - Execute experiments and collect real data
    - Provide comprehensive analysis and conclusions
    """

    try:
        # Delegate the entire research goal to OpenHands
        result = await runner.run_goal(
            workspace_path=workspace_path,
            goal=research_goal,
            progress_cb=progress_callback,
        )

        # Process and return results
        return {
            "success": result.success,
            "engine": "scientific_research_openhands",
            "approach": "full_goal_delegation",
            "results": {
                "message": result.message,
                "artifacts": result.artifacts,
                "workspace": str(result.workspace_path),
                "generated_files": result.artifacts.get("generated_files", []),
            },
            "session_id": session_id,
        }

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to delegate to OpenHands: {e}")

        return {
            "success": False,
            "engine": "scientific_research_openhands",
            "error": str(e),
            "session_id": session_id,
        }


async def route_to_code_research_openhands(
    query: str,
    session_id: str,
    repository_url: Optional[str] = None,
    workspace_base: Path = Path("/home/wuy/AI/uagent-workspace"),
) -> Dict[str, Any]:
    """Route code research queries to OpenHands with full goal delegation.

    Args:
        query: The code research goal
        session_id: Session identifier
        repository_url: Optional repository to analyze
        workspace_base: Base directory for workspaces

    Returns:
        Code research results from OpenHands execution
    """
    workspace_path = workspace_base / f"code_session_{session_id}"
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Progress callback
    async def progress_callback(event_type: str, data: Dict[str, Any]):
        await websocket_manager.broadcast_progress(
            session_id=session_id,
            message={
                "type": event_type,
                "engine": "code_research_openhands",
                "data": data,
            }
        )

    runner = OpenHandsGoalRunner()

    # Format code research goal
    code_research_goal = f"""
    Code Research Task:
    {query}

    {"Repository: " + repository_url if repository_url else ""}

    Instructions:
    1. Analyze the codebase structure and architecture
    2. Identify key components relevant to the research question
    3. Examine code patterns, dependencies, and relationships
    4. Document findings with code examples
    5. Provide actionable insights and recommendations

    Create analysis files in the code/ directory.
    """

    result = await runner.run_goal(
        workspace_path=workspace_path,
        goal=code_research_goal,
        progress_cb=progress_callback,
    )

    return {
        "success": result.success,
        "engine": "code_research_openhands",
        "approach": "full_goal_delegation",
        "results": {
            "message": result.message,
            "artifacts": result.artifacts,
            "workspace": str(result.workspace_path),
        },
        "session_id": session_id,
    }


# Update the main router to use OpenHands delegation
async def smart_route_with_openhands_delegation(
    query: str,
    session_id: str,
    engine_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Smart router that delegates complete goals to OpenHands.

    This avoids the repetitive step-by-step CodeAct approach and instead
    lets OpenHands handle the entire research goal autonomously.
    """
    # Determine engine type if not specified
    if not engine_type:
        # Use existing classification logic
        if "hypothesis" in query.lower() or "experiment" in query.lower():
            engine_type = "scientific"
        elif "code" in query.lower() or "repository" in query.lower():
            engine_type = "code"
        else:
            engine_type = "deep"

    # Route to appropriate engine with OpenHands delegation
    if engine_type == "scientific":
        return await route_to_scientific_research_openhands(query, session_id)
    elif engine_type == "code":
        return await route_to_code_research_openhands(query, session_id)
    else:
        # Deep research can still use existing approach or also delegate
        return {
            "engine": "deep_research",
            "message": "Deep research engine (implement delegation if needed)",
            "query": query,
        }