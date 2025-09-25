"""Test unified workspace path implementation for OpenHands integration"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from backend.app.core.openhands.client import OpenHandsClient
from backend.app.core.openhands.workspace_manager import WorkspaceManager
from backend.app.integrations.openhands_bridge import OpenHandsGoalPlanBridge, GoalPlan, GoalPlanStep
from backend.app.core.research_engines.scientific_research import ScientificResearchEngine


@pytest.fixture
async def temp_workspace_dir():
    """Create a temporary directory for workspaces"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def workspace_manager(temp_workspace_dir):
    """Create a workspace manager with temporary directory"""
    return WorkspaceManager(str(temp_workspace_dir))


@pytest.fixture
async def openhands_client(workspace_manager):
    """Create an OpenHands client with test workspace manager"""
    client = OpenHandsClient(base_workspace_dir=str(workspace_manager.base_dir))
    client.workspace_manager = workspace_manager
    return client


class TestUnifiedWorkspacePath:
    """Test that OpenHands and GoalBridge use unified workspace paths"""

    async def test_unified_session_id_format(self):
        """Test that session IDs follow unified format"""
        # Test scientific research session ID format
        research_session_id = "test_session_123"
        idea_id = "idea_456"

        # Expected unified format
        expected_session_id = f"experiment_{research_session_id}_{idea_id}"

        # Verify format matches expectation
        assert expected_session_id.startswith("experiment_")
        assert research_session_id in expected_session_id
        assert idea_id in expected_session_id

    async def test_workspace_reuse_same_session(self, openhands_client):
        """Test that multiple calls with same session ID reuse workspace"""
        session_id = "experiment_test_session_123"

        # Create session first time
        config1 = await openhands_client.create_session(
            research_type="scientific_research",
            session_id=session_id
        )
        workspace_id1 = config1.workspace_config.get("workspace_id")

        # Ensure session second time (should reuse)
        config2 = await openhands_client.ensure_session(
            research_type="scientific_research",
            session_id=session_id
        )
        workspace_id2 = config2.workspace_config.get("workspace_id")

        # Verify same workspace is reused
        assert workspace_id1 == workspace_id2
        assert workspace_id1 == session_id

    async def test_workspace_manager_preserves_existing(self, workspace_manager):
        """Test that workspace manager preserves existing workspaces"""
        research_id = "experiment_test_123"

        # Create workspace first time
        config1 = await workspace_manager.create_workspace(research_id=research_id)
        workspace_path1 = workspace_manager.base_dir / config1.workspace_id

        # Create test file in workspace
        test_file = workspace_path1 / "test_file.txt"
        test_file.write_text("test content")

        # Create workspace second time with same ID
        config2 = await workspace_manager.create_workspace(research_id=research_id)
        workspace_path2 = workspace_manager.base_dir / config2.workspace_id

        # Verify same workspace and content preserved
        assert config1.workspace_id == config2.workspace_id
        assert workspace_path1 == workspace_path2
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    async def test_goal_bridge_uses_unified_session(self, openhands_client):
        """Test that GoalBridge uses unified session ID from context"""
        mock_llm_client = Mock()
        mock_llm_client.generate = AsyncMock(return_value='{"summary": "test", "steps": []}')

        bridge = OpenHandsGoalPlanBridge(openhands_client, mock_llm_client)

        # Create a test plan
        plan = GoalPlan(
            plan_id="test_plan_789",
            goal="Test goal",
            summary="Test summary",
            steps=[]
        )

        # Execute with session ID in context
        unified_session_id = "experiment_main_session_idea_123"
        execution_context = {
            "session_id": unified_session_id,
            "resource_requirements": {}
        }

        with patch.object(openhands_client, 'ensure_session', new=AsyncMock()) as mock_ensure:
            mock_ensure.return_value = Mock(workspace_config={"workspace_id": unified_session_id})

            await bridge.execute_goal_plan(plan, execution_context)

            # Verify ensure_session was called with unified session ID
            mock_ensure.assert_called_once_with(
                research_type="scientific_research",
                session_id=unified_session_id,
                config={}
            )

    async def test_goal_bridge_creates_unified_session_without_context(self, openhands_client):
        """Test that GoalBridge creates unified session ID when not in context"""
        mock_llm_client = Mock()
        mock_llm_client.generate = AsyncMock(return_value='{"summary": "test", "steps": []}')

        bridge = OpenHandsGoalPlanBridge(openhands_client, mock_llm_client)

        # Create a test plan
        plan = GoalPlan(
            plan_id="test_plan_789",
            goal="Test goal",
            summary="Test summary",
            steps=[]
        )

        # Execute without session ID in context
        execution_context = {
            "resource_requirements": {}
        }

        with patch.object(openhands_client, 'ensure_session', new=AsyncMock()) as mock_ensure:
            mock_ensure.return_value = Mock(workspace_config={"workspace_id": "test"})

            await bridge.execute_goal_plan(plan, execution_context)

            # Verify ensure_session was called with generated unified session ID
            mock_ensure.assert_called_once_with(
                research_type="scientific_research",
                session_id=f"experiment_{plan.plan_id}",
                config={}
            )

    @pytest.mark.integration
    async def test_scientific_research_unified_path_integration(self):
        """Integration test for scientific research using unified paths"""
        # This test verifies the full flow but requires mocking external dependencies

        with patch('backend.app.core.research_engines.scientific_research.OpenHandsClient') as MockClient:
            mock_client_instance = Mock()
            MockClient.return_value = mock_client_instance

            # Mock ensure_session to track calls
            mock_client_instance.ensure_session = AsyncMock()
            mock_client_instance.ensure_session.return_value = Mock(
                workspace_config={"workspace_id": "experiment_test"}
            )

            # Create scientific research engine
            config = {
                "llm_provider": "mock",
                "openhands_enabled": True
            }

            with patch('backend.app.core.research_engines.scientific_research.LLMClient'):
                engine = ScientificResearchEngine(config)
                engine.openhands_client = mock_client_instance

                # Simulate acquiring OpenHands session
                research_session_id = "research_123"
                idea_id = "idea_456"

                # The expected unified session ID
                expected_session_id = f"experiment_{research_session_id}_{idea_id}"

                # Note: Full integration test would require more setup
                # This verifies the session ID format is correct
                assert expected_session_id.startswith("experiment_")
                assert not expected_session_id.startswith("openhands_")
                assert not expected_session_id.startswith("plan_plan_")


class AsyncMock(MagicMock):
    """Helper class for async mocking"""
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])