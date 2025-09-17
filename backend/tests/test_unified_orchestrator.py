"""
Comprehensive test suite for the unified uagent orchestrator
Tests all major components and integration workflows
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.core.unified_orchestrator import (
    UnifiedOrchestrator, WorkflowType, OrchestrationStrategy, WorkflowConfig
)
from app.core.meta_agent import TaskType, AgentRole
from app.core.agent_laboratory import CollaborationPattern


@pytest.fixture
def orchestrator():
    """Create a test orchestrator instance"""
    return UnifiedOrchestrator()


@pytest.fixture
def sample_workflow_config():
    """Sample workflow configuration for testing"""
    return WorkflowConfig(
        workflow_type=WorkflowType.AUTOMATED_RESEARCH,
        strategy=OrchestrationStrategy.ROMA_RECURSIVE,
        components=["ai_scientist", "search_engine"],
        parameters={"collaboration_enabled": True}
    )


@pytest.fixture
def sample_inputs():
    """Sample inputs for workflow testing"""
    return {
        "title": "Test Research Project",
        "description": "A test research project for validation",
        "research_questions": ["How effective is the unified agent approach?"]
    }


class TestUnifiedOrchestrator:
    """Test suite for UnifiedOrchestrator"""

    def test_orchestrator_initialization(self, orchestrator):
        """Test that orchestrator initializes all components correctly"""
        assert orchestrator.meta_agent is not None
        assert orchestrator.agent_lab is not None
        assert orchestrator.ai_scientist is not None
        assert orchestrator.repo_master is not None
        assert orchestrator.search_engine is not None
        assert len(orchestrator.workflow_templates) > 0

    def test_workflow_templates_loaded(self, orchestrator):
        """Test that workflow templates are properly loaded"""
        templates = orchestrator.workflow_templates
        assert "full_research_cycle" in templates
        assert "code_deep_dive" in templates
        assert "collaborative_project" in templates
        assert "intelligent_search" in templates

        # Verify template structure
        research_template = templates["full_research_cycle"]
        assert research_template.workflow_type == WorkflowType.AUTOMATED_RESEARCH
        assert "ai_scientist" in research_template.components

    @pytest.mark.asyncio
    async def test_execute_workflow_with_template(self, orchestrator, sample_inputs):
        """Test workflow execution using a template"""
        # Mock the async execution to avoid actually running workflows
        with patch.object(orchestrator, '_execute_workflow_async') as mock_execute:
            workflow_id = await orchestrator.execute_workflow(
                workflow_config="full_research_cycle",
                inputs=sample_inputs
            )

            assert workflow_id is not None
            assert workflow_id in orchestrator.active_workflows
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow_with_config(self, orchestrator, sample_workflow_config, sample_inputs):
        """Test workflow execution using a custom config"""
        with patch.object(orchestrator, '_execute_workflow_async') as mock_execute:
            workflow_id = await orchestrator.execute_workflow(
                workflow_config=sample_workflow_config,
                inputs=sample_inputs
            )

            assert workflow_id is not None
            assert workflow_id in orchestrator.active_workflows

            # Verify workflow result structure
            workflow_result = orchestrator.active_workflows[workflow_id]
            assert workflow_result.workflow_type == WorkflowType.AUTOMATED_RESEARCH
            assert workflow_result.status == "started"

    @pytest.mark.asyncio
    async def test_workflow_status_tracking(self, orchestrator, sample_inputs):
        """Test workflow status tracking"""
        with patch.object(orchestrator, '_execute_workflow_async') as mock_execute:
            workflow_id = await orchestrator.execute_workflow(
                workflow_config="full_research_cycle",
                inputs=sample_inputs
            )

            # Test getting workflow status
            status = await orchestrator.get_workflow_status(workflow_id)
            assert status is not None
            assert status.workflow_id == workflow_id

            # Test listing active workflows
            active_workflows = await orchestrator.list_active_workflows()
            assert len(active_workflows) >= 1
            assert any(w.workflow_id == workflow_id for w in active_workflows)

    @pytest.mark.asyncio
    async def test_cancel_workflow(self, orchestrator, sample_inputs):
        """Test workflow cancellation"""
        with patch.object(orchestrator, '_execute_workflow_async') as mock_execute:
            workflow_id = await orchestrator.execute_workflow(
                workflow_config="full_research_cycle",
                inputs=sample_inputs
            )

            # Cancel the workflow
            success = await orchestrator.cancel_workflow(workflow_id)
            assert success is True

            # Verify workflow is cancelled
            status = await orchestrator.get_workflow_status(workflow_id)
            assert status.status == "cancelled"

    @pytest.mark.asyncio
    async def test_system_status(self, orchestrator):
        """Test system status reporting"""
        status = await orchestrator.get_system_status()

        assert "components" in status
        assert "workflows" in status
        assert "templates_available" in status

        # Verify component status structure
        components = status["components"]
        assert "meta_agent" in components
        assert "agent_laboratory" in components
        assert "ai_scientist" in components
        assert "repo_master" in components


class TestResearchWorkflow:
    """Test suite for research workflow execution"""

    @pytest.mark.asyncio
    async def test_research_workflow_execution(self, orchestrator):
        """Test automated research workflow"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.AUTOMATED_RESEARCH,
            components=["ai_scientist", "search_engine"]
        )

        inputs = {
            "title": "AI Agent Research",
            "description": "Research on AI agent architectures",
            "research_questions": ["What are the best practices for AI agent design?"]
        }

        # Mock AI scientist methods
        with patch.object(orchestrator.ai_scientist, 'start_research_project', return_value="project_123"):
            with patch.object(orchestrator.ai_scientist, 'execute_research_phase', return_value={"phase": "completed"}):
                result = await orchestrator._execute_research_workflow(config, inputs)

                assert "project_id" in result
                assert result["project_id"] == "project_123"
                assert "research_questions" in result

    @pytest.mark.asyncio
    async def test_extract_research_questions(self, orchestrator):
        """Test research question extraction from search results"""
        mock_search_results = {
            "academic": [
                {"title": "How do neural networks learn?"},
                {"title": "What makes AI agents effective?"}
            ],
            "web": [
                {"title": "AI Development Best Practices"},
                {"title": "Machine Learning Tutorial"}
            ]
        }

        questions = await orchestrator._extract_research_questions(mock_search_results)

        assert len(questions) > 0
        assert any("?" in q for q in questions)  # At least one should be a proper question


class TestCodeAnalysisWorkflow:
    """Test suite for code analysis workflow"""

    @pytest.mark.asyncio
    async def test_code_analysis_workflow(self, orchestrator):
        """Test code analysis workflow execution"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.CODE_ANALYSIS,
            components=["repo_master", "agent_lab"]
        )

        inputs = {
            "repository_path": "/test/repo"
        }

        # Mock repo master methods
        mock_repo_summary = {
            "repository_info": {"name": "test_repo", "language": "python"},
            "code_metrics": {"complexity": {"average": 5}},
            "patterns_summary": {"by_type": {"code_smell": 2}}
        }

        with patch.object(orchestrator.repo_master, 'analyze_repository', return_value="repo_123"):
            with patch.object(orchestrator.repo_master, 'get_repository_summary', return_value=mock_repo_summary):
                with patch.object(orchestrator.agent_lab, 'create_collaboration_session', return_value="session_123"):
                    with patch.object(orchestrator.agent_lab, 'execute_collaborative_task', return_value={"analysis": "complete"}):
                        result = await orchestrator._execute_code_analysis_workflow(config, inputs)

                        assert "repository_analysis" in result
                        assert "collaborative_insights" in result
                        assert "recommendations" in result


class TestCollaborativeWorkflow:
    """Test suite for collaborative development workflow"""

    @pytest.mark.asyncio
    async def test_collaborative_workflow(self, orchestrator):
        """Test collaborative development workflow"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.COLLABORATIVE_DEVELOPMENT,
            components=["agent_lab", "meta_agent"],
            collaboration_pattern=CollaborationPattern.HIERARCHICAL
        )

        inputs = {
            "project_name": "Test Project",
            "requirements": ["Feature A", "Feature B"]
        }

        # Mock collaboration methods
        with patch.object(orchestrator.agent_lab, 'create_collaboration_session', return_value="session_123"):
            with patch.object(orchestrator.meta_agent, 'create_task', side_effect=["task_1", "task_2"]):
                with patch.object(orchestrator.agent_lab, 'execute_collaborative_task', return_value={"task": "completed"}):
                    with patch.object(orchestrator.agent_lab, 'get_collaboration_metrics', return_value={"effectiveness": 0.8}):
                        result = await orchestrator._execute_collaborative_workflow(config, inputs)

                        assert "collaboration_session" in result
                        assert "task_results" in result
                        assert "collaboration_metrics" in result


class TestSearchWorkflow:
    """Test suite for search workflow"""

    @pytest.mark.asyncio
    async def test_search_workflow(self, orchestrator):
        """Test intelligent search workflow"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.MULTI_MODAL_SEARCH,
            components=["search_engine"]
        )

        inputs = {
            "query": "machine learning algorithms"
        }

        mock_search_results = {
            "web": [{"title": "ML Algorithms Guide", "url": "example.com"}],
            "academic": [{"title": "Deep Learning Paper", "url": "arxiv.org"}],
            "metadata": {"intent": {"type": "academic"}}
        }

        with patch.object(orchestrator.search_engine, 'intelligent_search', return_value=mock_search_results):
            result = await orchestrator._execute_search_workflow(config, inputs)

            assert "web" in result
            assert "academic" in result
            assert "metadata" in result


class TestIntegrationScenarios:
    """Test suite for end-to-end integration scenarios"""

    @pytest.mark.asyncio
    async def test_full_research_cycle_integration(self, orchestrator):
        """Test complete research cycle from start to finish"""
        inputs = {
            "title": "AI Agent Architecture Research",
            "description": "Comprehensive research on AI agent architectures",
            "query": "AI agent frameworks"
        }

        # Mock all necessary components
        with patch.object(orchestrator.search_engine, 'intelligent_search') as mock_search:
            with patch.object(orchestrator.ai_scientist, 'start_research_project') as mock_start:
                with patch.object(orchestrator.ai_scientist, 'execute_research_phase') as mock_execute:
                    with patch.object(orchestrator.agent_lab, 'create_collaboration_session') as mock_collab:

                        # Configure mocks
                        mock_search.return_value = {"web": [], "academic": []}
                        mock_start.return_value = "project_123"
                        mock_execute.return_value = {"phase": "literature_review", "status": "completed"}
                        mock_collab.return_value = "session_123"

                        workflow_id = await orchestrator.execute_workflow(
                            workflow_config="full_research_cycle",
                            inputs=inputs
                        )

                        assert workflow_id is not None
                        assert workflow_id in orchestrator.active_workflows

    @pytest.mark.asyncio
    async def test_code_analysis_integration(self, orchestrator):
        """Test complete code analysis integration"""
        inputs = {
            "repository_path": "/test/python/project"
        }

        with patch.object(orchestrator.repo_master, 'analyze_repository') as mock_analyze:
            with patch.object(orchestrator.repo_master, 'get_repository_summary') as mock_summary:
                with patch.object(orchestrator.search_engine, 'unified_search') as mock_search:

                    # Configure mocks
                    mock_analyze.return_value = "repo_123"
                    mock_summary.return_value = {
                        "repository_info": {"name": "test_project", "language": "python"},
                        "code_metrics": {"complexity": {"average": 6}},
                        "patterns_summary": {"total_patterns": 5}
                    }
                    mock_search.return_value = {"code": [], "web": []}

                    workflow_id = await orchestrator.execute_workflow(
                        workflow_config="code_deep_dive",
                        inputs=inputs
                    )

                    assert workflow_id is not None

    @pytest.mark.asyncio
    async def test_hybrid_workflow_integration(self, orchestrator):
        """Test hybrid workflow combining multiple components"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.HYBRID_WORKFLOW,
            components=["ai_scientist", "repo_master", "search_engine"],
            parameters={
                "research": True,
                "code_analysis": True,
                "search": True
            }
        )

        inputs = {
            "title": "Hybrid Analysis Project",
            "repository_path": "/test/repo",
            "query": "test query"
        }

        # Mock all components
        with patch.object(orchestrator, '_execute_research_workflow') as mock_research:
            with patch.object(orchestrator, '_execute_code_analysis_workflow') as mock_code:
                with patch.object(orchestrator, '_execute_search_workflow') as mock_search:

                    mock_research.return_value = {"research": "completed"}
                    mock_code.return_value = {"analysis": "completed"}
                    mock_search.return_value = {"search": "completed"}

                    result = await orchestrator._execute_hybrid_workflow(config, inputs)

                    assert "research" in result
                    assert "code_analysis" in result
                    assert "search" in result


class TestErrorHandling:
    """Test suite for error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_invalid_workflow_type(self, orchestrator):
        """Test handling of invalid workflow types"""
        with pytest.raises(Exception):
            await orchestrator.execute_workflow(
                workflow_config="nonexistent_template",
                inputs={}
            )

    @pytest.mark.asyncio
    async def test_missing_inputs(self, orchestrator):
        """Test handling of missing required inputs"""
        # Test code analysis without repository path
        config = WorkflowConfig(
            workflow_type=WorkflowType.CODE_ANALYSIS,
            components=["repo_master"]
        )

        with pytest.raises(ValueError, match="Repository path required"):
            await orchestrator._execute_code_analysis_workflow(config, {})

    @pytest.mark.asyncio
    async def test_component_failure_handling(self, orchestrator):
        """Test handling of component failures"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.MULTI_MODAL_SEARCH,
            components=["search_engine"]
        )

        # Mock search engine to raise an exception
        with patch.object(orchestrator.search_engine, 'intelligent_search', side_effect=Exception("Search failed")):
            # Should not raise exception, but handle gracefully
            workflow_id = await orchestrator.execute_workflow(config, {"query": "test"})
            assert workflow_id is not None

            # Wait a bit for async execution
            await asyncio.sleep(0.1)

            # Check that workflow failed gracefully
            status = await orchestrator.get_workflow_status(workflow_id)
            assert status.status == "failed"
            assert "error" in status.results if hasattr(status, 'results') else True

    def test_workflow_config_validation(self):
        """Test workflow configuration validation"""
        # Test valid config
        config = WorkflowConfig(
            workflow_type=WorkflowType.AUTOMATED_RESEARCH,
            strategy=OrchestrationStrategy.ADAPTIVE,
            components=["ai_scientist"]
        )
        assert config.workflow_type == WorkflowType.AUTOMATED_RESEARCH

        # Test enum validation
        with pytest.raises(ValueError):
            WorkflowType("invalid_type")

        with pytest.raises(ValueError):
            OrchestrationStrategy("invalid_strategy")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])