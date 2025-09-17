"""
Integration tests for the complete uagent system
Tests component interactions and end-to-end workflows
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

from app.core.unified_orchestrator import UnifiedOrchestrator
from app.core.meta_agent import MetaAgent, TaskType, AgentRole
from app.core.agent_laboratory import AgentLaboratory, CollaborationPattern
from app.core.ai_scientist import AIScientist
from app.core.repo_master import RepoMaster, AnalysisDepth
from app.utils.multi_modal_search import MultiModalSearchEngine


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple Python project structure
        os.makedirs(os.path.join(temp_dir, "src"))

        # Create a main.py file
        with open(os.path.join(temp_dir, "src", "main.py"), "w") as f:
            f.write("""
def hello_world():
    '''A simple hello world function'''
    print("Hello, World!")
    return "Hello, World!"

class Calculator:
    '''A simple calculator class'''

    def add(self, a, b):
        '''Add two numbers'''
        return a + b

    def multiply(self, a, b):
        '''Multiply two numbers'''
        return a * b

if __name__ == "__main__":
    hello_world()
    calc = Calculator()
    print(calc.add(2, 3))
""")

        # Create a requirements.txt
        with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
            f.write("pytest==7.4.0\nrequests==2.31.0\n")

        yield temp_dir


class TestComponentIntegration:
    """Test integration between different components"""

    @pytest.mark.asyncio
    async def test_meta_agent_agent_lab_integration(self):
        """Test integration between MetaAgent and AgentLaboratory"""
        meta_agent = MetaAgent()
        agent_lab = AgentLaboratory()

        # Register meta-agent's agents with agent laboratory
        for agent_id, agent in meta_agent.agents.items():
            registered_id = agent_lab.register_agent(agent)
            assert registered_id == agent_id

        # Create a collaboration session
        session_id = await agent_lab.create_collaboration_session(
            name="Test Integration",
            pattern=CollaborationPattern.HIERARCHICAL,
            agent_roles=[AgentRole.RESEARCHER, AgentRole.CODER]
        )

        assert session_id is not None
        assert session_id in agent_lab.collaboration_sessions

        # Create task in meta-agent
        task_id = await meta_agent.create_task(
            name="Integration Test Task",
            description="Test task for integration",
            task_type=TaskType.CODE_GENERATION
        )

        task = meta_agent.tasks[task_id]
        assert task is not None

        # Execute task collaboratively
        with patch.object(agent_lab, '_execute_agent_step', return_value={"result": "success"}):
            result = await agent_lab.execute_collaborative_task(session_id, task)
            assert result is not None

    @pytest.mark.asyncio
    async def test_ai_scientist_search_integration(self):
        """Test integration between AI-Scientist and search engine"""
        ai_scientist = AIScientist()
        search_engine = MultiModalSearchEngine()

        # Mock search engine methods
        with patch.object(search_engine, 'unified_search') as mock_search:
            mock_search.return_value = {
                "academic": [
                    {
                        "title": "Machine Learning in Agent Systems",
                        "url": "https://arxiv.org/test",
                        "snippet": "This paper explores ML in agents",
                        "authors": ["John Doe"],
                        "source": "arxiv"
                    }
                ]
            }

            # Start research project
            project_id = await ai_scientist.start_research_project(
                title="AI Agent Research",
                description="Research on AI agents",
                research_questions=["How do AI agents learn?"]
            )

            # Mock literature search by integrating with search engine
            with patch.object(ai_scientist.literature_engine, 'search_papers') as mock_lit_search:
                mock_lit_search.return_value = [
                    ai_scientist.literature_engine.paper_database.get("test_paper",
                        type('Paper', (), {
                            'id': 'test_paper',
                            'title': 'ML Agent Research',
                            'authors': ['Test Author'],
                            'abstract': 'Test abstract',
                            'keywords': ['ai', 'agents']
                        })()
                    )
                ]

                # Execute literature review phase
                result = await ai_scientist.execute_research_phase(
                    project_id,
                    ai_scientist.ai_scientist.ResearchPhase.LITERATURE_REVIEW if hasattr(ai_scientist, 'ai_scientist') else None
                )

                assert result is not None

    @pytest.mark.asyncio
    async def test_repo_master_meta_agent_integration(self, temp_repo):
        """Test integration between RepoMaster and MetaAgent"""
        repo_master = RepoMaster()
        meta_agent = MetaAgent()

        # Analyze repository
        repo_id = await repo_master.analyze_repository(temp_repo, AnalysisDepth.SEMANTIC)
        assert repo_id is not None

        # Get repository summary
        repo_summary = await repo_master.get_repository_summary(repo_id)
        assert repo_summary is not None
        assert "repository_info" in repo_summary

        # Create analysis task in meta-agent
        task_id = await meta_agent.create_task(
            name="Repository Analysis",
            description="Analyze repository structure and code",
            task_type=TaskType.CODE_ANALYSIS,
            context={"repo_id": repo_id, "repo_summary": repo_summary}
        )

        # Assign and execute task
        agent_id = await meta_agent.assign_task(task_id)
        assert agent_id is not None

        # Execute workflow
        with patch.object(meta_agent, '_run_single_experiment', return_value={"analysis": "complete"}):
            result = await meta_agent.execute_workflow("test_workflow")
            assert result is not None


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""

    @pytest.mark.asyncio
    async def test_complete_research_workflow(self):
        """Test complete research workflow from start to finish"""
        orchestrator = UnifiedOrchestrator()

        # Define research inputs
        inputs = {
            "title": "AI Agent Architecture Study",
            "description": "Comprehensive study of AI agent architectures",
            "research_questions": [
                "What are the key components of effective AI agents?",
                "How do different architectures compare in performance?"
            ]
        }

        # Mock all external dependencies
        with patch.object(orchestrator.search_engine, 'intelligent_search') as mock_search:
            with patch.object(orchestrator.ai_scientist, 'start_research_project') as mock_start:
                with patch.object(orchestrator.ai_scientist, 'execute_research_phase') as mock_phase:
                    with patch.object(orchestrator.agent_lab, 'create_collaboration_session') as mock_session:

                        # Configure mocks
                        mock_search.return_value = {
                            "academic": [
                                {"title": "Agent Architecture Paper", "snippet": "Research on agents"}
                            ],
                            "web": [
                                {"title": "AI Agent Guide", "snippet": "Guide to AI agents"}
                            ]
                        }
                        mock_start.return_value = "research_project_123"
                        mock_phase.return_value = {
                            "phase": "literature_review_completed",
                            "papers_found": 5,
                            "next_phase": "hypothesis_generation"
                        }
                        mock_session.return_value = "collaboration_session_123"

                        # Execute workflow
                        workflow_id = await orchestrator.execute_workflow(
                            workflow_config="full_research_cycle",
                            inputs=inputs
                        )

                        assert workflow_id is not None

                        # Wait for workflow to start
                        await asyncio.sleep(0.1)

                        # Check workflow status
                        status = await orchestrator.get_workflow_status(workflow_id)
                        assert status is not None
                        assert status.workflow_id == workflow_id

    @pytest.mark.asyncio
    async def test_complete_code_analysis_workflow(self, temp_repo):
        """Test complete code analysis workflow"""
        orchestrator = UnifiedOrchestrator()

        inputs = {
            "repository_path": temp_repo
        }

        # Mock external search calls
        with patch.object(orchestrator.search_engine, 'unified_search') as mock_search:
            mock_search.return_value = {
                "code": [
                    {
                        "title": "Similar Python Project",
                        "url": "https://github.com/example/project",
                        "language": "python"
                    }
                ]
            }

            # Execute workflow
            workflow_id = await orchestrator.execute_workflow(
                workflow_config="code_deep_dive",
                inputs=inputs
            )

            assert workflow_id is not None

            # Wait for execution to start
            await asyncio.sleep(0.1)

            # Verify workflow was created
            status = await orchestrator.get_workflow_status(workflow_id)
            assert status is not None

    @pytest.mark.asyncio
    async def test_collaborative_development_workflow(self):
        """Test collaborative development workflow"""
        orchestrator = UnifiedOrchestrator()

        inputs = {
            "project_name": "AI Chat Bot",
            "requirements": [
                "Natural language processing",
                "Response generation",
                "User interface",
                "Database integration"
            ]
        }

        # Execute collaborative workflow
        workflow_id = await orchestrator.execute_workflow(
            workflow_config="collaborative_project",
            inputs=inputs
        )

        assert workflow_id is not None

        # Verify workflow tracking
        active_workflows = await orchestrator.list_active_workflows()
        assert any(w.workflow_id == workflow_id for w in active_workflows)

    @pytest.mark.asyncio
    async def test_intelligent_search_workflow(self):
        """Test intelligent search workflow"""
        orchestrator = UnifiedOrchestrator()

        # Mock search results
        with patch.object(orchestrator.search_engine, 'intelligent_search') as mock_search:
            mock_search.return_value = {
                "web": [
                    {"title": "AI Tutorial", "url": "example.com", "snippet": "Learn AI"}
                ],
                "academic": [
                    {"title": "AI Research Paper", "url": "arxiv.org", "snippet": "Latest AI research"}
                ],
                "code": [
                    {"title": "AI Library", "url": "github.com", "snippet": "Open source AI code"}
                ],
                "metadata": {
                    "intent": {"type": "general", "domain": "ai_ml"},
                    "search_types_used": ["web", "academic", "code"]
                }
            }

            # Execute search workflow
            workflow_id = await orchestrator.execute_workflow(
                workflow_config="intelligent_search",
                inputs={"query": "artificial intelligence machine learning"}
            )

            assert workflow_id is not None

    @pytest.mark.asyncio
    async def test_hybrid_workflow_execution(self, temp_repo):
        """Test hybrid workflow combining multiple workflow types"""
        orchestrator = UnifiedOrchestrator()

        from app.core.unified_orchestrator import WorkflowConfig, WorkflowType

        # Create hybrid workflow config
        hybrid_config = WorkflowConfig(
            workflow_type=WorkflowType.HYBRID_WORKFLOW,
            components=["ai_scientist", "repo_master", "search_engine", "agent_lab"],
            parameters={
                "research": True,
                "code_analysis": True,
                "search": True
            }
        )

        inputs = {
            "title": "Comprehensive AI Agent Analysis",
            "repository_path": temp_repo,
            "query": "AI agent frameworks comparison"
        }

        # Mock dependencies
        with patch.object(orchestrator.search_engine, 'intelligent_search') as mock_search:
            with patch.object(orchestrator.ai_scientist, 'start_research_project') as mock_research:
                mock_search.return_value = {"web": [], "academic": [], "code": []}
                mock_research.return_value = "hybrid_project_123"

                # Execute hybrid workflow
                workflow_id = await orchestrator.execute_workflow(
                    workflow_config=hybrid_config,
                    inputs=inputs
                )

                assert workflow_id is not None


class TestSystemResilience:
    """Test system resilience and error recovery"""

    @pytest.mark.asyncio
    async def test_component_failure_recovery(self):
        """Test system behavior when components fail"""
        orchestrator = UnifiedOrchestrator()

        # Test with failing search component
        with patch.object(orchestrator.search_engine, 'intelligent_search', side_effect=Exception("Search service down")):
            # Should still create workflow but handle failure gracefully
            workflow_id = await orchestrator.execute_workflow(
                workflow_config="intelligent_search",
                inputs={"query": "test query"}
            )

            assert workflow_id is not None

            # Wait for async execution
            await asyncio.sleep(0.2)

            # Check that failure was handled
            status = await orchestrator.get_workflow_status(workflow_id)
            assert status.status == "failed"

    @pytest.mark.asyncio
    async def test_partial_component_failure(self):
        """Test behavior when some components fail but others succeed"""
        orchestrator = UnifiedOrchestrator()

        # Mock partial failure scenario
        with patch.object(orchestrator.search_engine, 'unified_search') as mock_search:
            with patch.object(orchestrator.ai_scientist, 'start_research_project', side_effect=Exception("AI Scientist unavailable")):
                mock_search.return_value = {"web": [{"title": "Test", "url": "test.com"}]}

                # This should still work partially
                workflow_id = await orchestrator.execute_workflow(
                    workflow_config="full_research_cycle",
                    inputs={"title": "Test Research"}
                )

                assert workflow_id is not None

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """Test executing multiple workflows concurrently"""
        orchestrator = UnifiedOrchestrator()

        # Create multiple workflows
        workflow_ids = []

        for i in range(3):
            workflow_id = await orchestrator.execute_workflow(
                workflow_config="intelligent_search",
                inputs={"query": f"test query {i}"}
            )
            workflow_ids.append(workflow_id)

        # Verify all workflows were created
        assert len(workflow_ids) == 3
        assert len(set(workflow_ids)) == 3  # All unique

        # Check active workflows
        active = await orchestrator.list_active_workflows()
        assert len(active) >= 3

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self):
        """Test workflow timeout handling"""
        orchestrator = UnifiedOrchestrator()

        from app.core.unified_orchestrator import WorkflowConfig, WorkflowType

        # Create workflow with very short timeout
        config = WorkflowConfig(
            workflow_type=WorkflowType.MULTI_MODAL_SEARCH,
            components=["search_engine"],
            timeout=1  # 1 second timeout
        )

        # Mock long-running search
        with patch.object(orchestrator.search_engine, 'intelligent_search', side_effect=lambda **kwargs: asyncio.sleep(5)):
            workflow_id = await orchestrator.execute_workflow(
                workflow_config=config,
                inputs={"query": "test"}
            )

            assert workflow_id is not None

            # The workflow should eventually timeout (in real implementation)


class TestPerformanceAndScalability:
    """Test performance and scalability aspects"""

    @pytest.mark.asyncio
    async def test_large_repository_analysis(self, temp_repo):
        """Test analysis of larger repositories"""
        # Create additional files to simulate larger repo
        for i in range(10):
            file_path = os.path.join(temp_repo, f"module_{i}.py")
            with open(file_path, "w") as f:
                f.write(f"""
class Module{i}:
    def __init__(self):
        self.value = {i}

    def process(self, data):
        return data * {i}

    def calculate(self, x, y):
        result = 0
        for j in range({i}):
            result += x + y + j
        return result
""")

        orchestrator = UnifiedOrchestrator()

        # Analyze larger repository
        repo_id = await orchestrator.repo_master.analyze_repository(temp_repo)
        summary = await orchestrator.repo_master.get_repository_summary(repo_id)

        assert summary is not None
        assert summary["elements_summary"]["total_elements"] > 10  # Should have many elements

    @pytest.mark.asyncio
    async def test_system_status_under_load(self):
        """Test system status reporting under load"""
        orchestrator = UnifiedOrchestrator()

        # Create multiple workflows to simulate load
        workflow_ids = []
        for i in range(5):
            workflow_id = await orchestrator.execute_workflow(
                workflow_config="intelligent_search",
                inputs={"query": f"load test {i}"}
            )
            workflow_ids.append(workflow_id)

        # Get system status
        status = await orchestrator.get_system_status()

        assert status is not None
        assert status["workflows"]["active"] >= 5

        # Cancel all workflows
        for workflow_id in workflow_ids:
            await orchestrator.cancel_workflow(workflow_id)


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])