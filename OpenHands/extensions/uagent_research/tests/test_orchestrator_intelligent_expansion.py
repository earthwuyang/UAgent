"""
Integration tests for TreeSearchOrchestrator with intelligent node expansion
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from ..orchestrator.tree_orchestrator import TreeSearchOrchestrator
from ..services.idea_generation_service import IdeaGenerationService
from ..uagent_research.models.research_tree import (
    Budget,
    ResearchNode,
    NodeType,
    NodeStatus,
)


class MockIdeaGenerationService:
    """Mock IdeaGenerationService for testing."""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.generate_ideas_called = False
        self.generate_hypotheses_called = False
        self.generate_experiments_called = False
    
    async def generate_ideas(self, goal, context=None, max_ideas=3):
        """Mock generate_ideas."""
        self.generate_ideas_called = True
        if self.should_fail:
            raise Exception("Service failure")
        
        return [
            ResearchNode(
                id="intelligent-idea-1",
                type=NodeType.IDEA,
                title="LLM-Generated Idea: Advanced Neural Architecture Search",
                content="Use evolutionary algorithms combined with reinforcement learning",
                status=NodeStatus.PENDING,
                prior=0.85
            ),
            ResearchNode(
                id="intelligent-idea-2",
                type=NodeType.IDEA,
                title="LLM-Generated Idea: Transfer Learning Approach",
                content="Leverage pre-trained models and fine-tune for specific tasks",
                status=NodeStatus.PENDING,
                prior=0.80
            )
        ]
    
    async def generate_hypotheses(self, idea_content, parent_node, max_hypotheses=2):
        """Mock generate_hypotheses."""
        self.generate_hypotheses_called = True
        if self.should_fail:
            raise Exception("Service failure")
        
        return [
            ResearchNode(
                id="intelligent-hyp-1",
                type=NodeType.HYPOTHESIS,
                title="LLM-Generated Hypothesis 1",
                content=f"Hypothesis derived from: {idea_content[:30]}...",
                status=NodeStatus.PENDING,
                prior=0.75,
                parent_id=parent_node.id
            )
        ]
    
    async def generate_experiments(self, hypothesis_content, parent_node):
        """Mock generate_experiments."""
        self.generate_experiments_called = True
        if self.should_fail:
            raise Exception("Service failure")
        
        return [
            ResearchNode(
                id="intelligent-exp-1",
                type=NodeType.EXPERIMENT,
                title="LLM-Generated Experiment",
                content=f"Experiment to test: {hypothesis_content[:30]}...",
                status=NodeStatus.PENDING,
                prior=0.70,
                parent_id=parent_node.id
            )
        ]


@pytest.fixture
def mock_llm():
    """Mock LLM instance."""
    return Mock()


@pytest.fixture
def mock_idea_service():
    """Mock IdeaGenerationService."""
    return MockIdeaGenerationService()


@pytest.fixture
def mock_idea_service_failure():
    """Mock IdeaGenerationService that fails."""
    return MockIdeaGenerationService(should_fail=True)


class TestOrchestratorIntelligentExpansion:
    """Tests for orchestrator with intelligent expansion."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_intelligent_service(self, mock_llm, mock_idea_service):
        """Test orchestrator uses intelligent expansion when service is available."""
        orchestrator = TreeSearchOrchestrator(
            max_parallel=1,
            budget=Budget(max_iterations=1),
            llm=mock_llm,
            idea_service=mock_idea_service
        )
        
        assert orchestrator.use_intelligent_expansion is True
        assert orchestrator.idea_service == mock_idea_service

    @pytest.mark.asyncio
    async def test_orchestrator_without_service(self):
        """Test orchestrator falls back when no service is available."""
        orchestrator = TreeSearchOrchestrator(
            max_parallel=1,
            budget=Budget(max_iterations=1)
        )
        
        assert orchestrator.use_intelligent_expansion is False
        assert orchestrator.idea_service is None

    @pytest.mark.asyncio
    async def test_expand_root_with_intelligent_service(self, mock_idea_service):
        """Test ROOT node expansion uses intelligent service."""
        orchestrator = TreeSearchOrchestrator(
            max_parallel=1,
            budget=Budget(max_iterations=1),
            idea_service=mock_idea_service
        )
        
        # Initialize tree
        from ..uagent_research.models.research_tree import ResearchTree
        orchestrator.tree = ResearchTree()
        
        # Expand root node
        root = orchestrator.tree.root
        children = await orchestrator._expand_node(root, goal="Test research goal", context=None)
        
        # Verify intelligent service was called
        assert mock_idea_service.generate_ideas_called is True
        
        # Verify intelligent nodes were generated
        assert len(children) == 2
        assert children[0].title.startswith("LLM-Generated Idea:")
        assert children[0].id == "intelligent-idea-1"

    @pytest.mark.asyncio
    async def test_fallback_on_service_failure(self, mock_idea_service_failure):
        """Test orchestrator falls back to placeholders when service fails."""
        orchestrator = TreeSearchOrchestrator(
            max_parallel=1,
            budget=Budget(max_iterations=1),
            idea_service=mock_idea_service_failure
        )
        
        # Initialize tree
        from ..uagent_research.models.research_tree import ResearchTree
        orchestrator.tree = ResearchTree()
        
        # Expand root node
        root = orchestrator.tree.root
        children = await orchestrator._expand_node(root, goal="Test goal", context=None)
        
        # Service should have been called but failed
        assert mock_idea_service_failure.generate_ideas_called is True
        
        # Should fall back to placeholder nodes
        assert len(children) > 0
        # Fallback nodes have generic titles
        assert any("Web Research" in child.title or "Code Research" in child.title for child in children)

    @pytest.mark.asyncio
    async def test_expand_idea_node_with_service(self, mock_idea_service):
        """Test IDEA node expansion uses intelligent service."""
        orchestrator = TreeSearchOrchestrator(
            max_parallel=1,
            budget=Budget(max_iterations=1),
            idea_service=mock_idea_service
        )
        
        # Initialize tree
        from ..uagent_research.models.research_tree import ResearchTree
        orchestrator.tree = ResearchTree()
        
        # Create an IDEA node
        idea_node = ResearchNode(
            id="test-idea",
            type=NodeType.IDEA,
            title="Test Idea",
            content="Test idea content",
            status=NodeStatus.PENDING,
            prior=0.8
        )
        orchestrator.tree.add_node(idea_node, parent_id="root")
        
        # Expand the IDEA node
        children = await orchestrator._expand_node(idea_node, goal="Test goal", context=None)
        
        # Verify intelligent service was called
        assert mock_idea_service.generate_hypotheses_called is True
        
        # Verify intelligent hypotheses were generated
        assert len(children) == 1
        assert children[0].type == NodeType.HYPOTHESIS
        assert children[0].parent_id == idea_node.id

    @pytest.mark.asyncio
    async def test_tree_stats_updated_after_expansion(self, mock_idea_service):
        """Test that tree stats are updated after expansion."""
        orchestrator = TreeSearchOrchestrator(
            max_parallel=1,
            budget=Budget(max_iterations=1),
            idea_service=mock_idea_service
        )
        
        # Initialize tree
        from ..uagent_research.models.research_tree import ResearchTree
        orchestrator.tree = ResearchTree()
        
        initial_node_count = orchestrator.stats["total_nodes"]
        
        # Expand root node
        root = orchestrator.tree.root
        children = await orchestrator._expand_node(root, goal="Test goal", context=None)
        
        # Stats should be updated
        assert orchestrator.stats["total_nodes"] == initial_node_count + len(children)

    @pytest.mark.asyncio
    async def test_children_added_to_tree(self, mock_idea_service):
        """Test that generated children are added to the tree."""
        orchestrator = TreeSearchOrchestrator(
            max_parallel=1,
            budget=Budget(max_iterations=1),
            idea_service=mock_idea_service
        )
        
        # Initialize tree
        from ..uagent_research.models.research_tree import ResearchTree
        orchestrator.tree = ResearchTree()
        
        # Expand root node
        root = orchestrator.tree.root
        await orchestrator._expand_node(root, goal="Test goal", context=None)
        
        # Verify children were added to tree
        root_node = orchestrator.tree.nodes["root"]
        assert len(root_node.children) == 2
        
        # Verify children are in tree.nodes
        assert "intelligent-idea-1" in orchestrator.tree.nodes
        assert "intelligent-idea-2" in orchestrator.tree.nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
