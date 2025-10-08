"""
Tests for IdeaGenerationService
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List

from ..services.idea_generation_service import IdeaGenerationService
from ..uagent_research.models.research_tree import (
    ResearchNode,
    NodeType,
    NodeStatus,
)


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, responses: List[str] = None):
        self.responses = responses or []
        self.call_count = 0
    
    async def completion(self, messages, temperature=0.7):
        """Mock completion method."""
        if self.call_count < len(self.responses):
            response_text = self.responses[self.call_count]
        else:
            response_text = '[]'  # Default empty response
        
        self.call_count += 1
        
        # Create mock response object
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = response_text
        
        return mock_response


class MockResearchEngine:
    """Mock research engine for testing."""
    
    def __init__(self, ideas=None):
        self.ideas = ideas or []
    
    async def _generate_research_ideas(self, goal, context, max_ideas):
        """Mock idea generation."""
        return self.ideas[:max_ideas]


@pytest.fixture
def mock_llm_with_ideas():
    """Mock LLM that returns ideas."""
    responses = [
        '[{"title": "Idea 1", "description": "First research idea", "confidence": 0.8}, {"title": "Idea 2", "description": "Second research idea", "confidence": 0.75}]'
    ]
    return MockLLM(responses)


@pytest.fixture
def mock_llm_with_hypotheses():
    """Mock LLM that returns hypotheses."""
    responses = [
        '[{"title": "Hypothesis 1", "description": "First hypothesis to test", "confidence": 0.7}, {"title": "Hypothesis 2", "description": "Second hypothesis", "confidence": 0.65}]'
    ]
    return MockLLM(responses)


@pytest.fixture
def mock_llm_with_experiments():
    """Mock LLM that returns experiments."""
    responses = [
        '[{"title": "Experiment 1", "methodology": "Step 1, Step 2, Step 3", "expected_outcome": "Expected result", "confidence": 0.6}]'
    ]
    return MockLLM(responses)


@pytest.fixture
def mock_llm_failure():
    """Mock LLM that fails."""
    mock = Mock()
    mock.completion = AsyncMock(side_effect=Exception("LLM API error"))
    return mock


@pytest.fixture
def sample_idea_node():
    """Sample IDEA node for testing."""
    return ResearchNode(
        id="idea-123",
        type=NodeType.IDEA,
        title="Test Research Idea",
        content="This is a test research idea about machine learning",
        status=NodeStatus.PENDING,
        prior=0.8
    )


@pytest.fixture
def sample_hypothesis_node():
    """Sample HYPOTHESIS node for testing."""
    return ResearchNode(
        id="hypothesis-456",
        type=NodeType.HYPOTHESIS,
        title="Test Hypothesis",
        content="Neural networks can achieve 95% accuracy on this task",
        status=NodeStatus.PENDING,
        prior=0.7
    )


class TestIdeaGenerationService:
    """Test cases for IdeaGenerationService."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_llm_with_ideas):
        """Test service initialization."""
        service = IdeaGenerationService(llm=mock_llm_with_ideas)
        
        assert service.llm == mock_llm_with_ideas
        assert service.max_ideas == 3
        assert service.max_hypotheses == 2
        assert service.max_experiments == 1
        assert service.retry_count == 2

    @pytest.mark.asyncio
    async def test_initialization_with_config(self, mock_llm_with_ideas):
        """Test service initialization with custom config."""
        config = {
            'max_ideas': 5,
            'max_hypotheses': 3,
            'max_experiments': 2,
            'retry_count': 3
        }
        service = IdeaGenerationService(llm=mock_llm_with_ideas, config=config)
        
        assert service.max_ideas == 5
        assert service.max_hypotheses == 3
        assert service.max_experiments == 2
        assert service.retry_count == 3

    @pytest.mark.asyncio
    async def test_generate_hypotheses_success(self, mock_llm_with_hypotheses, sample_idea_node):
        """Test successful hypothesis generation."""
        service = IdeaGenerationService(llm=mock_llm_with_hypotheses)
        
        # Mock the engine to bypass it for this test
        service.engine = Mock()
        
        hypotheses = await service.generate_hypotheses(
            idea_content=sample_idea_node.content,
            parent_node=sample_idea_node
        )
        
        assert len(hypotheses) == 2
        assert all(node.type == NodeType.HYPOTHESIS for node in hypotheses)
        assert all(node.parent_id == sample_idea_node.id for node in hypotheses)
        assert hypotheses[0].title == "Hypothesis 1"
        assert hypotheses[1].title == "Hypothesis 2"
        assert 0.0 <= hypotheses[0].prior <= 1.0
        assert all(node.status == NodeStatus.PENDING for node in hypotheses)

    @pytest.mark.asyncio
    async def test_generate_experiments_success(self, mock_llm_with_experiments, sample_hypothesis_node):
        """Test successful experiment generation."""
        service = IdeaGenerationService(llm=mock_llm_with_experiments)
        
        # Mock the engine
        service.engine = Mock()
        
        experiments = await service.generate_experiments(
            hypothesis_content=sample_hypothesis_node.content,
            parent_node=sample_hypothesis_node
        )
        
        assert len(experiments) == 1
        assert all(node.type == NodeType.EXPERIMENT for node in experiments)
        assert all(node.parent_id == sample_hypothesis_node.id for node in experiments)
        assert experiments[0].title == "Experiment 1"
        assert "Methodology:" in experiments[0].content
        assert "Expected Outcome:" in experiments[0].content
        assert 0.0 <= experiments[0].prior <= 1.0

    @pytest.mark.asyncio
    async def test_llm_failure_handling(self, mock_llm_failure):
        """Test graceful handling of LLM failures."""
        service = IdeaGenerationService(llm=mock_llm_failure)
        
        # Mock the engine to raise an exception
        mock_engine = Mock()
        mock_engine._generate_research_ideas = AsyncMock(side_effect=Exception("API Error"))
        service.engine = mock_engine
        
        ideas = await service.generate_ideas(goal="Test goal", context="Test context")
        
        # Should return empty list on failure
        assert ideas == []

    @pytest.mark.asyncio
    async def test_no_engine_returns_empty(self, mock_llm_with_ideas):
        """Test that missing engine returns empty list."""
        service = IdeaGenerationService(llm=mock_llm_with_ideas)
        service.engine = None  # Simulate engine initialization failure
        
        ideas = await service.generate_ideas(goal="Test goal")
        hypotheses = await service.generate_hypotheses("test content", Mock())
        experiments = await service.generate_experiments("test hypothesis", Mock())
        
        assert ideas == []
        assert hypotheses == []
        assert experiments == []

    @pytest.mark.asyncio
    async def test_max_ideas_limit(self, mock_llm_with_ideas):
        """Test that max_ideas limit is respected."""
        # Create mock engine with many ideas
        mock_ideas = [Mock(title=f"Idea {i}", summary=f"Summary {i}", confidence=0.8) for i in range(10)]
        
        service = IdeaGenerationService(llm=mock_llm_with_ideas)
        
        # Mock the engine
        mock_engine = Mock()
        mock_engine._generate_research_ideas = AsyncMock(return_value=mock_ideas)
        service.engine = mock_engine
        
        ideas = await service.generate_ideas(goal="Test goal", max_ideas=3)
        
        # Should only return 3 ideas
        assert len(ideas) <= 3

    @pytest.mark.asyncio
    async def test_prior_calculation(self, mock_llm_with_ideas):
        """Test that prior values are calculated correctly."""
        # Create mock ideas with different confidence scores
        mock_ideas = [
            Mock(title="High Confidence", summary="Summary", confidence=0.9),
            Mock(title="Low Confidence", summary="Summary", confidence=0.5),
        ]
        
        service = IdeaGenerationService(llm=mock_llm_with_ideas)
        
        # Mock the engine
        mock_engine = Mock()
        mock_engine._generate_research_ideas = AsyncMock(return_value=mock_ideas)
        service.engine = mock_engine
        
        ideas = await service.generate_ideas(goal="Test goal")
        
        # Priors should be extracted from confidence
        assert ideas[0].prior == 0.9
        assert ideas[1].prior == 0.5

    @pytest.mark.asyncio
    async def test_node_structure(self, mock_llm_with_ideas):
        """Test that generated nodes have correct structure."""
        mock_ideas = [Mock(title="Test Idea", summary="Test summary", confidence=0.8)]
        
        service = IdeaGenerationService(llm=mock_llm_with_ideas)
        mock_engine = Mock()
        mock_engine._generate_research_ideas = AsyncMock(return_value=mock_ideas)
        service.engine = mock_engine
        
        ideas = await service.generate_ideas(goal="Test goal")
        
        assert len(ideas) == 1
        node = ideas[0]
        
        # Verify node structure
        assert isinstance(node, ResearchNode)
        assert node.id.startswith("idea-")
        assert node.type == NodeType.IDEA
        assert node.title == "Test Idea"
        assert node.content == "Test summary"
        assert node.status == NodeStatus.PENDING
        assert node.prior == 0.8
        assert node.parent_id is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestJSONParsingRobustness:
    """Tests for robust JSON parsing."""

    @pytest.mark.asyncio
    async def test_json_with_code_fences(self):
        """Test parsing JSON wrapped in code fences."""
        service = IdeaGenerationService(llm=Mock())
        
        text_with_fences = '''```json
[{"title": "Test", "description": "Content", "confidence": 0.8}]
```'''
        
        result = service._extract_json_from_text(text_with_fences)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['title'] == "Test"

    @pytest.mark.asyncio
    async def test_json_with_markdown_fences(self):
        """Test parsing JSON with markdown formatting."""
        service = IdeaGenerationService(llm=Mock())
        
        text = '''Here's the response:
```
[{"title": "Idea", "description": "Test", "confidence": 0.9}]
```
'''
        
        result = service._extract_json_from_text(text)
        assert result is not None
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_json_with_surrounding_text(self):
        """Test extracting JSON from text with surrounding content."""
        service = IdeaGenerationService(llm=Mock())
        
        text = '''Let me generate some ideas for you:
[{"title": "First", "description": "Desc", "confidence": 0.7}]
That should help!'''
        
        result = service._extract_json_from_text(text)
        assert result is not None
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self):
        """Test that malformed JSON returns None gracefully."""
        service = IdeaGenerationService(llm=Mock())
        
        text = "This is not JSON at all"
        result = service._extract_json_from_text(text)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self):
        """Test that empty text returns None."""
        service = IdeaGenerationService(llm=Mock())
        
        result = service._extract_json_from_text("")
        assert result is None
        
        result = service._extract_json_from_text(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_single_json_object_wrapped_in_array(self):
        """Test that single JSON object is wrapped in array."""
        service = IdeaGenerationService(llm=Mock())
        
        text = '{"title": "Single", "description": "Object", "confidence": 0.8}'
        result = service._extract_json_from_text(text)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_llm_text_extraction_openai_style(self):
        """Test extracting text from OpenAI-style response."""
        service = IdeaGenerationService(llm=Mock())
        
        # Mock OpenAI-style response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test content"
        
        result = service._extract_llm_text(mock_response)
        assert result == "Test content"

    @pytest.mark.asyncio
    async def test_llm_text_extraction_fallback(self):
        """Test fallback text extraction."""
        service = IdeaGenerationService(llm=Mock())
        
        # Mock response with content attribute
        mock_response = Mock()
        mock_response.content = "Direct content"
        
        result = service._extract_llm_text(mock_response)
        assert result == "Direct content"

    @pytest.mark.asyncio
    async def test_generate_hypotheses_with_malformed_response(self, sample_idea_node):
        """Test hypothesis generation handles malformed LLM response."""
        # Create mock LLM that returns non-JSON
        mock_llm = Mock()
        mock_llm.completion = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content="Sorry, I can't help with that"))]
        ))
        
        service = IdeaGenerationService(llm=mock_llm)
        service.engine = Mock()
        
        hypotheses = await service.generate_hypotheses(
            idea_content="test",
            parent_node=sample_idea_node
        )
        
        # Should return empty list, not crash
        assert hypotheses == []

    @pytest.mark.asyncio
    async def test_generate_experiments_with_code_fence_response(self, sample_hypothesis_node):
        """Test experiment generation with code-fenced response."""
        # Create mock LLM that returns JSON with code fences
        mock_llm = Mock()
        response_text = '''```json
[{"title": "Test Exp", "methodology": "Steps", "expected_outcome": "Result", "confidence": 0.6}]
```'''
        mock_llm.completion = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content=response_text))]
        ))
        
        service = IdeaGenerationService(llm=mock_llm)
        service.engine = Mock()
        
        experiments = await service.generate_experiments(
            hypothesis_content="test",
            parent_node=sample_hypothesis_node
        )
        
        # Should successfully parse despite code fences
        assert len(experiments) == 1
        assert experiments[0].title == "Test Exp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
