"""Unit tests for research tree models."""

import pytest
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from uagent_research.models.research_tree import (
    ResearchNode,
    ResearchTree,
    ResearchEdge,
    Budget,
    Task,
    Context,
    NodeType,
    NodeStatus,
)
from uagent_research.models.events import Artifact, ArtifactType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_artifact():
    """Factory fixture for creating Artifact instances."""
    def _create_artifact(
        kind: ArtifactType = ArtifactType.URL,
        locator: str = "https://example.com",
        content: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        return Artifact(
            kind=kind,
            locator=locator,
            content=content,
            summary=summary,
            metadata=metadata or {},
        )
    return _create_artifact


@pytest.fixture
def sample_node():
    """Factory fixture for creating ResearchNode instances."""
    def _create_node(
        node_id: str = "test-node-1",
        node_type: NodeType = NodeType.IDEA,
        title: str = "Test Node",
        content: str = "Test content",
        status: NodeStatus = NodeStatus.PENDING,
        parent_id: Optional[str] = None,
        score: Optional[float] = None,
        confidence: Optional[float] = None,
        artifacts: Optional[List[Artifact]] = None,
        cost: float = 0.0,
        tokens_used: int = 0,
        visits: int = 0,
        prior: float = 0.5,
        avg_value: float = 0.0,
        adapter: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        iterations: int = 0,
    ) -> ResearchNode:
        return ResearchNode(
            id=node_id,
            type=node_type,
            title=title,
            content=content,
            status=status,
            parent_id=parent_id,
            score=score,
            confidence=confidence,
            artifacts=artifacts or [],
            cost=cost,
            tokens_used=tokens_used,
            visits=visits,
            prior=prior,
            avg_value=avg_value,
            adapter=adapter,
            metadata=metadata or {},
            iterations=iterations,
        )
    return _create_node


@pytest.fixture
def sample_tree():
    """Factory fixture for creating ResearchTree instances."""
    def _create_tree(research_id: str = "test-research-1") -> ResearchTree:
        return ResearchTree(research_id=research_id)
    return _create_tree


@pytest.fixture
def populated_tree(sample_node, sample_tree):
    """Fixture providing a pre-populated research tree.
    
    Structure:
    ROOT
    ├── IDEA-1
    │   ├── HYPOTHESIS-1-1
    │   │   └── EXPERIMENT-1-1-1
    │   └── HYPOTHESIS-1-2
    │       └── EXPERIMENT-1-2-1
    ├── IDEA-2
    │   ├── HYPOTHESIS-2-1
    │   │   └── EXPERIMENT-2-1-1
    │   └── HYPOTHESIS-2-2
    │       └── EXPERIMENT-2-2-1
    └── IDEA-3
        └── HYPOTHESIS-3-1
            └── EXPERIMENT-3-1-1
    """
    tree = sample_tree()
    
    # Create root
    root = sample_node(
        node_id="root",
        node_type=NodeType.ROOT,
        title="Research Root",
        content="Root node"
    )
    tree.add_node(root)
    
    # Create ideas
    for i in range(1, 4):
        idea = sample_node(
            node_id=f"idea-{i}",
            node_type=NodeType.IDEA,
            title=f"Idea {i}",
            content=f"Idea {i} content"
        )
        tree.add_node(idea, parent_id="root")
        
        # Create hypotheses (2 for idea-1 and idea-2, 1 for idea-3)
        num_hypotheses = 2 if i <= 2 else 1
        for j in range(1, num_hypotheses + 1):
            hypothesis = sample_node(
                node_id=f"hypothesis-{i}-{j}",
                node_type=NodeType.HYPOTHESIS,
                title=f"Hypothesis {i}-{j}",
                content=f"Hypothesis {i}-{j} content"
            )
            tree.add_node(hypothesis, parent_id=f"idea-{i}")
            
            # Create experiment
            experiment = sample_node(
                node_id=f"experiment-{i}-{j}-1",
                node_type=NodeType.EXPERIMENT,
                title=f"Experiment {i}-{j}-1",
                content=f"Experiment {i}-{j}-1 content"
            )
            tree.add_node(experiment, parent_id=f"hypothesis-{i}-{j}")
    
    return tree


# ============================================================================
# Test ResearchNode
# ============================================================================

@pytest.mark.unit
class TestResearchNode:
    """Tests for ResearchNode model."""
    
    def test_node_creation_minimal(self, sample_node):
        """Test creating a node with minimal required fields."""
        node = sample_node(
            node_id="test-1",
            node_type=NodeType.IDEA,
            title="Test Idea",
            content="Test content"
        )
        
        assert node.id == "test-1"
        assert node.type == NodeType.IDEA
        assert node.title == "Test Idea"
        assert node.content == "Test content"
        # Verify defaults
        assert node.status == NodeStatus.PENDING
        assert node.visits == 0
        assert node.prior == 0.5
        assert node.avg_value == 0.0
        assert node.cost == 0.0
        assert node.tokens_used == 0
        assert node.artifacts == []
        assert node.metadata == {}
        assert node.iterations == 0
    
    def test_node_creation_full(self, sample_node, sample_artifact):
        """Test creating a node with all fields populated."""
        artifact = sample_artifact()
        node = sample_node(
            node_id="full-node",
            node_type=NodeType.HYPOTHESIS,
            title="Full Node",
            content="Full content",
            status=NodeStatus.COMPLETE,
            parent_id="parent-1",
            score=0.85,
            confidence=0.9,
            artifacts=[artifact],
            cost=1.5,
            tokens_used=1000,
            visits=5,
            prior=0.7,
            avg_value=0.6,
            adapter="test-adapter",
            metadata={"key": "value"},
            iterations=3,
        )
        
        assert node.id == "full-node"
        assert node.type == NodeType.HYPOTHESIS
        assert node.title == "Full Node"
        assert node.content == "Full content"
        assert node.status == NodeStatus.COMPLETE
        assert node.parent_id == "parent-1"
        assert node.score == 0.85
        assert node.confidence == 0.9
        assert len(node.artifacts) == 1
        assert node.cost == 1.5
        assert node.tokens_used == 1000
        assert node.visits == 5
        assert node.prior == 0.7
        assert node.avg_value == 0.6
        assert node.adapter == "test-adapter"
        assert node.metadata == {"key": "value"}
        assert node.iterations == 3
    
    def test_node_types(self, sample_node):
        """Test creation with each NodeType enum value."""
        node_types = [
            NodeType.ROOT, NodeType.IDEA, NodeType.HYPOTHESIS,
            NodeType.PLAN, NodeType.WEB_SEARCH, NodeType.CODE_SEARCH,
            NodeType.BROWSE, NodeType.ANALYSIS, NodeType.EXPERIMENT,
            NodeType.RESULT, NodeType.CRITIQUE, NodeType.SUMMARY
        ]
        
        for node_type in node_types:
            node = sample_node(node_id=f"node-{node_type.value}", node_type=node_type)
            assert node.type == node_type
    
    def test_status_pending_to_running(self, sample_node):
        """Test status transition from PENDING to RUNNING."""
        node = sample_node(status=NodeStatus.PENDING)
        assert node.started_at is None
        
        node.status = NodeStatus.RUNNING
        node.started_at = datetime.now()
        
        assert node.status == NodeStatus.RUNNING
        assert node.started_at is not None
        assert isinstance(node.started_at, datetime)
    
    def test_status_running_to_complete(self, sample_node):
        """Test status transition from RUNNING to COMPLETE."""
        node = sample_node(status=NodeStatus.RUNNING)
        node.started_at = datetime.now()
        assert node.completed_at is None
        
        node.status = NodeStatus.COMPLETE
        node.completed_at = datetime.now()
        
        assert node.status == NodeStatus.COMPLETE
        assert node.completed_at is not None
        assert isinstance(node.completed_at, datetime)
    
    def test_status_running_to_failed(self, sample_node):
        """Test status transition from RUNNING to FAILED."""
        node = sample_node(status=NodeStatus.RUNNING)
        node.started_at = datetime.now()
        
        node.status = NodeStatus.FAILED
        node.completed_at = datetime.now()
        
        assert node.status == NodeStatus.FAILED
        assert node.completed_at is not None
    
    def test_status_running_to_cancelled(self, sample_node):
        """Test status transition from RUNNING to CANCELLED."""
        node = sample_node(status=NodeStatus.RUNNING)
        node.started_at = datetime.now()
        
        node.status = NodeStatus.CANCELLED
        node.completed_at = datetime.now()
        
        assert node.status == NodeStatus.CANCELLED
        assert node.completed_at is not None
    
    def test_all_status_values(self, sample_node):
        """Test all NodeStatus enum values."""
        statuses = [
            NodeStatus.PENDING, NodeStatus.RUNNING,
            NodeStatus.COMPLETE, NodeStatus.FAILED, NodeStatus.CANCELLED
        ]
        
        for status in statuses:
            node = sample_node(node_id=f"node-{status.value}", status=status)
            assert node.status == status
    
    def test_puct_initial_values(self, sample_node):
        """Test initial PUCT values."""
        node = sample_node()
        
        assert node.visits == 0
        assert node.prior == 0.5
        assert node.avg_value == 0.0
    
    def test_puct_visits_increment(self, sample_node):
        """Test incrementing visits."""
        node = sample_node(visits=0)
        
        node.visits = 1
        assert node.visits == 1
        
        node.visits = 10
        assert node.visits == 10
    
    def test_puct_prior_update(self, sample_node):
        """Test updating prior probability."""
        node = sample_node(prior=0.5)
        
        node.prior = 0.0
        assert node.prior == 0.0
        
        node.prior = 1.0
        assert node.prior == 1.0
        
        node.prior = 0.75
        assert node.prior == 0.75
    
    def test_puct_avg_value_update(self, sample_node):
        """Test updating average value."""
        node = sample_node(avg_value=0.0)
        
        node.avg_value = -1.0
        assert node.avg_value == -1.0
        
        node.avg_value = 1.0
        assert node.avg_value == 1.0
        
        node.avg_value = 0.5
        assert node.avg_value == 0.5
    
    def test_add_single_artifact(self, sample_node, sample_artifact):
        """Test adding a single artifact to node."""
        node = sample_node()
        artifact = sample_artifact()
        
        node.artifacts.append(artifact)
        
        assert len(node.artifacts) == 1
        assert node.artifacts[0] == artifact
    
    def test_add_multiple_artifacts(self, sample_node, sample_artifact):
        """Test adding multiple artifacts."""
        node = sample_node()
        
        artifact1 = sample_artifact(locator="https://example1.com")
        artifact2 = sample_artifact(locator="https://example2.com")
        artifact3 = sample_artifact(locator="https://example3.com")
        
        node.artifacts.extend([artifact1, artifact2, artifact3])
        
        assert len(node.artifacts) == 3
        assert artifact1 in node.artifacts
        assert artifact2 in node.artifacts
        assert artifact3 in node.artifacts
    
    def test_artifacts_default_empty(self, sample_node):
        """Test artifacts list is empty by default."""
        node = sample_node()
        
        assert node.artifacts == []
        assert len(node.artifacts) == 0
    
    def test_artifact_types(self, sample_node, sample_artifact):
        """Test adding artifacts of different types."""
        node = sample_node()
        
        artifact_types = [
            (ArtifactType.URL, "https://example.com"),
            (ArtifactType.FILE, "/path/to/file"),
            (ArtifactType.CODE, "code_snippet"),
            (ArtifactType.SNIPPET, "text_snippet"),
            (ArtifactType.PLOT, "plot.png"),
            (ArtifactType.DATASET, "data.csv"),
            (ArtifactType.SUMMARY, "summary_text"),
        ]
        
        for artifact_type, locator in artifact_types:
            artifact = sample_artifact(kind=artifact_type, locator=locator)
            node.artifacts.append(artifact)
        
        assert len(node.artifacts) == 7
        for i, (artifact_type, _) in enumerate(artifact_types):
            assert node.artifacts[i].kind == artifact_type
    
    def test_cost_accumulation(self, sample_node):
        """Test setting cost."""
        node = sample_node(cost=0.0)
        
        node.cost = 1.5
        assert node.cost == 1.5
        
        node.cost = 10.0
        assert node.cost == 10.0
    
    def test_tokens_accumulation(self, sample_node):
        """Test setting tokens_used."""
        node = sample_node(tokens_used=0)
        
        node.tokens_used = 100
        assert node.tokens_used == 100
        
        node.tokens_used = 5000
        assert node.tokens_used == 5000
    
    def test_iterations_tracking(self, sample_node):
        """Test setting iterations."""
        node = sample_node(iterations=0)
        
        node.iterations = 1
        assert node.iterations == 1
        
        node.iterations = 10
        assert node.iterations == 10
    
    def test_cost_and_tokens_together(self, sample_node):
        """Test setting both cost and tokens."""
        node = sample_node(cost=2.5, tokens_used=1500)
        
        assert node.cost == 2.5
        assert node.tokens_used == 1500
    
    def test_created_at_auto_set(self, sample_node):
        """Test created_at is automatically set."""
        node = sample_node()
        
        assert node.created_at is not None
        assert isinstance(node.created_at, datetime)
    
    def test_started_at_initially_none(self, sample_node):
        """Test started_at is None initially."""
        node = sample_node()
        
        assert node.started_at is None
    
    def test_completed_at_initially_none(self, sample_node):
        """Test completed_at is None initially."""
        node = sample_node()
        
        assert node.completed_at is None
    
    def test_timestamp_ordering(self, sample_node):
        """Test timestamp ordering: created_at < started_at < completed_at."""
        node = sample_node()
        now = datetime.now()
        
        node.started_at = now + timedelta(seconds=1)
        node.completed_at = now + timedelta(seconds=2)
        
        assert node.created_at < node.started_at
        assert node.started_at < node.completed_at
    
    def test_adapter_field(self, sample_node):
        """Test setting adapter name."""
        node = sample_node(adapter="gpt-4")
        
        assert node.adapter == "gpt-4"
    
    def test_metadata_dict(self, sample_node):
        """Test adding metadata key-value pairs."""
        node = sample_node(metadata={"key1": "value1", "key2": 42})
        
        assert node.metadata["key1"] == "value1"
        assert node.metadata["key2"] == 42
    
    def test_metadata_default_empty(self, sample_node):
        """Test metadata is empty dict by default."""
        node = sample_node()
        
        assert node.metadata == {}
    
    def test_to_dict_minimal(self, sample_node):
        """Test to_dict on minimal node."""
        node = sample_node(
            node_id="min-1",
            node_type=NodeType.IDEA,
            title="Min Node",
            content="Min content"
        )
        
        result = node.to_dict()
        
        assert result["id"] == "min-1"
        assert result["type"] == "idea"
        assert result["title"] == "Min Node"
        assert result["content"] == "Min content"
        assert result["status"] == "pending"
        assert "created_at" in result
        assert "artifacts" in result
    
    def test_to_dict_full(self, sample_node, sample_artifact):
        """Test to_dict on fully populated node."""
        artifact = sample_artifact()
        node = sample_node(
            node_id="full-1",
            node_type=NodeType.HYPOTHESIS,
            title="Full Node",
            content="Full content",
            status=NodeStatus.COMPLETE,
            parent_id="parent-1",
            score=0.9,
            confidence=0.85,
            artifacts=[artifact],
            cost=2.0,
            tokens_used=2000,
            visits=10,
            prior=0.8,
            avg_value=0.7,
            adapter="gpt-4",
            metadata={"test": "data"},
            iterations=5,
        )
        
        result = node.to_dict()
        
        assert result["id"] == "full-1"
        assert result["type"] == "hypothesis"
        assert result["status"] == "complete"
        assert result["parent_id"] == "parent-1"
        assert result["score"] == 0.9
        assert result["confidence"] == 0.85
        assert result["cost"] == 2.0
        assert result["tokens_used"] == 2000
        assert result["visits"] == 10
        assert result["prior"] == 0.8
        assert result["avg_value"] == 0.7
        assert result["adapter"] == "gpt-4"
        assert result["metadata"] == {"test": "data"}
        assert result["iterations"] == 5
    
    def test_to_dict_artifacts(self, sample_node, sample_artifact):
        """Test artifacts are serialized as list of dicts."""
        artifact1 = sample_artifact(locator="https://example1.com")
        artifact2 = sample_artifact(locator="https://example2.com")
        node = sample_node(artifacts=[artifact1, artifact2])
        
        result = node.to_dict()
        
        assert "artifacts" in result
        assert isinstance(result["artifacts"], list)
        assert len(result["artifacts"]) == 2
        assert isinstance(result["artifacts"][0], dict)
    
    def test_to_dict_timestamps(self, sample_node):
        """Test timestamps are ISO format strings."""
        node = sample_node()
        node.started_at = datetime.now()
        node.completed_at = datetime.now()
        
        result = node.to_dict()
        
        assert "created_at" in result
        assert isinstance(result["created_at"], str)
        if result.get("started_at"):
            assert isinstance(result["started_at"], str)
        if result.get("completed_at"):
            assert isinstance(result["completed_at"], str)
    
    def test_to_dict_enums(self, sample_node):
        """Test enums are serialized as string values."""
        node = sample_node(
            node_type=NodeType.HYPOTHESIS,
            status=NodeStatus.RUNNING
        )
        
        result = node.to_dict()
        
        assert result["type"] == "hypothesis"
        assert isinstance(result["type"], str)
        assert result["status"] == "running"
        assert isinstance(result["status"], str)
    
    def test_to_dict_none_values(self, sample_node):
        """Test None values are preserved in dict."""
        node = sample_node(parent_id=None, score=None, confidence=None)
        
        result = node.to_dict()
        
        assert result["parent_id"] is None
        assert result["score"] is None
        assert result["confidence"] is None


    
    def test_to_dict_minimal_explicit(self, sample_node):
        """Test to_dict on minimal node with explicit field verification (Comment 1)."""
        node = sample_node(
            node_id="min-explicit",
            node_type=NodeType.IDEA,
            title="Min Node Explicit",
            content="Min content explicit"
        )
        
        result = node.to_dict()
        
        # Verify all required fields
        assert result["id"] == "min-explicit"
        assert result["type"] == "idea"  # String enum
        assert isinstance(result["type"], str)
        assert result["title"] == "Min Node Explicit"
        assert result["content"] == "Min content explicit"
        assert result["status"] == "pending"  # String enum
        assert isinstance(result["status"], str)
        
        # Verify PUCT defaults (Comment 1)
        assert result["visits"] == 0
        assert result["prior"] == 0.5
        assert result["avg_value"] == 0.0
        
        # Verify empty artifacts list (Comment 1)
        assert result["artifacts"] == []
        assert isinstance(result["artifacts"], list)
        
        # Verify ISO-formatted created_at (Comment 1)
        assert "created_at" in result
        assert isinstance(result["created_at"], str)
        # Verify it's a valid ISO format
        datetime.fromisoformat(result["created_at"].replace('Z', '+00:00'))
        
        # Verify None fields (Comment 1)
        assert result["parent_id"] is None
        assert result["started_at"] is None
        assert result["completed_at"] is None
    
    def test_to_dict_full_explicit(self, sample_node, sample_artifact):
        """Test to_dict on fully populated node with explicit verification (Comment 1)."""
        artifact = sample_artifact(locator="https://example.com", content="test content")
        now = datetime.now()
        started = now - timedelta(seconds=10)
        completed = now
        
        node = sample_node(
            node_id="full-explicit",
            node_type=NodeType.HYPOTHESIS,
            title="Full Node Explicit",
            content="Full content explicit",
            status=NodeStatus.COMPLETE,
            parent_id="parent-explicit",
            score=0.9,
            confidence=0.85,
            artifacts=[artifact],
            cost=2.0,
            tokens_used=2000,
            visits=10,
            prior=0.8,
            avg_value=0.7,
            adapter="gpt-4",
            metadata={"test": "data"},
            iterations=5,
        )
        node.started_at = started
        node.completed_at = completed
        
        result = node.to_dict()
        
        # Verify all numeric and string fields match (Comment 1)
        assert result["id"] == "full-explicit"
        assert result["type"] == "hypothesis"
        assert result["status"] == "complete"
        assert result["parent_id"] == "parent-explicit"
        assert result["score"] == 0.9
        assert result["confidence"] == 0.85
        assert result["cost"] == 2.0
        assert result["tokens_used"] == 2000
        assert result["visits"] == 10
        assert result["prior"] == 0.8
        assert result["avg_value"] == 0.7
        assert result["adapter"] == "gpt-4"
        assert result["metadata"] == {"test": "data"}
        assert result["iterations"] == 5
        
        # Verify artifacts are list of dicts (from model_dump()) (Comment 1)
        assert "artifacts" in result
        assert isinstance(result["artifacts"], list)
        assert len(result["artifacts"]) == 1
        assert isinstance(result["artifacts"][0], dict)
        assert "locator" in result["artifacts"][0]
        assert result["artifacts"][0]["locator"] == "https://example.com"
        
        # Verify started_at and completed_at are ISO strings (Comment 1)
        assert "started_at" in result
        assert result["started_at"] is not None
        assert isinstance(result["started_at"], str)
        assert "completed_at" in result
        assert result["completed_at"] is not None
        assert isinstance(result["completed_at"], str)
        
        # Verify ISO format is valid (Comment 1)
        datetime.fromisoformat(result["started_at"].replace('Z', '+00:00'))
        datetime.fromisoformat(result["completed_at"].replace('Z', '+00:00'))
    
    def test_puct_mutations(self, sample_node):
        """Test PUCT field mutations and serialization (Comment 5)."""
        node = sample_node()
        
        # Verify initial PUCT defaults
        assert node.visits == 0
        assert node.prior == 0.5
        assert node.avg_value == 0.0
        
        # Mutate PUCT fields
        node.visits = 3
        node.prior = 0.8
        node.avg_value = 0.25
        
        # Verify mutations are retained on node
        assert node.visits == 3
        assert node.prior == 0.8
        assert node.avg_value == 0.25
        
        # Verify mutations are reflected in to_dict()
        result = node.to_dict()
        assert result["visits"] == 3
        assert result["prior"] == 0.8
        assert result["avg_value"] == 0.25

# ============================================================================
# Test ResearchTree
# ============================================================================

@pytest.mark.unit
class TestResearchTree:
    """Tests for ResearchTree model."""
    
    def test_tree_creation(self, sample_tree):
        """Test creating an empty tree."""
        tree = sample_tree(research_id="test-research-1")
        
        assert tree.research_id == "test-research-1"
        assert tree.nodes == {}
        assert tree.edges == []
        assert tree.version == 0
        assert isinstance(tree.stats, dict)
    
    def test_tree_default_stats(self, sample_tree):
        """Test default stats dict has correct keys."""
        tree = sample_tree()
        
        assert "created" in tree.stats
        assert "expanded" in tree.stats
        assert "complete" in tree.stats
        assert "total_cost" in tree.stats
        assert "total_tokens" in tree.stats
        assert tree.stats["created"] == 0
        assert tree.stats["expanded"] == 0
        assert tree.stats["complete"] == 0
        assert tree.stats["total_cost"] == 0.0
        assert tree.stats["total_tokens"] == 0
    
    def test_add_root_node(self, sample_tree, sample_node):
        """Test adding a root node without parent."""
        tree = sample_tree()
        root = sample_node(node_id="root", node_type=NodeType.ROOT)
        
        initial_version = tree.version
        tree.add_node(root)
        
        assert "root" in tree.nodes
        assert tree.nodes["root"] == root
        assert len(tree.edges) == 0
        assert tree.stats["created"] == 1
        assert tree.version == initial_version + 1
    
    def test_add_child_node_with_parent_object(self, sample_tree, sample_node):
        """Test adding a child node with parent object."""
        tree = sample_tree()
        parent = sample_node(node_id="parent", node_type=NodeType.ROOT)
        child = sample_node(node_id="child", node_type=NodeType.IDEA)
        
        tree.add_node(parent)
        tree.add_node(child, parent=parent)
        
        assert "child" in tree.nodes
        assert child.parent_id == "parent"
        assert len(tree.edges) == 1
        assert tree.edges[0].parent_id == "parent"
        assert tree.edges[0].child_id == "child"
    
    def test_add_child_node_with_parent_id(self, sample_tree, sample_node):
        """Test adding a child node with parent_id string."""
        tree = sample_tree()
        parent = sample_node(node_id="parent", node_type=NodeType.ROOT)
        child = sample_node(node_id="child", node_type=NodeType.IDEA)
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent")
        
        assert "child" in tree.nodes
        assert child.parent_id == "parent"
        assert len(tree.edges) == 1
    
    def test_add_multiple_children(self, sample_tree, sample_node):
        """Test adding multiple children to same parent."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        child1 = sample_node(node_id="child1")
        child2 = sample_node(node_id="child2")
        child3 = sample_node(node_id="child3")
        
        tree.add_node(parent)
        tree.add_node(child1, parent_id="parent")
        tree.add_node(child2, parent_id="parent")
        tree.add_node(child3, parent_id="parent")
        
        assert len(tree.edges) == 3
        parent_edges = [e for e in tree.edges if e.parent_id == "parent"]
        assert len(parent_edges) == 3
    
    def test_add_node_increments_version(self, sample_tree, sample_node):
        """Test version increments with each node addition."""
        tree = sample_tree()
        
        assert tree.version == 0
        
        tree.add_node(sample_node(node_id="node1"))
        assert tree.version == 1
        
        tree.add_node(sample_node(node_id="node2"))
        assert tree.version == 2
        
        tree.add_node(sample_node(node_id="node3"))
        assert tree.version == 3
    
    def test_add_node_updates_stats(self, sample_tree, sample_node):
        """Test stats.created increments with node addition."""
        tree = sample_tree()
        
        assert tree.stats["created"] == 0
        
        tree.add_node(sample_node(node_id="node1"))
        assert tree.stats["created"] == 1
        
        tree.add_node(sample_node(node_id="node2"))
        assert tree.stats["created"] == 2
    
    def test_edge_creation(self, sample_tree, sample_node):
        """Test ResearchEdge is created with correct attributes."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        child = sample_node(node_id="child")
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent")
        
        edge = tree.edges[0]
        assert isinstance(edge, ResearchEdge)
        assert edge.parent_id == "parent"
        assert edge.child_id == "child"
    
    def test_edge_relation_default(self, sample_tree, sample_node):
        """Test edge relation defaults to 'child_of'."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        child = sample_node(node_id="child")
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent")
        
        edge = tree.edges[0]
        assert edge.relation == "child_of"
    
    def test_multiple_edges_same_parent(self, sample_tree, sample_node):
        """Test multiple edges can have same parent_id."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        
        tree.add_node(parent)
        tree.add_node(sample_node(node_id="child1"), parent_id="parent")
        tree.add_node(sample_node(node_id="child2"), parent_id="parent")
        
        parent_edges = [e for e in tree.edges if e.parent_id == "parent"]
        assert len(parent_edges) == 2

    
    def test_edge_creation_explicit(self, sample_tree, sample_node):
        """Test ResearchEdge creation with explicit structure verification (Comment 2)."""
        tree = sample_tree()
        parent = sample_node(node_id="parent-explicit")
        child = sample_node(node_id="child-explicit")
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent-explicit")
        
        # Verify edge structure (Comment 2)
        assert len(tree.edges) == 1
        edge = tree.edges[0]
        assert isinstance(edge, ResearchEdge)
        assert edge.parent_id == "parent-explicit"
        assert edge.child_id == "child-explicit"
        assert edge.relation == "child_of"  # Default relation
    
    def test_edge_relation_default_explicit(self, sample_tree, sample_node):
        """Test edge relation defaults to 'child_of' explicitly (Comment 2)."""
        tree = sample_tree()
        parent = sample_node(node_id="parent-rel")
        child = sample_node(node_id="child-rel")
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent-rel")
        
        # Explicitly verify relation field (Comment 2)
        edge = tree.edges[0]
        assert hasattr(edge, 'relation')
        assert edge.relation == "child_of"
        assert isinstance(edge.relation, str)
    
    def test_multiple_edges_same_parent_explicit(self, sample_tree, sample_node):
        """Test multiple edges with same parent_id explicitly (Comment 2)."""
        tree = sample_tree()
        parent = sample_node(node_id="parent-multi")
        
        tree.add_node(parent)
        tree.add_node(sample_node(node_id="child-multi-1"), parent_id="parent-multi")
        tree.add_node(sample_node(node_id="child-multi-2"), parent_id="parent-multi")
        tree.add_node(sample_node(node_id="child-multi-3"), parent_id="parent-multi")
        
        # Verify total edges (Comment 2)
        assert len(tree.edges) == 3
        
        # Verify all have same parent_id (Comment 2)
        parent_edges = [e for e in tree.edges if e.parent_id == "parent-multi"]
        assert len(parent_edges) == 3
        
        # Verify each edge has correct structure (Comment 2)
        child_ids = {e.child_id for e in parent_edges}
        assert child_ids == {"child-multi-1", "child-multi-2", "child-multi-3"}
        
        # Verify all edges have default relation (Comment 2)
        for edge in parent_edges:
            assert edge.relation == "child_of"
    
    def test_get_children_empty(self, sample_tree, sample_node):
        """Test get_children returns empty list for node with no children."""
        tree = sample_tree()
        node = sample_node(node_id="lonely")
        tree.add_node(node)
        
        children = tree.get_children("lonely")
        
        assert children == []
    
    def test_get_children_single(self, sample_tree, sample_node):
        """Test get_children returns single child."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        child = sample_node(node_id="child")
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent")
        
        children = tree.get_children("parent")
        
        assert len(children) == 1
        assert children[0].id == "child"
    
    def test_get_children_multiple(self, sample_tree, sample_node):
        """Test get_children returns all children."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        
        tree.add_node(parent)
        tree.add_node(sample_node(node_id="child1"), parent_id="parent")
        tree.add_node(sample_node(node_id="child2"), parent_id="parent")
        tree.add_node(sample_node(node_id="child3"), parent_id="parent")
        
        children = tree.get_children("parent")
        
        assert len(children) == 3
        child_ids = [c.id for c in children]
        assert "child1" in child_ids
        assert "child2" in child_ids
        assert "child3" in child_ids
    
    def test_get_children_nonexistent_node(self, sample_tree):
        """Test get_children returns empty list for nonexistent node."""
        tree = sample_tree()
        
        children = tree.get_children("does-not-exist")
        
        assert children == []
    
    def test_get_children_order(self, sample_tree, sample_node):
        """Test children are returned in order they were added."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        
        tree.add_node(parent)
        tree.add_node(sample_node(node_id="first"), parent_id="parent")
        tree.add_node(sample_node(node_id="second"), parent_id="parent")
        tree.add_node(sample_node(node_id="third"), parent_id="parent")
        
        children = tree.get_children("parent")
        
        assert children[0].id == "first"
        assert children[1].id == "second"
        assert children[2].id == "third"
    
    def test_get_parent_root_node(self, sample_tree, sample_node):
        """Test get_parent returns None for root node."""
        tree = sample_tree()
        root = sample_node(node_id="root")
        tree.add_node(root)
        
        parent_id = tree.get_parent("root")
        
        assert parent_id is None
    
    def test_get_parent_child_node(self, sample_tree, sample_node):
        """Test get_parent returns correct parent_id."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        child = sample_node(node_id="child")
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent")
        
        parent_id = tree.get_parent("child")
        
        assert parent_id == "parent"
    
    def test_get_parent_nonexistent_node(self, sample_tree):
        """Test get_parent returns None for nonexistent node."""
        tree = sample_tree()
        
        parent_id = tree.get_parent("does-not-exist")
        
        assert parent_id is None
    
    def test_path_to_root_single_node(self, sample_tree, sample_node):
        """Test path to root for a single root node."""
        tree = sample_tree()
        root = sample_node(node_id="root")
        tree.add_node(root)
        
        path = tree.get_path_to_root("root")
        
        assert len(path) == 1
        assert path[0].id == "root"
    
    def test_path_to_root_two_levels(self, sample_tree, sample_node):
        """Test path to root for child of root."""
        tree = sample_tree()
        root = sample_node(node_id="root")
        child = sample_node(node_id="child")
        
        tree.add_node(root)
        tree.add_node(child, parent_id="root")
        
        path = tree.get_path_to_root("child")
        
        assert len(path) == 2
        assert [n.id for n in path] == ["root", "child"]
    
    def test_path_to_root_three_levels(self, sample_tree, sample_node):
        """Test path to root for grandchild."""
        tree = sample_tree()
        root = sample_node(node_id="root")
        child = sample_node(node_id="child")
        grandchild = sample_node(node_id="grandchild")
        
        tree.add_node(root)
        tree.add_node(child, parent_id="root")
        tree.add_node(grandchild, parent_id="child")
        
        path = tree.get_path_to_root("grandchild")
        
        assert len(path) == 3
        assert [n.id for n in path] == ["root", "child", "grandchild"]
    
    def test_path_to_root_nonexistent_node(self, sample_tree):
        """Test path to root for nonexistent node."""
        tree = sample_tree()
        
        path = tree.get_path_to_root("does-not-exist")
        
        assert path == []
    
    def test_path_to_root_orphan_node(self, sample_tree, sample_node):
        """Test path stops at orphan node with invalid parent_id."""
        tree = sample_tree()
        orphan = sample_node(node_id="orphan", parent_id="missing-parent")
        tree.add_node(orphan)
        
        path = tree.get_path_to_root("orphan")
        
        # Should return path up to the orphan
        assert len(path) == 1
        assert path[0].id == "orphan"
    
    def test_max_depth_empty_tree(self, sample_tree):
        """Test max depth of empty tree is 0."""
        tree = sample_tree()
        
        depth = tree.calculate_max_depth()
        
        assert depth == 0
    
    def test_max_depth_single_node(self, sample_tree, sample_node):
        """Test max depth of tree with only root is 0."""
        tree = sample_tree()
        tree.add_node(sample_node(node_id="root"))
        
        depth = tree.calculate_max_depth()
        
        assert depth == 0
    
    def test_max_depth_two_levels(self, sample_tree, sample_node):
        """Test max depth of tree with root and children is 1."""
        tree = sample_tree()
        root = sample_node(node_id="root")
        
        tree.add_node(root)
        tree.add_node(sample_node(node_id="child1"), parent_id="root")
        tree.add_node(sample_node(node_id="child2"), parent_id="root")
        
        depth = tree.calculate_max_depth()
        
        assert depth == 1
    
    def test_max_depth_three_levels(self, sample_tree, sample_node):
        """Test max depth with root, children, and grandchildren."""
        tree = sample_tree()
        root = sample_node(node_id="root")
        child = sample_node(node_id="child")
        grandchild = sample_node(node_id="grandchild")
        
        tree.add_node(root)
        tree.add_node(child, parent_id="root")
        tree.add_node(grandchild, parent_id="child")
        
        depth = tree.calculate_max_depth()
        
        assert depth == 2
    
    def test_max_depth_unbalanced_tree(self, sample_tree, sample_node):
        """Test max depth with unbalanced tree."""
        tree = sample_tree()
        root = sample_node(node_id="root")
        
        tree.add_node(root)
        
        # Branch 1: depth 3
        tree.add_node(sample_node(node_id="c1"), parent_id="root")
        tree.add_node(sample_node(node_id="c1-1"), parent_id="c1")
        tree.add_node(sample_node(node_id="c1-1-1"), parent_id="c1-1")
        
        # Branch 2: depth 1
        tree.add_node(sample_node(node_id="c2"), parent_id="root")
        
        depth = tree.calculate_max_depth()
        
        assert depth == 3
    
    def test_max_depth_multiple_roots(self, sample_tree, sample_node):
        """Test max depth with multiple root nodes."""
        tree = sample_tree()
        
        # Root 1 with depth 2
        tree.add_node(sample_node(node_id="root1"))
        tree.add_node(sample_node(node_id="r1-c1"), parent_id="root1")
        tree.add_node(sample_node(node_id="r1-c1-1"), parent_id="r1-c1")
        
        # Root 2 with depth 1
        tree.add_node(sample_node(node_id="root2"))
        tree.add_node(sample_node(node_id="r2-c1"), parent_id="root2")
        
        depth = tree.calculate_max_depth()
        
        assert depth == 2

    
    def test_max_depth_with_cycle_protection(self, sample_tree, sample_node):
        """Test max depth calculation with cycle protection (Comment 4)."""
        tree = sample_tree()
        
        # Create a tree structure
        root = sample_node(node_id="root-cycle")
        child = sample_node(node_id="child-cycle")
        grandchild = sample_node(node_id="grandchild-cycle")
        
        tree.add_node(root)
        tree.add_node(child, parent_id="root-cycle")
        tree.add_node(grandchild, parent_id="child-cycle")
        
        # Normal depth should be 2
        depth = tree.calculate_max_depth()
        assert depth == 2
        
        # Manually create a cycle by injecting a back-edge
        # (This tests that _calculate_depth_from_node uses visited set)
        # Add an edge that creates a cycle: grandchild -> root
        tree.edges.append(ResearchEdge(parent_id="grandchild-cycle", child_id="root-cycle"))
        
        # Should still return finite depth due to visited set
        depth = tree.calculate_max_depth()
        assert isinstance(depth, int)
        assert depth >= 0
        assert depth < 100  # Ensure it doesn't loop infinitely
    
    def test_calculate_depth_from_leaf(self, sample_tree, sample_node):
        """Test calculating depth from a leaf node."""
        tree = sample_tree()
        tree.add_node(sample_node(node_id="leaf"))
        
        depth = tree._calculate_depth_from_node("leaf", set())
        
        assert depth == 0
    
    def test_calculate_depth_from_parent(self, sample_tree, sample_node):
        """Test calculating depth from a parent node."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        child = sample_node(node_id="child")
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent")
        
        depth = tree._calculate_depth_from_node("parent", set())
        
        assert depth == 1
    
    def test_calculate_depth_with_visited_set(self, sample_tree, sample_node):
        """Test depth calculation with visited set for cycle detection."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        child = sample_node(node_id="child")
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent")
        
        visited = {"parent"}
        depth = tree._calculate_depth_from_node("parent", visited)
        
        # Should return 0 since parent is already visited
        assert depth == 0
    
    def test_calculate_depth_nonexistent_node(self, sample_tree):
        """Test calculating depth from nonexistent node."""
        tree = sample_tree()
        
        depth = tree._calculate_depth_from_node("does-not-exist", set())
        
        assert depth == 0
    
    def test_stats_created_increments(self, sample_tree, sample_node):
        """Test stats.created increments with node additions."""
        tree = sample_tree()
        
        assert tree.stats["created"] == 0
        
        for i in range(5):
            tree.add_node(sample_node(node_id=f"node{i}"))
        
        assert tree.stats["created"] == 5
    
    def test_stats_initial_values(self, sample_tree):
        """Test stats has correct initial values."""
        tree = sample_tree()
        
        assert tree.stats["created"] == 0
        assert tree.stats["expanded"] == 0
        assert tree.stats["complete"] == 0
        assert tree.stats["total_cost"] == 0.0
        assert tree.stats["total_tokens"] == 0
    
    def test_stats_persistence(self, sample_tree, sample_node):
        """Test stats changes persist."""
        tree = sample_tree()
        
        tree.stats["expanded"] = 5
        tree.stats["complete"] = 3
        tree.stats["total_cost"] = 10.5
        tree.stats["total_tokens"] = 5000
        
        assert tree.stats["expanded"] == 5
        assert tree.stats["complete"] == 3
        assert tree.stats["total_cost"] == 10.5
        assert tree.stats["total_tokens"] == 5000
    
    def test_version_initial_zero(self, sample_tree):
        """Test version starts at 0."""
        tree = sample_tree()
        
        assert tree.version == 0
    
    def test_version_increments_on_add(self, sample_tree, sample_node):
        """Test version increments with each add."""
        tree = sample_tree()
        
        tree.add_node(sample_node(node_id="node1"))
        assert tree.version == 1
        
        tree.add_node(sample_node(node_id="node2"))
        assert tree.version == 2
    
    def test_tree_to_dict_empty(self, sample_tree):
        """Test serializing empty tree."""
        tree = sample_tree(research_id="test-1")
        
        result = tree.to_dict()
        
        assert result["research_id"] == "test-1"
        assert result["nodes"] == {}
        assert result["edges"] == []
        assert "stats" in result
        assert result["version"] == 0
    
    def test_tree_to_dict_with_nodes(self, sample_tree, sample_node):
        """Test serializing tree with nodes."""
        tree = sample_tree()
        node1 = sample_node(node_id="node1")
        node2 = sample_node(node_id="node2")
        
        tree.add_node(node1)
        tree.add_node(node2)
        
        result = tree.to_dict()
        
        assert "nodes" in result
        assert isinstance(result["nodes"], dict)
        assert "node1" in result["nodes"]
        assert "node2" in result["nodes"]
        assert isinstance(result["nodes"]["node1"], dict)
    
    def test_tree_to_dict_with_edges(self, sample_tree, sample_node):
        """Test serializing tree with edges."""
        tree = sample_tree()
        parent = sample_node(node_id="parent")
        child = sample_node(node_id="child")
        
        tree.add_node(parent)
        tree.add_node(child, parent_id="parent")
        
        result = tree.to_dict()
        
        assert "edges" in result
        assert isinstance(result["edges"], list)
        assert len(result["edges"]) == 1
        edge = result["edges"][0]
        assert edge["from"] == "parent"
        assert edge["to"] == "child"
        assert edge["relation"] == "child_of"
    
    def test_tree_to_dict_stats(self, sample_tree):
        """Test stats dict is included in serialization."""
        tree = sample_tree()
        tree.stats["expanded"] = 5
        
        result = tree.to_dict()
        
        assert "stats" in result
        assert isinstance(result["stats"], dict)
        assert result["stats"]["expanded"] == 5
    
    def test_tree_to_dict_version(self, sample_tree, sample_node):
        """Test version is included in serialization."""
        tree = sample_tree()
        tree.add_node(sample_node(node_id="node1"))
        
        result = tree.to_dict()
        
        assert "version" in result
        assert result["version"] == 1


# ============================================================================
# Test Budget
# ============================================================================

@pytest.mark.unit
class TestBudget:
    """Tests for Budget model."""
    
    def test_budget_defaults(self):
        """Test Budget with default values."""
        budget = Budget()
        
        assert budget.max_iterations == 10
        assert budget.max_cost == 10.0
        assert budget.max_tokens is None
        assert budget.deadline is None
    
    def test_budget_custom_values(self):
        """Test Budget with custom values."""
        deadline = datetime.now() + timedelta(hours=1)
        budget = Budget(
            max_iterations=20,
            max_cost=50.0,
            max_tokens=100000,
            deadline=deadline
        )
        
        assert budget.max_iterations == 20
        assert budget.max_cost == 50.0
        assert budget.max_tokens == 100000
        assert budget.deadline == deadline
    
    def test_budget_max_iterations(self):
        """Test setting max_iterations."""
        budget = Budget(max_iterations=15)
        
        assert budget.max_iterations == 15
    
    def test_budget_max_cost(self):
        """Test setting max_cost."""
        budget = Budget(max_cost=25.5)
        
        assert budget.max_cost == 25.5
    
    def test_budget_max_tokens(self):
        """Test setting max_tokens."""
        budget = Budget(max_tokens=50000)
        
        assert budget.max_tokens == 50000
    
    def test_budget_deadline(self):
        """Test setting deadline."""
        deadline = datetime.now() + timedelta(days=1)
        budget = Budget(deadline=deadline)
        
        assert budget.deadline == deadline
    
    def test_budget_zero_values(self):
        """Test Budget allows zero values."""
        budget = Budget(max_iterations=0, max_cost=0.0, max_tokens=0)
        
        assert budget.max_iterations == 0
        assert budget.max_cost == 0.0
        assert budget.max_tokens == 0


# ============================================================================
# Test Task
# ============================================================================

@pytest.mark.unit
class TestTask:
    """Tests for Task model."""
    
    def test_task_minimal(self):
        """Test Task with only goal."""
        task = Task(goal="Solve the problem")
        
        assert task.goal == "Solve the problem"
        assert task.id is None
        assert task.context is None
        assert isinstance(task.budget, Budget)
        assert task.constraints == {}
    
    def test_task_with_id(self):
        """Test Task with id."""
        task = Task(goal="Test", id="task-123")
        
        assert task.id == "task-123"
    
    def test_task_with_context(self):
        """Test Task with context string."""
        task = Task(goal="Test", context="Background information")
        
        assert task.context == "Background information"
    
    def test_task_with_budget(self):
        """Test Task with custom Budget."""
        custom_budget = Budget(max_iterations=20, max_cost=50.0)
        task = Task(goal="Test", budget=custom_budget)
        
        assert task.budget == custom_budget
        assert task.budget.max_iterations == 20
    
    def test_task_with_constraints(self):
        """Test Task with constraints dict."""
        constraints = {"min_confidence": 0.8, "required_sources": 3}
        task = Task(goal="Test", constraints=constraints)
        
        assert task.constraints == constraints
    
    def test_task_full(self):
        """Test Task with all fields."""
        budget = Budget(max_iterations=15)
        constraints = {"key": "value"}
        task = Task(
            goal="Full task",
            id="task-full",
            context="Full context",
            budget=budget,
            constraints=constraints
        )
        
        assert task.goal == "Full task"
        assert task.id == "task-full"
        assert task.context == "Full context"
        assert task.budget == budget
        assert task.constraints == constraints
    
    def test_task_budget_default(self):
        """Test budget defaults to Budget() instance."""
        task = Task(goal="Test")
        
        assert isinstance(task.budget, Budget)
        assert task.budget.max_iterations == 10
    
    def test_task_constraints_default(self):
        """Test constraints defaults to empty dict."""
        task = Task(goal="Test")
        
        assert task.constraints == {}


# ============================================================================
# Test Context
# ============================================================================

@pytest.mark.unit
class TestContext:
    """Tests for Context model."""
    
    def test_context_empty(self):
        """Test Context with no arguments."""
        context = Context()
        
        assert context.branch_id is None
        assert context.parent_nodes == []
        assert context.tools == []
        assert context.secrets == {}
        assert context.workspace_dir is None
        assert context.metadata == {}
    
    def test_context_with_branch_id(self):
        """Test Context with branch_id."""
        context = Context(branch_id="branch-123")
        
        assert context.branch_id == "branch-123"
    
    def test_context_with_parent_nodes(self, sample_node):
        """Test Context with list of ResearchNode objects."""
        node1 = sample_node(node_id="node1")
        node2 = sample_node(node_id="node2")
        context = Context(parent_nodes=[node1, node2])
        
        assert len(context.parent_nodes) == 2
        assert context.parent_nodes[0] == node1
        assert context.parent_nodes[1] == node2
    
    def test_context_with_tools(self):
        """Test Context with list of tool names."""
        tools = ["web_search", "code_search", "analysis"]
        context = Context(tools=tools)
        
        assert context.tools == tools
    
    def test_context_with_secrets(self):
        """Test Context with secrets dict."""
        secrets = {"api_key": "secret123", "token": "abc456"}
        context = Context(secrets=secrets)
        
        assert context.secrets == secrets
    
    def test_context_with_workspace_dir(self):
        """Test Context with workspace_dir path."""
        context = Context(workspace_dir="/path/to/workspace")
        
        assert context.workspace_dir == "/path/to/workspace"
    
    def test_context_with_metadata(self):
        """Test Context with metadata dict."""
        metadata = {"user": "test_user", "session": "abc123"}
        context = Context(metadata=metadata)
        
        assert context.metadata == metadata
    
    def test_context_full(self, sample_node):
        """Test Context with all fields."""
        node = sample_node(node_id="node1")
        context = Context(
            branch_id="branch-1",
            parent_nodes=[node],
            tools=["web_search"],
            secrets={"key": "value"},
            workspace_dir="/workspace",
            metadata={"info": "data"}
        )
        
        assert context.branch_id == "branch-1"
        assert len(context.parent_nodes) == 1
        assert context.tools == ["web_search"]
        assert context.secrets == {"key": "value"}
        assert context.workspace_dir == "/workspace"
        assert context.metadata == {"info": "data"}
    
    def test_context_parent_nodes_default(self):
        """Test parent_nodes defaults to empty list."""
        context = Context()
        
        assert context.parent_nodes == []
    
    def test_context_tools_default(self):
        """Test tools defaults to empty list."""
        context = Context()
        
        assert context.tools == []
    
    def test_context_secrets_default(self):
        """Test secrets defaults to empty dict."""
        context = Context()
        
        assert context.secrets == {}
    
    def test_context_metadata_default(self):
        """Test metadata defaults to empty dict."""
        context = Context()
        
        assert context.metadata == {}


# ============================================================================
# Integration Tests with populated_tree
# ============================================================================

@pytest.mark.unit
class TestPopulatedTree:
    """Tests using the populated_tree fixture."""
    
    def test_populated_tree_structure(self, populated_tree):
        """Test the populated tree has expected structure."""
        # Root should exist
        assert "root" in populated_tree.nodes
        
        # Should have 3 ideas
        ideas = [n for n in populated_tree.nodes.values() if n.type == NodeType.IDEA]
        assert len(ideas) == 3
        
        # Should have 5 hypotheses (2+2+1)
        hypotheses = [n for n in populated_tree.nodes.values() if n.type == NodeType.HYPOTHESIS]
        assert len(hypotheses) == 5
        
        # Should have 5 experiments
        experiments = [n for n in populated_tree.nodes.values() if n.type == NodeType.EXPERIMENT]
        assert len(experiments) == 5
    
    def test_populated_tree_max_depth(self, populated_tree):
        """Test max depth of populated tree."""
        depth = populated_tree.calculate_max_depth()
        
        # Root -> Idea -> Hypothesis -> Experiment = depth 3
        assert depth == 3
    
    def test_populated_tree_path_traversal(self, populated_tree):
        """Test path traversal in populated tree."""
        path = populated_tree.get_path_to_root("experiment-1-1-1")
        
        assert len(path) == 4
        assert [n.id for n in path] == ["root", "idea-1", "hypothesis-1-1", "experiment-1-1-1"]
    
    def test_populated_tree_children_count(self, populated_tree):
        """Test children counts at each level."""
        # Root should have 3 children (ideas)
        root_children = populated_tree.get_children("root")
        assert len(root_children) == 3
        
        # idea-1 should have 2 children (hypotheses)
        idea1_children = populated_tree.get_children("idea-1")
        assert len(idea1_children) == 2
        
        # hypothesis-1-1 should have 1 child (experiment)
        hyp_children = populated_tree.get_children("hypothesis-1-1")
        assert len(hyp_children) == 1
