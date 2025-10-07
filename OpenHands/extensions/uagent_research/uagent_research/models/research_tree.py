"""
Research Tree Models

Represents the hierarchical structure of research exploration.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from .events import Artifact


class NodeType(str, Enum):
    """Types of nodes in research tree"""
    ROOT = "root"
    IDEA = "idea"
    HYPOTHESIS = "hypothesis"
    PLAN = "plan"
    WEB_SEARCH = "web_search"
    CODE_SEARCH = "code_search"
    BROWSE = "browse"
    ANALYSIS = "analysis"
    EXPERIMENT = "experiment"
    RESULT = "result"
    CRITIQUE = "critique"
    SUMMARY = "summary"


class NodeStatus(str, Enum):
    """Status of a research node"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Budget(BaseModel):
    """Budget constraints for research execution"""
    max_iterations: int = 10
    max_cost: float = 10.0  # USD
    max_tokens: Optional[int] = None
    deadline: Optional[datetime] = None


class ResearchNode(BaseModel):
    """A node in the research tree"""
    id: str
    type: NodeType
    title: str
    content: str
    status: NodeStatus = NodeStatus.PENDING
    parent_id: Optional[str] = None

    # Scoring and metadata
    score: Optional[float] = None  # 0-1 quality score
    confidence: Optional[float] = None  # 0-1 confidence
    novelty: Optional[float] = None  # 0-1 novelty score

    # Execution tracking
    artifacts: List[Artifact] = Field(default_factory=list)
    cost: float = 0.0  # Accumulated cost in USD
    tokens_used: int = 0  # Total tokens
    iterations: int = 0  # Execution iterations

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Adapter metadata
    adapter: Optional[str] = None  # Which adapter executed this
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # PUCT scoring fields (for tree search)
    visits: int = 0  # Number of times visited
    prior: float = 0.5  # Prior probability
    avg_value: float = 0.0  # Average value from children

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "content": self.content,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "score": self.score,
            "confidence": self.confidence,
            "novelty": self.novelty,
            "artifacts": [a.model_dump() for a in self.artifacts],
            "cost": self.cost,
            "tokens_used": self.tokens_used,
            "iterations": self.iterations,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "adapter": self.adapter,
            "metadata": self.metadata,
            "visits": self.visits,
            "prior": self.prior,
            "avg_value": self.avg_value
        }


class ResearchEdge(BaseModel):
    """An edge in the research tree"""
    parent_id: str
    child_id: str
    relation: str = "child_of"  # Type of relationship


class ResearchTree(BaseModel):
    """Complete research tree"""
    research_id: str
    nodes: Dict[str, ResearchNode] = Field(default_factory=dict)
    edges: List[ResearchEdge] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=lambda: {
        "created": 0,
        "expanded": 0,
        "complete": False,
        "total_cost": 0.0,
        "total_tokens": 0
    })
    version: int = 0  # Incremented on each update

    def add_node(self, node: ResearchNode, parent: Optional[ResearchNode] = None, parent_id: Optional[str] = None):
        """Add node to tree

        Args:
            node: The node to add
            parent: Parent node object (deprecated, use parent_id instead)
            parent_id: Parent node ID string
        """
        self.nodes[node.id] = node

        # Support both parent object and parent_id for backwards compatibility
        actual_parent_id = None
        if parent:
            actual_parent_id = parent.id
        elif parent_id:
            actual_parent_id = parent_id

        if actual_parent_id:
            # Persist parent relationship on child node
            node.parent_id = actual_parent_id
            self.edges.append(ResearchEdge(
                parent_id=actual_parent_id,
                child_id=node.id
            ))

        self.stats["created"] += 1
        self.version += 1

    def get_children(self, node_id: str) -> List[ResearchNode]:
        """Get children of a node"""
        child_ids = [e.child_id for e in self.edges if e.parent_id == node_id]
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_parent(self, node_id: str) -> Optional[str]:
        """Get parent ID of a node"""
        if node_id not in self.nodes:
            return None
        return self.nodes[node_id].parent_id

    def get_path_to_root(self, node_id: str) -> List[ResearchNode]:
        """Get path from node to root"""
        path = []
        current_id = node_id

        while current_id:
            if current_id not in self.nodes:
                break

            node = self.nodes[current_id]
            path.append(node)
            current_id = node.parent_id

        return list(reversed(path))

    def calculate_max_depth(self) -> int:
        """Calculate maximum depth of the tree

        Returns:
            Maximum depth (root is depth 0)
        """
        if not self.nodes:
            return 0

        # Find root nodes (nodes without parents)
        root_ids = [nid for nid, node in self.nodes.items()
                   if not node.parent_id or node.parent_id not in self.nodes]

        if not root_ids:
            # No clear root, return 0
            return 0

        max_depth = 0
        for root_id in root_ids:
            # BFS to find max depth from this root
            depth = self._calculate_depth_from_node(root_id)
            max_depth = max(max_depth, depth)

        return max_depth

    def _calculate_depth_from_node(self, node_id: str, visited: Optional[set] = None) -> int:
        """Calculate depth from a specific node using DFS

        Args:
            node_id: Starting node ID
            visited: Set of visited nodes to avoid cycles

        Returns:
            Maximum depth from this node
        """
        if visited is None:
            visited = set()

        if node_id in visited or node_id not in self.nodes:
            return 0

        visited.add(node_id)
        children = self.get_children(node_id)

        if not children:
            return 0

        max_child_depth = 0
        for child in children:
            child_depth = self._calculate_depth_from_node(child.id, visited)
            max_child_depth = max(max_child_depth, child_depth)

        return 1 + max_child_depth

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            "research_id": self.research_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [{"from": e.parent_id, "to": e.child_id, "relation": e.relation} for e in self.edges],
            "stats": self.stats,
            "version": self.version
        }


class Task(BaseModel):
    """Task to be executed by an agent adapter"""
    id: Optional[str] = None  # Task/node ID
    goal: str
    context: Optional[str] = None
    budget: Budget = Field(default_factory=Budget)
    constraints: Dict[str, Any] = Field(default_factory=dict)


class Context(BaseModel):
    """Execution context for agent adapters"""
    branch_id: Optional[str] = None  # Current branch/node ID
    parent_nodes: List[ResearchNode] = Field(default_factory=list)  # Ancestor nodes
    tools: List[str] = Field(default_factory=list)  # Available tools
    secrets: Dict[str, str] = Field(default_factory=dict)  # API keys, etc.
    workspace_dir: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
