"""Test virtual conversation ID generation for EXPERIMENT nodes."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.tree_orchestrator import TreeSearchOrchestrator
from orchestrator.event_bus import EventBus
from control.control_bus import ControlBus
from adapters.base.agent_adapter import AgentAdapter, adapter_registry
from router.skill_router import SkillRouter
from uagent_research.models.research_tree import (
    ResearchNode,
    NodeType,
    NodeStatus,
    Task,
    Context,
)
from uagent_research.models.events import CompleteEvent


class MinimalAdapter(AgentAdapter):
    """Minimal adapter that completes immediately."""
    
    name = "minimal"
    
    def __init__(self):
        super().__init__(name=self.name)
    
    async def run(self, task: Task, context: Context):
        """Execute task and complete immediately."""
        yield CompleteEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            summary=f"Completed {task.goal}",
            artifacts=[],
            success=True,
        )
    
    async def cancel(self):
        self._cancelled = True


class TestRouter(SkillRouter):
    """Router that always selects minimal adapter."""
    
    def route(self, task: Task, context: Context) -> str:
        return "minimal"


@pytest.fixture
def minimal_adapter():
    """Register minimal adapter for testing."""
    original = adapter_registry._adapters.copy()
    adapter_registry._adapters.clear()
    adapter = MinimalAdapter()
    adapter_registry.register(adapter)
    yield adapter
    adapter_registry._adapters.clear()
    adapter_registry._adapters.update(original)


@pytest.fixture
async def orchestrator(minimal_adapter):
    """Create orchestrator with minimal adapter."""
    event_bus = EventBus(heartbeat_interval=0)
    control_bus = ControlBus()
    
    orch = TreeSearchOrchestrator(
        max_parallel=1,
        router=TestRouter(),
        event_bus=event_bus,
        control_bus=control_bus,
    )
    
    yield orch
    
    await orch.cancel()
    await event_bus.close()


@pytest.mark.asyncio
async def test_experiment_node_gets_conversation_id(orchestrator):
    """Test that EXPERIMENT nodes receive virtual conversation IDs."""
    
    # Run orchestrator with a simple goal
    tree = await orchestrator.run(
        goal="Test experiment conversation IDs",
        max_iterations=1,
        research_id="test_research_123"
    )
    
    # Find EXPERIMENT nodes in the tree
    experiment_nodes = [
        node for node in tree.nodes.values()
        if node.type == NodeType.EXPERIMENT
    ]
    
    # If there are experiment nodes, verify they have conversation IDs
    if experiment_nodes:
        for exp_node in experiment_nodes:
            # Check that metadata exists
            assert hasattr(exp_node, 'metadata'), f"Node {exp_node.id} missing metadata"
            assert exp_node.metadata is not None, f"Node {exp_node.id} metadata is None"
            
            # Check conversation_id is present
            assert 'conversation_id' in exp_node.metadata, \
                f"Node {exp_node.id} missing conversation_id in metadata"
            
            # Check worktree_branch is present
            assert 'worktree_branch' in exp_node.metadata, \
                f"Node {exp_node.id} missing worktree_branch in metadata"
            
            # Check parent_session_id is present
            assert 'parent_session_id' in exp_node.metadata, \
                f"Node {exp_node.id} missing parent_session_id in metadata"
            
            # Verify format of conversation_id
            conv_id = exp_node.metadata['conversation_id']
            assert conv_id.startswith('exp_test_research_123_'), \
                f"Conversation ID has wrong format: {conv_id}"
            
            # Verify worktree_branch format
            worktree = exp_node.metadata['worktree_branch']
            assert worktree.startswith('exp_exp_test_research_123_'), \
                f"Worktree branch has wrong format: {worktree}"
            
            # Verify parent_session_id
            assert exp_node.metadata['parent_session_id'] == 'test_research_123', \
                f"Parent session ID mismatch: {exp_node.metadata['parent_session_id']}"
            
            print(f"✅ EXPERIMENT node {exp_node.id} has valid conversation ID: {conv_id}")


@pytest.mark.asyncio
async def test_non_experiment_nodes_no_conversation_id(orchestrator):
    """Test that non-EXPERIMENT nodes (IDEA, HYPOTHESIS) don't get conversation IDs."""
    
    # Run orchestrator
    tree = await orchestrator.run(
        goal="Test non-experiment nodes",
        max_iterations=1,
        research_id="test_research_456"
    )
    
    # Find non-EXPERIMENT nodes
    non_experiment_nodes = [
        node for node in tree.nodes.values()
        if node.type in (NodeType.IDEA, NodeType.HYPOTHESIS, NodeType.ROOT)
    ]
    
    # Verify they don't have conversation_id in metadata
    for node in non_experiment_nodes:
        if hasattr(node, 'metadata') and node.metadata:
            # It's okay if metadata exists, just shouldn't have conversation_id
            assert 'conversation_id' not in node.metadata, \
                f"Non-EXPERIMENT node {node.id} (type={node.type}) should not have conversation_id"
        
        print(f"✅ Non-EXPERIMENT node {node.id} (type={node.type}) correctly has no conversation_id")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
