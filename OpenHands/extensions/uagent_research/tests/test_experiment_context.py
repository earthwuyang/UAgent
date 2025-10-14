"""Test ExperimentContext creation and passing to adapters."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uagent_research.models.research_tree import (
    ExperimentContext,
    Context,
    NodeType,
)


def test_experiment_context_creation():
    """Test that ExperimentContext can be created with all required fields."""
    
    exp_context = ExperimentContext(
        conversation_id="exp_research_abc123_idea-1-h",
        worktree_branch="exp_exp_research_abc123_idea-1-h",
        worktree_path="../worktrees/exp_exp_research_abc123_idea-1-h",
        parent_branch="main"
    )
    
    assert exp_context.conversation_id == "exp_research_abc123_idea-1-h"
    assert exp_context.worktree_branch == "exp_exp_research_abc123_idea-1-h"
    assert exp_context.worktree_path == "../worktrees/exp_exp_research_abc123_idea-1-h"
    assert exp_context.parent_branch == "main"
    
    print(f"✅ ExperimentContext created successfully: {exp_context}")


def test_experiment_context_default_parent_branch():
    """Test that ExperimentContext uses default parent_branch if not provided."""
    
    exp_context = ExperimentContext(
        conversation_id="exp_research_xyz789_idea-2-h",
        worktree_branch="exp_exp_research_xyz789_idea-2-h",
        worktree_path="../worktrees/exp_exp_research_xyz789_idea-2-h"
    )
    
    # Should default to "main"
    assert exp_context.parent_branch == "main"
    
    print(f"✅ ExperimentContext uses default parent_branch: {exp_context.parent_branch}")


def test_context_with_experiment_context():
    """Test that Context can include optional experiment_context."""
    
    exp_context = ExperimentContext(
        conversation_id="exp_research_test_node-1",
        worktree_branch="exp_exp_research_test_node-1",
        worktree_path="../worktrees/exp_exp_research_test_node-1",
        parent_branch="develop"
    )
    
    context = Context(
        branch_id="node-1",
        parent_nodes=[],
        experiment_context=exp_context
    )
    
    assert context.branch_id == "node-1"
    assert context.experiment_context is not None
    assert context.experiment_context.conversation_id == "exp_research_test_node-1"
    assert context.experiment_context.worktree_branch == "exp_exp_research_test_node-1"
    assert context.experiment_context.parent_branch == "develop"
    
    print(f"✅ Context with ExperimentContext created successfully")


def test_context_without_experiment_context():
    """Test that Context works without experiment_context (backward compatibility)."""
    
    context = Context(
        branch_id="idea-node",
        parent_nodes=[],
    )
    
    assert context.branch_id == "idea-node"
    assert context.experiment_context is None
    
    print(f"✅ Context without ExperimentContext works (backward compatible)")


def test_experiment_context_serialization():
    """Test that ExperimentContext can be serialized/deserialized."""
    
    exp_context = ExperimentContext(
        conversation_id="exp_research_serial_test",
        worktree_branch="exp_exp_research_serial_test",
        worktree_path="../worktrees/exp_exp_research_serial_test",
        parent_branch="feature-branch"
    )
    
    # Pydantic models have model_dump() method
    context_dict = exp_context.model_dump()
    
    assert context_dict['conversation_id'] == "exp_research_serial_test"
    assert context_dict['worktree_branch'] == "exp_exp_research_serial_test"
    assert context_dict['worktree_path'] == "../worktrees/exp_exp_research_serial_test"
    assert context_dict['parent_branch'] == "feature-branch"
    
    # Recreate from dict
    exp_context_restored = ExperimentContext(**context_dict)
    
    assert exp_context_restored.conversation_id == exp_context.conversation_id
    assert exp_context_restored.worktree_branch == exp_context.worktree_branch
    assert exp_context_restored.worktree_path == exp_context.worktree_path
    assert exp_context_restored.parent_branch == exp_context.parent_branch
    
    print(f"✅ ExperimentContext serialization/deserialization works")


if __name__ == "__main__":
    # Run tests
    test_experiment_context_creation()
    test_experiment_context_default_parent_branch()
    test_context_with_experiment_context()
    test_context_without_experiment_context()
    test_experiment_context_serialization()
    
    print("\n✅ All tests passed!")
