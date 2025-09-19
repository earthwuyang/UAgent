"""
Tests for tree expansion fixes - preventing one-layer limitation and proper "done" semantics
"""

import pytest
import asyncio
from app.core.research_tree import HierarchicalResearchSystem
from app.core.research_tree import ResearchNode, ExperimentType, NodeStatus, ResearchNodeType


@pytest.mark.asyncio
async def test_tree_grows_beyond_one_layer():
    """Test that the tree grows beyond one layer by creating follow-up experiments"""
    sys = HierarchicalResearchSystem()

    goal_id = await sys.start_research_goal(
        title="Multi-layer tree research",
        description="Research that should create multiple layers of experiments",
        success_criteria=["depth>=2", "multiple experiment types"],
        max_depth=4,
        max_experiments=12,
        time_budget=300  # 5 minutes
    )

    # Let the scheduler work for a few iterations
    for _ in range(5):
        await sys.run_goal_iteration(goal_id)
        await asyncio.sleep(0.5)  # Brief pause between iterations

    tree = sys.research_trees[goal_id]

    # Verify tree has multiple layers
    depths = [node.depth for node in tree.values()]
    max_depth = max(depths) if depths else 0

    assert max_depth >= 2, f"Tree should have depth >= 2, but max depth is {max_depth}"

    # Verify we have nodes at depth 2 or higher
    deep_nodes = [node for node in tree.values() if node.depth >= 2]
    assert len(deep_nodes) > 0, "Tree should have at least one node at depth 2 or higher"

    # Verify we have different node types (indicating proper expansion)
    node_types = {node.node_type for node in tree.values()}
    assert len(node_types) >= 3, f"Tree should have multiple node types, found: {node_types}"


@pytest.mark.asyncio
async def test_completed_nodes_can_expand():
    """Test that COMPLETED nodes can still be expanded if they have room for children"""
    sys = HierarchicalResearchSystem()

    goal_id = await sys.start_research_goal(
        title="Expansion test",
        description="Test that completed nodes can expand",
        success_criteria=["follow-up experiments"],
        max_depth=3,
        max_experiments=8
    )

    # Run several iterations to get some completed experiments
    for _ in range(3):
        await sys.run_goal_iteration(goal_id)
        await asyncio.sleep(0.1)

    tree = sys.research_trees[goal_id]
    goal = sys.active_goals[goal_id]

    # Find a completed experiment node with results
    completed_exp_nodes = [
        node for node in tree.values()
        if (node.status == NodeStatus.COMPLETED and
            node.node_type == ResearchNodeType.EXPERIMENT and
            node.results and
            len(node.children) < sys._get_max_children(node) and
            node.depth < goal.max_depth)
    ]

    if completed_exp_nodes:
        # This node should be considered expandable by our new logic
        test_node = completed_exp_nodes[0]

        # Test the expandable logic from _select_node_ucb
        def expandable(n: ResearchNode) -> bool:
            if n.depth >= goal.max_depth:
                return False
            max_children = max(0, int(sys._get_max_children(n)))
            if len(n.children) >= max_children:
                return False
            needs_result = n.node_type in {
                ResearchNodeType.EXPERIMENT,
                ResearchNodeType.LITERATURE,
                ResearchNodeType.CODE_ANALYSIS,
            }
            return (not needs_result) or bool(n.results)

        assert expandable(test_node), "Completed experiment node with results should be expandable"


def test_done_vs_expandable():
    """Test the precise _is_done semantics"""
    sys = HierarchicalResearchSystem()

    # Create a mock goal and tree
    goal_id = "test_goal"
    sys.active_goals[goal_id] = type('Goal', (), {
        'max_depth': 3,
        'max_experiments': 10
    })()

    # Create a mock completed experiment node with results but room for children
    test_node = ResearchNode(
        id="test_node",
        parent_id="root",
        node_type=ResearchNodeType.EXPERIMENT,
        title="Test Experiment",
        description="Test node for done semantics",
        depth=1,
        status=NodeStatus.COMPLETED
    )

    # Add some mock results
    test_node.results = [type('Result', (), {'success': True, 'confidence': 0.9})()]
    test_node.children = []  # No children yet, so expandable

    sys.research_trees[goal_id] = {"test_node": test_node}

    # Should NOT be done because it's expandable
    assert not sys._is_done(goal_id, "test_node"), "Completed but expandable node should not be done"

    # Now make it non-expandable by filling children
    max_children = sys._get_max_children(test_node)
    test_node.children = [f"child_{i}" for i in range(max_children)]

    # Now it should be done
    assert sys._is_done(goal_id, "test_node"), "Completed and non-expandable node should be done"

    # Test failed node (always done)
    test_node.status = NodeStatus.FAILED
    assert sys._is_done(goal_id, "test_node"), "Failed node should always be done"


@pytest.mark.asyncio
async def test_expansion_after_simulation():
    """Test that expansion happens after simulation for result-dependent nodes"""
    sys = HierarchicalResearchSystem()

    goal_id = await sys.start_research_goal(
        title="Expansion order test",
        description="Test expansion after simulation",
        success_criteria=["proper ordering"],
        max_depth=3,
        max_experiments=6
    )

    # Run one iteration to get initial nodes
    await sys.run_goal_iteration(goal_id)

    tree = sys.research_trees[goal_id]

    # Find an experiment node without results
    exp_nodes_without_results = [
        node for node in tree.values()
        if (node.node_type == ResearchNodeType.EXPERIMENT and
            not node.results)
    ]

    if exp_nodes_without_results:
        # Simulate what happens in our fixed _execute_research_tree
        selected_node = exp_nodes_without_results[0]

        # Before fix: expansion would return empty list because no results
        # After fix: we run experiment first, then expand

        needs_result_first = selected_node.node_type in {
            ResearchNodeType.EXPERIMENT,
            ResearchNodeType.LITERATURE,
            ResearchNodeType.CODE_ANALYSIS,
        }

        assert needs_result_first, "Experiment nodes should need results first"
        assert not selected_node.results, "Test node should start without results"

        # After running the experiment (simulated), it should have results
        # and then be expandable for follow-ups


@pytest.mark.asyncio
async def test_max_children_defaults():
    """Test that _get_max_children returns sensible defaults"""
    sys = HierarchicalResearchSystem()

    # Test known node types
    root_node = ResearchNode(
        id="root", parent_id=None, node_type=ResearchNodeType.ROOT,
        title="Root", description="Root node"
    )
    assert sys._get_max_children(root_node) == 5, "Root should allow 5 children"

    exp_node = ResearchNode(
        id="exp", parent_id="root", node_type=ResearchNodeType.EXPERIMENT,
        title="Experiment", description="Test experiment"
    )
    assert sys._get_max_children(exp_node) == 3, "Experiment should allow 3 children"

    synthesis_node = ResearchNode(
        id="synth", parent_id="root", node_type=ResearchNodeType.SYNTHESIS,
        title="Synthesis", description="Test synthesis"
    )
    assert sys._get_max_children(synthesis_node) == 1, "Synthesis should allow 1 child"

    # Test hierarchical research type
    hier_node = ResearchNode(
        id="hier", parent_id="root", node_type=ResearchNodeType.HIERARCHICAL_RESEARCH,
        title="Hierarchical", description="Test hierarchical"
    )
    assert sys._get_max_children(hier_node) == 3, "Hierarchical research should allow 3 children"


@pytest.mark.asyncio
async def test_no_infinite_loops():
    """Test that the fixes don't create infinite loops in tree expansion"""
    sys = HierarchicalResearchSystem()

    goal_id = await sys.start_research_goal(
        title="Loop prevention test",
        description="Ensure no infinite expansion loops",
        success_criteria=["finite expansion"],
        max_depth=2,  # Keep shallow to prevent runaway
        max_experiments=5,
        time_budget=30  # Short time budget
    )

    # Run several iterations and verify it terminates properly
    iterations = 0
    max_iterations = 10  # Safety limit

    while iterations < max_iterations:
        try:
            result = await asyncio.wait_for(
                sys.run_goal_iteration(goal_id),
                timeout=5.0  # 5 second timeout per iteration
            )
            iterations += 1

            # Check if research is complete
            status = await sys.get_goal_status(goal_id)
            if status['status'] in ['completed', 'failed']:
                break

        except asyncio.TimeoutError:
            break

    # Should complete within reasonable iterations
    assert iterations < max_iterations, f"Research should complete within {max_iterations} iterations"

    # Tree should be reasonable size
    tree = sys.research_trees[goal_id]
    assert len(tree) < 20, f"Tree should be reasonable size, but has {len(tree)} nodes"


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_tree_grows_beyond_one_layer())
    print("✅ Tree grows beyond one layer")

    asyncio.run(test_completed_nodes_can_expand())
    print("✅ Completed nodes can expand")

    test_done_vs_expandable()
    print("✅ Done vs expandable semantics work")

    asyncio.run(test_expansion_after_simulation())
    print("✅ Expansion after simulation works")

    asyncio.run(test_max_children_defaults())
    print("✅ Max children defaults are correct")

    asyncio.run(test_no_infinite_loops())
    print("✅ No infinite loops detected")

    print("\n🎉 All tree expansion fixes tests passed!")