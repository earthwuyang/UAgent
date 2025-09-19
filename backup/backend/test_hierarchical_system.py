"""
Test script for the complete hierarchical research system
Tests the integration of AI agents, research tree, and unified orchestrator
"""

import asyncio
import sys
import os
import pytest
sys.path.append(os.path.dirname(__file__))

from app.core.research_tree import HierarchicalResearchSystem, ResearchGoal, ResearchNodeType, ExperimentType
from app.core.unified_orchestrator import UnifiedOrchestrator

@pytest.mark.asyncio
async def test_hierarchical_research_system():
    """Test the complete hierarchical research system with real agents"""

    print("🚀 Testing Hierarchical Research System with Real AI Agents")
    print("=" * 80)

    # Initialize systems
    print("📦 Initializing systems...")
    research_system = HierarchicalResearchSystem()
    orchestrator = UnifiedOrchestrator()

    # Inject orchestrator into research system
    research_system.orchestrator = orchestrator
    print("✅ Orchestrator injected into research system")

    # Create a complex research goal that will trigger hierarchical agents
    research_goal = ResearchGoal(
        id="test_hierarchical_research_001",
        title="Advanced Multi-Modal AI for Scientific Research",
        description="Investigate cutting-edge multi-modal AI approaches that can process text, images, and data to accelerate scientific discovery",
        success_criteria=[
            "Comprehensive literature review of multi-modal AI in science",
            "Identification of promising research directions",
            "Design of hierarchical experiment framework",
            "Analysis of implementation strategies"
        ],
        domain="AI/ML Research",
        priority=1.0
    )

    print(f"🎯 Created research goal: {research_goal.title}")
    print(f"📝 Description: {research_goal.description}")

    # Start hierarchical research
    print("\n🔬 Starting hierarchical research with multiple specialized agents...")
    goal_id = await research_system.start_research_goal(
        research_goal.title,
        research_goal.description,
        research_goal.success_criteria
    )
    print(f"✅ Research started with ID: {goal_id}")

    # Monitor progress
    print("\n📊 Monitoring research progress...")
    max_iterations = 20  # Limit iterations for testing
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Get current status
        status = await research_system.get_goal_status(goal_id)
        print(f"🔍 Research Status: {status['status']}")
        print(f"📈 Progress: {status['progress']}")

        # Print tree information
        tree_info = await research_system.get_goal_tree(goal_id)
        print(f"🌳 Tree Nodes: {tree_info['total_nodes']}")
        print(f"✅ Completed: {tree_info['completed_nodes']}")
        print(f"⚠️  Failed: {tree_info['failed_nodes']}")

        # Print recent results
        if tree_info.get('nodes'):
            print("\n🔬 Recent Node Results:")
            for node_id, node in list(tree_info['nodes'].items())[-3:]:  # Last 3 nodes
                if node.get('results'):
                    result = node['results']
                    print(f"  📊 {node['title'][:50]}...")
                    print(f"    ✅ Success: {result.get('success', False)}")
                    print(f"    🎯 Confidence: {result.get('confidence', 0):.2f}")
                    if result.get('metrics'):
                        metrics = result['metrics']
                        if 'agents_used' in metrics:
                            print(f"    🤖 Agents Used: {metrics['agents_used']}")
                        if 'research_readiness_score' in metrics:
                            print(f"    📊 Research Readiness: {metrics['research_readiness_score']:.2f}")

                    # Print key insights
                    insights = result.get('insights', [])
                    if insights:
                        print(f"    💡 Key Insights:")
                        for insight in insights[:2]:  # Top 2 insights
                            print(f"      - {insight}")

        # Check if research is complete
        if status['status'] in ['completed', 'failed']:
            print(f"\n🏁 Research {status['status'].upper()}!")
            break

        # Run next iteration
        await research_system.run_goal_iteration(goal_id)

        # Brief pause
        await asyncio.sleep(2)

    # Get final results
    print("\n" + "="*80)
    print("📋 FINAL RESULTS")
    print("="*80)

    final_status = await research_system.get_goal_status(goal_id)
    final_tree = await research_system.get_goal_tree(goal_id)

    print(f"🎯 Final Status: {final_status['status']}")
    print(f"🌳 Total Nodes Created: {final_tree['total_nodes']}")
    print(f"✅ Successfully Completed: {final_tree['completed_nodes']}")
    print(f"⚠️  Failed Nodes: {final_tree['failed_nodes']}")
    print(f"📊 Success Rate: {final_tree['success_rate']:.1%}")

    # Print summary of all completed experiments
    print(f"\n🔬 EXPERIMENT SUMMARY:")
    hierarchical_experiments = 0
    total_agents_used = 0
    total_papers_analyzed = 0
    total_web_resources = 0

    for node_id, node in final_tree.get('nodes', {}).items():
        if node.get('results') and node['results'].get('success'):
            result = node['results']
            experiment_type = node.get('experiment_type', 'unknown')

            print(f"\n📊 {node['title']}")
            print(f"   Type: {experiment_type}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")

            # Track hierarchical experiments
            if experiment_type == 'hierarchical_multi_agent':
                hierarchical_experiments += 1
                metrics = result.get('metrics', {})
                agents = metrics.get('agents_used', 0)
                papers = metrics.get('literature_papers', 0)
                web = metrics.get('web_resources', 0)

                total_agents_used += agents
                total_papers_analyzed += papers
                total_web_resources += web

                print(f"   🤖 Agents Used: {agents}")
                print(f"   📚 Papers Analyzed: {papers}")
                print(f"   🌐 Web Resources: {web}")
                print(f"   📊 Research Readiness: {metrics.get('research_readiness_score', 0):.2f}")

    print(f"\n🎉 HIERARCHICAL RESEARCH SUMMARY:")
    print(f"   🤖 Total Hierarchical Experiments: {hierarchical_experiments}")
    print(f"   👥 Total AI Agents Used: {total_agents_used}")
    print(f"   📚 Total Papers Analyzed: {total_papers_analyzed}")
    print(f"   🌐 Total Web Resources: {total_web_resources}")

    # Check orchestrator workflows
    orchestrator_status = await orchestrator.get_system_status()
    print(f"\n🎛️  ORCHESTRATOR STATUS:")
    print(f"   📊 Active Workflows: {orchestrator_status['workflows']['active']}")
    print(f"   ✅ Completed Workflows: {orchestrator_status['workflows']['completed']}")
    print(f"   ⚠️  Failed Workflows: {orchestrator_status['workflows']['failed']}")

    print("\n" + "="*80)
    print("🎊 Hierarchical Research System Test Complete!")
    print("="*80)

    return {
        'goal_id': goal_id,
        'final_status': final_status,
        'final_tree': final_tree,
        'hierarchical_experiments': hierarchical_experiments,
        'total_agents_used': total_agents_used,
        'total_papers_analyzed': total_papers_analyzed,
        'total_web_resources': total_web_resources
    }

async def main():
    """Main test function"""
    try:
        results = await test_hierarchical_research_system()
        print("\n✅ Test completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the test
    results = asyncio.run(main())

    if results:
        print(f"\n📊 Test Results Summary:")
        print(f"   Goal ID: {results['goal_id']}")
        print(f"   Status: {results['final_status']['status']}")
        print(f"   Hierarchical Experiments: {results['hierarchical_experiments']}")
        print(f"   Total AI Agents Used: {results['total_agents_used']}")
        print(f"   Success! ✅")
    else:
        print(f"   Test Failed! ❌")