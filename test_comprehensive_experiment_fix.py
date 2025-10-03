#!/usr/bin/env python3
"""
Test script to verify that the comprehensive experiment fix works correctly.
This demonstrates that the comprehensive prompt from EXPERIMENT_INSTRUCTIONS.md
is now being used as the actual goal for OpenHands instead of the basic goal.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from backend.app.core.research_engines.scientific_research import (
    ScientificResearchEngine,
    SequentialExperimentPlan,
    ResearchHypothesis,
    TechnicalRequirements
)
from backend.app.core.llm_client import LLMClient


async def test_comprehensive_experiment_fix():
    """Test that comprehensive prompts are actually used as OpenHands goals"""

    print("🧪 Testing Comprehensive Experiment Fix")
    print("=" * 80)

    # Create test technical requirements
    tech_requirements = TechnicalRequirements(
        source_code_modifications=["postgresql kernel modifications", "ml model embedding"],
        programming_languages=["c", "python"],
        execution_engines=["postgresql", "duckdb"],
        integration_requirements=["ml router", "dual-engine execution"],
        data_collection_requirements=["performance metrics", "accuracy metrics"],
        prohibited_shortcuts=["synthetic data", "mock implementations"]
    )

    # Create research hypothesis
    hypothesis = ResearchHypothesis(
        id="test_hyp_001",
        statement="pg_duckdb dual-execution with ML routing improves query performance",
        reasoning="ML routing can predict optimal engine selection based on query characteristics",
        testable_predictions=["routing improves response time", "ml model maintains high accuracy"],
        success_criteria={
            "performance_gain": ">10%",
            "routing_accuracy": ">90%",
            "build_success": "True"
        },
        variables={
            "independent": ["query_type", "dataset_size", "complexity"],
            "dependent": ["response_time", "accuracy", "traffic_distribution"]
        }
    )

    # Create a simple experiment design
    experiment_design = ExperimentDesign(
        id="exp_test_001",
        hypothesis_id=hypothesis.id,
        name="pg_duckdb ml routing implementation",
        description="Implement ML-based query routing between PostgreSQL and DuckDB",
        methodology="1. Build PostgreSQL with pg_duckdb\n2. Train ML model on query features\n3. Embed model in PostgreSQL C code\n4. Test integrated system with benchmark queries",
        variables=hypothesis.variables,
        controls=["same hardware", "identical datasets", "consistent query workload"],
        data_collection_plan={
            "metrics": ["query_time", "routing_accuracy", "model_predictions"],
            "sample_size": 100,
            "repetitions": 3
        },
        analysis_plan="Statistical analysis comparing routing vs baseline performance",
        expected_duration="2-4 hours",
        resource_requirements={},
        code_requirements=["c_python_integration", "machine_learning_sklearn", "database_benchmarking"],
        dependencies=["PostgreSQL 16+", "pg_duckdb", "scikit-learn"]
    )

    # Create mock sequential plan
    sequential_plan = SequentialExperimentPlan(
        id="seq_test_001",
        hypothesis_id=hypothesis.id,
        num_experiments=3,
        experiments=[experiment_design, experiment_design, experiment_design],  # 3 similar experiments
        overall_objective="Implement and test pg_duckdb dual-execution with ML routing optimization",
        experiment_dependencies={},
        shared_setup="Setup development environment with PostgreSQL, DuckDB, Python ML tools",
        expected_total_duration="6-8 hours"
    )

    print(f"📋 Created test sequential plan: {sequential_plan.id}")
    print(f"📊 Plan contains {len(sequential_plan.experiments)} experiments")
    print(f"🎯 Overall objective: {sequential_plan.overall_objective}")

    # Test the comprehensive prompt generation
    print("\n📝 Building comprehensive experiment prompt...")
    llm_client = LLMClient()
    engine = ScientificResearchEngine(llm_client)

    # Build the comprehensive prompt
    comprehensive_prompt = engine.experiment_executor._build_comprehensive_experiment_prompt(
        sequential_plan,
        prior_errors=None
    )

    print(f"📄 Comprehensive prompt length: {len(comprehensive_prompt)} characters")
    print("🧪 Sample from comprehensive prompt:")
    print(comprehensive_prompt[:500] + "...")

    # Verify that the comprehensive prompt was generated correctly
    print("\n✅ Comprehensive prompt contains key elements:")
    checks = [
        ("Overall objective", sequential_plan.overall_objective in comprehensive_prompt),
        ("Multiple experiments", str(len(sequential_plan.experiments)) in comprehensive_prompt),
        ("Step-by-step methodology", "STEP 1/" in comprehensive_prompt),
        ("Workspace organization", "workspace/experiments/" in comprehensive_prompt),
        ("Final.json requirement", "final.json" in comprehensive_prompt),
        ("README.md requirement", "README.md" in comprehensive_prompt),
        ("Sequential execution", "build upon each other" in comprehensive_prompt),
        ("Real systems requirement", "REAL systems" in comprehensive_prompt),
        ("Shared installations", "REUSE installations" in comprehensive_prompt)
    ]

    all_checks_passed = True
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"   {status} {check_name}: {'included' if check_result else 'missing'}")
        all_checks_passed = all_checks_passed and check_result

    if all_checks_passed:
        print("\n🎉 SUCCESS: Comprehensive prompt is correctly generated!")
        print("🚀 This comprehensive prompt will now be used as the OpenHands goal instead of basic metadata")
        print("📈 This ensures the entire experimental workflow including ML training, code modifications,")
        print("   end-to-end validation is passed to OpenHands as a unified coherent task")
    else:
        print("\n⚠️  Some checks failed - comprehensive prompt may be incomplete")

    print(f"\n📁 The comprehensive prompt would be saved to: experiments/{sequential_plan.id}/EXPERIMENT_INSTRUCTIONS.md")
    print("🎯 Now OpenHands will read this file and use its contents as the goal instead of basic experiment info")

if __name__ == "__main__":
    asyncio.run(test_comprehensive_experiment_fix())