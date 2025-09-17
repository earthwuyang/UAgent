#!/usr/bin/env python3
"""
Test script for the markdown report generation system
"""

import asyncio
import sys
import os
import logging

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.research_tree import HierarchicalResearchSystem
from app.core.report_generator import MarkdownReportGenerator

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

async def test_report_generation():
    """Test the complete report generation workflow"""

    print("🧪 Testing Report Generation System")
    print("=" * 50)

    # Initialize the research system
    research_system = HierarchicalResearchSystem()

    # Create a test research goal
    print("\n1️⃣ Creating test research goal...")
    goal_id = await research_system.start_research_goal(
        title="Hello World Python Program",
        description="Write a simple hello world program in Python that prints 'Hello, World!' to the console",
        success_criteria=[
            "Working Python code that prints 'Hello, World!'",
            "Code executes without errors",
            "Output matches expected format"
        ],
        max_depth=2,
        max_experiments=5
    )

    print(f"✅ Research goal created: {goal_id}")

    # Execute the research tree to generate some results
    print("\n2️⃣ Executing research to generate results...")
    await research_system._execute_research_tree(goal_id)

    # Check if we have successful results
    tree = research_system.research_trees[goal_id]
    successful_results = any(
        node.results and any(r.success for r in node.results)
        for node in tree.values()
    )

    print(f"✅ Research execution completed. Has successful results: {successful_results}")

    # Test manual report generation
    print("\n3️⃣ Testing manual report generation...")
    report_generator = MarkdownReportGenerator(research_system)

    try:
        report_path = await report_generator.generate_completion_report(goal_id)
        print(f"✅ Report generated successfully: {report_path}")

        # Check if file exists and has content
        if os.path.exists(report_path):
            file_size = os.path.getsize(report_path)
            print(f"📄 Report file size: {file_size} bytes")

            # Read first few lines to verify content
            with open(report_path, 'r', encoding='utf-8') as f:
                first_lines = ''.join(f.readlines()[:5])
                print(f"📝 Report preview:\n{first_lines}...")
        else:
            print("❌ Report file not found")

    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False

    # Test automatic report generation on root completion
    print("\n4️⃣ Testing automatic report generation on root completion...")

    # Manually trigger root completion check to test automatic generation
    await research_system._check_root_completion(goal_id)

    # Check for additional reports
    report_path_auto = report_generator.get_report_path(goal_id)
    if report_path_auto:
        print(f"✅ Automatic report generation working: {report_path_auto}")
    else:
        print("⚠️ Automatic report generation may not have triggered")

    print("\n🎉 Report generation test completed!")
    print(f"📊 View reports in: {report_generator.reports_dir}")

    # List all generated reports
    reports_dir = report_generator.reports_dir
    if os.path.exists(reports_dir):
        reports = [f for f in os.listdir(reports_dir) if f.endswith('.md')]
        print(f"📋 Generated reports ({len(reports)}):")
        for report in reports:
            print(f"   - {report}")

    return True

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_report_generation())
    exit(0 if success else 1)