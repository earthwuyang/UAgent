#!/usr/bin/env python3
"""
Experiment Validation Script for AI Scientist
Detects synthetic data generation and enforces real experimentation requirements
"""
import os
import re
import ast
import json
import sys
import glob
from typing import Dict, List, Tuple, Any

class ExperimentValidator:
    """Validates that experiments use real data and real implementations"""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.violations = []
        self.warnings = []

    def validate_experiment(self) -> Dict[str, Any]:
        """Main validation function"""
        print(f"🔍 Validating experiment: {os.path.basename(self.experiment_dir)}")

        # Load the idea to understand what should be implemented
        idea_path = os.path.join(self.experiment_dir, "idea.json")
        if os.path.exists(idea_path):
            with open(idea_path, 'r') as f:
                self.idea = json.load(f)
        else:
            self.violations.append("Missing idea.json file")
            return self._create_report()

        # Find all Python experiment files
        python_files = glob.glob(os.path.join(self.experiment_dir, "**", "*.py"), recursive=True)
        experiment_files = [f for f in python_files if 'experiment' in os.path.basename(f)]

        if not experiment_files:
            self.violations.append("No experiment Python files found")
            return self._create_report()

        # Validate each experiment file
        for exp_file in experiment_files:
            self._validate_experiment_file(exp_file)

        # Check for required real implementations based on idea
        self._check_real_implementation_requirements()

        return self._create_report()

    def _validate_experiment_file(self, file_path: str):
        """Validate a single experiment Python file"""
        print(f"📄 Checking file: {os.path.basename(file_path)}")

        with open(file_path, 'r') as f:
            content = f.read()

        # Parse the AST to analyze the code structure
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.violations.append(f"Syntax error in {file_path}: {e}")
            return

        # Check for synthetic data generation patterns
        self._check_synthetic_data_patterns(file_path, content, tree)

        # Check for missing real implementation requirements
        self._check_missing_real_implementations(file_path, content, tree)

        # Check for proper source code cloning/modification
        self._check_source_code_requirements(file_path, content, tree)

    def _check_synthetic_data_patterns(self, file_path: str, content: str, tree: ast.AST):
        """Detect synthetic/fake data generation patterns"""

        # Pattern 1: Random data generation
        if re.search(r'np\.random\.|random\.|fake|synthetic', content, re.IGNORECASE):
            if 'Generate synthetic data' in content or 'random' in content.lower():
                self.violations.append(f"CRITICAL: {file_path} generates synthetic data instead of using real data")

        # Pattern 2: Hardcoded fake results
        if re.search(r'fake.*result|mock.*data|dummy.*data', content, re.IGNORECASE):
            self.violations.append(f"CRITICAL: {file_path} contains fake/mock data")

        # Pattern 3: Random choice for ground truth (biggest red flag)
        if 'np.random.choice' in content and ('faster_engine' in content or 'ground_truth' in content):
            self.violations.append(f"CRITICAL: {file_path} randomly generates ground truth labels - this is scientific fraud!")

        # Pattern 4: No actual data source
        data_sources = ['pd.read_csv', 'pd.read_sql', 'sqlite3.connect', 'psycopg2.connect', 'git clone', 'subprocess.run']
        has_real_data_source = any(source in content for source in data_sources)

        if not has_real_data_source and 'np.random' in content:
            self.warnings.append(f"WARNING: {file_path} may be using only synthetic data without real data sources")

    def _check_missing_real_implementations(self, file_path: str, content: str, tree: ast.AST):
        """Check if the experiment implements what the idea claims"""

        idea_name = self.idea.get('Name', '').lower()
        experiments = self.idea.get('Experiments', [])

        # Check for kernel modification claims
        if any('kernel' in exp.lower() or 'modify' in exp.lower() for exp in experiments):
            if 'subprocess' not in content and 'git clone' not in content:
                self.violations.append(f"CRITICAL: Idea claims kernel modification but {file_path} has no source code cloning/building")

        # Check for PostgreSQL modification claims
        if any('postgresql' in exp.lower() and 'modify' in exp.lower() for exp in experiments):
            postgres_indicators = ['git clone', 'make', 'configure', 'src/backend', 'planner']
            if not any(indicator in content for indicator in postgres_indicators):
                self.violations.append(f"CRITICAL: Idea claims PostgreSQL modification but {file_path} has no PostgreSQL build/modification code")

        # Check for real system integration
        if 'htap' in idea_name or 'query' in idea_name.lower():
            if 'psycopg2' not in content and 'pg_' not in content and 'postgresql' not in content.lower():
                self.warnings.append(f"WARNING: HTAP/query experiment should integrate with real PostgreSQL")

    def _check_source_code_requirements(self, file_path: str, content: str, tree: ast.AST):
        """Check if source code cloning and modification is properly implemented"""

        # Check for git operations
        git_operations = ['git clone', 'git checkout', 'git apply', 'git diff']
        has_git_ops = any(op in content for op in git_operations)

        # Check for build operations
        build_operations = ['make', 'cmake', 'configure', 'gcc', 'clang', './configure']
        has_build_ops = any(op in content for op in build_operations)

        # Check for source code modification
        modification_operations = ['patch', 'sed -i', 'with open.*w', 'write(', 'diff']
        has_modifications = any(op in content for op in modification_operations)

        # If the idea requires kernel/system modification, these should be present
        experiments = self.idea.get('Experiments', [])
        requires_modification = any(
            'modify' in exp.lower() or 'kernel' in exp.lower() or 'source' in exp.lower()
            for exp in experiments
        )

        if requires_modification:
            if not has_git_ops:
                self.violations.append(f"CRITICAL: {file_path} should clone source code but has no git operations")
            if not has_build_ops:
                self.violations.append(f"CRITICAL: {file_path} should build modified code but has no build operations")
            if not has_modifications:
                self.violations.append(f"CRITICAL: {file_path} should modify source code but has no modification operations")

    def _check_real_implementation_requirements(self):
        """Check experiment-wide requirements for real implementation"""

        # Check if workspace has actual cloned repositories
        workspace_dirs = glob.glob(os.path.join(self.experiment_dir, "**", "*"), recursive=True)
        git_repos = [d for d in workspace_dirs if os.path.isdir(d) and '.git' in os.listdir(d) if os.path.exists(os.path.join(d, '.git'))]

        experiments = self.idea.get('Experiments', [])
        requires_source_modification = any(
            'modify' in exp.lower() or 'kernel' in exp.lower() or 'clone' in exp.lower()
            for exp in experiments
        )

        if requires_source_modification and not git_repos:
            self.violations.append("CRITICAL: Experiment claims to modify source code but no git repositories found in workspace")

        # Check for actual data files vs synthetic generation
        data_files = glob.glob(os.path.join(self.experiment_dir, "**", "*.csv"), recursive=True)
        data_files.extend(glob.glob(os.path.join(self.experiment_dir, "**", "*.json"), recursive=True))
        data_files.extend(glob.glob(os.path.join(self.experiment_dir, "**", "*.db"), recursive=True))

        # Filter out config files
        real_data_files = [f for f in data_files if 'config' not in f and 'idea' not in f]

        if not real_data_files:
            self.warnings.append("WARNING: No real data files found - experiment may be using only synthetic data")

    def _create_report(self) -> Dict[str, Any]:
        """Create validation report"""

        # Determine overall status
        if self.violations:
            status = "FAILED"
            print("❌ EXPERIMENT VALIDATION FAILED")
        elif self.warnings:
            status = "WARNING"
            print("⚠️  EXPERIMENT VALIDATION PASSED WITH WARNINGS")
        else:
            status = "PASSED"
            print("✅ EXPERIMENT VALIDATION PASSED")

        print(f"\n📊 Validation Summary:")
        print(f"   Status: {status}")
        print(f"   Violations: {len(self.violations)}")
        print(f"   Warnings: {len(self.warnings)}")

        if self.violations:
            print("\n🚨 CRITICAL VIOLATIONS:")
            for i, violation in enumerate(self.violations, 1):
                print(f"   {i}. {violation}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")

        return {
            "status": status,
            "violations": self.violations,
            "warnings": self.warnings,
            "experiment_dir": self.experiment_dir,
            "idea": self.idea
        }

def main():
    """Main validation function"""
    if len(sys.argv) != 2:
        print("Usage: python experiment_validator.py <experiment_directory>")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    if not os.path.exists(experiment_dir):
        print(f"❌ Error: Experiment directory {experiment_dir} does not exist")
        sys.exit(1)

    validator = ExperimentValidator(experiment_dir)
    report = validator.validate_experiment()

    # Save validation report
    report_path = os.path.join(experiment_dir, "validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📝 Validation report saved to: {report_path}")

    # Exit with appropriate code
    if report["status"] == "FAILED":
        sys.exit(1)
    elif report["status"] == "WARNING":
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()