"""
Agent Laboratory Supervision Utility
Prevents wasteful dummy implementations and monitors progress
"""
import os
import re
import time
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class LabSupervisor:
    """Supervises Agent Laboratory execution to prevent waste and monitor progress"""

    def __init__(self, lab_dir: str, max_cost_threshold: float = 10.0):
        self.lab_dir = Path(lab_dir)
        self.max_cost_threshold = max_cost_threshold
        self.start_time = time.time()
        self.last_check_time = time.time()
        self.cost_history = []
        self.dummy_implementations_found = []
        self.progress_log = []

    def check_for_dummy_implementations(self, file_path: str) -> List[str]:
        """Check a Python file for dummy implementations that waste API calls"""
        dummy_patterns = [
            r'return\s*["\'].*[Dd]ummy.*["\']',
            r'return\s*f?["\'].*[Pp]laceholder.*["\']',
            r'return\s*["\'].*[Tt]odo.*["\']',
            r'return\s*["\'].*[Nn]ot\s+implemented.*["\']',
            r'return\s*["\'].*[Ss]tub.*["\']',
            r'def\s+query_model.*:\s*\n\s*#.*placeholder',
            r'def\s+query_model.*:\s*\n\s*return\s*["\']',
            r'\\boxed\{Dummy\s*Answer\}',
            r'This\s+is\s+a\s+placeholder',
            r'In\s+a\s+real\s+scenario',
        ]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            found_issues = []
            for pattern in dummy_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    found_issues.append(f"Found dummy pattern '{pattern}': {matches}")

            # Check for missing imports that would cause real API calls to fail
            if 'def query_model' in content and 'from inference import query_model' not in content:
                found_issues.append("Defines query_model locally instead of importing from inference.py")

            return found_issues

        except Exception as e:
            return [f"Error checking file {file_path}: {e}"]

    def scan_lab_directory(self) -> Dict[str, List[str]]:
        """Scan entire lab directory for dummy implementations"""
        issues = {}

        # Check all Python files in the lab directory
        for py_file in self.lab_dir.rglob("*.py"):
            file_issues = self.check_for_dummy_implementations(str(py_file))
            if file_issues:
                issues[str(py_file)] = file_issues

        return issues

    def detect_current_phase(self) -> str:
        """Detect which phase the Agent Laboratory is currently in"""
        # Check for phase indicators in the lab directory
        src_dir = self.lab_dir / "src"
        tex_dir = self.lab_dir / "tex"

        # Check for specific files that indicate phases
        if (src_dir / "run_experiments.py").exists():
            if (src_dir / "experiment_output.log").exists():
                return "running-experiments"
            else:
                return "data-preparation"
        elif src_dir.exists() and any(src_dir.glob("*.py")):
            return "data-preparation"
        elif tex_dir.exists() and any(tex_dir.glob("*.tex")):
            return "report-writing"
        else:
            # If no clear indicators, assume early phase
            return "literature-review"

    def check_experiment_output(self) -> Tuple[bool, str]:
        """Check if experiments are producing real results or dummy outputs"""
        current_phase = self.detect_current_phase()

        # Only check for experiment output if we should be in the experiments phase
        if current_phase in ["literature-review", "plan-formulation", "data-preparation"]:
            return True, f"In {current_phase} phase - experiment output not expected yet"

        log_file = self.lab_dir / "src" / "experiment_output.log"

        if not log_file.exists():
            if current_phase == "running-experiments":
                return False, "Expected experiment output log in running-experiments phase but none found"
            else:
                return True, f"In {current_phase} phase - experiment output not required yet"

        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Check for dummy outputs in the log
            if "Dummy Answer" in content:
                return False, "Experiment is producing 'Dummy Answer' outputs - wasting API calls!"

            # Check if accuracy is stuck at 0%
            accuracy_lines = [line for line in content.split('\n') if 'Accuracy:' in line]
            if len(accuracy_lines) > 10:  # After 10 examples
                recent_lines = accuracy_lines[-5:]  # Check last 5
                if all('Accuracy: 0.00%' in line for line in recent_lines):
                    return False, "Accuracy stuck at 0% - likely dummy implementations"

            return True, "Experiments appear to be running correctly"

        except Exception as e:
            return False, f"Error checking experiment output: {e}"

    def estimate_cost_from_log(self) -> float:
        """Estimate current cost from inference logs"""
        # This is a simple estimation - in a real system you'd integrate with actual cost tracking
        log_patterns = [
            r"Current experiment cost = \$([0-9\.]+)",
            r"Cost: \$([0-9\.]+)",
        ]

        total_cost = 0.0
        try:
            # Look for cost information in any log files
            for log_file in self.lab_dir.rglob("*.log"):
                with open(log_file, 'r') as f:
                    content = f.read()

                for pattern in log_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        total_cost += float(match)

        except Exception as e:
            print(f"Error estimating cost: {e}")

        return total_cost

    def should_stop_execution(self) -> Tuple[bool, str]:
        """Determine if execution should be stopped to prevent waste"""
        current_time = time.time()
        current_phase = self.detect_current_phase()

        # Check cost threshold
        estimated_cost = self.estimate_cost_from_log()
        if estimated_cost > self.max_cost_threshold:
            return True, f"Cost threshold exceeded: ${estimated_cost:.2f} > ${self.max_cost_threshold}"

        # Check for dummy implementations (only critical in data-preparation and running-experiments phases)
        issues = self.scan_lab_directory()
        if issues and current_phase in ["data-preparation", "running-experiments"]:
            dummy_files = [f for f, problems in issues.items() if any('dummy' in p.lower() for p in problems)]
            if dummy_files:
                return True, f"Dummy implementations detected in: {dummy_files}"

        # Check experiment output (phase-aware)
        output_ok, output_msg = self.check_experiment_output()
        if not output_ok:
            return True, f"Experiment issues: {output_msg}"

        # Phase-specific time limits
        phase_time_limits = {
            "literature-review": 1800,    # 30 minutes
            "plan-formulation": 900,      # 15 minutes
            "data-preparation": 1800,     # 30 minutes
            "running-experiments": 7200,  # 2 hours
            "report-writing": 1800,       # 30 minutes
        }

        time_limit = phase_time_limits.get(current_phase, 3600)  # Default 1 hour
        if current_time - self.start_time > time_limit:
            return True, f"Phase '{current_phase}' exceeded time limit of {time_limit/60:.0f} minutes"

        return False, "Execution appears normal"

    def generate_supervision_report(self) -> str:
        """Generate a comprehensive supervision report"""
        current_time = time.time()
        runtime = current_time - self.start_time

        # Scan for issues
        issues = self.scan_lab_directory()
        should_stop, stop_reason = self.should_stop_execution()
        estimated_cost = self.estimate_cost_from_log()
        output_ok, output_msg = self.check_experiment_output()
        current_phase = self.detect_current_phase()

        report = f"""
AGENT LABORATORY SUPERVISION REPORT
====================================
Runtime: {runtime/60:.1f} minutes
Estimated Cost: ${estimated_cost:.4f}
Max Cost Threshold: ${self.max_cost_threshold:.2f}
Current Phase: {current_phase}

DUMMY IMPLEMENTATION SCAN:
{'-'*40}
"""
        if issues:
            for file_path, file_issues in issues.items():
                report += f"\n❌ {file_path}:\n"
                for issue in file_issues:
                    report += f"   - {issue}\n"
        else:
            report += "✅ No dummy implementations detected\n"

        report += f"""
EXPERIMENT OUTPUT STATUS:
{'-'*40}
{"✅" if output_ok else "❌"} {output_msg}

RECOMMENDATION:
{'-'*40}
"""
        if should_stop:
            report += f"🛑 STOP EXECUTION: {stop_reason}\n"
        else:
            report += f"✅ Continue execution (within normal parameters)\n"

        return report

    def save_supervision_log(self):
        """Save supervision data to log file"""
        log_data = {
            "timestamp": time.time(),
            "runtime_minutes": (time.time() - self.start_time) / 60,
            "estimated_cost": self.estimate_cost_from_log(),
            "issues_found": self.scan_lab_directory(),
            "should_stop": self.should_stop_execution(),
        }

        log_file = self.lab_dir / "supervision_log.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)


def supervise_lab_execution(lab_dir: str, max_cost: float = 10.0, check_interval: int = 300):
    """Continuously supervise lab execution"""
    supervisor = LabSupervisor(lab_dir, max_cost)

    print(f"🔍 Starting Agent Laboratory supervision for {lab_dir}")
    print(f"💰 Max cost threshold: ${max_cost}")
    print(f"⏱️  Check interval: {check_interval} seconds")
    print("="*60)

    try:
        while True:
            # Generate and print report
            report = supervisor.generate_supervision_report()
            print(report)

            # Save log
            supervisor.save_supervision_log()

            # Check if we should stop
            should_stop, reason = supervisor.should_stop_execution()
            if should_stop:
                print(f"\n🛑 STOPPING SUPERVISION: {reason}")
                print("Consider terminating the Agent Laboratory process.")
                break

            # Wait for next check
            print(f"⏳ Next check in {check_interval} seconds...")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n⏹️  Supervision stopped by user")
    except Exception as e:
        print(f"\n❌ Supervision error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python supervision.py <lab_directory> [max_cost] [check_interval]")
        print("Example: python supervision.py MATH_research_dir/research_dir_0_lab_1 5.0 60")
        sys.exit(1)

    lab_dir = sys.argv[1]
    max_cost = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    check_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    supervise_lab_execution(lab_dir, max_cost, check_interval)