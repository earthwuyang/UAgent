"""
AIRA-style Research Operators
Based on "AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench"

Implements Draft, Improve, Debug, Analysis operators with strict output contracts:
- Exactly one idea + one self-contained script
- 5-fold CV evaluation enforced
- Prints score and writes submission.csv
"""

import os
import re
import ast
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Protocol
from pathlib import Path
import tempfile
import subprocess
import logging

from .research_tree import ResearchNode, ExperimentResult, ResearchNodeType, NodeStatus

logger = logging.getLogger(__name__)


@dataclass
class Artifact:
    """AIRA-style artifact representing a code solution"""
    id: str
    parent_id: Optional[str]
    depth: int
    code_path: Path
    plan_text: str
    operator: str  # draft/improve/debug/analysis
    status: str  # PENDING/RUNNING/COMPLETED/FAILED
    val_metric: Optional[float] = None  # higher is better
    cv_folds: int = 5
    runtime_sec: Optional[float] = None
    logs_path: Optional[Path] = None
    visits: int = 0  # for MCTS
    mean_fitness: float = 0.0  # for MCTS backup


@dataclass
class ProblemSpec:
    """Problem specification for operators"""
    task_name: str
    description: str
    data_info: str
    constraints: Dict[str, Any]
    success_criteria: List[str]


@dataclass
class Memory:
    """Lightweight scoped memory for operators"""
    previous_ideas: List[str]
    recent_errors: List[str]
    improvements_tried: List[str]
    max_items: int = 10

    def add_idea(self, idea: str):
        self.previous_ideas.append(idea)
        if len(self.previous_ideas) > self.max_items:
            self.previous_ideas.pop(0)

    def add_error(self, error: str):
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.max_items:
            self.recent_errors.pop(0)

    def add_improvement(self, improvement: str):
        self.improvements_tried.append(improvement)
        if len(self.improvements_tried) > self.max_items:
            self.improvements_tried.pop(0)


class Operator(Protocol):
    """AIRA operator interface"""
    name: str

    async def apply(self, ctx: ProblemSpec, parent: Optional[Artifact], memory: Memory) -> Artifact:
        """Apply operator to generate new artifact"""
        ...


class DraftOperator:
    """
    AIRA Draft operator - generates baseline solutions
    Modes: simple, normal, complex (staged over time)
    """
    name = "draft"

    def __init__(self, complexity: str = "normal", llm_client=None):
        self.complexity = complexity
        self.llm_client = llm_client

    async def apply(self, ctx: ProblemSpec, parent: Optional[Artifact], memory: Memory) -> Artifact:
        """Generate a new baseline idea and implementation"""

        # Generate unique idea considering previous attempts
        idea_prompt = self._build_draft_prompt(ctx, memory, self.complexity)

        try:
            if self.llm_client:
                response = await self.llm_client.generate_response(
                    prompt=idea_prompt,
                    system_prompt="You are a Kaggle Grandmaster. Generate exactly one distinct ML solution.",
                    temperature=0.6,
                    max_tokens=3000
                )
                content = response.get("content", "")
            else:
                # Fallback template for testing
                content = self._generate_template_solution(ctx, memory)

            # Extract plan and code
            plan_text, code = self._parse_llm_output(content)

            # Create artifact
            artifact_id = f"draft_{parent.id if parent else 'root'}_{hash(plan_text) % 10000}"
            artifact_dir = Path(tempfile.mkdtemp(prefix=f"artifact_{artifact_id}_"))

            # Write code to file
            code_file = artifact_dir / "solution.py"
            with open(code_file, 'w') as f:
                f.write(code)

            # Create artifact
            artifact = Artifact(
                id=artifact_id,
                parent_id=parent.id if parent else None,
                depth=(parent.depth + 1) if parent else 0,
                code_path=artifact_dir,
                plan_text=plan_text,
                operator="draft",
                status="PENDING"
            )

            # Add to memory
            memory.add_idea(plan_text)

            return artifact

        except Exception as e:
            logger.error(f"Draft operator failed: {e}")
            raise

    def _build_draft_prompt(self, ctx: ProblemSpec, memory: Memory, complexity: str) -> str:
        """Build prompt for draft generation"""

        complexity_guidance = {
            "simple": "Use basic algorithms (linear models, simple trees). Focus on solid implementation.",
            "normal": "Use standard ML approaches (ensemble methods, feature engineering). Balance complexity and reliability.",
            "complex": "Consider advanced techniques (neural networks, sophisticated ensembles). Optimize for performance."
        }

        previous_ideas = "\n".join(f"- {idea}" for idea in memory.previous_ideas[-5:])

        return f"""
You are a Kaggle Grandmaster. Propose exactly ONE distinct ML solution for this task.

**Task:** {ctx.task_name}
**Description:** {ctx.description}
**Data Info:** {ctx.data_info}
**Constraints:** {ctx.constraints}

**Complexity Level:** {complexity} - {complexity_guidance[complexity]}

**Previously Explored Ideas:**
{previous_ideas if previous_ideas else "None"}

**Requirements:**
1. Generate exactly ONE distinct idea (different from previous)
2. Output ONE self-contained Python script
3. Must use 5-fold cross-validation
4. Must print the CV score clearly
5. Must write 'submission.csv'
6. Keep stdout clean (no EDA output)

**Output Format:**
PLAN: [One paragraph describing your approach]

CODE:
```python
[Complete self-contained script]
```
"""

    def _parse_llm_output(self, content: str) -> tuple[str, str]:
        """Extract plan and code from LLM output"""

        # Extract plan
        plan_match = re.search(r'PLAN:\s*(.*?)(?=CODE:|$)', content, re.DOTALL)
        plan_text = plan_match.group(1).strip() if plan_match else "Generated baseline solution"

        # Extract code
        code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Fallback: try to find any code block
            code_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            code = code_match.group(1).strip() if code_match else self._generate_template_solution_code()

        return plan_text, code

    def _generate_template_solution(self, ctx: ProblemSpec, memory: Memory) -> str:
        """Generate template solution when LLM unavailable"""
        return f"""
PLAN: Baseline solution for {ctx.task_name} using simple machine learning approach.

CODE:
```python
{self._generate_template_solution_code()}
```
"""

    def _generate_template_solution_code(self) -> str:
        """Generate template Python code that follows AIRA contract"""
        return '''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    """Baseline ML solution with 5-fold CV"""
    print("Starting baseline ML solution...")

    # Load data (placeholder)
    # train = pd.read_csv('train.csv')
    # test = pd.read_csv('test.csv')

    # For demo: create synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    X_test = np.random.randn(100, 10)

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Validate
        y_pred = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, y_pred))
        cv_scores.append(score)

        print(f"Fold {fold+1}: RMSE = {score:.4f}")

    # Final CV score
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"5-fold CV Score: {cv_mean:.4f} ± {cv_std:.4f}")

    # Train on full data and predict
    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(X, y)
    test_pred = final_model.predict(X_test)

    # Write submission
    submission = pd.DataFrame({
        'id': range(len(test_pred)),
        'target': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

    return cv_mean

if __name__ == "__main__":
    main()
'''


class ImproveOperator:
    """AIRA Improve operator - enhances existing solutions"""
    name = "improve"

    def __init__(self, complexity: str = "normal", llm_client=None):
        self.complexity = complexity
        self.llm_client = llm_client

    async def apply(self, ctx: ProblemSpec, parent: Artifact, memory: Memory) -> Artifact:
        """Improve an existing solution"""

        if not parent or not parent.code_path:
            raise ValueError("Improve operator requires parent artifact with code")

        # Read parent code and logs
        parent_code = self._read_code(parent.code_path / "solution.py")
        parent_logs = self._read_logs(parent.logs_path) if parent.logs_path else ""

        # Generate improvement
        improve_prompt = self._build_improve_prompt(ctx, parent, parent_code, parent_logs, memory)

        try:
            if self.llm_client:
                response = await self.llm_client.generate_response(
                    prompt=improve_prompt,
                    system_prompt="You are a Kaggle expert. Improve the solution with one focused change.",
                    temperature=0.6,
                    max_tokens=3000
                )
                content = response.get("content", "")
            else:
                content = self._generate_template_improvement(parent_code)

            # Extract improvement and code
            improvement_text, code = self._parse_llm_output(content)

            # Create artifact
            artifact_id = f"improve_{parent.id}_{hash(improvement_text) % 10000}"
            artifact_dir = Path(tempfile.mkdtemp(prefix=f"artifact_{artifact_id}_"))

            # Write improved code
            code_file = artifact_dir / "solution.py"
            with open(code_file, 'w') as f:
                f.write(code)

            artifact = Artifact(
                id=artifact_id,
                parent_id=parent.id,
                depth=parent.depth + 1,
                code_path=artifact_dir,
                plan_text=f"Improvement: {improvement_text}",
                operator="improve",
                status="PENDING"
            )

            memory.add_improvement(improvement_text)
            return artifact

        except Exception as e:
            logger.error(f"Improve operator failed: {e}")
            raise

    def _build_improve_prompt(self, ctx: ProblemSpec, parent: Artifact, code: str, logs: str, memory: Memory) -> str:
        """Build prompt for improvement generation"""

        recent_improvements = "\n".join(f"- {imp}" for imp in memory.improvements_tried[-3:])

        return f"""
Given the previous solution and its performance, propose exactly ONE improvement.

**Previous Solution Plan:** {parent.plan_text}
**Previous Performance:** {parent.val_metric if parent.val_metric else 'Unknown'}

**Previous Code:**
```python
{code[:2000]}  # truncated for brevity
```

**Execution Logs:**
{logs[:1000] if logs else 'No logs available'}

**Recent Improvements Tried:**
{recent_improvements if recent_improvements else "None"}

**Instructions:**
1. Propose ONE focused improvement (features, model, hyperparams, ensemble, etc.)
2. Keep evaluation consistent (same 5-fold CV)
3. Output ONE complete, runnable script
4. Must print CV score and write submission.csv

**Output Format:**
IMPROVEMENT: [Describe the specific change]

CODE:
```python
[Complete improved script]
```
"""

    def _read_code(self, code_path: Path) -> str:
        """Read code from file"""
        try:
            with open(code_path, 'r') as f:
                return f.read()
        except Exception:
            return ""

    def _read_logs(self, logs_path: Path) -> str:
        """Read execution logs"""
        try:
            with open(logs_path, 'r') as f:
                return f.read()
        except Exception:
            return ""

    def _parse_llm_output(self, content: str) -> tuple[str, str]:
        """Extract improvement description and code"""

        # Extract improvement description
        imp_match = re.search(r'IMPROVEMENT:\s*(.*?)(?=CODE:|$)', content, re.DOTALL)
        improvement_text = imp_match.group(1).strip() if imp_match else "Performance improvement"

        # Extract code
        code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Fallback
            code = self._generate_template_improvement_code()

        return improvement_text, code

    def _generate_template_improvement(self, parent_code: str) -> str:
        """Generate template improvement"""
        return f"""
IMPROVEMENT: Add feature engineering and hyperparameter tuning

CODE:
```python
{self._generate_template_improvement_code()}
```
"""

    def _generate_template_improvement_code(self) -> str:
        """Generate improved template code"""
        return '''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    """Improved ML solution with feature engineering and tuning"""
    print("Starting improved ML solution...")

    # Load data (placeholder)
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    X_test = np.random.randn(100, 10)

    # Feature engineering
    X_enhanced = np.column_stack([
        X,
        X[:, 0] * X[:, 1],  # interaction feature
        np.sum(X**2, axis=1)  # polynomial feature
    ])

    X_test_enhanced = np.column_stack([
        X_test,
        X_test[:, 0] * X_test[:, 1],
        np.sum(X_test**2, axis=1)
    ])

    # Scale features
    scaler = StandardScaler()
    X_enhanced = scaler.fit_transform(X_enhanced)
    X_test_enhanced = scaler.transform(X_test_enhanced)

    # 5-fold cross-validation with tuned parameters
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_enhanced)):
        X_train, X_val = X_enhanced[train_idx], X_enhanced[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Improved model with tuning
        model = RandomForestRegressor(
            n_estimators=200,  # increased
            max_depth=10,      # tuned
            min_samples_split=5,  # tuned
            random_state=42
        )
        model.fit(X_train, y_train)

        # Validate
        y_pred = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, y_pred))
        cv_scores.append(score)

        print(f"Fold {fold+1}: RMSE = {score:.4f}")

    # Final CV score
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"5-fold CV Score: {cv_mean:.4f} ± {cv_std:.4f}")

    # Train final model and predict
    final_model = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5, random_state=42
    )
    final_model.fit(X_enhanced, y)
    test_pred = final_model.predict(X_test_enhanced)

    # Write submission
    submission = pd.DataFrame({
        'id': range(len(test_pred)),
        'target': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

    return cv_mean

if __name__ == "__main__":
    main()
'''


class DebugOperator:
    """AIRA Debug operator - fixes bugs without changing core approach"""
    name = "debug"

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def apply(self, ctx: ProblemSpec, parent: Artifact, memory: Memory) -> Artifact:
        """Debug a failed solution"""

        if not parent or not parent.logs_path:
            raise ValueError("Debug operator requires parent with error logs")

        # Read parent code and error logs
        parent_code = self._read_code(parent.code_path / "solution.py")
        error_logs = self._read_logs(parent.logs_path)

        # Generate debug fix
        debug_prompt = self._build_debug_prompt(ctx, parent_code, error_logs, memory)

        try:
            if self.llm_client:
                response = await self.llm_client.generate_response(
                    prompt=debug_prompt,
                    system_prompt="You are a Python debugging expert. Fix bugs without changing the core approach.",
                    temperature=0.3,  # Lower temperature for debugging
                    max_tokens=3000
                )
                content = response.get("content", "")
            else:
                content = self._generate_template_debug(parent_code)

            # Extract fixed code
            code = self._parse_debug_output(content)

            # Create artifact
            artifact_id = f"debug_{parent.id}_{hash(error_logs) % 10000}"
            artifact_dir = Path(tempfile.mkdtemp(prefix=f"artifact_{artifact_id}_"))

            # Write fixed code
            code_file = artifact_dir / "solution.py"
            with open(code_file, 'w') as f:
                f.write(code)

            artifact = Artifact(
                id=artifact_id,
                parent_id=parent.id,
                depth=parent.depth + 1,
                code_path=artifact_dir,
                plan_text=f"Debug fix for: {parent.plan_text}",
                operator="debug",
                status="PENDING"
            )

            memory.add_error(error_logs[:200])  # Store error summary
            return artifact

        except Exception as e:
            logger.error(f"Debug operator failed: {e}")
            raise

    def _build_debug_prompt(self, ctx: ProblemSpec, code: str, error_logs: str, memory: Memory) -> str:
        """Build prompt for debugging"""

        recent_errors = "\n".join(f"- {err}" for err in memory.recent_errors[-3:])

        return f"""
Fix the bugs in this code without changing the core approach.

**Buggy Code:**
```python
{code}
```

**Error Logs:**
{error_logs}

**Recent Similar Errors:**
{recent_errors if recent_errors else "None"}

**Instructions:**
1. Fix bugs ONLY - don't change the core method
2. Ensure code still uses 5-fold CV
3. Must print CV score and write submission.csv
4. Output ONE complete, executable script

**Output Format:**
```python
[Complete fixed script]
```
"""

    def _read_code(self, code_path: Path) -> str:
        """Read code from file"""
        try:
            with open(code_path, 'r') as f:
                return f.read()
        except Exception:
            return ""

    def _read_logs(self, logs_path: Path) -> str:
        """Read error logs"""
        try:
            with open(logs_path, 'r') as f:
                return f.read()
        except Exception:
            return ""

    def _parse_debug_output(self, content: str) -> str:
        """Extract fixed code"""

        # Extract code block
        code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Fallback: try any code block
        code_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Last resort: return template
        return self._generate_template_debug_code()

    def _generate_template_debug(self, original_code: str) -> str:
        """Generate template debug fix"""
        return f"""
```python
{self._generate_template_debug_code()}
```
"""

    def _generate_template_debug_code(self) -> str:
        """Generate debugged template code"""
        return '''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    """Debugged ML solution"""
    print("Starting debugged ML solution...")

    try:
        # Load data with error handling
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = np.random.randn(1000)
        X_test = np.random.randn(100, 10)

        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model with error handling
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Validate
            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_scores.append(score)

            print(f"Fold {fold+1}: RMSE = {score:.4f}")

        # Final CV score
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        print(f"5-fold CV Score: {cv_mean:.4f} ± {cv_std:.4f}")

        # Train final model
        final_model = RandomForestRegressor(n_estimators=100, random_state=42)
        final_model.fit(X, y)
        test_pred = final_model.predict(X_test)

        # Write submission with error handling
        submission = pd.DataFrame({
            'id': range(len(test_pred)),
            'target': test_pred
        })
        submission.to_csv('submission.csv', index=False)
        print("Submission saved to submission.csv")

        return cv_mean

    except Exception as e:
        print(f"Error in execution: {e}")
        return 0.0

if __name__ == "__main__":
    main()
'''


class CodeExecutor:
    """Execute code artifacts and extract 5-fold CV scores"""

    def __init__(self, timeout_sec: int = 300):
        self.timeout_sec = timeout_sec

    async def run_and_score(self, artifact: Artifact) -> float:
        """Execute artifact code and return CV score"""

        artifact.status = "RUNNING"

        try:
            # Execute Python script
            script_path = artifact.code_path / "solution.py"

            result = await asyncio.create_subprocess_exec(
                'python', str(script_path),
                cwd=str(artifact.code_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(),
                timeout=self.timeout_sec
            )

            stdout_text = stdout.decode('utf-8', errors='ignore')
            stderr_text = stderr.decode('utf-8', errors='ignore')

            # Parse CV score from stdout
            cv_score = self._parse_cv_score(stdout_text)

            # Save logs
            log_file = artifact.code_path / "execution.log"
            with open(log_file, 'w') as f:
                f.write(f"STDOUT:\n{stdout_text}\n\nSTDERR:\n{stderr_text}")

            artifact.logs_path = log_file
            artifact.val_metric = cv_score
            artifact.status = "COMPLETED" if result.returncode == 0 else "FAILED"

            return cv_score if cv_score is not None else 0.0

        except asyncio.TimeoutError:
            artifact.status = "FAILED"
            logger.error(f"Artifact {artifact.id} execution timed out")
            return 0.0

        except Exception as e:
            artifact.status = "FAILED"
            logger.error(f"Artifact {artifact.id} execution failed: {e}")
            return 0.0

    def _parse_cv_score(self, stdout: str) -> Optional[float]:
        """Parse 5-fold CV score from stdout"""

        # Look for patterns like "5-fold CV Score: 0.1234"
        patterns = [
            r'5-fold CV Score:\s*([0-9\.]+)',
            r'CV Score:\s*([0-9\.]+)',
            r'RMSE.*?([0-9\.]+)',
            r'Score.*?([0-9\.]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # Fallback: look for any float in the last few lines
        lines = stdout.strip().split('\n')[-5:]
        for line in lines:
            numbers = re.findall(r'([0-9]+\.[0-9]+)', line)
            if numbers:
                try:
                    return float(numbers[-1])  # Take last number
                except ValueError:
                    continue

        return None


# Operator factory
class OperatorFactory:
    """Factory for creating AIRA operators"""

    @staticmethod
    def create_draft(complexity: str = "normal", llm_client=None) -> DraftOperator:
        return DraftOperator(complexity, llm_client)

    @staticmethod
    def create_improve(complexity: str = "normal", llm_client=None) -> ImproveOperator:
        return ImproveOperator(complexity, llm_client)

    @staticmethod
    def create_debug(llm_client=None) -> DebugOperator:
        return DebugOperator(llm_client)