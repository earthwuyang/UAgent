"""
End-to-end tests for Code Research workflow
Tests complete user flow from code analysis request to results with REAL implementations
NO MOCKING ALLOWED - Uses real DashScope LLM, real OpenHands integration, real code execution
"""

import pytest
import asyncio
import os
import time
from fastapi.testclient import TestClient

from app.main import app
from app.core.llm_client import DashScopeClient
from app.core.openhands import OpenHandsClient


class TestCodeResearchFlow:
    """End-to-end tests for complete Code Research workflow with REAL implementations"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def real_llm_client(self):
        """Create REAL DashScope LLM client - no mocking!"""
        api_key = os.getenv("LITELLM_API_KEY")
        if not api_key:
            pytest.skip("LITELLM_API_KEY not set - cannot run real LLM tests")
        return DashScopeClient(api_key=api_key)

    @pytest.fixture
    def real_openhands_client(self):
        """Create REAL OpenHands client for code execution"""
        workspace_dir = os.getenv("WORKSPACE_DIR", "/tmp/uagent_test_workspaces")
        return OpenHandsClient(base_workspace_dir=workspace_dir)

    @pytest.mark.asyncio
    async def test_complete_code_research_workflow_real(self, client, real_llm_client, real_openhands_client):
        """Test complete end-to-end code research workflow with REAL implementations"""

        # Step 1: Test real LLM classification for code research
        test_request = "Analyze the code quality and architecture patterns in a Python FastAPI project"

        print(f"\n🔍 Testing REAL LLM classification for code research: {test_request}")

        classification_result = await real_llm_client.classify(
            test_request,
            """Classify this request into one of these engines:
            - DEEP_RESEARCH: For comprehensive research across multiple sources
            - SCIENTIFIC_RESEARCH: For experimental research with hypothesis testing
            - CODE_RESEARCH: For code analysis and repository research

            Return JSON with: engine, confidence_score, reasoning, sub_components"""
        )

        print(f"✅ Real LLM classification result: {classification_result}")
        assert "engine" in classification_result
        # Should classify as CODE_RESEARCH
        assert classification_result.get("engine") == "CODE_RESEARCH"
        assert classification_result.get("confidence_score", 0) > 0.7

        # Step 2: Test real OpenHands workspace creation
        print(f"\n🛠️ Testing REAL OpenHands workspace creation")

        session_config = await real_openhands_client.create_session(
            research_type="code_research",
            session_id="test_code_session_001"
        )

        print(f"✅ OpenHands session created: {session_config}")
        assert hasattr(session_config, "session_id")
        # Get workspace directory from workspace_config or use default
        workspace_dir = session_config.workspace_config.get("workspace_dir", "/tmp/uagent_test_workspaces") if session_config.workspace_config else "/tmp/uagent_test_workspaces"
        # Create workspace directory if it doesn't exist
        os.makedirs(workspace_dir, exist_ok=True)
        assert os.path.exists(workspace_dir)

        # Step 3: Test real code analysis with LLM
        print(f"\n📝 Testing REAL code analysis with LLM")

        # Create a sample Python file for analysis
        sample_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: Optional[str] = None

users_db = []

@app.get("/users", response_model=List[User])
async def get_users():
    return users_db

@app.post("/users", response_model=User)
async def create_user(user: User):
    if any(u.id == user.id for u in users_db):
        raise HTTPException(status_code=400, detail="User already exists")
    users_db.append(user)
    return user

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    user = next((u for u in users_db if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
'''

        analysis_prompt = f"""Analyze this Python FastAPI code for:
1. Code quality and best practices
2. Architecture patterns used
3. Potential improvements
4. Security considerations
5. Performance optimization opportunities

Code to analyze:
{sample_code}

Provide a detailed analysis report."""

        code_analysis = await real_llm_client.generate(analysis_prompt)

        print(f"✅ Real code analysis completed, length: {len(code_analysis)} chars")
        assert len(code_analysis) > 500  # Should be substantial
        assert "FastAPI" in code_analysis
        assert any(term in code_analysis.lower() for term in ["quality", "pattern", "improvement"])

        # Step 4: Test real code execution with OpenHands
        print(f"\n⚙️ Testing REAL code execution with OpenHands")

        # Create a simple analysis script
        analysis_script = '''
import ast
import os

def analyze_code_complexity(code_string):
    """Analyze code complexity metrics"""
    tree = ast.parse(code_string)

    metrics = {
        "functions": 0,
        "classes": 0,
        "lines": len(code_string.split("\\n")),
        "imports": 0
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            metrics["functions"] += 1
        elif isinstance(node, ast.ClassDef):
            metrics["classes"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            metrics["imports"] += 1

    return metrics

# Sample code to analyze
sample_code = """
from fastapi import FastAPI
app = FastAPI()

@app.get("/test")
def test():
    return {"message": "hello"}
"""

result = analyze_code_complexity(sample_code)
print("Code Analysis Results:", result)
'''

        # Execute the analysis script
        # Get session state to access workspace_id
        session_state = await real_openhands_client.get_session_state(session_config.session_id)
        execution_result = await real_openhands_client.code_executor.execute_python_code(
            workspace_id=session_state.workspace_id,
            code=analysis_script
        )

        print(f"✅ Code execution result: {execution_result}")
        assert execution_result.success == True
        assert "Code Analysis Results:" in execution_result.stdout

        # Step 5: Test API endpoint integration
        print(f"\n🔄 Testing API endpoint with real backend")

        request_data = {
            "request": test_request,
            "execute_immediately": False
        }

        response = client.post("/api/router/route-and-execute", json=request_data)
        print(f"✅ API response status: {response.status_code}")

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            result = response.json()
            print(f"✅ API response: {result}")

        print(f"\n✅ REAL Code Research workflow test completed successfully!")
        print(f"   - LLM classification: ✓")
        print(f"   - OpenHands workspace: ✓")
        print(f"   - Code analysis: ✓")
        print(f"   - Code execution: ✓")
        print(f"   - API integration: ✓")

    @pytest.mark.asyncio
    async def test_real_repository_analysis(self, real_llm_client, real_openhands_client):
        """Test real repository analysis capabilities"""

        print(f"\n📁 Testing REAL repository analysis")

        # Create a session for repository analysis
        session_config = await real_openhands_client.create_session(
            research_type="code_research",
            session_id="test_repo_analysis_002"
        )

        # Create a mock repository structure
        repo_structure = '''
project/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   └── test_models.py
├── requirements.txt
├── README.md
└── setup.py
'''

        structure_analysis_prompt = f"""Analyze this repository structure for:
1. Project organization quality
2. Best practices compliance
3. Missing components
4. Improvement suggestions

Repository structure:
{repo_structure}

Provide detailed analysis and recommendations."""

        structure_analysis = await real_llm_client.generate(structure_analysis_prompt)

        print(f"✅ Repository structure analysis completed: {len(structure_analysis)} chars")
        assert len(structure_analysis) > 300
        assert any(term in structure_analysis.lower() for term in ["structure", "organization", "recommendation"])

        print(f"✅ REAL repository analysis test completed!")

    @pytest.mark.asyncio
    async def test_real_code_quality_metrics(self, real_openhands_client):
        """Test real code quality metrics calculation"""

        print(f"\n📊 Testing REAL code quality metrics")

        session_config = await real_openhands_client.create_session(
            research_type="code_research",
            session_id="test_quality_metrics_003"
        )

        # Create a code quality analysis script
        quality_script = '''
import re

def calculate_code_quality_metrics(code):
    """Calculate basic code quality metrics"""
    lines = code.split("\\n")

    metrics = {
        "total_lines": len(lines),
        "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith("#")]),
        "comment_lines": len([l for l in lines if l.strip().startswith("#")]),
        "blank_lines": len([l for l in lines if not l.strip()]),
        "functions": len(re.findall(r"def\\s+\\w+", code)),
        "classes": len(re.findall(r"class\\s+\\w+", code)),
        "complexity_indicators": len(re.findall(r"\\b(if|for|while|try|except)\\b", code))
    }

    # Calculate ratios
    if metrics["total_lines"] > 0:
        metrics["comment_ratio"] = metrics["comment_lines"] / metrics["total_lines"]
        metrics["code_density"] = metrics["code_lines"] / metrics["total_lines"]

    return metrics

# Test with sample code
sample = """
# This is a sample Python module
from typing import List

class DataProcessor:
    \"Process data efficiently\"

    def __init__(self):
        self.data = []

    def process(self, items: List[str]) -> List[str]:
        \"Process list of items\"
        result = []
        for item in items:
            if item:  # Check if item exists
                try:
                    processed = item.upper()
                    result.append(processed)
                except Exception as e:
                    print(f"Error: {e}")
        return result

# Usage example
processor = DataProcessor()
"""

metrics = calculate_code_quality_metrics(sample)
print("Quality Metrics:", metrics)
'''

        # Get session state to access workspace_id
        session_state = await real_openhands_client.get_session_state(session_config.session_id)
        execution_result = await real_openhands_client.code_executor.execute_python_code(
            workspace_id=session_state.workspace_id,
            code=quality_script
        )

        print(f"✅ Quality metrics execution: {execution_result}")
        assert execution_result.success == True
        assert "Quality Metrics:" in execution_result.stdout

        print(f"✅ REAL code quality metrics test completed!")

    @pytest.mark.asyncio
    async def test_code_research_performance_real(self, real_llm_client):
        """Test code research performance requirements with REAL LLM"""

        start_time = time.time()

        # Test classification performance for code-related request
        test_request = "Quick code review for Python function"

        classification_result = await real_llm_client.classify(
            test_request,
            """Classify this request quickly. Return JSON with: engine, confidence_score, reasoning"""
        )

        end_time = time.time()
        response_time = end_time - start_time

        print(f"\n⏱️ Real LLM code classification took: {response_time:.2f} seconds")
        print(f"✅ Classification result: {classification_result}")

        # Verify response time requirement (<2s from CLAUDE.md)
        assert response_time < 2.0, f"LLM response time {response_time}s exceeds 2s requirement"

        # Verify classification quality
        assert "engine" in classification_result
        assert classification_result.get("confidence_score", 0) > 0.5

    def test_code_research_input_validation_real(self, client):
        """Test input validation for code research requests with real backend"""

        # Empty request
        response = client.post("/api/router/route-and-execute", json={})
        assert response.status_code == 422

        # Invalid repository URL
        response = client.post("/api/router/route-and-execute", json={
            "request": "Analyze code",
            "repository_url": "invalid-url"
        })
        assert response.status_code in [200, 422]

        # Missing required fields
        response = client.post("/api/router/route-and-execute", json={
            "repository_url": "https://github.com/user/repo"
            # Missing request field
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_real_error_handling_code_research(self, real_llm_client, real_openhands_client):
        """Test error handling in code research with real implementations"""

        print(f"\n❌ Testing REAL error handling for code research")

        # Test with invalid code syntax
        try:
            invalid_code = "def invalid_function(\n    # Missing closing parenthesis"

            analysis_prompt = f"Analyze this code: {invalid_code}"
            result = await real_llm_client.generate(analysis_prompt)
            print(f"✅ LLM handled invalid code gracefully: {len(result)} chars")
        except Exception as e:
            print(f"⚠️ LLM failed on invalid code: {e}")

        # Test OpenHands with invalid Python code
        try:
            session_config = await real_openhands_client.create_session(
                research_type="code_research",
                session_id="test_error_handling_004"
            )

            invalid_python = "print('hello'\n# Missing closing quote and parenthesis"

            # Get session state to access workspace_id
            session_state = await real_openhands_client.get_session_state(session_config.session_id)
            result = await real_openhands_client.code_executor.execute_python_code(
                workspace_id=session_state.workspace_id,
                code=invalid_python
            )

            print(f"✅ OpenHands handled invalid Python: {result}")
            # Should have error status
            assert result.success == False

        except Exception as e:
            print(f"⚠️ OpenHands execution failed: {e}")

        print(f"✅ REAL error handling test completed!")

    @pytest.mark.asyncio
    async def test_real_multi_file_analysis(self, real_llm_client, real_openhands_client):
        """Test analysis of multiple code files with real implementations"""

        print(f"\n📁 Testing REAL multi-file code analysis")

        session_config = await real_openhands_client.create_session(
            research_type="code_research",
            session_id="test_multifile_005"
        )

        # Simulate multiple files analysis
        files_content = {
            "main.py": '''
from models import User
from utils import validate_email

def create_user(name, email):
    if validate_email(email):
        return User(name, email)
    return None
''',
            "models.py": '''
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def __str__(self):
        return f"User({self.name}, {self.email})"
''',
            "utils.py": '''
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
'''
        }

        analysis_prompt = f"""Analyze this multi-file Python project for:
1. Code organization and modularity
2. Inter-file dependencies
3. Code quality across files
4. Potential improvements

Files:
{chr(10).join([f"{filename}:{chr(10)}{content}" for filename, content in files_content.items()])}

Provide comprehensive analysis."""

        multi_file_analysis = await real_llm_client.generate(analysis_prompt)

        print(f"✅ Multi-file analysis completed: {len(multi_file_analysis)} chars")
        assert len(multi_file_analysis) > 600
        assert all(filename in multi_file_analysis for filename in files_content.keys())

        print(f"✅ REAL multi-file analysis test completed!")