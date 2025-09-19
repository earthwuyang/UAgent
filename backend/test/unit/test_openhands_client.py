"""
Unit tests for OpenHands Client
"""

import asyncio
import tempfile
import shutil
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from app.core.openhands.client import (
    OpenHandsClient,
    SessionConfig,
    SessionState,
    CodeGenerationRequest,
    CodeGenerationResult
)


class TestOpenHandsClient:
    """Test OpenHandsClient functionality"""

    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def openhands_client(self, temp_base_dir):
        """Create OpenHandsClient instance for testing"""
        return OpenHandsClient(temp_base_dir)

    @pytest.mark.asyncio
    async def test_client_initialization(self, temp_base_dir):
        """Test OpenHandsClient initialization"""
        client = OpenHandsClient(temp_base_dir)

        assert client.workspace_manager is not None
        assert client.code_executor is not None
        assert client.sessions == {}
        assert client.session_states == {}

    @pytest.mark.asyncio
    async def test_session_creation(self, openhands_client):
        """Test session creation"""
        research_type = "scientific_research"
        session_id = "test_session_123"

        config = await openhands_client.create_session(
            research_type=research_type,
            session_id=session_id
        )

        assert config.session_id == session_id
        assert config.research_type == research_type
        assert config.max_iterations == 10  # Default value
        assert config.timeout_per_step == 300  # Default value
        assert config.enable_code_execution is True

        # Check session is tracked
        assert session_id in openhands_client.sessions
        assert session_id in openhands_client.session_states

        # Check session state
        state = openhands_client.session_states[session_id]
        assert state.session_id == session_id
        assert state.status == "created"
        assert state.current_step == 0
        assert state.total_steps == 0

    @pytest.mark.asyncio
    async def test_session_creation_with_auto_id(self, openhands_client):
        """Test session creation with automatic ID generation"""
        research_type = "deep_research"

        config = await openhands_client.create_session(research_type=research_type)

        assert config.session_id.startswith("session_")
        assert config.research_type == research_type
        assert len(config.session_id) > 8  # Should have unique suffix

    @pytest.mark.asyncio
    async def test_session_creation_with_config(self, openhands_client):
        """Test session creation with custom configuration"""
        research_type = "code_research"
        custom_config = {
            "max_iterations": 5,
            "timeout_per_step": 600,
            "custom_setting": "test_value"
        }

        config = await openhands_client.create_session(
            research_type=research_type,
            config=custom_config
        )

        assert config.max_iterations == 5
        assert config.timeout_per_step == 600
        assert "custom_setting" in config.workspace_config

    @pytest.mark.asyncio
    async def test_get_session_state(self, openhands_client):
        """Test session state retrieval"""
        session_id = "test_state_session"
        await openhands_client.create_session("scientific_research", session_id)

        state = await openhands_client.get_session_state(session_id)

        assert state is not None
        assert state.session_id == session_id
        assert state.status == "created"

        # Test non-existent session
        invalid_state = await openhands_client.get_session_state("nonexistent")
        assert invalid_state is None

    @pytest.mark.asyncio
    async def test_get_workspace_status(self, openhands_client):
        """Test workspace status retrieval"""
        session_id = "test_workspace_status"
        await openhands_client.create_session("scientific_research", session_id)

        status = await openhands_client.get_workspace_status(session_id)

        assert status is not None
        assert status.workspace_id == session_id
        assert status.status in ["active", "stopped", "created"]

        # Test non-existent session
        invalid_status = await openhands_client.get_workspace_status("nonexistent")
        assert invalid_status is None

    @pytest.mark.asyncio
    async def test_code_generation_and_execution(self, openhands_client):
        """Test code generation and execution"""
        session_id = "test_code_gen"
        await openhands_client.create_session("scientific_research", session_id)

        request = CodeGenerationRequest(
            session_id=session_id,
            task_description="Create a simple Python function that adds two numbers",
            context={"research_type": "scientific_research"},
            execute_immediately=True
        )

        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.generate = AsyncMock(return_value="""
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(f"5 + 3 = {result}")
""")

        result = await openhands_client.generate_and_execute_code(
            request, mock_llm_client
        )

        assert result.session_id == session_id
        assert result.task_description == request.task_description
        assert result.generated_code is not None
        assert len(result.generated_code) > 0
        assert result.execution_result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_code_generation_without_llm(self, openhands_client):
        """Test code generation fallback without LLM client"""
        session_id = "test_no_llm"
        await openhands_client.create_session("scientific_research", session_id)

        request = CodeGenerationRequest(
            session_id=session_id,
            task_description="Test template generation",
            execute_immediately=True
        )

        result = await openhands_client.generate_and_execute_code(request)

        assert result.session_id == session_id
        assert result.generated_code is not None
        assert "TODO: Implement the actual task logic here" in result.generated_code
        assert result.execution_result is not None

    @pytest.mark.asyncio
    async def test_code_generation_session_not_found(self, openhands_client):
        """Test code generation with non-existent session"""
        request = CodeGenerationRequest(
            session_id="nonexistent_session",
            task_description="This should fail"
        )

        result = await openhands_client.generate_and_execute_code(request)

        assert result.success is False
        assert "Session not found" in result.analysis

    @pytest.mark.asyncio
    async def test_code_generation_with_execution_error(self, openhands_client):
        """Test code generation with execution error"""
        session_id = "test_exec_error"
        await openhands_client.create_session("scientific_research", session_id)

        request = CodeGenerationRequest(
            session_id=session_id,
            task_description="Generate code with error",
            execute_immediately=True
        )

        mock_llm_client = MagicMock()
        mock_llm_client.generate = AsyncMock(return_value="""
print("Before error")
x = 1 / 0  # This will cause an error
print("After error")
""")

        result = await openhands_client.generate_and_execute_code(
            request, mock_llm_client
        )

        assert result.session_id == session_id
        assert result.execution_result is not None
        assert result.execution_result.success is False
        assert "ZeroDivisionError" in result.execution_result.stderr

    @pytest.mark.asyncio
    async def test_streaming_code_execution(self, openhands_client):
        """Test streaming code execution"""
        session_id = "test_streaming"
        await openhands_client.create_session("scientific_research", session_id)

        python_code = """
import time
for i in range(3):
    print(f"Step {i+1}")
    time.sleep(0.1)
print("Completed")
"""

        events = []
        async for event in openhands_client.execute_code_streaming(
            session_id, python_code
        ):
            events.append(event)

        # Should have start, stdout events, and complete event
        assert len(events) >= 2
        assert any(event["type"] == "start" for event in events)
        assert any(event["type"] == "stdout" for event in events)

    @pytest.mark.asyncio
    async def test_streaming_execution_session_not_found(self, openhands_client):
        """Test streaming execution with non-existent session"""
        events = []
        async for event in openhands_client.execute_code_streaming(
            "nonexistent_session", "print('test')"
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "Session not found" in events[0]["message"]

    @pytest.mark.asyncio
    async def test_streaming_execution_unsupported_language(self, openhands_client):
        """Test streaming execution with unsupported language"""
        session_id = "test_unsupported_lang"
        await openhands_client.create_session("scientific_research", session_id)

        events = []
        async for event in openhands_client.execute_code_streaming(
            session_id, "console.log('test')", language="javascript"
        ):
            events.append(event)

        assert len(events) >= 1
        error_event = next(e for e in events if e["type"] == "error")
        assert "not supported" in error_event["message"]

    @pytest.mark.asyncio
    async def test_session_closure(self, openhands_client):
        """Test session closure and cleanup"""
        session_id = "test_closure"
        await openhands_client.create_session("scientific_research", session_id)

        # Verify session exists
        assert session_id in openhands_client.sessions
        assert session_id in openhands_client.session_states

        # Close session
        success = await openhands_client.close_session(session_id)

        assert success is True
        assert session_id not in openhands_client.sessions
        assert session_id not in openhands_client.session_states

    @pytest.mark.asyncio
    async def test_close_nonexistent_session(self, openhands_client):
        """Test closing non-existent session"""
        success = await openhands_client.close_session("nonexistent_session")
        assert success is False

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self, openhands_client):
        """Test multiple concurrent sessions"""
        session_ids = [f"concurrent_session_{i}" for i in range(5)]

        # Create sessions concurrently
        tasks = [
            openhands_client.create_session("scientific_research", session_id)
            for session_id in session_ids
        ]
        configs = await asyncio.gather(*tasks)

        # Verify all sessions created
        assert len(configs) == 5
        for i, config in enumerate(configs):
            assert config.session_id == session_ids[i]
            assert session_ids[i] in openhands_client.sessions

        # Clean up sessions
        cleanup_tasks = [
            openhands_client.close_session(session_id)
            for session_id in session_ids
        ]
        results = await asyncio.gather(*cleanup_tasks)
        assert all(results)

    @pytest.mark.asyncio
    async def test_session_state_updates(self, openhands_client):
        """Test session state updates during operations"""
        session_id = "test_state_updates"
        await openhands_client.create_session("scientific_research", session_id)

        # Initial state
        state = await openhands_client.get_session_state(session_id)
        assert state.status == "created"
        assert state.current_step == 0

        # Execute code (should update state)
        request = CodeGenerationRequest(
            session_id=session_id,
            task_description="Simple test task",
            execute_immediately=True
        )

        await openhands_client.generate_and_execute_code(request)

        # State should be updated
        updated_state = await openhands_client.get_session_state(session_id)
        assert updated_state.current_step > state.current_step
        assert updated_state.last_activity > state.last_activity

    def test_session_config_creation(self):
        """Test SessionConfig object creation"""
        config = SessionConfig(
            session_id="test_config",
            research_type="scientific_research",
            max_iterations=5,
            timeout_per_step=600,
            enable_code_execution=True,
            workspace_config={"test": "value"}
        )

        assert config.session_id == "test_config"
        assert config.research_type == "scientific_research"
        assert config.max_iterations == 5
        assert config.timeout_per_step == 600
        assert config.enable_code_execution is True
        assert config.workspace_config == {"test": "value"}

    def test_session_state_creation(self):
        """Test SessionState object creation"""
        state = SessionState(
            session_id="test_state",
            status="running",
            current_step=3,
            total_steps=10,
            workspace_id="workspace_123",
            created_at=1234567890.0,
            last_activity=1234567900.0,
            error_message="Test error"
        )

        assert state.session_id == "test_state"
        assert state.status == "running"
        assert state.current_step == 3
        assert state.total_steps == 10
        assert state.workspace_id == "workspace_123"
        assert state.error_message == "Test error"

    def test_code_generation_request_creation(self):
        """Test CodeGenerationRequest object creation"""
        request = CodeGenerationRequest(
            session_id="test_request",
            task_description="Test task",
            context={"key": "value"},
            execute_immediately=False,
            language="python",
            timeout=120
        )

        assert request.session_id == "test_request"
        assert request.task_description == "Test task"
        assert request.context == {"key": "value"}
        assert request.execute_immediately is False
        assert request.language == "python"
        assert request.timeout == 120

    def test_code_generation_result_creation(self):
        """Test CodeGenerationResult object creation"""
        result = CodeGenerationResult(
            session_id="test_result",
            task_description="Test task",
            generated_code="print('test')",
            execution_result=None,
            analysis="Test analysis",
            success=True,
            next_steps=["step1", "step2"]
        )

        assert result.session_id == "test_result"
        assert result.task_description == "Test task"
        assert result.generated_code == "print('test')"
        assert result.analysis == "Test analysis"
        assert result.success is True
        assert result.next_steps == ["step1", "step2"]

    @pytest.mark.asyncio
    async def test_code_generation_prompt_creation(self, openhands_client):
        """Test code generation prompt creation"""
        request = CodeGenerationRequest(
            session_id="test_prompt",
            task_description="Create a sorting function",
            context={"data_type": "integers", "algorithm": "quicksort"}
        )

        prompt = openhands_client._create_code_generation_prompt(request)

        assert "Create a sorting function" in prompt
        assert "python" in prompt.lower()
        assert "data_type" in prompt
        assert "quicksort" in prompt

    @pytest.mark.asyncio
    async def test_template_code_generation(self, openhands_client):
        """Test template code generation"""
        request = CodeGenerationRequest(
            session_id="test_template",
            task_description="Generate template code"
        )

        template_code = openhands_client._create_template_code(request)

        assert "def main():" in template_code
        assert "Generate template code" in template_code
        assert "test_template" in template_code
        assert "if __name__ == \"__main__\":" in template_code

    @pytest.mark.asyncio
    async def test_llm_code_generation_error_handling(self, openhands_client):
        """Test LLM code generation with error handling"""
        request = CodeGenerationRequest(
            session_id="test_llm_error",
            task_description="Test error handling"
        )

        # Mock LLM client that raises an exception
        mock_llm_client = MagicMock()
        mock_llm_client.generate = AsyncMock(side_effect=Exception("LLM error"))

        code = await openhands_client._generate_code_with_llm(
            mock_llm_client, "test prompt", request
        )

        # Should fall back to template code
        assert "def main():" in code
        assert "Test error handling" in code

    @pytest.mark.asyncio
    async def test_analysis_generation(self, openhands_client):
        """Test execution result analysis generation"""
        from app.core.openhands.code_executor import ExecutionResult

        # Test successful execution
        success_result = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="Hello World\nResult: 42",
            stderr="",
            execution_time=1.5,
            files_created=["output.txt"],
            files_modified=[]
        )

        analysis = openhands_client._analyze_execution_result(
            success_result, "print('Hello World')"
        )

        assert "✅ Code executed successfully" in analysis
        assert "1.50 seconds" in analysis
        assert "2 lines" in analysis
        assert "1" in analysis  # files created

        # Test failed execution
        failure_result = ExecutionResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Error: Something went wrong",
            execution_time=0.5,
            files_created=[],
            files_modified=[]
        )

        error_analysis = openhands_client._analyze_execution_result(
            failure_result, "broken code"
        )

        assert "❌ Code execution failed" in error_analysis
        assert "exit code: 1" in error_analysis

    @pytest.mark.asyncio
    async def test_next_steps_generation(self, openhands_client):
        """Test next steps generation based on execution results"""
        from app.core.openhands.code_executor import ExecutionResult

        request = CodeGenerationRequest(
            session_id="test_next_steps",
            task_description="Test task",
            context={"research_type": "scientific_research"}
        )

        # Test successful execution
        success_result = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="Success",
            stderr="",
            execution_time=1.0,
            files_created=[],
            files_modified=[]
        )

        next_steps = openhands_client._generate_next_steps(request, success_result)

        assert len(next_steps) <= 5
        assert any("analyze" in step.lower() for step in next_steps)
        assert any("scientific" in step.lower() or "experiment" in step.lower() for step in next_steps)

        # Test failed execution
        failure_result = ExecutionResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Error occurred",
            execution_time=0.5,
            files_created=[],
            files_modified=[]
        )

        error_steps = openhands_client._generate_next_steps(request, failure_result)

        assert len(error_steps) <= 5
        assert any("debug" in step.lower() for step in error_steps)