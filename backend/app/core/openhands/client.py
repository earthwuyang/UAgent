"""OpenHands Client for UAgent system integration"""

import asyncio
import json
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, AsyncGenerator
import logging

from .workspace_manager import WorkspaceManager, WorkspaceConfig, WorkspaceStatus
from .code_executor import CodeExecutor, ExecutionResult, ExecutionCommand

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for OpenHands session"""
    session_id: str
    research_type: str  # 'deep_research', 'code_research', 'scientific_research'
    max_iterations: int = 10
    timeout_per_step: int = 300
    enable_code_execution: bool = True
    workspace_config: Dict[str, Any] = None


@dataclass
class SessionState:
    """Current state of an OpenHands session"""
    session_id: str
    status: str  # 'created', 'running', 'paused', 'completed', 'error'
    current_step: int
    total_steps: int
    workspace_id: str
    created_at: float
    last_activity: float
    error_message: Optional[str] = None


@dataclass
class CodeGenerationRequest:
    """Request for code generation and execution"""
    session_id: str
    task_description: str
    context: Dict[str, Any] = None
    execute_immediately: bool = True
    language: str = "python"
    timeout: int = 300


@dataclass
class CodeGenerationResult:
    """Result of code generation and execution"""
    session_id: str
    task_description: str
    generated_code: str
    execution_result: Optional[ExecutionResult] = None
    analysis: str = ""
    success: bool = False
    next_steps: List[str] = None


class OpenHandsClient:
    """Main client for OpenHands integration with UAgent"""

    def __init__(self, base_workspace_dir: str = None):
        """Initialize OpenHands client

        Args:
            base_workspace_dir: Base directory for workspaces
        """
        self.workspace_manager = WorkspaceManager(base_workspace_dir)
        self.code_executor = CodeExecutor(self.workspace_manager)
        self.sessions: Dict[str, SessionConfig] = {}
        self.session_states: Dict[str, SessionState] = {}
        logger.info("OpenHandsClient initialized")

    async def create_session(
        self,
        research_type: str,
        session_id: str = None,
        config: Dict[str, Any] = None
    ) -> SessionConfig:
        """Create a new OpenHands session

        Args:
            research_type: Type of research ('deep_research', 'code_research', 'scientific_research')
            session_id: Optional session ID
            config: Additional configuration

        Returns:
            SessionConfig: Configuration for the created session
        """
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        # Create workspace for the session
        workspace_config = await self.workspace_manager.create_workspace(
            research_id=session_id,
            config=config
        )

        # Create session configuration
        session_config = SessionConfig(
            session_id=session_id,
            research_type=research_type,
            workspace_config=config or {},
            **(config or {})
        )

        # Initialize session state
        session_state = SessionState(
            session_id=session_id,
            status="created",
            current_step=0,
            total_steps=0,
            workspace_id=workspace_config.workspace_id,
            created_at=asyncio.get_event_loop().time(),
            last_activity=asyncio.get_event_loop().time()
        )

        self.sessions[session_id] = session_config
        self.session_states[session_id] = session_state

        logger.info(f"Created OpenHands session: {session_id} for {research_type}")
        return session_config

    async def get_session_state(self, session_id: str) -> Optional[SessionState]:
        """Get current state of a session

        Args:
            session_id: ID of the session

        Returns:
            SessionState or None if session doesn't exist
        """
        return self.session_states.get(session_id)

    async def get_workspace_status(self, session_id: str) -> Optional[WorkspaceStatus]:
        """Get workspace status for a session

        Args:
            session_id: ID of the session

        Returns:
            WorkspaceStatus or None if session doesn't exist
        """
        session_state = self.session_states.get(session_id)
        if not session_state:
            return None

        return await self.workspace_manager.get_workspace_status(session_state.workspace_id)

    async def generate_and_execute_code(
        self,
        request: CodeGenerationRequest,
        llm_client=None
    ) -> CodeGenerationResult:
        """Generate code using LLM and execute it

        Args:
            request: CodeGenerationRequest object
            llm_client: LLM client for code generation

        Returns:
            CodeGenerationResult: Result of generation and execution
        """
        session_state = self.session_states.get(request.session_id)
        if not session_state:
            return CodeGenerationResult(
                session_id=request.session_id,
                task_description=request.task_description,
                generated_code="",
                success=False,
                analysis="Session not found"
            )

        # Update session state
        session_state.status = "running"
        session_state.current_step += 1
        session_state.last_activity = asyncio.get_event_loop().time()

        try:
            # Generate code using LLM
            code_generation_prompt = self._create_code_generation_prompt(request)

            if llm_client:
                # Use LLM to generate code
                generated_code = await self._generate_code_with_llm(
                    llm_client,
                    code_generation_prompt,
                    request
                )
            else:
                # Fallback: create template code
                generated_code = self._create_template_code(request)

            # Save generated code to workspace
            code_filename = f"generated_{request.session_id}_{session_state.current_step}.py"
            await self.workspace_manager.write_file(
                session_state.workspace_id,
                f"code/{code_filename}",
                generated_code
            )

            execution_result = None
            if request.execute_immediately:
                # Execute the generated code
                execution_result = await self.code_executor.execute_python_code(
                    workspace_id=session_state.workspace_id,
                    code=generated_code,
                    file_name=code_filename,
                    timeout=request.timeout
                )

            # Analyze results
            analysis = self._analyze_execution_result(execution_result, generated_code)
            success = execution_result.success if execution_result else True

            # Generate next steps
            next_steps = self._generate_next_steps(request, execution_result)

            session_state.status = "completed" if success else "error"
            if not success and execution_result:
                session_state.error_message = execution_result.stderr

            return CodeGenerationResult(
                session_id=request.session_id,
                task_description=request.task_description,
                generated_code=generated_code,
                execution_result=execution_result,
                analysis=analysis,
                success=success,
                next_steps=next_steps
            )

        except Exception as e:
            session_state.status = "error"
            session_state.error_message = str(e)
            logger.error(f"Code generation failed for session {request.session_id}: {e}")

            return CodeGenerationResult(
                session_id=request.session_id,
                task_description=request.task_description,
                generated_code="",
                success=False,
                analysis=f"Code generation failed: {str(e)}"
            )

    async def execute_code_streaming(
        self,
        session_id: str,
        code: str,
        language: str = "python"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute code with streaming output

        Args:
            session_id: ID of the session
            code: Code to execute
            language: Programming language

        Yields:
            Dict[str, Any]: Stream of execution events
        """
        session_state = self.session_states.get(session_id)
        if not session_state:
            yield {"type": "error", "message": "Session not found"}
            return

        session_state.status = "running"
        session_state.last_activity = asyncio.get_event_loop().time()

        try:
            if language.lower() == "python":
                # Write code to workspace
                code_filename = f"streaming_{session_state.current_step}.py"
                await self.workspace_manager.write_file(
                    session_state.workspace_id,
                    f"code/{code_filename}",
                    code
                )

                # Execute with streaming
                command = ExecutionCommand(
                    command=f"python3 code/{code_filename}",
                    timeout=300
                )

                async for event in self.code_executor.stream_execution(
                    session_state.workspace_id,
                    command
                ):
                    yield event

            else:
                yield {"type": "error", "message": f"Language {language} not supported"}

        except Exception as e:
            yield {"type": "error", "message": str(e)}

        finally:
            session_state.status = "completed"

    async def close_session(self, session_id: str) -> bool:
        """Close and cleanup a session

        Args:
            session_id: ID of the session to close

        Returns:
            bool: Success status
        """
        if session_id not in self.sessions:
            return False

        try:
            session_state = self.session_states.get(session_id)
            if session_state:
                # Cleanup workspace
                await self.workspace_manager.cleanup_workspace(session_state.workspace_id)

            # Remove session
            del self.sessions[session_id]
            if session_id in self.session_states:
                del self.session_states[session_id]

            logger.info(f"Closed session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to close session {session_id}: {e}")
            return False

    def _create_code_generation_prompt(self, request: CodeGenerationRequest) -> str:
        """Create prompt for code generation

        Args:
            request: CodeGenerationRequest object

        Returns:
            str: Generated prompt
        """
        context_str = ""
        if request.context:
            context_str = f"\nContext: {json.dumps(request.context, indent=2)}"

        return f"""
Generate {request.language} code for the following task:

Task: {request.task_description}
{context_str}

Requirements:
1. Write clean, well-documented code
2. Include error handling
3. Add comments explaining the logic
4. Use appropriate libraries and best practices
5. Make the code executable and testable

Please provide only the code without explanations or markdown formatting.
"""

    async def _generate_code_with_llm(
        self,
        llm_client,
        prompt: str,
        request: CodeGenerationRequest
    ) -> str:
        """Generate code using LLM client

        Args:
            llm_client: LLM client instance
            prompt: Code generation prompt
            request: Original request

        Returns:
            str: Generated code
        """
        try:
            # Use the LLM client's generation method
            if hasattr(llm_client, 'generate'):
                response = await llm_client.generate(prompt)
            elif hasattr(llm_client, 'chat'):
                response = await llm_client.chat([{"role": "user", "content": prompt}])
            else:
                # Fallback to template
                return self._create_template_code(request)

            # Extract code from response
            code = response
            if isinstance(response, dict):
                code = response.get('content', response.get('text', str(response)))

            return str(code).strip()

        except Exception as e:
            logger.warning(f"LLM code generation failed: {e}, using template")
            return self._create_template_code(request)

    def _create_template_code(self, request: CodeGenerationRequest) -> str:
        """Create template code when LLM is not available

        Args:
            request: CodeGenerationRequest object

        Returns:
            str: Template code
        """
        return f'''#!/usr/bin/env python3
"""
Generated code for: {request.task_description}
Session: {request.session_id}
"""

import os
import sys
import json
from pathlib import Path

def main():
    """
    Main function for: {request.task_description}
    """
    print("Starting task: {request.task_description}")

    # TODO: Implement the actual task logic here
    # This is a template - replace with actual implementation

    try:
        # Placeholder logic
        result = {{"status": "success", "message": "Template code executed"}}
        print(f"Result: {{json.dumps(result, indent=2)}}")
        return result

    except Exception as e:
        print(f"Error: {{e}}")
        return {{"status": "error", "message": str(e)}}

if __name__ == "__main__":
    result = main()
    print("Task completed.")
'''

    def _analyze_execution_result(
        self,
        execution_result: Optional[ExecutionResult],
        generated_code: str
    ) -> str:
        """Analyze execution result and provide insights

        Args:
            execution_result: Result of code execution
            generated_code: The code that was executed

        Returns:
            str: Analysis text
        """
        if not execution_result:
            return "Code generated but not executed."

        analysis_parts = []

        # Basic execution status
        if execution_result.success:
            analysis_parts.append("✅ Code executed successfully")
        else:
            analysis_parts.append(f"❌ Code execution failed (exit code: {execution_result.exit_code})")

        # Execution time
        analysis_parts.append(f"⏱️ Execution time: {execution_result.execution_time:.2f} seconds")

        # Output analysis
        if execution_result.stdout:
            lines = execution_result.stdout.strip().split('\n')
            analysis_parts.append(f"📤 Output: {len(lines)} lines")

        if execution_result.stderr:
            analysis_parts.append(f"⚠️ Errors/Warnings: {len(execution_result.stderr.split('\\n'))} lines")

        # File operations
        if execution_result.files_created:
            analysis_parts.append(f"📁 Files created: {len(execution_result.files_created)}")

        # Code quality insights
        code_lines = len([line for line in generated_code.split('\n') if line.strip()])
        analysis_parts.append(f"📝 Code size: {code_lines} lines")

        return " | ".join(analysis_parts)

    def _generate_next_steps(
        self,
        request: CodeGenerationRequest,
        execution_result: Optional[ExecutionResult]
    ) -> List[str]:
        """Generate suggested next steps

        Args:
            request: Original request
            execution_result: Result of execution

        Returns:
            List[str]: List of suggested next steps
        """
        next_steps = []

        if not execution_result or execution_result.success:
            next_steps.extend([
                "Analyze the output and results",
                "Consider extending functionality",
                "Add error handling and edge cases",
                "Optimize performance if needed"
            ])
        else:
            next_steps.extend([
                "Debug the execution errors",
                "Check input data and parameters",
                "Review code logic and syntax",
                "Add debugging prints and logging"
            ])

        # Research-type specific suggestions
        if request.context:
            research_type = request.context.get('research_type', '')
            if research_type == 'scientific_research':
                next_steps.extend([
                    "Validate experimental results",
                    "Generate visualizations and plots",
                    "Prepare data for further analysis"
                ])
            elif research_type == 'code_research':
                next_steps.extend([
                    "Compare with existing implementations",
                    "Extract reusable patterns",
                    "Generate integration examples"
                ])

        return next_steps[:5]  # Limit to 5 suggestions