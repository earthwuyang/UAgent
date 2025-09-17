"""
Code execution system for computational experiments
Integrates AI Scientist interpreter with iterative code generation and debugging
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import AI Scientist interpreter
import sys
import os
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../ai_scientist_local/ai_scientist/treesearch'))
    from interpreter import Interpreter, ExecutionResult
    INTERPRETER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI Scientist interpreter not available: {e}")
    INTERPRETER_AVAILABLE = False

    # Fallback mock classes
    class ExecutionResult:
        def __init__(self, term_out=None, exec_time=0.0, exc_type=None, exc_info=None, exc_stack=None):
            self.term_out = term_out or []
            self.exec_time = exec_time
            self.exc_type = exc_type
            self.exc_info = exc_info
            self.exc_stack = exc_stack

    class Interpreter:
        def __init__(self, working_dir, timeout=30):
            self.working_dir = working_dir
            self.timeout = timeout

        def run(self, code, reset_session=True):
            return ExecutionResult(exc_type="ImportError")

        def cleanup_session(self):
            pass

from .llm_client import llm_client

logger = logging.getLogger(__name__)


@dataclass
class CodeExecutionResult:
    """Result of code execution with debugging information"""
    success: bool
    confidence: float
    code_blocks: List[str]
    execution_outputs: List[Dict[str, Any]]
    final_code: Optional[str]
    execution_time: float
    iterations: int
    debug_attempts: int
    insights: List[str]
    metrics: Dict[str, Any]


class CodeExecutor:
    """Executes generated code with iterative debugging"""

    def __init__(self, max_iterations: int = 3, max_debug_attempts: int = 2, timeout: int = 30):
        self.max_iterations = max_iterations
        self.max_debug_attempts = max_debug_attempts
        self.timeout = timeout

    async def execute_computational_task(self, task_description: str, config: Dict[str, Any]) -> CodeExecutionResult:
        """
        Execute a computational task with iterative code generation and debugging
        """
        start_time = time.time()

        if not INTERPRETER_AVAILABLE:
            # Fallback to LLM-only code generation without execution
            logger.warning("Interpreter not available, falling back to LLM-only code generation")
            return await self._fallback_llm_only_generation(task_description, config, start_time)

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            interpreter = Interpreter(working_dir=workspace, timeout=self.timeout)

            try:
                # Generate initial code
                code_blocks = await self._generate_initial_code(task_description, config)

                execution_outputs = []
                debug_attempts = 0
                final_code = None
                success = False
                insights = []

                for iteration in range(self.max_iterations):
                    logger.info(f"Code execution iteration {iteration + 1}/{self.max_iterations}")

                    # Try to execute each code block
                    current_code = code_blocks[min(iteration, len(code_blocks) - 1)]

                    # Execute code with debugging
                    execution_result, debug_info = await self._execute_with_debugging(
                        interpreter, current_code, task_description
                    )

                    execution_outputs.append({
                        'iteration': iteration + 1,
                        'code': current_code,
                        'execution_result': execution_result.__dict__ if execution_result else None,
                        'debug_info': debug_info
                    })

                    debug_attempts += debug_info.get('debug_attempts', 0)

                    # Check if execution was successful
                    if execution_result and execution_result.exc_type is None:
                        success = True
                        final_code = current_code
                        insights.extend(self._extract_insights_from_execution(execution_result, task_description))
                        break

                    # If execution failed, try to fix the code
                    if iteration < self.max_iterations - 1:
                        logger.info(f"Code execution failed, attempting to fix...")
                        fixed_code = await self._fix_code(current_code, execution_result, task_description)
                        if fixed_code and fixed_code != current_code:
                            code_blocks.append(fixed_code)

                execution_time = time.time() - start_time

                # Calculate confidence based on success and execution quality
                confidence = self._calculate_confidence(success, execution_outputs, debug_attempts)

                # Generate comprehensive metrics
                metrics = self._generate_metrics(execution_outputs, debug_attempts, execution_time, task_description)

                return CodeExecutionResult(
                    success=success,
                    confidence=confidence,
                    code_blocks=code_blocks,
                    execution_outputs=execution_outputs,
                    final_code=final_code,
                    execution_time=execution_time,
                    iterations=len(execution_outputs),
                    debug_attempts=debug_attempts,
                    insights=insights,
                    metrics=metrics
                )

            except Exception as e:
                logger.error(f"Code execution failed with exception: {e}")
                execution_time = time.time() - start_time

                return CodeExecutionResult(
                    success=False,
                    confidence=0.0,
                    code_blocks=[],
                    execution_outputs=[],
                    final_code=None,
                    execution_time=execution_time,
                    iterations=0,
                    debug_attempts=0,
                    insights=[f"Code execution failed with exception: {str(e)}"],
                    metrics={'error': str(e), 'execution_time': execution_time}
                )

            finally:
                # Clean up interpreter
                try:
                    interpreter.cleanup_session()
                except:
                    pass

    async def _generate_initial_code(self, task_description: str, config: Dict[str, Any]) -> List[str]:
        """Generate initial code implementations"""
        system_prompt = """You are an expert programmer. Generate working Python code that solves the given computational task.

        Requirements:
        1. Write complete, executable Python code
        2. Include proper error handling
        3. Add print statements to show results
        4. Make the code self-contained (import required modules)
        5. Focus on correctness and clarity

        Generate 2-3 different approaches if possible."""

        prompt = f"""Task: {task_description}
        Configuration: {config}

        Please generate working Python code to solve this task. Provide multiple approaches if applicable.
        Each code block should be complete and executable on its own.

        Format your response with code blocks clearly marked with ```python tags."""

        try:
            llm_response = await llm_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=3000
            )

            if llm_response.get("success"):
                content = llm_response.get("content", "")
                code_blocks = self._extract_code_blocks(content)
                return code_blocks if code_blocks else [content]
            else:
                # Fallback: basic template code
                return [f'# Task: {task_description}\nprint("Task: {task_description}")']

        except Exception as e:
            logger.error(f"Failed to generate initial code: {e}")
            return [f'# Task: {task_description}\nprint("Failed to generate code: {str(e)}")']

    async def _execute_with_debugging(self, interpreter: Interpreter, code: str, task_description: str) -> Tuple[Optional[ExecutionResult], Dict[str, Any]]:
        """Execute code with debugging attempts"""
        debug_info = {'debug_attempts': 0, 'debug_logs': []}

        try:
            # First attempt: direct execution
            result = interpreter.run(code, reset_session=True)

            if result.exc_type is None:
                debug_info['debug_logs'].append("Code executed successfully on first attempt")
                return result, debug_info

            # If failed, try debugging
            debug_attempts = 0
            current_code = code

            while debug_attempts < self.max_debug_attempts:
                debug_attempts += 1
                debug_info['debug_attempts'] = debug_attempts

                logger.info(f"Debug attempt {debug_attempts}/{self.max_debug_attempts}")

                # Analyze the error and fix the code
                fixed_code = await self._debug_and_fix_code(current_code, result, task_description)

                if fixed_code and fixed_code != current_code:
                    current_code = fixed_code
                    debug_info['debug_logs'].append(f"Debug attempt {debug_attempts}: Fixed code")

                    # Try executing the fixed code
                    result = interpreter.run(current_code, reset_session=True)

                    if result.exc_type is None:
                        debug_info['debug_logs'].append(f"Debug attempt {debug_attempts}: Success!")
                        return result, debug_info
                else:
                    debug_info['debug_logs'].append(f"Debug attempt {debug_attempts}: No fix generated")
                    break

            debug_info['debug_logs'].append(f"All debug attempts failed")
            return result, debug_info

        except Exception as e:
            debug_info['debug_logs'].append(f"Execution failed with exception: {str(e)}")
            return None, debug_info

    async def _debug_and_fix_code(self, code: str, execution_result: ExecutionResult, task_description: str) -> Optional[str]:
        """Debug failed code and generate a fix"""
        system_prompt = """You are an expert Python debugger. Given code that failed to execute, analyze the error and provide a corrected version.

        Focus on:
        1. Fixing syntax errors
        2. Handling missing imports
        3. Correcting logic errors
        4. Adding proper error handling

        Return only the corrected Python code, nothing else."""

        error_info = ""
        if execution_result.exc_type:
            error_info = f"Error Type: {execution_result.exc_type}\n"
            if execution_result.exc_info:
                error_info += f"Error Details: {execution_result.exc_info}\n"
            if execution_result.term_out:
                error_info += f"Output: {''.join(execution_result.term_out)}\n"

        prompt = f"""Task: {task_description}

        Failed Code:
        ```python
        {code}
        ```

        Error Information:
        {error_info}

        Please provide the corrected Python code that fixes these errors:"""

        try:
            llm_response = await llm_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=2000
            )

            if llm_response.get("success"):
                content = llm_response.get("content", "")
                # Extract code blocks or use full content
                code_blocks = self._extract_code_blocks(content)
                return code_blocks[0] if code_blocks else content.strip()

        except Exception as e:
            logger.error(f"Failed to debug code: {e}")

        return None

    async def _fix_code(self, code: str, execution_result: Optional[ExecutionResult], task_description: str) -> Optional[str]:
        """Generate an improved version of the code"""
        return await self._debug_and_fix_code(code, execution_result, task_description)

    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract Python code blocks from markdown-formatted content"""
        import re

        # Find code blocks between ```python and ```
        python_pattern = r"```python(.*?)```"
        matches = re.findall(python_pattern, content, re.DOTALL)

        if matches:
            return [match.strip() for match in matches if match.strip()]

        # Fallback: find any code blocks between ```
        general_pattern = r"```(.*?)```"
        matches = re.findall(general_pattern, content, re.DOTALL)

        return [match.strip() for match in matches if match.strip()]

    def _extract_insights_from_execution(self, execution_result: ExecutionResult, task_description: str) -> List[str]:
        """Extract insights from successful code execution"""
        insights = []

        if execution_result.term_out:
            output_lines = execution_result.term_out
            insights.append(f"Code executed successfully with {len(output_lines)} output lines")

            # Look for numeric results, success indicators, etc.
            for line in output_lines:
                if any(word in line.lower() for word in ['result', 'answer', 'output', 'success']):
                    insights.append(f"Key output: {line.strip()}")

        insights.append(f"Execution completed in {execution_result.exec_time:.2f} seconds")

        return insights

    def _calculate_confidence(self, success: bool, execution_outputs: List[Dict], debug_attempts: int) -> float:
        """Calculate confidence score based on execution results"""
        if not success:
            return 0.0

        base_confidence = 0.9

        # Reduce confidence for debugging attempts
        debug_penalty = debug_attempts * 0.1

        # Reduce confidence for multiple iterations needed
        iteration_penalty = max(0, len(execution_outputs) - 1) * 0.05

        final_confidence = max(0.1, base_confidence - debug_penalty - iteration_penalty)
        return final_confidence

    def _generate_metrics(self, execution_outputs: List[Dict], debug_attempts: int, execution_time: float, task_description: str) -> Dict[str, Any]:
        """Generate comprehensive metrics"""
        return {
            'total_iterations': len(execution_outputs),
            'debug_attempts': debug_attempts,
            'execution_time': execution_time,
            'has_working_code': any(
                output.get('execution_result', {}).get('exc_type') is None
                for output in execution_outputs
            ),
            'task_complexity_score': min(1.0, len(task_description.split()) / 20),
            'code_generation_success': len(execution_outputs) > 0,
            'debugging_success_rate': (
                1.0 - (debug_attempts / max(1, self.max_debug_attempts))
                if debug_attempts <= self.max_debug_attempts else 0.0
            )
        }

    async def _fallback_llm_only_generation(self, task_description: str, config: Dict[str, Any], start_time: float) -> CodeExecutionResult:
        """Fallback method when code execution is not available"""
        try:
            # Generate code using LLM only
            code_blocks = await self._generate_initial_code(task_description, config)
            execution_time = time.time() - start_time

            # Simulate basic analysis without execution
            insights = [
                "Code generation completed (execution not available)",
                f"Generated {len(code_blocks)} code implementations",
                "Code execution would be needed for full validation"
            ]

            if code_blocks:
                insights.append(f"First code block has {len(code_blocks[0].split())} lines")

            return CodeExecutionResult(
                success=len(code_blocks) > 0,
                confidence=0.6 if code_blocks else 0.2,  # Lower confidence without execution
                code_blocks=code_blocks,
                execution_outputs=[],
                final_code=code_blocks[0] if code_blocks else None,
                execution_time=execution_time,
                iterations=1,
                debug_attempts=0,
                insights=insights,
                metrics={
                    'execution_time': execution_time,
                    'code_generation_success': len(code_blocks) > 0,
                    'execution_available': False,
                    'fallback_mode': True
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return CodeExecutionResult(
                success=False,
                confidence=0.0,
                code_blocks=[],
                execution_outputs=[],
                final_code=None,
                execution_time=execution_time,
                iterations=0,
                debug_attempts=0,
                insights=[f"Fallback code generation failed: {str(e)}"],
                metrics={'error': str(e), 'execution_time': execution_time, 'fallback_mode': True}
            )