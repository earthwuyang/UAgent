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

logger = logging.getLogger(__name__)

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
from .workspace_manager import WorkspaceManager
from ..utils.multi_modal_search import MultiModalSearchEngine


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
    """Executes generated code with iterative debugging and comprehensive setup"""

    def __init__(self, max_iterations: int = 3, max_debug_attempts: int = 2, timeout: int = 30):
        self.max_iterations = max_iterations
        self.max_debug_attempts = max_debug_attempts
        self.timeout = timeout
        self.workspace_manager = WorkspaceManager()
        self.search_engine = MultiModalSearchEngine()

    async def execute_computational_task(self, task_description: str, config: Dict[str, Any], node_id: str = None) -> CodeExecutionResult:
        """
        Execute a computational task with iterative code generation and debugging
        """
        start_time = time.time()

        existing_workspace = config.get("workspace_path") if isinstance(config, dict) else None

        if existing_workspace and Path(existing_workspace).exists():
            workspace_path = existing_workspace
            workspace = Path(existing_workspace)
            workspace_metadata = {
                "workspace_name": workspace.name,
                "workspace_path": str(workspace.resolve()),
            }
            for subdir in ["src", "scripts", "tests", "docs", "data", "output", "logs"]:
                (workspace / subdir).mkdir(exist_ok=True)
        else:
            # Create dedicated workspace for this task
            workspace_metadata = self.workspace_manager.create_workspace(
                task_name=task_description[:100],  # Limit length for filename
                task_type="computational"
            )
            workspace_path = workspace_metadata["workspace_path"]
            workspace = Path(workspace_path)

        logger.info(f"Created workspace for task: {workspace_path}")

        assessment = await self._assess_task_feasibility(task_description, config)
        supporting_materials = None
        knowledge_metrics: Dict[str, Any] = {
            "llm_confident": assessment.get("confident"),
            "confidence_notes": assessment.get("notes"),
        }

        if not assessment.get("confident") or assessment.get("recommend_web_research"):
            supporting_materials = await self._gather_supporting_materials(task_description, workspace)
            knowledge_metrics["supporting_materials"] = supporting_materials.get("sources", []) if supporting_materials else []
            knowledge_metrics["used_supporting_materials"] = True
        else:
            knowledge_metrics["used_supporting_materials"] = False

        if not INTERPRETER_AVAILABLE:
            # Fallback to LLM-only code generation without execution
            logger.warning("Interpreter not available, falling back to comprehensive LLM generation")
            result = await self._fallback_comprehensive_generation(task_description, config, start_time, node_id, workspace_path, supporting_materials)
            result.metrics.setdefault("knowledge_assessment", knowledge_metrics)
            return result

        # Create interpreter in the workspace
        interpreter = Interpreter(working_dir=workspace, timeout=self.timeout)

        try:
            # Generate initial code and parse comprehensive output
            llm_output = await self._generate_comprehensive_solution(task_description, config, node_id, supporting_materials)

            # Parse and save all generated files
            saved_files = self.workspace_manager.parse_and_save_comprehensive_output(workspace_path, llm_output)

            # Extract code blocks for execution
            code_blocks = self._extract_code_blocks(llm_output)

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
            metrics = self._generate_metrics(
                execution_outputs,
                debug_attempts,
                execution_time,
                task_description,
                workspace_path=str(workspace),
                saved_files=saved_files if isinstance(saved_files, dict) else {}
            )
            metrics["knowledge_assessment"] = knowledge_metrics

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

    async def _generate_initial_code(self, task_description: str, config: Dict[str, Any], node_id: str = None) -> List[str]:
        """Generate initial code implementations with comprehensive setup"""
        system_prompt = """You are an expert DevOps engineer and programmer. Generate a COMPLETE deployment solution that includes:

        1. ALL necessary code files (Python, Dockerfile, docker-compose.yml, etc.)
        2. DETAILED step-by-step reproduction instructions
        3. AUTOMATED scripts for building, testing, and deployment
        4. ERROR handling and logging for every step
        5. VALIDATION scripts to verify the deployment works

        CRITICAL: Always log every step, command, and action. Include verbose output for debugging.
        Generate production-ready, complete solutions, not just basic examples."""

        prompt = f"""Task: {task_description}
        Configuration: {config}

        Generate a COMPLETE solution including:

        ## 1. Code Implementation
        - All necessary source files
        - Complete Dockerfile and docker-compose.yml
        - Configuration files

        ## 2. Automated Scripts
        - build.sh: Build the application
        - run.sh: Run the application locally
        - test.sh: Test the deployment
        - deploy.sh: Deploy to production

        ## 3. Step-by-Step Instructions
        - Prerequisites and dependencies
        - Exact commands to reproduce
        - How to verify it's working
        - Troubleshooting common issues

        ## 4. Logging and Monitoring
        - Log every operation
        - Include health checks
        - Error reporting mechanisms

        Format each file clearly with filename headers and proper code blocks."""

        try:
            llm_response = await llm_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=3000,
                node_id=node_id,
                context={"task": "code_generation", "step": "initial"}
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

    async def _generate_comprehensive_solution(self, task_description: str, config: Dict[str, Any], node_id: str = None, supporting_materials: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive solution with all files and instructions"""
        supplemental = ""
        if supporting_materials and supporting_materials.get("summary"):
            supplemental = f"\n\n## Supporting References\n{supporting_materials['summary']}\n"

        llm_response = await llm_client.generate_response(
            prompt=f"""Task: {task_description}
Configuration: {config}
Supporting Context:{supplemental}

Generate a COMPLETE solution including:

## 1. Code Implementation
- All necessary source files
- Complete Dockerfile and docker-compose.yml
- Configuration files

## 2. Automated Scripts
- build.sh: Build the application
- run.sh: Run the application locally
- test.sh: Test the deployment
- deploy.sh: Deploy to production

## 3. Step-by-Step Instructions
- Prerequisites and dependencies
- Exact commands to reproduce
- How to verify it's working
- Troubleshooting common issues

## 4. Logging and Monitoring
- Log every operation
- Include health checks
- Error reporting mechanisms

Format each file clearly with filename headers and proper code blocks.""",
            system_prompt="""You are an expert DevOps engineer and programmer. Generate a COMPLETE deployment solution that includes:

1. ALL necessary code files (Python, Dockerfile, docker-compose.yml, etc.)
2. DETAILED step-by-step reproduction instructions
3. AUTOMATED scripts for building, testing, and deployment
4. ERROR handling and logging for every step
5. VALIDATION scripts to verify the deployment works

CRITICAL: Always log every step, command, and action. Include verbose output for debugging.
Generate production-ready, complete solutions, not just basic examples.""",
            temperature=0.3,
            max_tokens=4000,
            node_id=node_id,
            context={"task": "comprehensive_deployment", "step": "generation"}
        )

        if llm_response.get("success"):
            return llm_response.get("content", "")
        else:
            return f"# Failed to generate comprehensive solution\n# Error: {llm_response.get('error', 'Unknown error')}"

    async def _fallback_comprehensive_generation(self, task_description: str, config: Dict[str, Any], start_time: float, node_id: str = None, workspace_path: str = None, supporting_materials: Optional[Dict[str, Any]] = None) -> CodeExecutionResult:
        """Enhanced fallback method that generates comprehensive deployment setup"""
        try:
            # Generate comprehensive solution
            llm_output = await self._generate_comprehensive_solution(task_description, config, node_id, supporting_materials)

            # Parse and save all generated files
            if workspace_path:
                saved_files = self.workspace_manager.parse_and_save_comprehensive_output(workspace_path, llm_output)
            else:
                saved_files = {"code_files": [], "scripts": [], "docs": [], "configs": []}

            execution_time = time.time() - start_time

            # Enhanced insights based on what was generated
            insights = [
                "Comprehensive deployment solution generated",
                f"Generated {len(saved_files.get('code_files', []))} code files",
                f"Generated {len(saved_files.get('scripts', []))} automation scripts",
                f"Generated {len(saved_files.get('configs', []))} configuration files",
                f"Generated {len(saved_files.get('docs', []))} documentation files"
            ]

            if workspace_path:
                insights.append(f"All files saved to: {workspace_path}")
                insights.append("DEPLOYMENT.md created with step-by-step instructions")

            return CodeExecutionResult(
                success=len(saved_files.get('code_files', [])) > 0,
                confidence=0.8 if saved_files.get('scripts') else 0.6,
                code_blocks=self._extract_code_blocks(llm_output),
                execution_outputs=[],
                final_code=llm_output,
                execution_time=execution_time,
                iterations=1,
                debug_attempts=0,
                insights=insights,
                metrics={
                    'execution_time': execution_time,
                    'comprehensive_generation': True,
                    'files_generated': sum(len(files) for files in saved_files.values()),
                    'workspace_path': workspace_path,
                    'saved_files': saved_files
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
                insights=[f"Comprehensive generation failed: {str(e)}"],
                metrics={'error': str(e), 'execution_time': execution_time}
            )

    async def _assess_task_feasibility(self, task_description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are an experienced DevOps engineer.
Task: {task_description}
Configuration: {config}

Assess whether your current knowledge is sufficient to implement this task accurately without consulting external sources.
Respond in JSON with keys:
  confident: boolean
  confidence_reason: short string
  recommend_web_research: boolean (true if additional references or updated docs are needed)
"""
        try:
            response = await llm_client.generate_response(
                prompt=prompt,
                system_prompt="You provide concise JSON assessments.",
                temperature=0.0,
                max_tokens=200,
            )
            if response.get("success"):
                import json
                import re
                content = response.get("content", "")
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    data = json.loads(json_match.group())
                    return {
                        "confident": bool(data.get("confident", False)),
                        "notes": data.get("confidence_reason"),
                        "recommend_web_research": bool(data.get("recommend_web_research", False)),
                    }
        except Exception as exc:
            logger.debug(f"LLM knowledge assessment failed: {exc}")
        return {"confident": False, "notes": "Assessment unavailable", "recommend_web_research": True}

    async def _gather_supporting_materials(self, task_description: str, workspace: Path) -> Optional[Dict[str, Any]]:
        try:
            search_results = await self.search_engine.unified_search(
                query=f"{task_description} Docker deployment",
                search_types=["web", "code"],
                limit=5,
            )
            sources = []
            summary_lines = []
            for category, items in search_results.items():
                if not isinstance(items, list):
                    continue
                for item in items[:3]:
                    title = item.get("title") or item.get("headline")
                    url = item.get("url") or item.get("link")
                    snippet = item.get("snippet") or item.get("summary")
                    if title and url:
                        sources.append({"title": title, "url": url, "snippet": snippet})
                        summary_lines.append(f"- [{title}]({url}) — {snippet}")

            if sources:
                docs_dir = workspace / "docs"
                docs_dir.mkdir(exist_ok=True)
                support_file = docs_dir / "SUPPORTING_REFERENCES.md"
                support_file.write_text(
                    "# Supporting References\n\n" + "\n".join(summary_lines), encoding="utf-8"
                )
                return {"sources": sources, "summary": "\n".join(summary_lines)}
        except Exception as exc:
            logger.debug(f"Failed gathering supporting materials: {exc}")
        return None

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

    def _generate_metrics(
        self,
        execution_outputs: List[Dict],
        debug_attempts: int,
        execution_time: float,
        task_description: str,
        workspace_path: Optional[str] = None,
        saved_files: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
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
            ),
            'workspace_path': workspace_path,
            'generated_files': saved_files,
        }

    async def _fallback_llm_only_generation(self, task_description: str, config: Dict[str, Any], start_time: float, node_id: str = None) -> CodeExecutionResult:
        """Fallback method when code execution is not available"""
        try:
            # Generate code using LLM only
            code_blocks = await self._generate_initial_code(task_description, config, node_id)
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
