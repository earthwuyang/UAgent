"""
Code Research Engine - Repository analysis and understanding

Adapted for OpenHands integration with RepoMaster capabilities.
"""

import logging
import uuid
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

# OpenHands imports
from openhands.llm.llm import LLM
from openhands.runtime.runtime import Runtime
from openhands.events.stream import EventStream
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation
from ...orchestrator.event_bus import EventBus, get_event_bus
from ...uagent_research.models.events import StepEvent, CompleteEvent, ErrorEvent

logger = logging.getLogger(__name__)


class CodeResearchEngine:
    """
    Code research engine for repository analysis.

    Provides:
    - Repository structure analysis
    - Code comprehension
    - Architecture understanding
    - Dependency mapping
    - Function/class search
    """

    def __init__(self, llm: LLM, config: Optional[Dict[str, Any]] = None, event_bus: Optional[EventBus] = None):
        """
        Initialize code research engine.

        Args:
            llm: OpenHands LLM instance
            config: Optional configuration
        """
        self.llm = llm
        self.event_bus = event_bus or get_event_bus()
        self.config = config or {}

        self.max_repo_size_gb = self.config.get('max_repo_size_gb', 10)
        self.analysis_depth = self.config.get('analysis_depth', 'quick')  # quick or deep

        logger.info(f"CodeResearchEngine initialized: depth={self.analysis_depth}")

    async def analyze_repository(
        self,
        query: str,
        workspace: str,
        runtime: Runtime,
        event_stream: Optional[EventStream] = None,
        branch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze code repository to answer query.

        Args:
            query: Question about the codebase
            workspace: Path to repository root
            runtime: OpenHands runtime for command execution
            event_stream: Optional event stream for progress updates

        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing repository: {workspace}")
        logger.info(f"Query: {query}")

        workspace_path = Path(workspace)

        if not workspace_path.exists():
            raise ValueError(f"Workspace does not exist: {workspace}")

        bid = branch_id or f"code_research_{uuid.uuid4().hex[:8]}"

        try:
            await self.event_bus.publish(
                StepEvent(
                    branch_id=bid,
                    action="Analyzing repository structure",
                    reasoning="Inspecting files and directories",
                )
            )
            structure = await self._analyze_structure(workspace_path, runtime)

            await self.event_bus.publish(
                StepEvent(
                    branch_id=bid,
                    action="Finding relevant files",
                    reasoning="Searching for files matching the query",
                )
            )
            relevant_files = await self._find_relevant_files(
                query, structure, workspace_path, runtime
            )

            await self.event_bus.publish(
                StepEvent(
                    branch_id=bid,
                    action="Analyzing code",
                    reasoning="Reviewing candidate files for insights",
                )
            )
            code_analysis = await self._analyze_code(
                query, relevant_files, workspace_path, runtime
            )

            await self.event_bus.publish(
                StepEvent(
                    branch_id=bid,
                    action="Synthesizing answer",
                    reasoning="Generating comprehensive answer from analysis",
                )
            )
            answer = await self._synthesize_answer(
                query, structure, relevant_files, code_analysis
            )

            await self.event_bus.publish(
                CompleteEvent(
                    branch_id=bid,
                    summary=answer[:200],
                    artifacts=[],
                )
            )

            return {
                "query": query,
                "workspace": workspace,
                "structure": structure,
                "relevant_files": relevant_files,
                "code_analysis": code_analysis,
                "answer": answer,
            }
        except Exception as e:
            await self.event_bus.publish(
                ErrorEvent(
                    branch_id=bid,
                    message=str(e),
                )
            )
            raise

    async def _analyze_structure(
        self,
        workspace_path: Path,
        runtime: Runtime
    ) -> Dict[str, Any]:
        """Analyze repository structure"""

        logger.info("Analyzing repository structure...")

        # Get directory tree
        action = CmdRunAction(
            command=f"find {workspace_path} -type f -name '*.py' | head -100",
            thought="Finding Python files in repository"
        )
        observation = await runtime.run_action(action)

        python_files = []
        if isinstance(observation, CmdOutputObservation) and observation.exit_code == 0:
            python_files = observation.content.strip().split('\n')
            python_files = [f for f in python_files if f]

        # Get directory structure
        action = CmdRunAction(
            command=f"find {workspace_path} -maxdepth 3 -type d",
            thought="Finding directories"
        )
        observation = await runtime.run_action(action)

        directories = []
        if isinstance(observation, CmdOutputObservation) and observation.exit_code == 0:
            directories = observation.content.strip().split('\n')
            directories = [d for d in directories if d]

        return {
            'python_files': python_files[:50],  # Limit for LLM
            'directories': directories[:30],
            'file_count': len(python_files),
            'directory_count': len(directories),
        }

    async def _find_relevant_files(
        self,
        query: str,
        structure: Dict[str, Any],
        workspace_path: Path,
        runtime: Runtime
    ) -> List[Dict[str, Any]]:
        """Find files relevant to query using grep and LLM"""

        logger.info("Finding relevant files...")

        # Extract keywords from query
        prompt = f"""Extract 3-5 key technical terms or concepts from this question that would appear in code:
Question: {query}

Respond with ONLY a JSON array of strings:
["keyword1", "keyword2", "keyword3"]
"""

        response = await self.llm.completion(
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            keywords = json.loads(json_match.group())
        else:
            keywords = json.loads(content)

        # Search for keywords in code
        relevant_files = []

        for keyword in keywords[:3]:  # Limit searches
            action = CmdRunAction(
                command=f"grep -r -l '{keyword}' {workspace_path} --include='*.py' 2>/dev/null | head -10",
                thought=f"Searching for '{keyword}' in Python files"
            )
            observation = await runtime.run_action(action)

            if isinstance(observation, CmdOutputObservation) and observation.exit_code == 0:
                files = observation.content.strip().split('\n')
                for file_path in files:
                    if file_path and file_path not in [f['path'] for f in relevant_files]:
                        relevant_files.append({
                            'path': file_path,
                            'keyword': keyword,
                            'relevance_score': 1.0,  # Could be refined
                        })

        return relevant_files[:10]  # Limit for analysis

    async def _analyze_code(
        self,
        query: str,
        relevant_files: List[Dict[str, Any]],
        workspace_path: Path,
        runtime: Runtime
    ) -> Dict[str, Any]:
        """Analyze code in relevant files"""

        logger.info(f"Analyzing {len(relevant_files)} relevant files...")

        file_analyses = []

        for file_info in relevant_files[:5]:  # Limit to avoid context overflow
            file_path = file_info['path']

            # Read file content
            action = CmdRunAction(
                command=f"cat {file_path}",
                thought=f"Reading {file_path}"
            )
            observation = await runtime.run_action(action)

            if isinstance(observation, CmdOutputObservation) and observation.exit_code == 0:
                content = observation.content

                # Analyze with LLM
                prompt = f"""Analyze this code file to help answer the question:
Question: {query}

File: {file_path}
```python
{content[:2000]}  # Truncate for LLM context
```

Provide:
1. Summary of what this file does
2. Key functions/classes relevant to the question
3. How this file relates to the question

Respond with ONLY valid JSON:
{{
    "summary": "brief summary",
    "key_components": ["component1", "component2"],
    "relevance": "how this answers the question"
}}
"""

                response = await self.llm.completion(
                    messages=[{"role": "user", "content": prompt}]
                )

                analysis_content = response.choices[0].message.content
                json_match = re.search(r'\{.*\}', analysis_content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    analysis = json.loads(analysis_content)

                file_analyses.append({
                    'file': file_path,
                    'analysis': analysis,
                })

        return {
            'analyzed_files': file_analyses,
            'total_analyzed': len(file_analyses),
        }

    async def _synthesize_answer(
        self,
        query: str,
        structure: Dict[str, Any],
        relevant_files: List[Dict[str, Any]],
        code_analysis: Dict[str, Any]
    ) -> str:
        """Synthesize final answer from analysis"""

        logger.info("Synthesizing answer...")

        context = {
            'query': query,
            'structure_summary': {
                'file_count': structure['file_count'],
                'directory_count': structure['directory_count'],
            },
            'relevant_files': [f['path'] for f in relevant_files],
            'code_analysis': code_analysis['analyzed_files'],
        }

        prompt = f"""Based on this code repository analysis, answer the question:

Question: {query}

Analysis:
{json.dumps(context, indent=2)}

Provide a comprehensive answer that:
1. Directly answers the question
2. References specific files and code
3. Explains the architecture/implementation
4. Provides examples if relevant

Write your answer in clear, technical prose.
"""

        response = await self.llm.completion(
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        return answer
