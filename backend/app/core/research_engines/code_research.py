"""Code Research Engine - Repository analysis and code understanding"""

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from ..llm_client import LLMClient
from ..websocket_manager import progress_tracker, websocket_manager
from ...integrations.repomaster_bridge import RepoMasterBridge, RepoMasterBridgeResult


@dataclass
class CodeRepository:
    """Code repository information"""
    name: str
    url: str
    description: str
    language: str
    stars: int
    forks: int
    last_updated: str
    topics: List[str]
    license: Optional[str] = None


@dataclass
class CodePattern:
    """Code pattern or best practice"""
    name: str
    description: str
    code_example: str
    language: str
    category: str  # 'design_pattern', 'best_practice', 'optimization', 'security'
    file_path: str
    repository: str


@dataclass
class CodeAnalysis:
    """Analysis of code repository or pattern"""
    repository: str
    summary: str
    architecture_overview: str
    key_features: List[str]
    technologies_used: List[str]
    patterns_identified: List[CodePattern]
    quality_metrics: Dict[str, Any]
    recommendations: List[str]


@dataclass
class CodeResearchResult:
    """Comprehensive code research result"""
    query: str
    repositories: List[CodeRepository]
    analysis: List[CodeAnalysis]
    best_practices: List[CodePattern]
    integration_guide: str
    recommendations: List[str]
    confidence_score: float


class GitHubSearchEngine:
    """GitHub repository search and analysis"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self._default_quality_scores = {
            "maintainability": 0.5,
            "test_coverage": 0.5,
            "documentation": 0.5,
        }

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def search_repositories(self, query: str, language: Optional[str] = None, limit: int = 10) -> List[CodeRepository]:
        """Search GitHub repositories"""
        self.logger.info(f"Searching GitHub repositories for: {query}")

        # In production, integrate with GitHub API
        # For now, simulate realistic repository results

        repositories = []
        for i in range(min(limit, 5)):  # Simulate up to 5 repositories
            repo = CodeRepository(
                name=f"{query.replace(' ', '-').lower()}-{i+1}",
                url=f"https://github.com/example/{query.replace(' ', '-').lower()}-{i+1}",
                description=f"A comprehensive implementation of {query} with modern best practices",
                language=language or "Python",
                stars=1000 - (i * 100),
                forks=200 - (i * 20),
                last_updated="2024-01-15",
                topics=[query.lower(), "open-source", "library"],
                license="MIT"
            )
            repositories.append(repo)

        return repositories

    async def analyze_repository(self, repository: CodeRepository) -> CodeAnalysis:
        """Analyze a specific repository"""
        self.logger.info(f"Analyzing repository: {repository.name}")

        # In production, this would:
        # 1. Clone or fetch repository content
        # 2. Analyze file structure and dependencies
        # 3. Extract code patterns and architecture
        # 4. Run static analysis tools
        # 5. Generate comprehensive analysis

        analysis_prompt = f"""
        Analyze the repository "{repository.name}" with the following details:
        - Description: {repository.description}
        - Language: {repository.language}
        - Stars: {repository.stars}
        - Topics: {', '.join(repository.topics)}

        Provide analysis in JSON format with:
        1. "summary": Brief overview of the repository's purpose and functionality
        2. "architecture_overview": Description of the codebase architecture
        3. "key_features": List of 3-5 key features or capabilities
        4. "technologies_used": List of technologies, frameworks, and libraries used
        5. "quality_metrics": Object with quality indicators (maintainability, test_coverage, documentation)
        6. "recommendations": List of 3-5 recommendations for usage or improvement

        Respond with valid JSON only.
        """

        try:
            response = await self.llm_client.generate(analysis_prompt)

            import json
            try:
                analysis_data = json.loads(response)
            except json.JSONDecodeError:
                # Fallback structured analysis - return fallback CodeAnalysis object directly
                return self._create_fallback_analysis(repository)

            # Create CodePattern objects for identified patterns
            patterns = await self._extract_code_patterns(repository, analysis_data)

            return CodeAnalysis(
                repository=repository.name,
                summary=analysis_data.get("summary", f"Analysis of {repository.name}"),
                architecture_overview=analysis_data.get("architecture_overview", "Standard modular architecture"),
                key_features=analysis_data.get("key_features", ["Core functionality", "API interface", "Documentation"]),
                technologies_used=analysis_data.get("technologies_used", [repository.language]),
                patterns_identified=patterns,
                quality_metrics=analysis_data.get("quality_metrics", {"maintainability": 0.8, "test_coverage": 0.7, "documentation": 0.9}),
                recommendations=analysis_data.get("recommendations", ["Review implementation", "Consider integration", "Test thoroughly"])
            )

        except Exception as e:
            self.logger.error(f"Error analyzing repository {repository.name}: {e}")
            return self._create_fallback_analysis(repository)

    async def _extract_code_patterns(self, repository: CodeRepository, analysis_data: Dict[str, Any]) -> List[CodePattern]:
        """Extract code patterns from repository analysis"""
        # In production, this would analyze actual code files
        patterns = []

        # Simulate pattern extraction based on repository characteristics
        if "api" in repository.description.lower() or "fastapi" in repository.topics:
            patterns.append(CodePattern(
                name="REST API Pattern",
                description="RESTful API implementation with proper routing and error handling",
                code_example=f"# Example from {repository.name}\n@app.get('/api/v1/items')\nasync def get_items():\n    return items",
                language=repository.language,
                category="design_pattern",
                file_path="app/routes/api.py",
                repository=repository.name
            ))

        if "database" in repository.description.lower():
            patterns.append(CodePattern(
                name="Repository Pattern",
                description="Data access layer abstraction for database operations",
                code_example=f"# Repository pattern from {repository.name}\nclass ItemRepository:\n    async def get_all(self):\n        return await db.fetch_all()",
                language=repository.language,
                category="design_pattern",
                file_path="app/repositories/item_repository.py",
                repository=repository.name
            ))

        return patterns

    def _create_fallback_analysis(self, repository: CodeRepository) -> CodeAnalysis:
        """Create fallback analysis when LLM analysis fails"""
        return CodeAnalysis(
            repository=repository.name,
            summary=f"Repository {repository.name} implements {repository.description}",
            architecture_overview="Standard modular architecture with separation of concerns",
            key_features=["Core functionality", "Well-documented API", "Test coverage"],
            technologies_used=[repository.language],
            patterns_identified=[],
            quality_metrics={"maintainability": 0.7, "test_coverage": 0.6, "documentation": 0.8},
            recommendations=["Review code structure", "Consider integration testing", "Evaluate performance"]
        )


class CodePatternEngine:
    """Engine for identifying and extracting code patterns"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)

    async def identify_patterns(self, code_content: str, language: str) -> List[CodePattern]:
        """Identify patterns in code content"""
        self.logger.info(f"Identifying patterns in {language} code")

        pattern_prompt = f"""
        Analyze the following {language} code and identify design patterns, best practices, and notable implementations:

        ```{language}
        {code_content[:2000]}  # Limit code content for analysis
        ```

        Identify patterns and provide response in JSON format with array of objects containing:
        1. "name": Pattern name
        2. "description": Brief description of the pattern
        3. "category": One of [design_pattern, best_practice, optimization, security]
        4. "code_example": Relevant code snippet demonstrating the pattern
        5. "benefits": Benefits of using this pattern

        Respond with valid JSON only.
        """

        try:
            response = await self.llm_client.generate(pattern_prompt)

            import json
            patterns_data = json.loads(response)

            patterns = []
            for pattern_data in patterns_data:
                pattern = CodePattern(
                    name=pattern_data.get("name", "Unknown Pattern"),
                    description=pattern_data.get("description", "Pattern description"),
                    code_example=pattern_data.get("code_example", "# Code example"),
                    language=language,
                    category=pattern_data.get("category", "best_practice"),
                    file_path="analyzed_code.py",
                    repository="code_analysis"
                )
                patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error identifying patterns: {e}")
            return []

    async def extract_best_practices(self, repositories: List[CodeRepository]) -> List[CodePattern]:
        """Extract best practices from multiple repositories"""
        self.logger.info(f"Extracting best practices from {len(repositories)} repositories")

        all_patterns = []
        for repo in repositories:
            # In production, analyze actual repository code
            # For now, simulate pattern extraction based on repository metadata
            if "fastapi" in repo.description.lower():
                pattern = CodePattern(
                    name="Async Route Handlers",
                    description="Use async/await for non-blocking request handling",
                    code_example="@app.get('/items')\nasync def get_items():\n    return await fetch_items()",
                    language=repo.language,
                    category="best_practice",
                    file_path="app/routes.py",
                    repository=repo.name
                )
                all_patterns.append(pattern)

            if "security" in repo.topics:
                pattern = CodePattern(
                    name="Input Validation",
                    description="Validate all user inputs to prevent security vulnerabilities",
                    code_example="from pydantic import BaseModel\n\nclass UserInput(BaseModel):\n    email: str\n    age: int",
                    language=repo.language,
                    category="security",
                    file_path="app/models.py",
                    repository=repo.name
                )
                all_patterns.append(pattern)

        return all_patterns


class CodeResearchEngine:
    """Code Research Engine for repository analysis and code understanding"""

    def __init__(self, llm_client: LLMClient, openhands_client: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize code research engine

        Args:
            llm_client: LLM client for analysis
            config: Configuration options
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.openhands_client = openhands_client

        # Initialize sub-engines
        self.github_engine = GitHubSearchEngine(llm_client)
        self.pattern_engine = CodePatternEngine(llm_client)
        self._session_phase_nodes: Dict[str, Dict[str, str]] = {}
        self._session_root_nodes: Dict[str, str] = {}
        self._repomaster_bridge: Optional[RepoMasterBridge] = None
        self._use_repo_master_default = bool(self.config.get("use_repo_master", True))
        self._default_quality_scores = {
            "maintainability": 0.5,
            "test_coverage": 0.5,
            "documentation": 0.5,
        }

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_root_node_id(self, session_id: Optional[str]) -> str:
        if not session_id:
            return "code_research-root"
        if session_id not in self._session_root_nodes:
            self._session_root_nodes[session_id] = f"{session_id}-code_research-root"
        return self._session_root_nodes[session_id]

    async def _log_progress(
        self,
        session_id: Optional[str],
        phase: str,
        progress: float,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_phase: Optional[str] = None,
        node_type: str = "step"
    ) -> str:
        if not session_id:
            return ""

        metadata = dict(metadata or {})
        node_id = metadata.get("node_id")

        phase_nodes = self._session_phase_nodes.setdefault(session_id, {})
        if node_id is None:
            node_id = phase_nodes.get(phase) or f"{session_id}-code-{phase}-{uuid.uuid4().hex[:6]}"
        parent_id = metadata.get("parent_id")
        if not parent_id:
            if parent_phase and parent_phase in phase_nodes:
                parent_id = phase_nodes[parent_phase]
            else:
                parent_id = self._get_root_node_id(session_id)
        metadata["parent_id"] = parent_id
        phase_nodes[phase] = node_id

        metadata["node_id"] = node_id
        metadata.setdefault("node_type", node_type)
        metadata.setdefault("title", message)
        metadata.setdefault("phase", phase)

        try:
            await progress_tracker.log_research_progress(
                session_id=session_id or "unknown",
                engine="code_research",
                phase=phase,
                progress=progress,
                message=message,
                metadata=metadata
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            self.logger.debug(
                "Failed to log code research progress for %s (phase=%s): %s",
                session_id,
                phase,
                exc
            )

        return node_id

    def _get_repomaster_bridge(self) -> RepoMasterBridge:
        if self._repomaster_bridge is None:
            self._repomaster_bridge = RepoMasterBridge()
        return self._repomaster_bridge

    def _should_use_repomaster(self, workflow_plan: Optional[Dict[str, Any]]) -> bool:
        if workflow_plan and "use_repo_master" in workflow_plan:
            return bool(workflow_plan["use_repo_master"])
        return self._use_repo_master_default

    async def _handle_repomaster_progress(self, session_id: str, event: Dict[str, Any]) -> None:
        if not session_id:
            return

        event_type = event.get("event")
        metadata = {k: v for k, v in event.items() if k not in {"event", "session_id"}}

        if event_type == "task_started":
            await self._log_progress(
                session_id,
                phase="repomaster_initializing",
                progress=5.0,
                message="RepoMaster initializing repository workflow",
                metadata=metadata,
            )
        elif event_type == "repo_search_results":
            repositories = metadata.get("repositories", [])
            repo_count = len(repositories)
            node_id = await self._log_progress(
                session_id,
                phase="discovering_repositories",
                progress=35.0,
                message=f"RepoMaster identified {repo_count} candidate repositories",
                metadata={"repositories": repositories},
                parent_phase="repomaster_initializing",
            )
            await self._log_repomaster_repositories(session_id, repositories, node_id)
        elif event_type == "repository_task_started":
            await self._log_progress(
                session_id,
                phase="repomaster_execution",
                progress=65.0,
                message="RepoMaster executing selected repository",
                metadata=metadata,
                parent_phase="discovering_repositories",
            )
        elif event_type == "repository_task_completed":
            await self._log_progress(
                session_id,
                phase="repomaster_execution_complete",
                progress=90.0,
                message="RepoMaster repository execution completed",
                metadata=metadata,
                parent_phase="repomaster_execution",
            )
        elif event_type == "final_answer_extracted":
            await self._log_progress(
                session_id,
                phase="synthesizing_guidance",
                progress=96.0,
                message="RepoMaster synthesizing final findings",
                metadata=metadata,
            )
        elif event_type == "task_completed":
            await self._log_progress(
                session_id,
                phase="repomaster_completed",
                progress=100.0,
                message="RepoMaster workflow completed",
                metadata=metadata,
            )
        elif event_type == "repo_execution_step":
            phase_nodes = self._session_phase_nodes.get(session_id, {})
            parent_id = metadata.get("parent_id")
            if not parent_id:
                parent_id = phase_nodes.get("repomaster_execution")
            code_blocks = metadata.get("details", {}).get("code_blocks", [])
            code_preview = code_blocks[0]["code"].strip() if code_blocks else "Repository execution step"
            node_id = metadata.get("node_id") or f"{session_id}-code-repomaster_exec_step-{uuid.uuid4().hex[:6]}"
            exec_details = metadata.get("details", {}) or {}
            exec_metadata = {
                "title": code_preview.split("\n", 1)[0][:120],
                "code_blocks": code_blocks,
                "exit_code": exec_details.get("exit_code"),
                "output": exec_details.get("output"),
                "parent_id": parent_id,
                "node_id": node_id,
                "node_type": "result",
                "phase": "repomaster_execution_step",
                "details": exec_details,
            }
            await self._log_progress(
                session_id,
                phase=f"repomaster_execution_step_{uuid.uuid4().hex[:4]}",
                progress=78.0,
                message=exec_metadata["title"],
                metadata=exec_metadata,
                parent_phase="repomaster_execution",
                node_type="result",
            )
        elif event_type in {"repo_tool_call", "repo_tool_result"}:
            phase_nodes = self._session_phase_nodes.get(session_id, {})
            parent_id = phase_nodes.get("repomaster_execution") or self._get_root_node_id(session_id)
            tool_name = metadata.get("tool") or "tool"
            query = metadata.get("query") or ""
            result_preview = metadata.get("result")
            if isinstance(result_preview, (list, dict)):
                try:
                    result_preview = json.dumps(result_preview)[:200]
                except Exception:
                    result_preview = str(result_preview)[:200]
            elif result_preview:
                result_preview = str(result_preview)[:200]
            message = f"{tool_name} | {query[:80]}" if query else tool_name
            node_id = f"{session_id}-code-repomaster_{tool_name}_{uuid.uuid4().hex[:6]}"
            await self._log_progress(
                session_id,
                phase=f"repomaster_{tool_name}_{event_type}",
                progress=50.0 if event_type == "repo_tool_call" else 60.0,
                message=message,
                metadata={
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "node_type": "step" if event_type == "repo_tool_call" else "result",
                    "phase": f"repomaster_{tool_name}",
                    "tool": tool_name,
                    "query": query,
                    "result": metadata.get("result"),
                },
                parent_phase="repomaster_execution",
                node_type="step" if event_type == "repo_tool_call" else "result",
            )

    async def _log_repomaster_repositories(
        self,
        session_id: str,
        repositories: List[Dict[str, Any]],
        parent_node_id: Optional[str],
    ) -> None:
        if not session_id or not repositories:
            return

        for idx, repo in enumerate(repositories[:10]):
            name = repo.get("repo_name") or repo.get("name") or f"repository_{idx + 1}"
            description = repo.get("repo_description") or repo.get("description") or "Candidate repository"
            url = repo.get("repo_url") or repo.get("url")
            metadata = {
                "title": name,
                "description": description,
                "url": url,
                "rank": idx + 1,
                "parent_id": parent_node_id,
                "node_type": "result",
            }
            await self._log_progress(
                session_id,
                phase=f"repomaster_repo_{idx + 1}",
                progress=40.0,
                message=name,
                metadata=metadata,
                parent_phase="discovering_repositories",
                node_type="result",
            )

    async def _handle_repomaster_llm(self, session_id: str, payload: Dict[str, Any]) -> None:
        if not session_id:
            return

        message = payload.get("message") or {}
        content = message.get("content")
        if not content:
            return

        role = message.get("role", "assistant")
        timestamp = datetime.utcnow().isoformat()
        event_type = "llm_prompt_complete" if role == "assistant" else "llm_prompt_start"
        event_payload: Dict[str, Any] = {
            "type": event_type,
            "session_id": session_id,
            "timestamp": timestamp,
            "engine": payload.get("llm_config", {}).get("model", "repomaster"),
        }
        if event_type == "llm_prompt_complete":
            event_payload["response"] = content
        else:
            event_payload["prompt"] = content

        websocket_manager.store_llm_event(session_id, event_payload)

        connections = websocket_manager.llm_stream_connections.get(session_id)
        if connections:
            await websocket_manager._broadcast_to_connections(connections, event_payload)

    def _extract_repositories_from_events(self, events: List[Dict[str, Any]]) -> List[CodeRepository]:
        repo_map: Dict[str, CodeRepository] = {}
        for event in events:
            if event.get("event") != "repo_search_results":
                continue
            for repo_data in event.get("repositories", []) or []:
                name = repo_data.get("repo_name") or repo_data.get("name")
                if not name:
                    continue
                url = repo_data.get("repo_url") or repo_data.get("url", "")
                description = repo_data.get("repo_description") or repo_data.get("description", "Candidate repository")
                language = (
                    repo_data.get("language")
                    or repo_data.get("repo_language")
                    or repo_data.get("primary_language")
                    or "Unknown"
                )
                try:
                    stars = int(repo_data.get("stars") or repo_data.get("stargazers_count") or 0)
                except (TypeError, ValueError):
                    stars = 0
                try:
                    forks = int(repo_data.get("forks") or repo_data.get("forks_count") or 0)
                except (TypeError, ValueError):
                    forks = 0
                topics_raw = repo_data.get("topics") or []
                if isinstance(topics_raw, str):
                    topics = [topic.strip() for topic in topics_raw.split(",") if topic.strip()]
                elif isinstance(topics_raw, list):
                    topics = [str(topic) for topic in topics_raw]
                else:
                    topics = []
                last_updated = repo_data.get("updated_at") or repo_data.get("last_updated") or ""
                repo_map[name] = CodeRepository(
                    name=name,
                    url=url,
                    description=description,
                    language=language,
                    stars=stars,
                    forks=forks,
                    last_updated=last_updated or datetime.utcnow().strftime("%Y-%m-%d"),
                    topics=topics,
                    license=repo_data.get("license"),
                )

        return list(repo_map.values())

    def _convert_repomaster_result(
        self,
        query: str,
        bridge_result: RepoMasterBridgeResult,
    ) -> CodeResearchResult:
        repositories = self._extract_repositories_from_events(
            bridge_result.metadata.get("progress_events", [])
        )
        confidence = 0.5 + min(0.1 * len(repositories), 0.3)

        code_result = CodeResearchResult(
            query=query,
            repositories=repositories,
            analysis=[],
            best_practices=[],
            integration_guide="",
            recommendations=[],
            confidence_score=round(confidence, 2),
        )
        setattr(code_result, "report_markdown", bridge_result.report_markdown or bridge_result.final_answer)
        setattr(code_result, "repomaster_metadata", bridge_result.metadata)
        return code_result

    async def _execute_repomaster(
        self,
        query: str,
        session_id: str,
        workflow_plan: Optional[Dict[str, Any]],
    ) -> RepoMasterBridgeResult:
        bridge = self._get_repomaster_bridge()
        workspace_root = Path(
            os.getenv(
                "UAGENT_WORKSPACE_DIR",
                os.getenv("WORKSPACE_DIR", "/tmp/uagent_workspaces"),
            )
        )
        workspace = workspace_root / session_id / "repomaster"

        repository_hint = None
        input_data = None
        if workflow_plan:
            repository_hint = (
                workflow_plan.get("repository_url")
                or workflow_plan.get("repository")
                or workflow_plan.get("repo_url")
            )
            input_data = workflow_plan.get("input_data")

        async def progress_handler(event: Dict[str, Any]) -> None:
            await self._handle_repomaster_progress(session_id, event)

        async def llm_handler(message: Dict[str, Any]) -> None:
            await self._handle_repomaster_llm(session_id, message)

        await self._log_progress(
            session_id,
            phase="repomaster_invocation",
            progress=20.0,
            message="Invoking RepoMaster repository pipeline",
            metadata={"title": "RepoMaster Invocation"},
            parent_phase="initializing",
        )

        bridge_result = await bridge.run_task(
            session_id=session_id,
            query=query,
            workspace=workspace,
            repository_hint=repository_hint,
            input_data=input_data,
            progress_handler=progress_handler,
            llm_handler=llm_handler,
        )

        await self._log_progress(
            session_id,
            phase="repomaster_finalizing",
            progress=98.0,
            message="RepoMaster execution finalized",
            metadata={"content_preview": bridge_result.final_answer[:200]},
            parent_phase="repomaster_invocation",
        )

        await self._log_progress(
            session_id,
            phase="repomaster_summary_ready",
            progress=100.0,
            message="RepoMaster results ready",
            metadata={"title": "RepoMaster run completed"},
            parent_phase="repomaster_finalizing",
        )

        return bridge_result

    async def research_code(
        self,
        query: str,
        language: Optional[str] = None,
        include_analysis: bool = True,
        session_id: Optional[str] = None,
        workflow_plan: Optional[Dict[str, Any]] = None,
    ) -> CodeResearchResult:
        """Conduct comprehensive code research

        Args:
            query: Code research query (e.g., "FastAPI REST API", "machine learning pipelines")
            language: Programming language filter
            include_analysis: Whether to include detailed code analysis
            session_id: Optional session identifier for progress streaming

        Returns:
            Comprehensive code research result
        """
        self.logger.info(f"Starting code research for: {query}")

        root_id = self._get_root_node_id(session_id)

        await self._log_progress(
            session_id,
            phase="initializing",
            progress=5.0,
            message="Preparing repository discovery pipeline",
            metadata={"query": query, "language": language, "parent_id": root_id}
        )

        use_repomaster = bool(session_id and self._should_use_repomaster(workflow_plan))

        try:
            if use_repomaster:
                bridge_result = await self._execute_repomaster(query, session_id, workflow_plan)
                return self._convert_repomaster_result(query, bridge_result)

            # Search for relevant repositories (fallback path)
            repositories = await self.github_engine.search_repositories(query, language, limit=10)

            repo_phase_node = await self._log_progress(
                session_id,
                phase="discovering_repositories",
                progress=30.0,
                message=f"Found {len(repositories)} candidate repositories",
                metadata={"repositories": [repo.name for repo in repositories[:5]]},
                parent_phase="initializing"
            )

            # Analyze repositories if requested
            analyses = []
            if include_analysis and repositories:
                # Analyze top repositories concurrently
                analysis_tasks = [
                    self.github_engine.analyze_repository(repo)
                    for repo in repositories[:5]  # Limit to top 5 for detailed analysis
                ]
                analyses = await asyncio.gather(*analysis_tasks)

                await self._log_progress(
                    session_id,
                    phase="analyzing_repositories",
                    progress=55.0,
                    message="Completed deep analysis on top repositories",
                    metadata={"analyzed": [analysis.repository for analysis in analyses]},
                    parent_phase="discovering_repositories"
                )

            # Extract best practices and patterns
            best_practices = await self.pattern_engine.extract_best_practices(repositories)

            await self._log_progress(
                session_id,
                phase="extracting_patterns",
                progress=70.0,
                message=f"Identified {len(best_practices)} reusable patterns",
                metadata={"pattern_count": len(best_practices)},
                parent_phase="analyzing_repositories"
            )

            if session_id and repositories:
                for idx, repo in enumerate(repositories[:5]):
                    await self._log_progress(
                        session_id,
                        phase=f"repository_{idx + 1}",
                        progress=45.0,
                        message=f"Repository: {repo.name}",
                        metadata={
                            "title": repo.name,
                            "description": repo.description,
                            "url": repo.url,
                            "language": repo.language,
                            "stars": repo.stars,
                            "parent_id": repo_phase_node,
                        },
                        parent_phase="discovering_repositories",
                        node_type="result"
                    )

            # Generate integration guide
            integration_guide = await self._generate_integration_guide(query, repositories, analyses)

            await self._log_progress(
                session_id,
                phase="synthesizing_guidance",
                progress=82.0,
                message="Prepared integration guidance and recommendations",
                metadata={"guide_length": len(integration_guide)}
            )

            # Optionally run a CodeAct demo that builds a tiny example based on the integration guide
            if self.config.get("use_codeact_in_code_engine", True) and self.openhands_client and session_id:
                try:
                    session_cfg = await self.openhands_client.ensure_session(
                        research_type="code_research",
                        session_id=f"openhands_code_{session_id}",
                        config=self.config.get("default_openhands_resources", {}),
                    )
                    workspace_path = self.openhands_client.workspace_manager.get_workspace_path(session_cfg.workspace_config["workspace_id"])  # type: ignore[index]
                    if workspace_path and self.openhands_client.action_runner.is_available:
                        await self._log_progress(
                            session_id,
                            phase="codeact_demo_start",
                            progress=84.0,
                            message="Starting CodeAct demo execution",
                            metadata={"workspace": str(workspace_path)},
                            parent_phase="synthesizing_guidance",
                        )
                        # Migrate to V2 runtime
                        from ...integrations.openhands_runtime import OpenHandsClientV2
                        from ...services.codeact_runner import CodeActRunnerV2
                        goal = (
                            "Create a Python file at code/integration_demo.py that prints 'Integration demo OK' and then run it "
                            "using bash. If it fails, inspect files and fix. Finish when the output is correct."
                        )
                        # Ensure deterministic bootstrap via v2 (idempotent)
                        try:
                            allowed_roots = [workspace_path / "code", workspace_path / "experiments", workspace_path / "logs", workspace_path / "output", workspace_path / "workspace"]
                            v2_client = OpenHandsClientV2(workspace_path=workspace_path, allowed_write_roots=allowed_roots)
                            try:
                                v2_runner = CodeActRunnerV2(v2_client, workspace_path)
                                await v2_runner.ensure_bootstrap()
                            finally:
                                try:
                                    await v2_client.close()
                                except Exception:
                                    pass
                        except Exception as exc:
                            self.logger.debug("V2 bootstrap skipped in code engine: %s", exc)

                        runner = CodeActRunnerV2(v2_client, workspace_path)
                        async def _cb(event: str, data: Dict[str, Any]):
                            try:
                                await self._log_progress(
                                    session_id,
                                    phase=f"codeact_{event}",
                                    progress=86.0,
                                    message=f"CodeAct {event}",
                                    metadata=data,
                                    parent_phase="codeact_demo_start",
                                )
                            except Exception:
                                pass
                        codeact_result = await runner.run(
                            workspace_path=workspace_path,
                            goal=goal,
                            max_steps=int(self.config.get("codeact_max_steps", 8)),
                            timeout_per_action=int(self.config.get("codeact_action_timeout", 120)),
                            progress_cb=_cb,
                        )
                        await self._log_progress(
                            session_id,
                            phase="codeact_demo_complete",
                            progress=88.0,
                            message="Completed CodeAct demo execution",
                            metadata={"success": bool(codeact_result.get("success"))},
                            parent_phase="codeact_demo_start",
                        )
                except Exception as exc:
                    self.logger.warning("CodeAct demo failed: %s", exc)

            # Generate recommendations
            recommendations = await self._generate_recommendations(query, repositories, analyses, best_practices)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(repositories, analyses)

            return CodeResearchResult(
                query=query,
                repositories=repositories,
                analysis=analyses,
                best_practices=best_practices,
                integration_guide=integration_guide,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
        finally:
            if session_id:
                self._session_phase_nodes.pop(session_id, None)
                self._session_root_nodes.pop(session_id, None)

    async def analyze_specific_repository(self, repository_url: str) -> CodeAnalysis:
        """Analyze a specific repository by URL

        Args:
            repository_url: GitHub repository URL

        Returns:
            Detailed code analysis
        """
        self.logger.info(f"Analyzing specific repository: {repository_url}")

        # Extract repository info from URL
        repo_name = repository_url.split('/')[-1]

        # Create repository object (in production, fetch from GitHub API)
        repository = CodeRepository(
            name=repo_name,
            url=repository_url,
            description=f"Repository {repo_name}",
            language="Python",  # Default, should be detected
            stars=0,
            forks=0,
            last_updated="2024-01-01",
            topics=[]
        )

        return await self.github_engine.analyze_repository(repository)

    async def find_implementation_examples(self, concept: str, language: str) -> List[CodePattern]:
        """Find implementation examples for a specific concept

        Args:
            concept: Programming concept (e.g., "singleton pattern", "async queue")
            language: Programming language

        Returns:
            List of implementation examples
        """
        self.logger.info(f"Finding {language} implementations for: {concept}")

        # Search for repositories implementing the concept
        repositories = await self.github_engine.search_repositories(
            f"{concept} {language}", language, limit=5
        )

        # Extract patterns from found repositories
        all_patterns = []
        for repo in repositories:
            analysis = await self.github_engine.analyze_repository(repo)
            all_patterns.extend(analysis.patterns_identified)

        return all_patterns

    async def _generate_integration_guide(
        self,
        query: str,
        repositories: List[CodeRepository],
        analyses: List[CodeAnalysis]
    ) -> str:
        """Generate integration guide for the researched code"""

        if not repositories:
            return "No repositories found for integration guidance."

        guide_prompt = f"""
        Generate an integration guide for using code related to "{query}".

        Available repositories:
        {chr(10).join([f"- {repo.name}: {repo.description}" for repo in repositories[:3]])}

        Provide a step-by-step integration guide including:
        1. Installation/setup steps
        2. Basic usage examples
        3. Configuration requirements
        4. Common integration patterns
        5. Potential issues and solutions

        Keep the guide practical and focused on implementation.
        """

        try:
            response = await self.llm_client.generate(guide_prompt)
            return response
        except Exception as e:
            self.logger.error(f"Error generating integration guide: {e}")
            return f"Integration guide for {query}: Review the identified repositories and their documentation for implementation details."

    async def _generate_recommendations(
        self,
        query: str,
        repositories: List[CodeRepository],
        analyses: List[CodeAnalysis],
        best_practices: List[CodePattern]
    ) -> List[str]:
        """Generate recommendations based on research results"""

        recommendations = []

        if repositories:
            # Repository-based recommendations
            top_repo = repositories[0]
            recommendations.append(f"Consider using {top_repo.name} ({top_repo.stars} stars) as a starting point")

            # Language recommendations
            languages = list(set(repo.language for repo in repositories))
            if len(languages) > 1:
                recommendations.append(f"Multiple language options available: {', '.join(languages)}")

        if analyses:
            # Analysis-based recommendations
            avg_quality = sum(
                self._to_float(
                    analysis.quality_metrics.get('maintainability'),
                    self._default_quality_scores['maintainability'],
                )
                for analysis in analyses
            ) / len(analyses)
            if avg_quality > 0.7:
                recommendations.append("Found high-quality implementations with good maintainability")
            else:
                recommendations.append("Review code quality carefully before integration")

        if best_practices:
            # Pattern-based recommendations
            security_patterns = [p for p in best_practices if p.category == 'security']
            if security_patterns:
                recommendations.append("Security patterns identified - ensure proper implementation")

        # Default recommendations
        if not recommendations:
            recommendations = [
                f"Research implementations thoroughly for {query}",
                "Test all code before production use",
                "Follow established coding standards"
            ]

        return recommendations

    def _calculate_confidence_score(self, repositories: List[CodeRepository], analyses: List[CodeAnalysis]) -> float:
        """Calculate confidence score based on research quality"""

        if not repositories:
            return 0.1

        # Factor in repository popularity
        total_stars = sum(repo.stars for repo in repositories)
        star_score = min(total_stars / 1000.0, 1.0)  # Normalize to 0-1

        # Factor in number of repositories found
        repo_count_score = min(len(repositories) / 10.0, 1.0)

        # Factor in analysis quality
        analysis_score = 0.5
        if analyses:
            avg_maintainability = sum(
                self._to_float(
                    analysis.quality_metrics.get('maintainability'),
                    self._default_quality_scores['maintainability'],
                )
                for analysis in analyses
            ) / len(analyses)
            analysis_score = avg_maintainability

        # Calculate weighted average
        confidence = (star_score * 0.4 + repo_count_score * 0.3 + analysis_score * 0.3)
        return round(confidence, 2)

    async def compare_implementations(self, implementations: List[str], language: str) -> Dict[str, Any]:
        """Compare different implementations of the same concept

        Args:
            implementations: List of implementation names or repository URLs
            language: Programming language

        Returns:
            Comparison analysis
        """
        self.logger.info(f"Comparing {len(implementations)} implementations in {language}")

        comparison_results = {}
        for impl in implementations:
            repositories = await self.github_engine.search_repositories(impl, language, limit=3)
            if repositories:
                analysis = await self.github_engine.analyze_repository(repositories[0])
                comparison_results[impl] = {
                    "repository": repositories[0],
                    "analysis": analysis,
                    "quality_score": analysis.quality_metrics.get('maintainability', 0.5)
                }

        return comparison_results
