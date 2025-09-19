"""Code Research Engine - Repository analysis and code understanding"""

import asyncio
import logging
import re
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..llm_client import LLMClient
from ..websocket_manager import progress_tracker


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

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        """Initialize code research engine

        Args:
            llm_client: LLM client for analysis
            config: Configuration options
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize sub-engines
        self.github_engine = GitHubSearchEngine(llm_client)
        self.pattern_engine = CodePatternEngine(llm_client)
        self._session_phase_nodes: Dict[str, Dict[str, str]] = {}
        self._session_root_nodes: Dict[str, str] = {}

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

    async def research_code(
        self,
        query: str,
        language: Optional[str] = None,
        include_analysis: bool = True,
        session_id: Optional[str] = None
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

        # Search for relevant repositories
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

        # Generate recommendations
        recommendations = await self._generate_recommendations(query, repositories, analyses, best_practices)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(repositories, analyses)

        try:
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
            avg_quality = sum(analysis.quality_metrics.get('maintainability', 0.5) for analysis in analyses) / len(analyses)
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
                analysis.quality_metrics.get('maintainability', 0.5)
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
