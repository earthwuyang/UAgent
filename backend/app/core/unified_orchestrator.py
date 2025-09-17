"""
Unified Orchestrator - Main coordination system for uagent
Combines ROMA meta-agent, AgentLaboratory collaboration, AI-Scientist automation,
RepoMaster analysis, and multi-modal search into a cohesive workflow system
"""

import asyncio
import uuid
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from .meta_agent import MetaAgent, Task, TaskType, TaskStatus, Agent, AgentRole
from .agent_laboratory import AgentLaboratory, CollaborationPattern, CollaborationSession
from .ai_scientist import AIScientist, ResearchProject, ResearchPhase
from .repo_master import RepoMaster, Repository, AnalysisDepth
from .hierarchical_agents import HierarchicalAgentCoordinator, AgentRole as HierarchicalAgentRole
from ..utils.multi_modal_search import MultiModalSearchEngine

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of workflows the unified orchestrator can execute"""
    RESEARCH_PROJECT = "research_project"
    CODE_ANALYSIS = "code_analysis"
    COLLABORATIVE_DEVELOPMENT = "collaborative_development"
    AUTOMATED_RESEARCH = "automated_research"
    MULTI_MODAL_SEARCH = "multi_modal_search"
    HYBRID_WORKFLOW = "hybrid_workflow"


class OrchestrationStrategy(Enum):
    """Strategies for orchestrating different components"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    ROMA_RECURSIVE = "roma_recursive"


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution"""
    workflow_type: WorkflowType
    strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE
    components: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 3600  # Default 1 hour timeout
    max_iterations: int = 10
    collaboration_pattern: Optional[CollaborationPattern] = None


@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    workflow_id: str
    workflow_type: WorkflowType
    status: str
    components_used: List[str]
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class UnifiedOrchestrator:
    """
    Main orchestrator that coordinates all uagent components
    Provides intelligent workflow management and component integration
    """

    def __init__(self):
        # Initialize all component systems
        self.meta_agent = MetaAgent()
        self.agent_lab = AgentLaboratory()
        self.ai_scientist = AIScientist()
        self.repo_master = RepoMaster()
        self.search_engine = MultiModalSearchEngine()
        self.hierarchical_coordinator = HierarchicalAgentCoordinator()

        # Workflow management
        self.active_workflows: Dict[str, WorkflowResult] = {}
        self.workflow_templates: Dict[str, WorkflowConfig] = self._initialize_workflow_templates()

        # Component integration
        self._integrate_components()

    def _initialize_workflow_templates(self) -> Dict[str, WorkflowConfig]:
        """Initialize predefined workflow templates"""
        return {
            "full_research_cycle": WorkflowConfig(
                workflow_type=WorkflowType.AUTOMATED_RESEARCH,
                strategy=OrchestrationStrategy.ROMA_RECURSIVE,
                components=["ai_scientist", "search_engine", "agent_lab", "meta_agent"],
                parameters={
                    "research_depth": "comprehensive",
                    "collaboration_enabled": True,
                    "auto_iteration": True
                }
            ),
            "code_deep_dive": WorkflowConfig(
                workflow_type=WorkflowType.CODE_ANALYSIS,
                strategy=OrchestrationStrategy.SEQUENTIAL,
                components=["repo_master", "agent_lab", "search_engine"],
                parameters={
                    "analysis_depth": "deep",
                    "pattern_detection": True,
                    "documentation_generation": True
                }
            ),
            "collaborative_project": WorkflowConfig(
                workflow_type=WorkflowType.COLLABORATIVE_DEVELOPMENT,
                strategy=OrchestrationStrategy.PARALLEL,
                components=["agent_lab", "meta_agent", "repo_master"],
                collaboration_pattern=CollaborationPattern.HIERARCHICAL,
                parameters={
                    "team_size": 4,
                    "specialization_level": "high"
                }
            ),
            "intelligent_search": WorkflowConfig(
                workflow_type=WorkflowType.MULTI_MODAL_SEARCH,
                strategy=OrchestrationStrategy.ADAPTIVE,
                components=["search_engine", "ai_scientist", "repo_master"],
                parameters={
                    "search_breadth": "comprehensive",
                    "result_synthesis": True
                }
            )
        }

    def _integrate_components(self):
        """Integrate components for seamless collaboration"""
        # Register meta-agent's specialized agents with agent laboratory
        for agent_id, agent in self.meta_agent.agents.items():
            self.agent_lab.register_agent(agent)

    async def execute_workflow(
        self,
        workflow_config: Union[str, WorkflowConfig],
        inputs: Dict[str, Any] = None
    ) -> str:
        """Execute a workflow and return workflow ID for tracking"""
        # Handle template names
        if isinstance(workflow_config, str):
            if workflow_config in self.workflow_templates:
                config = self.workflow_templates[workflow_config]
            else:
                raise ValueError(f"Unknown workflow template: {workflow_config}")
        else:
            config = workflow_config

        inputs = inputs or {}
        workflow_id = str(uuid.uuid4())

        # Create workflow result tracking
        workflow_result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=config.workflow_type,
            status="started",
            components_used=config.components
        )
        self.active_workflows[workflow_id] = workflow_result

        # Execute workflow asynchronously
        asyncio.create_task(self._execute_workflow_async(workflow_id, config, inputs))

        return workflow_id

    async def _execute_workflow_async(
        self,
        workflow_id: str,
        config: WorkflowConfig,
        inputs: Dict[str, Any]
    ):
        """Execute workflow asynchronously"""
        workflow_result = self.active_workflows[workflow_id]
        start_time = datetime.now()

        try:
            workflow_result.status = "running"

            # Execute based on workflow type
            if config.workflow_type == WorkflowType.AUTOMATED_RESEARCH:
                results = await self._execute_research_workflow(config, inputs)
            elif config.workflow_type == WorkflowType.CODE_ANALYSIS:
                results = await self._execute_code_analysis_workflow(config, inputs)
            elif config.workflow_type == WorkflowType.COLLABORATIVE_DEVELOPMENT:
                results = await self._execute_collaborative_workflow(config, inputs)
            elif config.workflow_type == WorkflowType.MULTI_MODAL_SEARCH:
                results = await self._execute_search_workflow(config, inputs)
            elif config.workflow_type == WorkflowType.HYBRID_WORKFLOW:
                results = await self._execute_hybrid_workflow(config, inputs)
            else:
                results = {"error": "Unknown workflow type"}

            workflow_result.results = results
            workflow_result.status = "completed"

        except Exception as e:
            workflow_result.results = {"error": str(e)}
            workflow_result.status = "failed"

        finally:
            end_time = datetime.now()
            workflow_result.execution_time = (end_time - start_time).total_seconds()
            workflow_result.completed_at = end_time

    async def _execute_research_workflow(
        self,
        config: WorkflowConfig,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute automated research workflow"""
        # Extract research parameters
        title = inputs.get("title", "Automated Research Project")
        description = inputs.get("description", "AI-driven research investigation")
        research_questions = inputs.get("research_questions", [])

        if not research_questions:
            # Generate research questions using search
            query = inputs.get("query", title)
            search_results = await self.search_engine.intelligent_search(query)
            # Extract potential research questions from search results
            research_questions = await self._extract_research_questions(search_results)

        # Start AI-Scientist research project
        project_id = await self.ai_scientist.start_research_project(
            title=title,
            description=description,
            research_questions=research_questions,
            initial_context=inputs
        )

        # Create collaboration session for research
        if config.parameters.get("collaboration_enabled", True):
            collab_session = await self.agent_lab.create_collaboration_session(
                name=f"Research: {title}",
                pattern=CollaborationPattern.HIERARCHICAL,
                agent_roles=[AgentRole.RESEARCHER, AgentRole.ANALYZER, AgentRole.SYNTHESIZER]
            )
        else:
            collab_session = None

        results = {
            "project_id": project_id,
            "collaboration_session": collab_session,
            "research_questions": research_questions
        }

        # Execute research phases with ROMA-style recursion
        if config.strategy == OrchestrationStrategy.ROMA_RECURSIVE:
            results["phases"] = await self._execute_recursive_research(project_id, collab_session)
        else:
            results["phases"] = await self._execute_sequential_research(project_id)

        return results

    async def _execute_code_analysis_workflow(
        self,
        config: WorkflowConfig,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute code analysis workflow"""
        repo_path = inputs.get("repository_path")
        if not repo_path:
            raise ValueError("Repository path required for code analysis")

        analysis_depth = AnalysisDepth(config.parameters.get("analysis_depth", "semantic"))

        # Analyze repository with RepoMaster
        repo_id = await self.repo_master.analyze_repository(repo_path, analysis_depth)

        # Get repository summary
        repo_summary = await self.repo_master.get_repository_summary(repo_id)

        # Create collaborative analysis session
        if "agent_lab" in config.components:
            collab_session = await self.agent_lab.create_collaboration_session(
                name=f"Code Analysis: {repo_path}",
                pattern=CollaborationPattern.PEER_TO_PEER,
                agent_roles=[AgentRole.ANALYZER, AgentRole.CODER, AgentRole.REVIEWER]
            )

            # Execute collaborative analysis
            analysis_task = Task(
                id=str(uuid.uuid4()),
                name="Collaborative Code Analysis",
                description=f"Analyze repository: {repo_path}",
                task_type=TaskType.CODE_ANALYSIS,
                context={"repo_id": repo_id, "summary": repo_summary}
            )

            collaborative_results = await self.agent_lab.execute_collaborative_task(
                collab_session, analysis_task
            )
        else:
            collaborative_results = {}

        # Enhanced search for related projects
        if "search_engine" in config.components:
            search_query = f"similar projects {repo_summary['repository_info']['language']}"
            search_results = await self.search_engine.unified_search(
                query=search_query,
                search_types=['code', 'web'],
                limit=10
            )
        else:
            search_results = {}

        return {
            "repository_analysis": repo_summary,
            "collaborative_insights": collaborative_results,
            "related_projects": search_results,
            "recommendations": await self._generate_code_recommendations(repo_summary)
        }

    async def _execute_collaborative_workflow(
        self,
        config: WorkflowConfig,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute collaborative development workflow"""
        project_name = inputs.get("project_name", "Collaborative Project")
        requirements = inputs.get("requirements", [])

        # Create collaboration session
        collab_pattern = config.collaboration_pattern or CollaborationPattern.HIERARCHICAL
        collab_session = await self.agent_lab.create_collaboration_session(
            name=project_name,
            pattern=collab_pattern,
            agent_roles=[AgentRole.CODER, AgentRole.REVIEWER, AgentRole.TESTER, AgentRole.SYNTHESIZER]
        )

        # Break down project into tasks using meta-agent
        tasks = []
        for req in requirements:
            task_id = await self.meta_agent.create_task(
                name=f"Implement: {req}",
                description=f"Collaborative implementation of {req}",
                task_type=TaskType.CODE_GENERATION,
                context=inputs
            )
            tasks.append(task_id)

        # Execute tasks collaboratively
        task_results = {}
        for task_id in tasks:
            task = self.meta_agent.tasks[task_id]
            result = await self.agent_lab.execute_collaborative_task(collab_session, task)
            task_results[task_id] = result

        # Get collaboration metrics
        metrics = await self.agent_lab.get_collaboration_metrics(collab_session)

        return {
            "collaboration_session": collab_session,
            "task_results": task_results,
            "collaboration_metrics": metrics,
            "project_summary": await self._synthesize_project_results(task_results)
        }

    async def _execute_search_workflow(
        self,
        config: WorkflowConfig,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute intelligent search workflow"""
        query = inputs.get("query")
        if not query:
            raise ValueError("Query required for search workflow")

        # Perform intelligent search
        search_results = await self.search_engine.intelligent_search(
            query=query,
            context=inputs
        )

        # Enhance results with AI-Scientist analysis
        if "ai_scientist" in config.components and search_results.get("academic"):
            enhanced_academic = await self._enhance_academic_results(search_results["academic"])
            search_results["academic_enhanced"] = enhanced_academic

        # Enhance code results with RepoMaster analysis
        if "repo_master" in config.components and search_results.get("code"):
            enhanced_code = await self._enhance_code_results(search_results["code"])
            search_results["code_enhanced"] = enhanced_code

        # Synthesize results
        if config.parameters.get("result_synthesis", True):
            synthesis = await self._synthesize_search_results(search_results, query)
            search_results["synthesis"] = synthesis

        return search_results

    async def _execute_hybrid_workflow(
        self,
        config: WorkflowConfig,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute hybrid workflow combining multiple workflow types"""
        # This is a flexible workflow that can combine elements from other workflows
        results = {}

        # Execute sub-workflows based on configuration
        if "research" in config.parameters:
            research_config = WorkflowConfig(
                workflow_type=WorkflowType.AUTOMATED_RESEARCH,
                components=["ai_scientist", "search_engine"]
            )
            results["research"] = await self._execute_research_workflow(research_config, inputs)

        if "code_analysis" in config.parameters:
            code_config = WorkflowConfig(
                workflow_type=WorkflowType.CODE_ANALYSIS,
                components=["repo_master", "agent_lab"]
            )
            results["code_analysis"] = await self._execute_code_analysis_workflow(code_config, inputs)

        if "search" in config.parameters:
            search_config = WorkflowConfig(
                workflow_type=WorkflowType.MULTI_MODAL_SEARCH,
                components=["search_engine"]
            )
            results["search"] = await self._execute_search_workflow(search_config, inputs)

        return results

    async def _extract_research_questions(self, search_results: Dict[str, Any]) -> List[str]:
        """Extract potential research questions from search results"""
        questions = []

        # Extract from academic results
        for result in search_results.get("academic", []):
            title = result.get("title", "")
            if "?" in title:
                questions.append(title)
            elif any(word in title.lower() for word in ["how", "what", "why", "when", "where"]):
                questions.append(f"How does {title.lower()}?")

        # Generate questions from web results
        for result in search_results.get("web", [])[:5]:
            title = result.get("title", "")
            questions.append(f"What are the implications of {title.lower()}?")

        return questions[:5]

    async def _execute_recursive_research(
        self,
        project_id: str,
        collab_session: Optional[str]
    ) -> Dict[str, Any]:
        """Execute research with ROMA-style recursive decomposition"""
        phases_results = {}

        research_phases = [
            ResearchPhase.LITERATURE_REVIEW,
            ResearchPhase.HYPOTHESIS_GENERATION,
            ResearchPhase.EXPERIMENTAL_DESIGN,
            ResearchPhase.EXPERIMENT_EXECUTION,
            ResearchPhase.RESULT_ANALYSIS
        ]

        for phase in research_phases:
            # Execute phase with AI-Scientist
            phase_result = await self.ai_scientist.execute_research_phase(project_id, phase)

            # If collaboration enabled, get peer review
            if collab_session:
                review_task = Task(
                    id=str(uuid.uuid4()),
                    name=f"Review {phase.value}",
                    description=f"Peer review of {phase.value} results",
                    task_type=TaskType.VALIDATION,
                    context={"phase_result": phase_result}
                )
                peer_review = await self.agent_lab.execute_collaborative_task(
                    collab_session, review_task, "consensus_decision"
                )
                phase_result["peer_review"] = peer_review

            phases_results[phase.value] = phase_result

            # ROMA-style self-reflection after each phase
            reflection = await self.meta_agent.self_reflect()
            phase_result["meta_reflection"] = reflection

        return phases_results

    async def _execute_sequential_research(self, project_id: str) -> Dict[str, Any]:
        """Execute research phases sequentially without collaboration"""
        # Define the phases to execute in order
        sequential_phases = [
            ResearchPhase.LITERATURE_REVIEW,
            ResearchPhase.HYPOTHESIS_GENERATION,
            ResearchPhase.EXPERIMENTAL_DESIGN,
            ResearchPhase.EXPERIMENT_EXECUTION,
            ResearchPhase.RESULT_ANALYSIS
        ]

        results = {}

        for phase in sequential_phases:
            logger.info(f"Executing {phase.value} phase")
            phase_result = await self.ai_scientist.execute_research_phase(project_id, phase)
            results[phase.value] = phase_result

            # Stop if phase failed
            if phase_result.get("error"):
                logger.error(f"Phase {phase.value} failed: {phase_result['error']}")
                break
            else:
                logger.info(f"Phase {phase.value} completed successfully")

        return {
            "method": "sequential",
            "phases_completed": list(results.keys()),
            "phase_results": results
        }

    async def _generate_code_recommendations(self, repo_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on repository analysis"""
        recommendations = []

        # Extract metrics and patterns
        metrics = repo_summary.get("code_metrics", {})
        patterns = repo_summary.get("patterns_summary", {})

        # Complexity recommendations
        complexity = metrics.get("complexity", {})
        if complexity.get("average", 0) > 8:
            recommendations.append("Consider refactoring high-complexity functions")

        # Pattern-based recommendations
        code_smells = patterns.get("by_type", {}).get("code_smell", 0)
        if code_smells > 0:
            recommendations.append(f"Address {code_smells} code smell issues")

        # Architecture recommendations
        arch_score = metrics.get("architecture_score", 0)
        if arch_score < 0.7:
            recommendations.append("Improve overall architecture and design patterns")

        return recommendations

    async def _enhance_academic_results(self, academic_results: List[Dict]) -> List[Dict]:
        """Enhance academic search results with AI-Scientist insights"""
        enhanced = []

        for result in academic_results:
            enhanced_result = result.copy()

            # Add relevance analysis
            title = result.get("title", "")
            enhanced_result["ai_analysis"] = {
                "research_area": await self._classify_research_area(title),
                "novelty_score": await self._estimate_novelty(result),
                "methodology_type": await self._identify_methodology(result)
            }

            enhanced.append(enhanced_result)

        return enhanced

    async def _enhance_code_results(self, code_results: List[Dict]) -> List[Dict]:
        """Enhance code search results with RepoMaster analysis"""
        enhanced = []

        for result in code_results:
            enhanced_result = result.copy()

            # Add code quality assessment
            enhanced_result["quality_assessment"] = {
                "estimated_complexity": "medium",  # Would use actual analysis
                "language_best_practices": True,
                "documentation_quality": "good"
            }

            enhanced.append(enhanced_result)

        return enhanced

    async def _synthesize_search_results(
        self,
        search_results: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """Synthesize search results across different sources"""
        synthesis = {
            "query": query,
            "total_results": sum(len(results) for results in search_results.values() if isinstance(results, list)),
            "sources_used": list(search_results.keys()),
            "key_themes": [],
            "recommendations": []
        }

        # Extract key themes from all results
        all_titles = []
        for source, results in search_results.items():
            if isinstance(results, list):
                all_titles.extend([r.get("title", "") for r in results])

        # Simple theme extraction (in production, would use NLP)
        common_words = {}
        for title in all_titles:
            words = title.lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    common_words[word] = common_words.get(word, 0) + 1

        # Get top themes
        synthesis["key_themes"] = [
            word for word, count in sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        # Generate recommendations
        if search_results.get("academic"):
            synthesis["recommendations"].append("Consider reviewing academic literature for theoretical foundation")

        if search_results.get("code"):
            synthesis["recommendations"].append("Examine existing code implementations for practical insights")

        return synthesis

    async def _synthesize_project_results(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from collaborative project tasks"""
        return {
            "total_tasks": len(task_results),
            "completed_successfully": len([r for r in task_results.values() if "error" not in r]),
            "summary": "Collaborative project execution completed",
            "next_steps": ["Review and integrate all task outputs", "Conduct final testing", "Prepare documentation"]
        }

    async def _classify_research_area(self, title: str) -> str:
        """Classify research area from paper title"""
        title_lower = title.lower()
        if any(term in title_lower for term in ["machine learning", "ai", "neural"]):
            return "artificial_intelligence"
        elif any(term in title_lower for term in ["algorithm", "computational", "optimization"]):
            return "computer_science"
        elif any(term in title_lower for term in ["data", "analysis", "statistics"]):
            return "data_science"
        else:
            return "general"

    async def _estimate_novelty(self, paper: Dict[str, Any]) -> float:
        """Estimate novelty score of research paper"""
        # Simple heuristic based on publication date and citation count
        citations = paper.get("citations", 0)
        year = paper.get("published", "2020")

        try:
            pub_year = int(year[:4]) if isinstance(year, str) else int(year)
            current_year = datetime.now().year
            recency = max(0, (current_year - pub_year) / 10)  # Normalize to 0-1
            citation_score = min(citations / 100, 1.0)  # Normalize citations
            return (recency + citation_score) / 2
        except:
            return 0.5

    async def _identify_methodology(self, paper: Dict[str, Any]) -> str:
        """Identify research methodology from paper"""
        abstract = paper.get("snippet", "").lower()
        if "experiment" in abstract:
            return "experimental"
        elif "survey" in abstract or "review" in abstract:
            return "survey"
        elif "theoretical" in abstract or "theory" in abstract:
            return "theoretical"
        else:
            return "empirical"

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get status of workflow execution"""
        return self.active_workflows.get(workflow_id)

    async def list_active_workflows(self) -> List[WorkflowResult]:
        """List all active workflows"""
        return [
            workflow for workflow in self.active_workflows.values()
            if workflow.status in ["started", "running"]
        ]

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            if workflow.status in ["started", "running"]:
                workflow.status = "cancelled"
                workflow.completed_at = datetime.now()
                return True
        return False

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "components": {
                "meta_agent": {
                    "active_tasks": len([t for t in self.meta_agent.tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
                    "total_agents": len(self.meta_agent.agents)
                },
                "agent_laboratory": {
                    "active_sessions": len([s for s in self.agent_lab.collaboration_sessions.values() if s.status == "active"]),
                    "registered_agents": len(self.agent_lab.specialized_agents)
                },
                "ai_scientist": {
                    "active_projects": len(self.ai_scientist.active_projects)
                },
                "repo_master": {
                    "analyzed_repositories": len(self.repo_master.analyzed_repositories)
                }
            },
            "workflows": {
                "active": len([w for w in self.active_workflows.values() if w.status in ["started", "running"]]),
                "completed": len([w for w in self.active_workflows.values() if w.status == "completed"]),
                "failed": len([w for w in self.active_workflows.values() if w.status == "failed"])
            },
            "templates_available": list(self.workflow_templates.keys())
        }

    async def execute_hierarchical_research(
        self,
        research_query: str,
        domain: str = "",
        research_context: Dict[str, Any] = None
    ) -> str:
        """Execute hierarchical multi-agent research pipeline"""

        session_id = await self.hierarchical_coordinator.start_hierarchical_research(
            research_query=research_query,
            domain=domain,
            research_context=research_context or {}
        )

        # Create a workflow result to track in the unified system
        workflow_result = WorkflowResult(
            workflow_id=session_id,
            workflow_type=WorkflowType.AUTOMATED_RESEARCH,
            status="running",
            components_used=["hierarchical_coordinator"],
            results={"hierarchical_session_id": session_id}
        )

        self.active_workflows[session_id] = workflow_result

        return session_id

    async def get_hierarchical_research_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of hierarchical research session"""
        return await self.hierarchical_coordinator.get_session_status(session_id)

    async def get_hierarchical_research_results(self, session_id: str) -> Dict[str, Any]:
        """Get final results of hierarchical research session"""
        results = await self.hierarchical_coordinator.get_session_results(session_id)

        # Update workflow status if completed
        if session_id in self.active_workflows:
            workflow = self.active_workflows[session_id]
            session_status = await self.hierarchical_coordinator.get_session_status(session_id)

            if session_status.get("status") == "completed":
                workflow.status = "completed"
                workflow.completed_at = datetime.now()
                workflow.results.update(results)
            elif session_status.get("status") == "failed":
                workflow.status = "failed"
                workflow.completed_at = datetime.now()
                workflow.results.update({"error": results.get("error", "Unknown error")})

        return results

    async def _execute_hierarchical_research_with_tree_integration(
        self,
        project_id: str,
        research_goal: str = None
    ) -> Dict[str, Any]:
        """Execute hierarchical research integrated with the tree system"""

        # If no research goal provided, try to extract from project
        if not research_goal:
            # Here we would normally fetch from the database, but for now use a default
            research_goal = f"Research project {project_id}"

        # Start hierarchical research session
        session_id = await self.execute_hierarchical_research(
            research_query=research_goal,
            domain="AI/ML Research",
            research_context={"project_id": project_id}
        )

        # Wait for completion (in a real system, this would be done asynchronously)
        max_wait_time = 300  # 5 minutes timeout
        wait_interval = 5    # Check every 5 seconds
        total_waited = 0

        while total_waited < max_wait_time:
            status = await self.get_hierarchical_research_status(session_id)

            if status.get("status") == "completed":
                results = await self.get_hierarchical_research_results(session_id)
                return {
                    "method": "hierarchical_agents",
                    "session_id": session_id,
                    "results": results,
                    "status": "completed"
                }
            elif status.get("status") == "failed":
                return {
                    "method": "hierarchical_agents",
                    "session_id": session_id,
                    "error": "Hierarchical research failed",
                    "status": "failed"
                }

            await asyncio.sleep(wait_interval)
            total_waited += wait_interval

        return {
            "method": "hierarchical_agents",
            "session_id": session_id,
            "error": "Research timeout - still in progress",
            "status": "timeout"
        }