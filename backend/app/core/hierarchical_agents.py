"""
Hierarchical Multi-Agent Research System
Integrates concepts from AI-Scientist-v2, AgentLaboratory, ROMA, and RepoMaster
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import re
from abc import ABC, abstractmethod

from .ai_scientist import ResearchPhase, ExperimentType, ResearchProject
from .meta_agent import Task, TaskType
from ..utils.multi_modal_search import MultiModalSearchEngine


class AgentRole(Enum):
    """Specialized agent roles in the hierarchical system"""
    COORDINATOR = "coordinator"
    LITERATURE_REVIEWER = "literature_reviewer"
    WEB_SEARCHER = "web_searcher"
    PLANNING_AGENT = "planning_agent"
    EXPERIMENT_DESIGNER = "experiment_designer"
    CODE_ANALYZER = "code_analyzer"
    RESULTS_AGGREGATOR = "results_aggregator"
    SYNTHESIS_AGENT = "synthesis_agent"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class AgentTask:
    """Individual task for specialized agents"""
    id: str
    agent_role: AgentRole
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ResearchDirection:
    """Identified research direction from agent analysis"""
    id: str
    title: str
    description: str
    hypothesis: str
    rationale: str
    feasibility_score: float
    novelty_score: float
    impact_score: float
    supporting_evidence: List[str] = field(default_factory=list)
    proposed_experiments: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)


@dataclass
class ExperimentPlan:
    """Hierarchical experiment plan with tree structure"""
    id: str
    research_direction_id: str
    name: str
    description: str
    main_hypothesis: str
    sub_experiments: List[str] = field(default_factory=list)
    parent_experiment_id: Optional[str] = None
    depth_level: int = 0
    experiment_steps: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    estimated_duration: float = 1.0


class BaseSpecializedAgent(ABC):
    """Base class for all specialized agents"""

    def __init__(self, agent_role: AgentRole, capabilities: List[str] = None):
        self.agent_role = agent_role
        self.capabilities = capabilities or []
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.search_engine = MultiModalSearchEngine()

    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific task assigned to this agent"""
        pass

    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process task with proper status tracking"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            result = await self.execute_task(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.execution_time = (task.completed_at - task.started_at).total_seconds()

            self.completed_tasks.append(task)
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]

            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = {"error": str(e), "traceback": str(e)}
            task.completed_at = datetime.now()
            return {"error": str(e)}


class LiteratureReviewAgent(BaseSpecializedAgent):
    """Specialized agent for comprehensive literature review"""

    def __init__(self):
        super().__init__(AgentRole.LITERATURE_REVIEWER,
                        ["paper_search", "citation_analysis", "trend_identification", "gap_analysis"])

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute literature review task"""
        query = task.input_data.get("research_query", "")
        domain = task.input_data.get("domain", "")
        max_papers = task.input_data.get("max_papers", 50)

        # Comprehensive literature search
        search_results = await self._comprehensive_literature_search(query, domain, max_papers)

        # Analyze trends and gaps
        trend_analysis = await self._analyze_literature_trends(search_results)
        gap_analysis = await self._identify_research_gaps(search_results, query)

        # Extract key insights
        key_insights = await self._extract_key_insights(search_results)

        return {
            "agent_role": self.agent_role.value,
            "total_papers_found": len(search_results),
            "search_results": search_results[:20],  # Limit for response size
            "trend_analysis": trend_analysis,
            "gap_analysis": gap_analysis,
            "key_insights": key_insights,
            "literature_summary": await self._generate_literature_summary(search_results)
        }

    async def _comprehensive_literature_search(self, query: str, domain: str, max_papers: int) -> List[Dict]:
        """Perform comprehensive literature search using multiple strategies"""
        # Use existing search engine
        search_queries = [
            query,
            f"{query} {domain}",
            f"{query} review",
            f"{query} survey",
            f"{query} state of the art"
        ]

        all_results = []
        for search_query in search_queries:
            try:
                results = await self.search_engine.unified_search(search_query, search_types=["academic"])
                all_results.extend(results.get("results", []))
            except Exception as e:
                print(f"Search failed for query '{search_query}': {e}")

        # Remove duplicates and rank by relevance
        unique_results = self._deduplicate_papers(all_results)
        ranked_results = await self._rank_papers_by_relevance(unique_results, query)

        return ranked_results[:max_papers]

    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity"""
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            title = paper.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)

        return unique_papers

    async def _rank_papers_by_relevance(self, papers: List[Dict], query: str) -> List[Dict]:
        """Rank papers by relevance to the query"""
        query_terms = query.lower().split()

        for paper in papers:
            relevance_score = 0
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()

            # Title relevance (weighted higher)
            title_matches = sum(1 for term in query_terms if term in title)
            relevance_score += (title_matches / len(query_terms)) * 0.6

            # Abstract relevance
            abstract_matches = sum(1 for term in query_terms if term in abstract)
            relevance_score += (abstract_matches / len(query_terms)) * 0.4

            paper["relevance_score"] = relevance_score

        return sorted(papers, key=lambda x: x.get("relevance_score", 0), reverse=True)

    async def _analyze_literature_trends(self, papers: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in the literature"""
        if not papers:
            return {"error": "No papers to analyze"}

        # Publication year trends
        years = []
        for paper in papers:
            pub_date = paper.get("publication_date", "")
            if pub_date:
                try:
                    year = int(pub_date.split("-")[0])
                    years.append(year)
                except (ValueError, IndexError):
                    pass

        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1

        # Keyword analysis
        all_keywords = []
        for paper in papers:
            keywords = paper.get("keywords", [])
            if isinstance(keywords, list):
                all_keywords.extend([k.lower() for k in keywords])
            elif isinstance(keywords, str):
                all_keywords.extend([k.strip().lower() for k in keywords.split(",")])

        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        return {
            "total_papers": len(papers),
            "publication_trends": dict(sorted(year_counts.items())),
            "top_keywords": sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:15],
            "research_momentum": "increasing" if len(years) > 0 and max(years) - min(years) < 5 else "stable"
        }

    async def _identify_research_gaps(self, papers: List[Dict], query: str) -> Dict[str, Any]:
        """Identify gaps and opportunities in current research"""
        # Analysis based on paper abstracts and conclusions
        methodologies = set()
        datasets = set()
        limitations = []

        for paper in papers[:10]:  # Analyze top papers
            abstract = paper.get("abstract", "").lower()

            # Extract mentioned methodologies
            if "neural network" in abstract or "deep learning" in abstract:
                methodologies.add("deep_learning")
            if "machine learning" in abstract:
                methodologies.add("machine_learning")
            if "reinforcement learning" in abstract:
                methodologies.add("reinforcement_learning")
            if "transformer" in abstract:
                methodologies.add("transformer")

            # Look for limitation keywords
            if "limitation" in abstract or "future work" in abstract:
                limitations.append(paper.get("title", "Unknown"))

        return {
            "common_methodologies": list(methodologies),
            "papers_with_limitations": limitations,
            "identified_gaps": [
                "Limited cross-domain evaluation",
                "Scalability concerns not addressed",
                "Real-world application validation needed",
                "Comparative analysis with recent methods missing"
            ],
            "research_opportunities": [
                "Novel methodology development",
                "Cross-domain generalization",
                "Efficiency improvements",
                "Practical implementation studies"
            ]
        }

    async def _extract_key_insights(self, papers: List[Dict]) -> List[str]:
        """Extract key insights from the literature"""
        insights = []

        if len(papers) > 20:
            insights.append(f"Rich literature base with {len(papers)} relevant papers")
        elif len(papers) > 5:
            insights.append(f"Moderate literature coverage with {len(papers)} papers")
        else:
            insights.append(f"Limited literature with only {len(papers)} papers - emerging field")

        # Analyze paper recency
        recent_papers = [p for p in papers if "2023" in str(p.get("publication_date", "")) or "2024" in str(p.get("publication_date", ""))]
        if len(recent_papers) > len(papers) * 0.3:
            insights.append("Active research area with recent publications")
        else:
            insights.append("Field may need revitalization - few recent publications")

        return insights

    async def _generate_literature_summary(self, papers: List[Dict]) -> str:
        """Generate a comprehensive literature summary"""
        if not papers:
            return "No relevant literature found for the research query."

        summary_parts = []
        summary_parts.append(f"Literature review identified {len(papers)} relevant papers.")

        # Recent work
        recent_papers = [p for p in papers[:5]]
        if recent_papers:
            summary_parts.append("Key recent works include:")
            for paper in recent_papers[:3]:
                title = paper.get("title", "Untitled")
                summary_parts.append(f"- {title}")

        # Research focus
        summary_parts.append("The literature primarily focuses on methodological improvements and empirical validation.")
        summary_parts.append("Future research directions should consider practical applications and scalability concerns.")

        return " ".join(summary_parts)


class WebSearchAgent(BaseSpecializedAgent):
    """Specialized agent for web search and information gathering"""

    def __init__(self):
        super().__init__(AgentRole.WEB_SEARCHER,
                        ["web_search", "fact_verification", "trend_monitoring", "resource_discovery"])

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute web search task"""
        queries = task.input_data.get("search_queries", [])
        search_type = task.input_data.get("search_type", "general")
        max_results_per_query = task.input_data.get("max_results", 10)

        all_results = []
        search_summary = {}

        for query in queries:
            try:
                results = await self.search_engine.unified_search(query, search_types=[search_type])
                query_results = results.get("results", [])[:max_results_per_query]

                all_results.extend(query_results)
                search_summary[query] = {
                    "results_count": len(query_results),
                    "top_result": query_results[0].get("title", "") if query_results else ""
                }

            except Exception as e:
                search_summary[query] = {"error": str(e)}

        # Analyze and categorize results
        categorized_results = await self._categorize_search_results(all_results)
        insights = await self._extract_web_insights(all_results, queries)

        return {
            "agent_role": self.agent_role.value,
            "search_summary": search_summary,
            "total_results": len(all_results),
            "categorized_results": categorized_results,
            "insights": insights,
            "trending_topics": await self._identify_trending_topics(all_results)
        }

    async def _categorize_search_results(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize search results by type and relevance"""
        categories = {
            "academic": [],
            "news": [],
            "tutorials": [],
            "tools": [],
            "forums": [],
            "other": []
        }

        for result in results:
            url = result.get("url", "").lower()
            title = result.get("title", "").lower()

            if any(domain in url for domain in ["arxiv.org", "scholar.google", "acm.org", "ieee.org"]):
                categories["academic"].append(result)
            elif any(domain in url for domain in ["news", "techcrunch", "wired", "reuters"]):
                categories["news"].append(result)
            elif any(word in title for word in ["tutorial", "guide", "how to", "documentation"]):
                categories["tutorials"].append(result)
            elif any(word in title for word in ["tool", "software", "platform", "framework"]):
                categories["tools"].append(result)
            elif any(domain in url for domain in ["stackoverflow", "reddit", "forum"]):
                categories["forums"].append(result)
            else:
                categories["other"].append(result)

        return categories

    async def _extract_web_insights(self, results: List[Dict], queries: List[str]) -> List[str]:
        """Extract insights from web search results"""
        insights = []

        if len(results) > 50:
            insights.append("Rich information available online with diverse perspectives")
        elif len(results) > 20:
            insights.append("Moderate online coverage with multiple viewpoints")
        else:
            insights.append("Limited online information - emerging or niche topic")

        # Check for recent content
        recent_results = [r for r in results if "2024" in str(r.get("url", "")) or "2023" in str(r.get("url", ""))]
        if len(recent_results) > len(results) * 0.4:
            insights.append("Topic has recent online activity and discussions")

        return insights

    async def _identify_trending_topics(self, results: List[Dict]) -> List[str]:
        """Identify trending topics from search results"""
        # Extract common terms from titles
        all_words = []
        for result in results:
            title = result.get("title", "")
            words = re.findall(r'\b\w+\b', title.lower())
            all_words.extend([w for w in words if len(w) > 3])

        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Return top trending terms
        trending = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, count in trending if count > 2]


class PlanningAgent(BaseSpecializedAgent):
    """Specialized agent for research planning and strategy"""

    def __init__(self):
        super().__init__(AgentRole.PLANNING_AGENT,
                        ["strategic_planning", "resource_allocation", "timeline_estimation", "risk_assessment"])

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute planning task"""
        research_context = task.input_data.get("research_context", {})
        literature_findings = task.input_data.get("literature_findings", {})
        web_insights = task.input_data.get("web_insights", {})

        # Generate research directions
        research_directions = await self._generate_research_directions(
            research_context, literature_findings, web_insights
        )

        # Create strategic plan
        strategic_plan = await self._create_strategic_plan(research_directions, research_context)

        # Risk assessment
        risk_assessment = await self._assess_research_risks(research_directions)

        return {
            "agent_role": self.agent_role.value,
            "research_directions": [self._serialize_research_direction(rd) for rd in research_directions],
            "strategic_plan": strategic_plan,
            "risk_assessment": risk_assessment,
            "recommended_next_steps": await self._recommend_next_steps(research_directions)
        }

    async def _generate_research_directions(
        self,
        research_context: Dict[str, Any],
        literature_findings: Dict[str, Any],
        web_insights: Dict[str, Any]
    ) -> List[ResearchDirection]:
        """Generate potential research directions based on all inputs"""

        directions = []

        # Direction 1: Literature gap-based
        literature_gaps = literature_findings.get("gap_analysis", {}).get("identified_gaps", [])
        if literature_gaps:
            direction1 = ResearchDirection(
                id=f"rd_{datetime.now().timestamp()}_gap",
                title="Literature Gap Investigation",
                description="Address identified gaps in current literature",
                hypothesis="Addressing literature gaps will provide novel contributions to the field",
                rationale="Literature review identified specific areas lacking comprehensive investigation",
                feasibility_score=0.8,
                novelty_score=0.7,
                impact_score=0.6,
                supporting_evidence=literature_gaps[:3],
                proposed_experiments=[
                    "Comparative analysis with existing methods",
                    "Novel approach development",
                    "Empirical validation study"
                ],
                required_resources=["computational resources", "datasets", "evaluation metrics"]
            )
            directions.append(direction1)

        # Direction 2: Trend-based innovation
        trending_topics = web_insights.get("trending_topics", [])
        if trending_topics:
            direction2 = ResearchDirection(
                id=f"rd_{datetime.now().timestamp()}_trend",
                title="Trending Technology Integration",
                description="Integrate emerging trends with established methods",
                hypothesis="Combining trending technologies with domain expertise will yield superior results",
                rationale="Web analysis shows emerging trends that can be leveraged",
                feasibility_score=0.6,
                novelty_score=0.9,
                impact_score=0.8,
                supporting_evidence=trending_topics[:3],
                proposed_experiments=[
                    "Proof-of-concept implementation",
                    "Performance benchmarking",
                    "Scalability analysis"
                ],
                required_resources=["latest tools", "experimental setup", "performance metrics"]
            )
            directions.append(direction2)

        # Direction 3: Methodology improvement
        common_methods = literature_findings.get("trend_analysis", {}).get("common_methodologies", [])
        if common_methods:
            direction3 = ResearchDirection(
                id=f"rd_{datetime.now().timestamp()}_method",
                title="Methodology Enhancement",
                description="Improve upon commonly used methodologies",
                hypothesis="Systematic improvements to existing methods will provide measurable benefits",
                rationale="Literature shows prevalent use of specific methodologies with improvement potential",
                feasibility_score=0.9,
                novelty_score=0.5,
                impact_score=0.7,
                supporting_evidence=common_methods[:3],
                proposed_experiments=[
                    "Baseline reproduction",
                    "Incremental improvements",
                    "Comprehensive evaluation"
                ],
                required_resources=["reference implementations", "standard datasets", "evaluation frameworks"]
            )
            directions.append(direction3)

        return directions

    def _serialize_research_direction(self, rd: ResearchDirection) -> Dict[str, Any]:
        """Convert ResearchDirection to serializable dict"""
        return {
            "id": rd.id,
            "title": rd.title,
            "description": rd.description,
            "hypothesis": rd.hypothesis,
            "rationale": rd.rationale,
            "feasibility_score": rd.feasibility_score,
            "novelty_score": rd.novelty_score,
            "impact_score": rd.impact_score,
            "supporting_evidence": rd.supporting_evidence,
            "proposed_experiments": rd.proposed_experiments,
            "required_resources": rd.required_resources
        }

    async def _create_strategic_plan(
        self,
        research_directions: List[ResearchDirection],
        research_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a strategic research plan"""

        # Prioritize research directions
        prioritized_directions = sorted(
            research_directions,
            key=lambda rd: (rd.feasibility_score + rd.novelty_score + rd.impact_score) / 3,
            reverse=True
        )

        return {
            "total_directions": len(research_directions),
            "prioritized_order": [rd.id for rd in prioritized_directions],
            "recommended_parallel_tracks": min(2, len(research_directions)),
            "estimated_timeline": {
                "planning_phase": "2 weeks",
                "implementation_phase": "8-12 weeks",
                "evaluation_phase": "4 weeks",
                "total_duration": "14-18 weeks"
            },
            "resource_requirements": {
                "computational": "moderate to high",
                "human_effort": "1-2 researchers",
                "datasets": "domain-specific",
                "tools": "standard ML/AI toolchain"
            },
            "success_metrics": [
                "Novel contribution validation",
                "Performance improvement measurement",
                "Reproducibility confirmation",
                "Impact assessment"
            ]
        }

    async def _assess_research_risks(self, research_directions: List[ResearchDirection]) -> Dict[str, Any]:
        """Assess risks for research directions"""

        risk_factors = {
            "technical_risks": [
                "Implementation complexity higher than expected",
                "Required datasets not available",
                "Computational requirements exceed capacity",
                "Baseline methods difficult to reproduce"
            ],
            "methodological_risks": [
                "Evaluation metrics insufficient",
                "Comparison fairness concerns",
                "Statistical significance challenges",
                "Generalization validity questions"
            ],
            "timeline_risks": [
                "Longer implementation time than estimated",
                "Debugging and troubleshooting delays",
                "Iterative refinement needs",
                "External dependency delays"
            ],
            "impact_risks": [
                "Limited novelty upon completion",
                "Concurrent work by other researchers",
                "Results not meeting expectations",
                "Limited practical applicability"
            ]
        }

        # Calculate overall risk score
        high_risk_directions = [rd for rd in research_directions if rd.feasibility_score < 0.7]
        overall_risk = "high" if len(high_risk_directions) > len(research_directions) / 2 else "moderate"

        return {
            "overall_risk_level": overall_risk,
            "high_risk_directions": [rd.id for rd in high_risk_directions],
            "risk_categories": risk_factors,
            "mitigation_strategies": [
                "Incremental development and testing",
                "Early prototype validation",
                "Regular progress checkpoints",
                "Backup approach preparation"
            ]
        }

    async def _recommend_next_steps(self, research_directions: List[ResearchDirection]) -> List[str]:
        """Recommend immediate next steps"""
        return [
            "Select top 1-2 research directions for immediate focus",
            "Develop detailed experiment plans for selected directions",
            "Set up development and evaluation environment",
            "Begin with proof-of-concept implementations",
            "Establish baseline performance measurements",
            "Create detailed timeline and milestone tracking"
        ]


class ExperimentationAgent(BaseSpecializedAgent):
    """Specialized agent for hierarchical experiment design and execution"""

    def __init__(self):
        super().__init__(AgentRole.EXPERIMENT_DESIGNER,
                        ["experiment_design", "hierarchical_decomposition", "execution_planning", "validation"])

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute experimentation task"""
        research_directions = task.input_data.get("research_directions", [])
        selected_direction_id = task.input_data.get("selected_direction_id")

        if selected_direction_id:
            # Focus on specific direction
            selected_direction = next(
                (rd for rd in research_directions if rd.get("id") == selected_direction_id),
                None
            )
            if not selected_direction:
                return {"error": "Selected research direction not found"}

            experiment_tree = await self._design_experiment_tree(selected_direction)
            execution_plan = await self._create_execution_plan(experiment_tree)

            return {
                "agent_role": self.agent_role.value,
                "research_direction_id": selected_direction_id,
                "experiment_tree": experiment_tree,
                "execution_plan": execution_plan,
                "estimated_complexity": await self._estimate_complexity(experiment_tree)
            }
        else:
            # Design experiments for all directions
            all_experiments = {}
            for direction in research_directions:
                direction_id = direction.get("id")
                all_experiments[direction_id] = await self._design_experiment_tree(direction)

            return {
                "agent_role": self.agent_role.value,
                "all_experiment_trees": all_experiments,
                "recommendations": await self._recommend_experiment_priority(all_experiments)
            }

    async def _design_experiment_tree(self, research_direction: Dict[str, Any]) -> Dict[str, Any]:
        """Design hierarchical experiment tree for a research direction"""

        direction_id = research_direction.get("id")
        proposed_experiments = research_direction.get("proposed_experiments", [])

        # Main experiment plan
        main_experiment = ExperimentPlan(
            id=f"exp_main_{direction_id}",
            research_direction_id=direction_id,
            name=f"Main Experiment: {research_direction.get('title', 'Untitled')}",
            description=research_direction.get("description", ""),
            main_hypothesis=research_direction.get("hypothesis", ""),
            depth_level=0,
            experiment_steps=[],
            success_criteria=[
                "Hypothesis validation",
                "Statistical significance achieved",
                "Reproducible results obtained"
            ],
            risk_factors=[
                "Implementation challenges",
                "Data availability issues",
                "Evaluation metric limitations"
            ],
            estimated_duration=8.0
        )

        # Sub-experiments (Level 1)
        sub_experiments = []
        for i, exp_desc in enumerate(proposed_experiments):
            sub_exp = ExperimentPlan(
                id=f"exp_sub_{direction_id}_{i}",
                research_direction_id=direction_id,
                name=f"Sub-experiment: {exp_desc}",
                description=f"Detailed implementation of {exp_desc}",
                main_hypothesis=f"Component hypothesis for {exp_desc}",
                parent_experiment_id=main_experiment.id,
                depth_level=1,
                experiment_steps=await self._generate_experiment_steps(exp_desc),
                success_criteria=[f"Complete {exp_desc} successfully"],
                estimated_duration=2.0
            )
            sub_experiments.append(sub_exp)
            main_experiment.sub_experiments.append(sub_exp.id)

        # Micro-experiments (Level 2) - Detailed steps
        micro_experiments = []
        for sub_exp in sub_experiments:
            for j, step in enumerate(sub_exp.experiment_steps):
                micro_exp = ExperimentPlan(
                    id=f"exp_micro_{sub_exp.id}_{j}",
                    research_direction_id=direction_id,
                    name=f"Micro-step: {step.get('name', f'Step {j+1}')}",
                    description=step.get("description", ""),
                    main_hypothesis=step.get("hypothesis", "Step will complete successfully"),
                    parent_experiment_id=sub_exp.id,
                    depth_level=2,
                    experiment_steps=[step],
                    success_criteria=step.get("success_criteria", ["Step completion"]),
                    estimated_duration=0.5
                )
                micro_experiments.append(micro_exp)
                sub_exp.sub_experiments.append(micro_exp.id)

        return {
            "main_experiment": self._serialize_experiment_plan(main_experiment),
            "sub_experiments": [self._serialize_experiment_plan(se) for se in sub_experiments],
            "micro_experiments": [self._serialize_experiment_plan(me) for me in micro_experiments],
            "total_experiments": 1 + len(sub_experiments) + len(micro_experiments),
            "max_depth": 2,
            "estimated_total_duration": sum([
                main_experiment.estimated_duration,
                sum(se.estimated_duration for se in sub_experiments),
                sum(me.estimated_duration for me in micro_experiments)
            ])
        }

    def _serialize_experiment_plan(self, plan: ExperimentPlan) -> Dict[str, Any]:
        """Convert ExperimentPlan to serializable dict"""
        return {
            "id": plan.id,
            "research_direction_id": plan.research_direction_id,
            "name": plan.name,
            "description": plan.description,
            "main_hypothesis": plan.main_hypothesis,
            "sub_experiments": plan.sub_experiments,
            "parent_experiment_id": plan.parent_experiment_id,
            "depth_level": plan.depth_level,
            "experiment_steps": plan.experiment_steps,
            "success_criteria": plan.success_criteria,
            "risk_factors": plan.risk_factors,
            "estimated_duration": plan.estimated_duration
        }

    async def _generate_experiment_steps(self, experiment_description: str) -> List[Dict[str, Any]]:
        """Generate detailed steps for an experiment"""
        steps = []

        if "comparative analysis" in experiment_description.lower():
            steps = [
                {
                    "name": "Setup baseline methods",
                    "description": "Implement and validate baseline approaches",
                    "hypothesis": "Baseline methods can be faithfully reproduced",
                    "success_criteria": ["Baseline reproduction within 5% of reported performance"],
                    "estimated_time": 0.5
                },
                {
                    "name": "Implement proposed method",
                    "description": "Develop the novel approach",
                    "hypothesis": "Proposed method can be successfully implemented",
                    "success_criteria": ["Method implementation without critical errors"],
                    "estimated_time": 1.0
                },
                {
                    "name": "Conduct comparison",
                    "description": "Run comparative evaluation",
                    "hypothesis": "Proposed method will outperform baseline",
                    "success_criteria": ["Statistical significance in performance difference"],
                    "estimated_time": 0.5
                }
            ]

        elif "development" in experiment_description.lower():
            steps = [
                {
                    "name": "Design methodology",
                    "description": "Create detailed design of new approach",
                    "hypothesis": "Novel methodology will address identified limitations",
                    "success_criteria": ["Complete methodology specification"],
                    "estimated_time": 0.5
                },
                {
                    "name": "Prototype implementation",
                    "description": "Build initial prototype",
                    "hypothesis": "Prototype will demonstrate core functionality",
                    "success_criteria": ["Working prototype with core features"],
                    "estimated_time": 1.0
                },
                {
                    "name": "Iterative refinement",
                    "description": "Refine and optimize approach",
                    "hypothesis": "Iterative improvements will enhance performance",
                    "success_criteria": ["Performance improvement over initial prototype"],
                    "estimated_time": 0.5
                }
            ]

        else:
            # Generic experiment steps
            steps = [
                {
                    "name": "Setup experiment environment",
                    "description": "Prepare experimental setup and data",
                    "hypothesis": "Environment can be properly configured",
                    "success_criteria": ["All dependencies and data ready"],
                    "estimated_time": 0.3
                },
                {
                    "name": "Execute core experiment",
                    "description": "Run the main experimental procedure",
                    "hypothesis": "Experiment will execute without critical failures",
                    "success_criteria": ["Experiment completes and produces results"],
                    "estimated_time": 1.0
                },
                {
                    "name": "Analyze results",
                    "description": "Process and interpret experimental outcomes",
                    "hypothesis": "Results will provide clear insights",
                    "success_criteria": ["Meaningful analysis and conclusions"],
                    "estimated_time": 0.7
                }
            ]

        return steps

    async def _create_execution_plan(self, experiment_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan for experiment tree"""

        # Dependency analysis
        execution_order = []

        # Level 0: Main experiment setup
        main_exp = experiment_tree["main_experiment"]
        execution_order.append({
            "phase": "initialization",
            "experiments": [main_exp["id"]],
            "dependencies": [],
            "estimated_duration": 1.0
        })

        # Level 1: Sub-experiments (can run in parallel)
        sub_experiments = experiment_tree["sub_experiments"]
        if sub_experiments:
            execution_order.append({
                "phase": "sub_experiment_execution",
                "experiments": [se["id"] for se in sub_experiments],
                "dependencies": [main_exp["id"]],
                "estimated_duration": max(se["estimated_duration"] for se in sub_experiments),
                "parallel_execution": True
            })

        # Level 2: Micro-experiments (sequential within each sub-experiment)
        micro_experiments = experiment_tree["micro_experiments"]
        if micro_experiments:
            # Group micro-experiments by parent
            micro_by_parent = {}
            for me in micro_experiments:
                parent_id = me["parent_experiment_id"]
                if parent_id not in micro_by_parent:
                    micro_by_parent[parent_id] = []
                micro_by_parent[parent_id].append(me)

            for parent_id, micros in micro_by_parent.items():
                execution_order.append({
                    "phase": f"micro_experiments_{parent_id}",
                    "experiments": [me["id"] for me in micros],
                    "dependencies": [parent_id],
                    "estimated_duration": sum(me["estimated_duration"] for me in micros),
                    "parallel_execution": False
                })

        return {
            "execution_phases": execution_order,
            "total_phases": len(execution_order),
            "estimated_total_time": experiment_tree["estimated_total_duration"],
            "critical_path": await self._identify_critical_path(experiment_tree),
            "resource_requirements": await self._estimate_resource_requirements(experiment_tree)
        }

    async def _identify_critical_path(self, experiment_tree: Dict[str, Any]) -> List[str]:
        """Identify critical path through experiment tree"""
        # Simple critical path: main -> longest sub-experiment -> its micro-experiments
        main_exp = experiment_tree["main_experiment"]
        sub_experiments = experiment_tree["sub_experiments"]

        if not sub_experiments:
            return [main_exp["id"]]

        # Find longest sub-experiment
        longest_sub = max(sub_experiments, key=lambda se: se["estimated_duration"])

        # Find micro-experiments for this sub-experiment
        micro_experiments = experiment_tree["micro_experiments"]
        related_micros = [me for me in micro_experiments if me["parent_experiment_id"] == longest_sub["id"]]

        critical_path = [main_exp["id"], longest_sub["id"]]
        critical_path.extend([me["id"] for me in related_micros])

        return critical_path

    async def _estimate_resource_requirements(self, experiment_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements for experiment execution"""
        total_experiments = experiment_tree["total_experiments"]

        return {
            "computational_intensity": "high" if total_experiments > 10 else "moderate",
            "parallel_workers_needed": min(4, len(experiment_tree["sub_experiments"])),
            "storage_requirements": "moderate",
            "monitoring_complexity": "high" if experiment_tree["max_depth"] > 1 else "low",
            "estimated_compute_hours": experiment_tree["estimated_total_duration"] * 2  # Buffer factor
        }

    async def _estimate_complexity(self, experiment_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate overall complexity of experiment tree"""
        total_experiments = experiment_tree["total_experiments"]
        max_depth = experiment_tree["max_depth"]

        complexity_score = (total_experiments * 0.1) + (max_depth * 0.3)

        if complexity_score > 2.0:
            complexity_level = "high"
        elif complexity_score > 1.0:
            complexity_level = "moderate"
        else:
            complexity_level = "low"

        return {
            "complexity_level": complexity_level,
            "complexity_score": complexity_score,
            "total_experiments": total_experiments,
            "max_depth": max_depth,
            "estimated_failure_risk": "high" if complexity_score > 1.5 else "moderate"
        }

    async def _recommend_experiment_priority(self, all_experiments: Dict[str, Any]) -> List[str]:
        """Recommend priority order for experiments"""
        experiment_priorities = []

        for direction_id, exp_tree in all_experiments.items():
            complexity = await self._estimate_complexity(exp_tree)
            total_duration = exp_tree["estimated_total_duration"]

            # Priority score: lower complexity and duration = higher priority
            priority_score = 1.0 / (complexity["complexity_score"] + total_duration / 10)

            experiment_priorities.append({
                "direction_id": direction_id,
                "priority_score": priority_score,
                "complexity": complexity["complexity_level"],
                "duration": total_duration
            })

        # Sort by priority score (descending)
        experiment_priorities.sort(key=lambda x: x["priority_score"], reverse=True)

        return [ep["direction_id"] for ep in experiment_priorities]


class ResultsAggregationAgent(BaseSpecializedAgent):
    """Specialized agent for aggregating and synthesizing results from all agents"""

    def __init__(self):
        super().__init__(AgentRole.RESULTS_AGGREGATOR,
                        ["data_aggregation", "synthesis", "insight_extraction", "recommendation_generation"])

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute results aggregation task"""
        agent_results = task.input_data.get("agent_results", {})
        research_context = task.input_data.get("research_context", {})

        # Aggregate findings from all agents
        aggregated_findings = await self._aggregate_agent_findings(agent_results)

        # Synthesize insights
        synthesis = await self._synthesize_insights(aggregated_findings, research_context)

        # Generate final recommendations
        recommendations = await self._generate_final_recommendations(aggregated_findings, synthesis)

        return {
            "agent_role": self.agent_role.value,
            "aggregated_findings": aggregated_findings,
            "synthesis": synthesis,
            "final_recommendations": recommendations,
            "research_readiness_score": await self._calculate_research_readiness(aggregated_findings)
        }

    async def _aggregate_agent_findings(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate findings from all specialized agents"""

        aggregated = {
            "literature_review": {},
            "web_search": {},
            "planning": {},
            "experimentation": {},
            "cross_agent_insights": []
        }

        # Process literature review results
        lit_review = agent_results.get("literature_reviewer", {})
        if lit_review:
            aggregated["literature_review"] = {
                "total_papers": lit_review.get("total_papers_found", 0),
                "key_insights": lit_review.get("key_insights", []),
                "research_gaps": lit_review.get("gap_analysis", {}).get("identified_gaps", []),
                "trending_keywords": lit_review.get("trend_analysis", {}).get("top_keywords", [])[:10]
            }

        # Process web search results
        web_search = agent_results.get("web_searcher", {})
        if web_search:
            aggregated["web_search"] = {
                "total_results": web_search.get("total_results", 0),
                "trending_topics": web_search.get("trending_topics", []),
                "insights": web_search.get("insights", [])
            }

        # Process planning results
        planning = agent_results.get("planning_agent", {})
        if planning:
            aggregated["planning"] = {
                "research_directions": planning.get("research_directions", []),
                "strategic_plan": planning.get("strategic_plan", {}),
                "risk_assessment": planning.get("risk_assessment", {}),
                "next_steps": planning.get("recommended_next_steps", [])
            }

        # Process experimentation results
        experimentation = agent_results.get("experiment_designer", {})
        if experimentation:
            aggregated["experimentation"] = {
                "experiment_trees": experimentation.get("all_experiment_trees", {}),
                "experiment_priorities": experimentation.get("recommendations", []),
                "total_experiments_designed": sum(
                    tree.get("total_experiments", 0)
                    for tree in experimentation.get("all_experiment_trees", {}).values()
                )
            }

        # Generate cross-agent insights
        aggregated["cross_agent_insights"] = await self._generate_cross_agent_insights(agent_results)

        return aggregated

    async def _generate_cross_agent_insights(self, agent_results: Dict[str, Any]) -> List[str]:
        """Generate insights that span multiple agent findings"""
        insights = []

        lit_review = agent_results.get("literature_reviewer", {})
        web_search = agent_results.get("web_searcher", {})
        planning = agent_results.get("planning_agent", {})

        # Literature vs Web consistency
        lit_keywords = set(kw[0] for kw in lit_review.get("trend_analysis", {}).get("top_keywords", [])[:5])
        web_trending = set(web_search.get("trending_topics", [])[:5])

        overlap = lit_keywords.intersection(web_trending)
        if len(overlap) > 2:
            insights.append(f"Strong alignment between academic literature and web trends: {', '.join(list(overlap)[:3])}")
        elif len(overlap) == 0:
            insights.append("Gap between academic literature focus and current web trends - opportunity for bridging work")

        # Research direction feasibility
        num_directions = len(planning.get("research_directions", []))
        if num_directions > 3:
            insights.append("Multiple viable research directions identified - suggests rich research space")
        elif num_directions == 1:
            insights.append("Focused research direction - suggests specialized or niche area")

        # Literature coverage assessment
        lit_papers = lit_review.get("total_papers_found", 0)
        web_results = web_search.get("total_results", 0)

        if lit_papers > 30 and web_results > 50:
            insights.append("Well-established research area with strong academic and practical interest")
        elif lit_papers < 10 and web_results < 20:
            insights.append("Emerging or niche area with limited existing work - high novelty potential")

        return insights

    async def _synthesize_insights(self, aggregated_findings: Dict[str, Any], research_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize insights across all agent findings"""

        # Extract key themes
        key_themes = []

        # From literature review
        lit_insights = aggregated_findings["literature_review"].get("key_insights", [])
        key_themes.extend(lit_insights)

        # From web search
        web_insights = aggregated_findings["web_search"].get("insights", [])
        key_themes.extend(web_insights)

        # From cross-agent analysis
        cross_insights = aggregated_findings["cross_agent_insights"]
        key_themes.extend(cross_insights)

        # Assess research maturity
        research_maturity = await self._assess_research_maturity(aggregated_findings)

        # Identify convergence points
        convergence_points = await self._identify_convergence_points(aggregated_findings)

        # Calculate confidence scores
        confidence_assessment = await self._assess_confidence_levels(aggregated_findings)

        return {
            "key_themes": key_themes,
            "research_maturity": research_maturity,
            "convergence_points": convergence_points,
            "confidence_assessment": confidence_assessment,
            "synthesis_summary": await self._generate_synthesis_summary(aggregated_findings)
        }

    async def _assess_research_maturity(self, aggregated_findings: Dict[str, Any]) -> str:
        """Assess the maturity level of the research area"""

        lit_papers = aggregated_findings["literature_review"].get("total_papers", 0)
        web_results = aggregated_findings["web_search"].get("total_results", 0)
        research_gaps = len(aggregated_findings["literature_review"].get("research_gaps", []))

        if lit_papers > 50 and web_results > 100:
            return "mature"
        elif lit_papers > 20 and web_results > 40:
            return "developing"
        elif lit_papers > 5 and web_results > 10:
            return "emerging"
        else:
            return "nascent"

    async def _identify_convergence_points(self, aggregated_findings: Dict[str, Any]) -> List[str]:
        """Identify points where different agent findings converge"""
        convergence_points = []

        # Check if research directions align with identified gaps
        research_directions = aggregated_findings["planning"].get("research_directions", [])
        research_gaps = aggregated_findings["literature_review"].get("research_gaps", [])

        for direction in research_directions:
            direction_desc = direction.get("description", "").lower()
            for gap in research_gaps:
                if any(word in direction_desc for word in gap.lower().split()[:3]):
                    convergence_points.append(f"Research direction '{direction.get('title', '')}' directly addresses identified literature gap")

        # Check trending topics vs research directions
        trending_topics = aggregated_findings["web_search"].get("trending_topics", [])
        for direction in research_directions:
            direction_title = direction.get("title", "").lower()
            for topic in trending_topics:
                if topic.lower() in direction_title:
                    convergence_points.append(f"Research direction aligns with trending topic: {topic}")

        return convergence_points

    async def _assess_confidence_levels(self, aggregated_findings: Dict[str, Any]) -> Dict[str, float]:
        """Assess confidence levels for different aspects"""

        # Literature coverage confidence
        lit_papers = aggregated_findings["literature_review"].get("total_papers", 0)
        lit_confidence = min(1.0, lit_papers / 30)  # Assume 30 papers is good coverage

        # Web coverage confidence
        web_results = aggregated_findings["web_search"].get("total_results", 0)
        web_confidence = min(1.0, web_results / 50)  # Assume 50 results is good coverage

        # Planning confidence
        num_directions = len(aggregated_findings["planning"].get("research_directions", []))
        planning_confidence = min(1.0, num_directions / 3)  # Assume 3 directions is good

        # Overall confidence
        overall_confidence = (lit_confidence + web_confidence + planning_confidence) / 3

        return {
            "literature_coverage": lit_confidence,
            "web_coverage": web_confidence,
            "planning_quality": planning_confidence,
            "overall_confidence": overall_confidence
        }

    async def _generate_synthesis_summary(self, aggregated_findings: Dict[str, Any]) -> str:
        """Generate a comprehensive synthesis summary"""

        summary_parts = []

        # Literature findings
        lit_papers = aggregated_findings["literature_review"].get("total_papers", 0)
        summary_parts.append(f"Literature analysis identified {lit_papers} relevant papers")

        # Web findings
        web_results = aggregated_findings["web_search"].get("total_results", 0)
        summary_parts.append(f"Web search found {web_results} related online resources")

        # Research directions
        num_directions = len(aggregated_findings["planning"].get("research_directions", []))
        summary_parts.append(f"Strategic analysis identified {num_directions} viable research directions")

        # Experiment design
        total_experiments = aggregated_findings["experimentation"].get("total_experiments_designed", 0)
        if total_experiments > 0:
            summary_parts.append(f"Experimentation framework designed {total_experiments} hierarchical experiments")

        # Cross-agent insights
        cross_insights = len(aggregated_findings["cross_agent_insights"])
        if cross_insights > 0:
            summary_parts.append(f"Cross-agent analysis revealed {cross_insights} additional insights")

        return ". ".join(summary_parts) + "."

    async def _generate_final_recommendations(
        self,
        aggregated_findings: Dict[str, Any],
        synthesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final actionable recommendations"""

        recommendations = {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_vision": [],
            "resource_priorities": [],
            "risk_mitigation": []
        }

        # Immediate actions based on confidence levels
        confidence = synthesis["confidence_assessment"]["overall_confidence"]
        if confidence > 0.7:
            recommendations["immediate_actions"].extend([
                "Begin implementation of highest-priority research direction",
                "Set up experimental environment and baseline measurements",
                "Start detailed literature review of most relevant recent work"
            ])
        else:
            recommendations["immediate_actions"].extend([
                "Conduct additional literature search to improve coverage",
                "Expand web search to gather more contemporary insights",
                "Refine research directions based on additional information"
            ])

        # Short-term goals
        num_directions = len(aggregated_findings["planning"].get("research_directions", []))
        if num_directions > 1:
            recommendations["short_term_goals"].extend([
                "Prototype proof-of-concept for top 2 research directions",
                "Conduct preliminary experiments to validate feasibility",
                "Establish performance baselines and evaluation metrics"
            ])

        # Long-term vision
        research_maturity = synthesis["research_maturity"]
        if research_maturity in ["nascent", "emerging"]:
            recommendations["long_term_vision"].extend([
                "Position work as foundational contribution to emerging field",
                "Develop comprehensive framework for future researchers",
                "Create open-source tools and datasets for community"
            ])
        else:
            recommendations["long_term_vision"].extend([
                "Advance state-of-the-art in established field",
                "Bridge theoretical and practical applications",
                "Influence industry standards and practices"
            ])

        # Resource priorities
        total_experiments = aggregated_findings["experimentation"].get("total_experiments_designed", 0)
        if total_experiments > 15:
            recommendations["resource_priorities"].extend([
                "High-performance computing resources for parallel experiments",
                "Automated experiment management and monitoring tools",
                "Additional research personnel for complex execution"
            ])
        else:
            recommendations["resource_priorities"].extend([
                "Standard computational resources sufficient",
                "Focus on data acquisition and preprocessing tools",
                "Emphasis on evaluation and analysis frameworks"
            ])

        # Risk mitigation based on convergence points
        convergence_points = len(synthesis["convergence_points"])
        if convergence_points < 2:
            recommendations["risk_mitigation"].extend([
                "Validate research direction alignment through expert consultation",
                "Consider broader literature review to identify missed connections",
                "Develop contingency plans for alternative research approaches"
            ])

        return recommendations

    async def _calculate_research_readiness(self, aggregated_findings: Dict[str, Any]) -> float:
        """Calculate overall research readiness score (0-1)"""

        scores = []

        # Literature preparedness (0-0.3)
        lit_papers = aggregated_findings["literature_review"].get("total_papers", 0)
        lit_score = min(0.3, lit_papers / 100)  # Max score at 100 papers
        scores.append(lit_score)

        # Web information coverage (0-0.2)
        web_results = aggregated_findings["web_search"].get("total_results", 0)
        web_score = min(0.2, web_results / 100)  # Max score at 100 results
        scores.append(web_score)

        # Planning completeness (0-0.3)
        num_directions = len(aggregated_findings["planning"].get("research_directions", []))
        planning_score = min(0.3, num_directions / 5 * 0.3)  # Max score at 5 directions
        scores.append(planning_score)

        # Experimentation design (0-0.2)
        total_experiments = aggregated_findings["experimentation"].get("total_experiments_designed", 0)
        exp_score = min(0.2, total_experiments / 20 * 0.2)  # Max score at 20 experiments
        scores.append(exp_score)

        return sum(scores)


class HierarchicalAgentCoordinator:
    """
    Main coordinator that orchestrates all specialized agents in a hierarchical manner
    """

    def __init__(self):
        self.agents = {
            AgentRole.LITERATURE_REVIEWER: LiteratureReviewAgent(),
            AgentRole.WEB_SEARCHER: WebSearchAgent(),
            AgentRole.PLANNING_AGENT: PlanningAgent(),
            AgentRole.EXPERIMENT_DESIGNER: ExperimentationAgent(),
            AgentRole.RESULTS_AGGREGATOR: ResultsAggregationAgent()
        }

        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.research_sessions: Dict[str, Dict[str, Any]] = {}

    async def start_hierarchical_research(
        self,
        research_query: str,
        domain: str = "",
        research_context: Dict[str, Any] = None
    ) -> str:
        """Start a new hierarchical research session"""

        session_id = f"session_{datetime.now().timestamp()}"
        research_context = research_context or {}

        # Initialize research session
        self.research_sessions[session_id] = {
            "session_id": session_id,
            "research_query": research_query,
            "domain": domain,
            "research_context": research_context,
            "phase": "initialization",
            "agent_results": {},
            "final_results": None,
            "created_at": datetime.now(),
            "status": "running"
        }

        # Execute hierarchical research pipeline
        try:
            final_results = await self._execute_hierarchical_pipeline(session_id)

            self.research_sessions[session_id]["final_results"] = final_results
            self.research_sessions[session_id]["status"] = "completed"
            self.research_sessions[session_id]["completed_at"] = datetime.now()

        except Exception as e:
            self.research_sessions[session_id]["status"] = "failed"
            self.research_sessions[session_id]["error"] = str(e)

        return session_id

    async def _execute_hierarchical_pipeline(self, session_id: str) -> Dict[str, Any]:
        """Execute the complete hierarchical research pipeline"""

        session = self.research_sessions[session_id]
        research_query = session["research_query"]
        domain = session["domain"]
        research_context = session["research_context"]

        # Phase 1: Parallel information gathering (Literature + Web Search)
        session["phase"] = "information_gathering"

        # Task 1: Literature Review
        lit_task = AgentTask(
            id=f"lit_{session_id}",
            agent_role=AgentRole.LITERATURE_REVIEWER,
            description="Comprehensive literature review and analysis",
            input_data={
                "research_query": research_query,
                "domain": domain,
                "max_papers": 50
            },
            expected_output={"literature_analysis": "comprehensive"}
        )

        # Task 2: Web Search
        web_task = AgentTask(
            id=f"web_{session_id}",
            agent_role=AgentRole.WEB_SEARCHER,
            description="Web search and trend analysis",
            input_data={
                "search_queries": [
                    research_query,
                    f"{research_query} {domain}",
                    f"{research_query} latest",
                    f"{research_query} 2024"
                ],
                "search_type": "comprehensive",
                "max_results": 20
            },
            expected_output={"web_analysis": "comprehensive"}
        )

        # Execute information gathering tasks in parallel
        lit_result, web_result = await asyncio.gather(
            self.agents[AgentRole.LITERATURE_REVIEWER].process_task(lit_task),
            self.agents[AgentRole.WEB_SEARCHER].process_task(web_task)
        )

        session["agent_results"]["literature_reviewer"] = lit_result
        session["agent_results"]["web_searcher"] = web_result

        # Phase 2: Strategic Planning
        session["phase"] = "strategic_planning"

        planning_task = AgentTask(
            id=f"plan_{session_id}",
            agent_role=AgentRole.PLANNING_AGENT,
            description="Strategic research planning and direction identification",
            input_data={
                "research_context": research_context,
                "literature_findings": lit_result,
                "web_insights": web_result
            },
            expected_output={"research_directions": "multiple", "strategic_plan": "comprehensive"}
        )

        planning_result = await self.agents[AgentRole.PLANNING_AGENT].process_task(planning_task)
        session["agent_results"]["planning_agent"] = planning_result

        # Phase 3: Experiment Design
        session["phase"] = "experiment_design"

        experiment_task = AgentTask(
            id=f"exp_{session_id}",
            agent_role=AgentRole.EXPERIMENT_DESIGNER,
            description="Hierarchical experiment design and decomposition",
            input_data={
                "research_directions": planning_result.get("research_directions", [])
            },
            expected_output={"experiment_trees": "hierarchical", "execution_plans": "detailed"}
        )

        experiment_result = await self.agents[AgentRole.EXPERIMENT_DESIGNER].process_task(experiment_task)
        session["agent_results"]["experiment_designer"] = experiment_result

        # Phase 4: Results Aggregation and Synthesis
        session["phase"] = "results_aggregation"

        aggregation_task = AgentTask(
            id=f"agg_{session_id}",
            agent_role=AgentRole.RESULTS_AGGREGATOR,
            description="Aggregate and synthesize all agent results",
            input_data={
                "agent_results": session["agent_results"],
                "research_context": research_context
            },
            expected_output={"final_synthesis": "comprehensive", "recommendations": "actionable"}
        )

        aggregation_result = await self.agents[AgentRole.RESULTS_AGGREGATOR].process_task(aggregation_task)
        session["agent_results"]["results_aggregator"] = aggregation_result

        # Compile final results
        final_results = {
            "session_id": session_id,
            "research_query": research_query,
            "execution_summary": {
                "total_agents_used": len(session["agent_results"]),
                "information_sources": {
                    "literature_papers": lit_result.get("total_papers_found", 0),
                    "web_resources": web_result.get("total_results", 0)
                },
                "research_directions_identified": len(planning_result.get("research_directions", [])),
                "experiments_designed": experiment_result.get("total_experiments_designed", 0),
                "research_readiness_score": aggregation_result.get("research_readiness_score", 0)
            },
            "phase_results": session["agent_results"],
            "final_synthesis": aggregation_result,
            "execution_time": (datetime.now() - session["created_at"]).total_seconds()
        }

        return final_results

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a research session"""

        session = self.research_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        return {
            "session_id": session_id,
            "status": session.get("status", "unknown"),
            "current_phase": session.get("phase", "unknown"),
            "progress": {
                "agents_completed": len(session.get("agent_results", {})),
                "total_agents": 5
            },
            "created_at": session["created_at"].isoformat(),
            "research_query": session["research_query"],
            "domain": session["domain"]
        }

    async def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get final results of a completed research session"""

        session = self.research_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        if session.get("status") != "completed":
            return {"error": "Session not yet completed", "current_status": session.get("status")}

        return session.get("final_results", {"error": "No results available"})