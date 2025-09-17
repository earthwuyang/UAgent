"""
Core Meta-Agent Framework - ROMA-style recursive agent orchestration
Combines the best of ROMA, AI-Scientist, AgentLaboratory, and RepoMaster
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from pydantic import BaseModel


class AgentRole(Enum):
    """Agent roles from AgentLaboratory pattern"""
    META_ORCHESTRATOR = "meta_orchestrator"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"


class TaskType(Enum):
    """Task types inspired by AI-Scientist and ROMA"""
    RESEARCH = "research"
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    EXPERIMENTATION = "experimentation"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Task:
    """Unified task representation"""
    id: str
    name: str
    description: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_agent: Optional[str] = None


@dataclass
class Agent:
    """Base agent with ROMA-style self-reflection capabilities"""
    id: str
    name: str
    role: AgentRole
    capabilities: List[AgentCapability]
    status: str = "idle"
    current_task: Optional[str] = None
    memory: Dict[str, Any] = field(default_factory=dict)
    parent_agent: Optional[str] = None
    child_agents: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class AgentInterface(ABC):
    """Abstract interface for all agents"""

    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a given task"""
        pass

    @abstractmethod
    async def self_reflect(self) -> Dict[str, Any]:
        """ROMA-style self-reflection"""
        pass

    @abstractmethod
    async def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle the task"""
        pass


class MetaAgent:
    """
    Core Meta-Agent with ROMA-style recursive orchestration
    Manages other agents and performs hierarchical task decomposition
    """

    def __init__(self, agent_id: str = None):
        self.id = agent_id or str(uuid.uuid4())
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        self.active_workflows: Dict[str, Dict] = {}

        # Initialize with core agent roles
        self._initialize_core_agents()

    def _initialize_core_agents(self):
        """Initialize core agents based on AgentLaboratory pattern"""
        core_agents = [
            Agent(
                id=str(uuid.uuid4()),
                name="Researcher",
                role=AgentRole.RESEARCHER,
                capabilities=[
                    AgentCapability(
                        name="literature_search",
                        description="Search and analyze research papers",
                        input_types=["query", "domain"],
                        output_types=["papers", "insights"]
                    ),
                    AgentCapability(
                        name="hypothesis_generation",
                        description="Generate research hypotheses",
                        input_types=["context", "data"],
                        output_types=["hypotheses"]
                    )
                ]
            ),
            Agent(
                id=str(uuid.uuid4()),
                name="Coder",
                role=AgentRole.CODER,
                capabilities=[
                    AgentCapability(
                        name="code_generation",
                        description="Generate code from specifications",
                        input_types=["requirements", "context"],
                        output_types=["code", "documentation"]
                    ),
                    AgentCapability(
                        name="code_refactoring",
                        description="Refactor and optimize code",
                        input_types=["code", "requirements"],
                        output_types=["refactored_code"]
                    )
                ]
            ),
            Agent(
                id=str(uuid.uuid4()),
                name="Analyzer",
                role=AgentRole.ANALYZER,
                capabilities=[
                    AgentCapability(
                        name="semantic_analysis",
                        description="Deep semantic code analysis (RepoMaster-style)",
                        input_types=["code", "repository"],
                        output_types=["analysis", "insights"]
                    ),
                    AgentCapability(
                        name="pattern_detection",
                        description="Detect patterns and anomalies",
                        input_types=["data", "code"],
                        output_types=["patterns", "recommendations"]
                    )
                ]
            )
        ]

        for agent in core_agents:
            self.agents[agent.id] = agent

    async def create_task(
        self,
        name: str,
        description: str,
        task_type: TaskType,
        context: Dict[str, Any] = None,
        parent_task_id: str = None,
        priority: int = 1
    ) -> str:
        """Create a new task with ROMA-style decomposition"""
        task_id = str(uuid.uuid4())

        task = Task(
            id=task_id,
            name=name,
            description=description,
            task_type=task_type,
            context=context or {},
            parent_task_id=parent_task_id,
            priority=priority
        )

        self.tasks[task_id] = task

        # ROMA-style recursive decomposition
        if self._should_decompose_task(task):
            subtasks = await self._decompose_task(task)
            task.subtasks = subtasks
        else:
            self.task_queue.append(task_id)

        return task_id

    def _should_decompose_task(self, task: Task) -> bool:
        """Determine if task should be decomposed further (ROMA pattern)"""
        complex_types = [TaskType.RESEARCH, TaskType.EXPERIMENTATION]
        return task.task_type in complex_types and not task.parent_task_id

    async def _decompose_task(self, task: Task) -> List[str]:
        """Decompose complex task into subtasks (ROMA-style)"""
        subtasks = []

        if task.task_type == TaskType.RESEARCH:
            # AI-Scientist style research decomposition
            research_steps = [
                ("Literature Review", "Review existing research and papers"),
                ("Hypothesis Generation", "Generate testable hypotheses"),
                ("Experimental Design", "Design experiments to test hypotheses"),
                ("Analysis", "Analyze experimental results"),
                ("Synthesis", "Synthesize findings into insights")
            ]

            for step_name, step_desc in research_steps:
                subtask_id = await self.create_task(
                    name=step_name,
                    description=step_desc,
                    task_type=TaskType.RESEARCH,
                    parent_task_id=task.id,
                    context=task.context
                )
                subtasks.append(subtask_id)

        elif task.task_type == TaskType.CODE_ANALYSIS:
            # RepoMaster style code analysis decomposition
            analysis_steps = [
                ("Structure Analysis", "Analyze repository structure and dependencies"),
                ("Semantic Analysis", "Deep semantic understanding of code"),
                ("Pattern Detection", "Identify patterns and code smells"),
                ("Documentation Analysis", "Analyze and extract documentation")
            ]

            for step_name, step_desc in analysis_steps:
                subtask_id = await self.create_task(
                    name=step_name,
                    description=step_desc,
                    task_type=TaskType.CODE_ANALYSIS,
                    parent_task_id=task.id,
                    context=task.context
                )
                subtasks.append(subtask_id)

        return subtasks

    async def assign_task(self, task_id: str) -> Optional[str]:
        """Assign task to most suitable agent (AgentLaboratory pattern)"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        best_agent = None
        best_score = 0

        for agent in self.agents.values():
            if agent.status != "idle":
                continue

            score = await self._calculate_agent_suitability(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent

        if best_agent:
            task.assigned_agent = best_agent.id
            best_agent.current_task = task_id
            best_agent.status = "working"
            return best_agent.id

        return None

    async def _calculate_agent_suitability(self, agent: Agent, task: Task) -> float:
        """Calculate how suitable an agent is for a task"""
        score = 0.0

        # Role matching
        role_mapping = {
            TaskType.RESEARCH: [AgentRole.RESEARCHER],
            TaskType.CODE_ANALYSIS: [AgentRole.ANALYZER, AgentRole.CODER],
            TaskType.CODE_GENERATION: [AgentRole.CODER],
            TaskType.VALIDATION: [AgentRole.TESTER, AgentRole.REVIEWER]
        }

        if agent.role in role_mapping.get(task.task_type, []):
            score += 1.0

        # Capability matching
        for capability in agent.capabilities:
            if any(req in capability.name for req in task.required_capabilities):
                score += 0.5

        return score

    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute complete workflow with ROMA-style orchestration"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return {"error": "Workflow not found"}

        results = {}

        # Process tasks in priority order
        while self.task_queue:
            self.task_queue.sort(key=lambda tid: self.tasks[tid].priority, reverse=True)
            task_id = self.task_queue.pop(0)

            agent_id = await self.assign_task(task_id)
            if agent_id:
                result = await self._execute_task(task_id, agent_id)
                results[task_id] = result

                # ROMA-style result propagation to parent
                await self._propagate_results(task_id, result)

        return results

    async def _execute_task(self, task_id: str, agent_id: str) -> Dict[str, Any]:
        """Execute individual task"""
        task = self.tasks[task_id]
        agent = self.agents[agent_id]

        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()

        try:
            # Simulate task execution based on type
            result = await self._simulate_task_execution(task, agent)

            task.status = TaskStatus.COMPLETED
            task.results = result

            # Free up agent
            agent.status = "idle"
            agent.current_task = None

            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.results = {"error": str(e)}
            agent.status = "idle"
            agent.current_task = None
            return {"error": str(e)}

    async def _simulate_task_execution(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Simulate task execution (to be replaced with actual implementations)"""
        await asyncio.sleep(0.1)  # Simulate work

        return {
            "task_id": task.id,
            "agent_id": agent.id,
            "result": f"Completed {task.name} using {agent.role.value}",
            "timestamp": datetime.now().isoformat()
        }

    async def _propagate_results(self, task_id: str, result: Dict[str, Any]):
        """Propagate results to parent task (ROMA pattern)"""
        task = self.tasks[task_id]
        if not task.parent_task_id:
            return

        parent_task = self.tasks[task.parent_task_id]
        if not parent_task.results:
            parent_task.results = {}

        parent_task.results[task_id] = result

        # Check if all subtasks are complete
        all_complete = all(
            self.tasks[subtask_id].status == TaskStatus.COMPLETED
            for subtask_id in parent_task.subtasks
        )

        if all_complete:
            # Synthesize subtask results
            synthesized = await self._synthesize_results(parent_task)
            parent_task.results["synthesized"] = synthesized
            parent_task.status = TaskStatus.COMPLETED

    async def _synthesize_results(self, parent_task: Task) -> Dict[str, Any]:
        """Synthesize results from subtasks"""
        return {
            "summary": f"Completed {len(parent_task.subtasks)} subtasks",
            "subtask_results": [
                self.tasks[subtask_id].results
                for subtask_id in parent_task.subtasks
            ]
        }

    async def self_reflect(self) -> Dict[str, Any]:
        """ROMA-style meta-agent self-reflection"""
        return {
            "active_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "agent_utilization": {
                agent.id: agent.status for agent in self.agents.values()
            },
            "recommendations": await self._generate_recommendations()
        }

    async def _generate_recommendations(self) -> List[str]:
        """Generate self-improvement recommendations"""
        recommendations = []

        idle_agents = [a for a in self.agents.values() if a.status == "idle"]
        if len(idle_agents) > len(self.agents) * 0.5:
            recommendations.append("Consider consolidating idle agents or creating more tasks")

        failed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
        if failed_tasks:
            recommendations.append(f"Investigate {len(failed_tasks)} failed tasks")

        return recommendations