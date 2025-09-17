"""
AgentLaboratory Integration - Multi-agent collaboration and orchestration
Specialized agents working together in collaborative workflows
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .meta_agent import Agent, AgentRole, Task, TaskType, AgentCapability


class CollaborationPattern(Enum):
    """Collaboration patterns for multi-agent workflows"""
    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    CONSENSUS = "consensus"


class CommunicationProtocol(Enum):
    """Communication protocols between agents"""
    DIRECT_MESSAGE = "direct_message"
    BROADCAST = "broadcast"
    QUERY_RESPONSE = "query_response"
    NEGOTIATION = "negotiation"
    VOTE = "vote"


@dataclass
class AgentMessage:
    """Message between agents"""
    id: str
    sender_id: str
    receiver_id: Optional[str]
    message_type: CommunicationProtocol
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    conversation_id: Optional[str] = None


@dataclass
class CollaborationSession:
    """A collaborative session between multiple agents"""
    id: str
    name: str
    pattern: CollaborationPattern
    participating_agents: List[str]
    coordinator_agent: Optional[str]
    shared_context: Dict[str, Any] = field(default_factory=dict)
    messages: List[AgentMessage] = field(default_factory=list)
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)


class SpecializedAgent:
    """
    Specialized agent implementations from AgentLaboratory
    Each agent has unique capabilities and collaboration patterns
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.collaboration_history: List[str] = []
        self.peer_ratings: Dict[str, float] = {}

    async def send_message(
        self,
        receiver_id: str,
        message_type: CommunicationProtocol,
        content: Dict[str, Any],
        conversation_id: str = None
    ) -> AgentMessage:
        """Send message to another agent"""
        message = AgentMessage(
            id=f"msg_{datetime.now().timestamp()}",
            sender_id=self.agent.id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            conversation_id=conversation_id
        )
        return message

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and optionally respond"""
        response_content = await self._generate_response(message)
        if response_content:
            return await self.send_message(
                receiver_id=message.sender_id,
                message_type=CommunicationProtocol.QUERY_RESPONSE,
                content=response_content,
                conversation_id=message.conversation_id
            )
        return None

    async def _generate_response(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Generate response based on agent role and message content"""
        if self.agent.role == AgentRole.RESEARCHER:
            return await self._researcher_response(message)
        elif self.agent.role == AgentRole.CODER:
            return await self._coder_response(message)
        elif self.agent.role == AgentRole.REVIEWER:
            return await self._reviewer_response(message)
        elif self.agent.role == AgentRole.TESTER:
            return await self._tester_response(message)
        return None

    async def _researcher_response(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Researcher agent response logic"""
        content = message.content

        if message.message_type == CommunicationProtocol.QUERY_RESPONSE:
            if "research_query" in content:
                return {
                    "type": "research_findings",
                    "query": content["research_query"],
                    "findings": f"Research findings for: {content['research_query']}",
                    "sources": ["source1", "source2"],
                    "confidence": 0.85
                }

        return None

    async def _coder_response(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Coder agent response logic"""
        content = message.content

        if "code_request" in content:
            return {
                "type": "code_solution",
                "request": content["code_request"],
                "code": f"# Generated code for: {content['code_request']}\npass",
                "language": "python",
                "documentation": "Auto-generated solution"
            }

        return None

    async def _reviewer_response(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Reviewer agent response logic"""
        content = message.content

        if "review_request" in content:
            return {
                "type": "review_feedback",
                "item_reviewed": content["review_request"],
                "feedback": "Review feedback and suggestions",
                "rating": 4.2,
                "suggestions": ["suggestion1", "suggestion2"]
            }

        return None

    async def _tester_response(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Tester agent response logic"""
        content = message.content

        if "test_request" in content:
            return {
                "type": "test_results",
                "tested_item": content["test_request"],
                "passed": True,
                "test_cases": 5,
                "coverage": 0.92,
                "issues": []
            }

        return None


class AgentLaboratory:
    """
    Main coordination system for multi-agent collaboration
    Manages agent interactions, workflows, and collaborative patterns
    """

    def __init__(self):
        self.specialized_agents: Dict[str, SpecializedAgent] = {}
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}
        self.message_history: List[AgentMessage] = []
        self.workflow_templates: Dict[str, Dict] = self._initialize_workflow_templates()

    def _initialize_workflow_templates(self) -> Dict[str, Dict]:
        """Initialize common collaborative workflow templates"""
        return {
            "code_development": {
                "pattern": CollaborationPattern.PIPELINE,
                "steps": [
                    {"role": AgentRole.RESEARCHER, "task": "requirements_analysis"},
                    {"role": AgentRole.CODER, "task": "implementation"},
                    {"role": AgentRole.REVIEWER, "task": "code_review"},
                    {"role": AgentRole.TESTER, "task": "testing"}
                ]
            },
            "research_synthesis": {
                "pattern": CollaborationPattern.PARALLEL,
                "steps": [
                    {"role": AgentRole.RESEARCHER, "task": "literature_review"},
                    {"role": AgentRole.ANALYZER, "task": "data_analysis"},
                    {"role": AgentRole.SYNTHESIZER, "task": "synthesis"}
                ]
            },
            "consensus_decision": {
                "pattern": CollaborationPattern.CONSENSUS,
                "steps": [
                    {"role": "all", "task": "initial_proposals"},
                    {"role": "all", "task": "discussion"},
                    {"role": "all", "task": "voting"},
                    {"role": AgentRole.SYNTHESIZER, "task": "final_decision"}
                ]
            }
        }

    def register_agent(self, agent: Agent) -> str:
        """Register a specialized agent"""
        specialized_agent = SpecializedAgent(agent)
        self.specialized_agents[agent.id] = specialized_agent
        return agent.id

    async def create_collaboration_session(
        self,
        name: str,
        pattern: CollaborationPattern,
        agent_roles: List[AgentRole],
        coordinator_role: AgentRole = None
    ) -> str:
        """Create a new collaboration session"""
        session_id = f"collab_{datetime.now().timestamp()}"

        # Find agents with required roles
        participating_agents = []
        coordinator_agent = None

        for agent_id, spec_agent in self.specialized_agents.items():
            if spec_agent.agent.role in agent_roles:
                participating_agents.append(agent_id)
                if coordinator_role and spec_agent.agent.role == coordinator_role:
                    coordinator_agent = agent_id

        session = CollaborationSession(
            id=session_id,
            name=name,
            pattern=pattern,
            participating_agents=participating_agents,
            coordinator_agent=coordinator_agent
        )

        self.collaboration_sessions[session_id] = session
        return session_id

    async def execute_collaborative_task(
        self,
        session_id: str,
        task: Task,
        workflow_template: str = None
    ) -> Dict[str, Any]:
        """Execute a task using collaborative pattern"""
        session = self.collaboration_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        if workflow_template and workflow_template in self.workflow_templates:
            return await self._execute_template_workflow(session, task, workflow_template)
        else:
            return await self._execute_pattern_workflow(session, task)

    async def _execute_template_workflow(
        self,
        session: CollaborationSession,
        task: Task,
        template_name: str
    ) -> Dict[str, Any]:
        """Execute workflow based on predefined template"""
        template = self.workflow_templates[template_name]
        results = {}

        if template["pattern"] == CollaborationPattern.PIPELINE:
            # Sequential execution
            current_result = {"initial_context": task.context}

            for step in template["steps"]:
                required_role = step["role"]
                step_task = step["task"]

                agent_id = self._find_agent_by_role(session, required_role)
                if agent_id:
                    step_result = await self._execute_agent_step(
                        agent_id, step_task, current_result
                    )
                    results[step_task] = step_result
                    current_result.update(step_result)

        elif template["pattern"] == CollaborationPattern.PARALLEL:
            # Parallel execution
            step_tasks = []
            for step in template["steps"]:
                agent_id = self._find_agent_by_role(session, step["role"])
                if agent_id:
                    step_tasks.append(
                        self._execute_agent_step(agent_id, step["task"], task.context)
                    )

            step_results = await asyncio.gather(*step_tasks)
            for i, step in enumerate(template["steps"]):
                results[step["task"]] = step_results[i]

        return results

    async def _execute_pattern_workflow(
        self,
        session: CollaborationSession,
        task: Task
    ) -> Dict[str, Any]:
        """Execute workflow based on collaboration pattern"""
        if session.pattern == CollaborationPattern.CONSENSUS:
            return await self._execute_consensus_workflow(session, task)
        elif session.pattern == CollaborationPattern.PEER_TO_PEER:
            return await self._execute_peer_to_peer_workflow(session, task)
        else:
            return await self._execute_hierarchical_workflow(session, task)

    async def _execute_consensus_workflow(
        self,
        session: CollaborationSession,
        task: Task
    ) -> Dict[str, Any]:
        """Execute consensus-based decision making"""
        proposals = {}

        # Phase 1: Get proposals from all agents
        for agent_id in session.participating_agents:
            proposal = await self._get_agent_proposal(agent_id, task)
            proposals[agent_id] = proposal

        # Phase 2: Discussion and refinement
        discussion_rounds = 2
        for round_num in range(discussion_rounds):
            refined_proposals = {}
            for agent_id in session.participating_agents:
                refined = await self._refine_proposal(
                    agent_id, proposals[agent_id], proposals
                )
                refined_proposals[agent_id] = refined
            proposals = refined_proposals

        # Phase 3: Voting
        votes = {}
        for voter_id in session.participating_agents:
            vote = await self._cast_vote(voter_id, proposals)
            votes[voter_id] = vote

        # Phase 4: Final decision
        final_decision = await self._determine_consensus(proposals, votes)

        return {
            "consensus_result": final_decision,
            "proposals": proposals,
            "votes": votes,
            "participants": session.participating_agents
        }

    async def _execute_peer_to_peer_workflow(
        self,
        session: CollaborationSession,
        task: Task
    ) -> Dict[str, Any]:
        """Execute peer-to-peer collaborative workflow"""
        # Create conversation threads between agents
        conversations = {}

        for i, agent1_id in enumerate(session.participating_agents):
            for agent2_id in session.participating_agents[i+1:]:
                conv_id = f"conv_{agent1_id}_{agent2_id}"
                conversation = await self._initiate_conversation(
                    agent1_id, agent2_id, task, conv_id
                )
                conversations[conv_id] = conversation

        # Synthesize results from all conversations
        synthesis = await self._synthesize_conversations(conversations)

        return {
            "peer_collaboration_result": synthesis,
            "conversations": conversations,
            "participants": session.participating_agents
        }

    async def _execute_hierarchical_workflow(
        self,
        session: CollaborationSession,
        task: Task
    ) -> Dict[str, Any]:
        """Execute hierarchical workflow with coordinator"""
        coordinator_id = session.coordinator_agent
        if not coordinator_id:
            coordinator_id = session.participating_agents[0]

        # Coordinator distributes subtasks
        subtasks = await self._coordinator_decompose_task(coordinator_id, task)

        # Execute subtasks
        subtask_results = {}
        for subtask_id, subtask_info in subtasks.items():
            assigned_agent = subtask_info["assigned_agent"]
            result = await self._execute_agent_step(
                assigned_agent, subtask_info["description"], subtask_info["context"]
            )
            subtask_results[subtask_id] = result

        # Coordinator synthesizes results
        final_result = await self._coordinator_synthesize(
            coordinator_id, subtask_results
        )

        return {
            "hierarchical_result": final_result,
            "subtask_results": subtask_results,
            "coordinator": coordinator_id
        }

    def _find_agent_by_role(self, session: CollaborationSession, role: AgentRole) -> Optional[str]:
        """Find agent in session with specific role"""
        for agent_id in session.participating_agents:
            if self.specialized_agents[agent_id].agent.role == role:
                return agent_id
        return None

    async def _execute_agent_step(
        self,
        agent_id: str,
        task_description: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single step by an agent"""
        # Simulate agent execution
        await asyncio.sleep(0.1)

        agent = self.specialized_agents[agent_id]
        return {
            "agent_id": agent_id,
            "role": agent.agent.role.value,
            "task": task_description,
            "result": f"Completed {task_description}",
            "context_used": context,
            "timestamp": datetime.now().isoformat()
        }

    async def _get_agent_proposal(self, agent_id: str, task: Task) -> Dict[str, Any]:
        """Get proposal from agent for consensus workflow"""
        return {
            "agent_id": agent_id,
            "proposal": f"Proposal from {agent_id} for {task.name}",
            "reasoning": "Agent reasoning for this proposal",
            "confidence": 0.8
        }

    async def _cast_vote(self, voter_id: str, proposals: Dict[str, Any]) -> Dict[str, Any]:
        """Cast vote in consensus workflow"""
        # Simulate voting logic
        proposal_scores = {
            agent_id: 0.7 + (hash(f"{voter_id}{agent_id}") % 30) / 100
            for agent_id in proposals.keys()
        }

        best_proposal = max(proposal_scores.keys(), key=lambda x: proposal_scores[x])

        return {
            "voter_id": voter_id,
            "voted_for": best_proposal,
            "scores": proposal_scores
        }

    async def _determine_consensus(
        self,
        proposals: Dict[str, Any],
        votes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine final consensus decision"""
        vote_counts = {}
        for vote in votes.values():
            voted_for = vote["voted_for"]
            vote_counts[voted_for] = vote_counts.get(voted_for, 0) + 1

        winner = max(vote_counts.keys(), key=lambda x: vote_counts[x])

        return {
            "consensus_choice": winner,
            "vote_counts": vote_counts,
            "winning_proposal": proposals[winner],
            "consensus_strength": vote_counts[winner] / len(votes)
        }

    async def get_collaboration_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get metrics for collaboration session"""
        session = self.collaboration_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        return {
            "session_id": session_id,
            "participants": len(session.participating_agents),
            "messages_exchanged": len([m for m in self.message_history if m.conversation_id == session_id]),
            "session_duration": (datetime.now() - session.created_at).total_seconds(),
            "collaboration_effectiveness": await self._calculate_effectiveness(session)
        }

    async def _calculate_effectiveness(self, session: CollaborationSession) -> float:
        """Calculate collaboration effectiveness score"""
        # Simple effectiveness metric based on agent participation
        base_score = 0.5

        # Bonus for diverse agent roles
        unique_roles = len(set(
            self.specialized_agents[agent_id].agent.role
            for agent_id in session.participating_agents
        ))
        role_bonus = min(unique_roles * 0.1, 0.3)

        # Bonus for message exchange
        message_count = len([
            m for m in self.message_history
            if any(agent_id in [m.sender_id, m.receiver_id] for agent_id in session.participating_agents)
        ])
        message_bonus = min(message_count * 0.02, 0.2)

        return base_score + role_bonus + message_bonus