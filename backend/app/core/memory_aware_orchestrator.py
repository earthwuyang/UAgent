"""
Memory-Aware Orchestrator for UAgent
Integrates memory management into the unified orchestrator workflow
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .unified_orchestrator import UnifiedOrchestrator, WorkflowConfig, WorkflowResult, WorkflowType
from .memory_service import MemoryService, MemoryEvent, MemoryEventKind, MemorySnapshotType
from .llm_client import QwenLLMClient

logger = logging.getLogger(__name__)


class MemoryAwareOrchestrator:
    """
    Enhanced orchestrator that integrates memory management throughout the workflow
    """

    def __init__(self):
        # Initialize core components
        self.orchestrator = UnifiedOrchestrator()
        self.memory_service = MemoryService()
        self.llm_client = QwenLLMClient()

        # Memory-enhanced event listeners
        self.orchestrator.add_event_listener(self._handle_workflow_event)

        logger.info("Memory-aware orchestrator initialized")

    async def _handle_workflow_event(self, event: Dict[str, Any]):
        """Handle workflow events and log them to memory"""
        try:
            event_type = event.get("event")
            workflow_id = event.get("workflow_id")
            goal_id = event.get("goal_id", workflow_id)  # Use workflow_id as fallback
            node_id = event.get("node_id", "orchestrator")  # Default node

            if not goal_id:
                logger.debug("Skipping memory logging for event without goal_id")
                return

            # Determine event importance based on type
            importance = self._calculate_event_importance(event_type, event)

            # Log workflow event to memory
            await self.memory_service.log_event(
                run_id=goal_id,
                node_id=node_id,
                source="orchestrator",
                kind=self._map_event_to_memory_kind(event_type),
                body=event,
                importance=importance,
                maybe_artifact=len(str(event)) > 1000  # Offload large events
            )

            # Record statistics
            await self.memory_service.record_memory_stats(
                run_id=goal_id,
                stat_type="workflow_event",
                stat_data={
                    "event_type": event_type,
                    "timestamp": datetime.now().isoformat(),
                    "event_size": len(str(event))
                },
                node_id=node_id
            )

        except Exception as e:
            logger.error(f"Failed to handle workflow event in memory: {e}")

    def _calculate_event_importance(self, event_type: str, event: Dict[str, Any]) -> float:
        """Calculate importance score for different event types"""
        importance_map = {
            "workflow_started": 0.8,
            "workflow_completed": 0.9,
            "workflow_failed": 0.9,
            "ai_scientist_project_created": 0.7,
            "research_phase_completed": 0.6,
            "collaboration_session_started": 0.5,
            "collaborative_task_completed": 0.6,
            "repository_analyzed": 0.7,
            "error": 0.8,
            "decision": 0.7
        }

        base_importance = importance_map.get(event_type, 0.3)

        # Boost importance for errors or failures
        if "error" in event or "failed" in event_type:
            base_importance = min(base_importance + 0.2, 1.0)

        # Boost importance for successful completions
        if "completed" in event_type and not event.get("error"):
            base_importance = min(base_importance + 0.1, 1.0)

        return base_importance

    def _map_event_to_memory_kind(self, event_type: str) -> MemoryEventKind:
        """Map workflow event types to memory event kinds"""
        if "error" in event_type or "failed" in event_type:
            return MemoryEventKind.ERROR
        elif "completed" in event_type or "ready" in event_type:
            return MemoryEventKind.OBSERVATION
        elif "started" in event_type or "created" in event_type:
            return MemoryEventKind.DECISION
        else:
            return MemoryEventKind.NOTE

    async def execute_memory_enhanced_workflow(
        self,
        workflow_config: Union[str, WorkflowConfig],
        inputs: Dict[str, Any] = None
    ) -> str:
        """Execute workflow with enhanced memory context injection"""
        inputs = inputs or {}

        # Inject memory context if available
        if "goal_id" in inputs:
            goal_id = inputs["goal_id"]
            node_id = inputs.get("node_id", "root")

            try:
                # Assemble memory context
                query_terms = self._extract_query_terms(inputs)
                context = await self.memory_service.assemble_context(
                    run_id=goal_id,
                    node_id=node_id,
                    query_terms=query_terms
                )

                # Inject context into inputs
                inputs["memory_context"] = context
                inputs["memory_enhanced"] = True

                logger.info(f"Injected memory context for {goal_id}/{node_id}: {context['total_tokens_used']} tokens")

            except Exception as e:
                logger.error(f"Failed to inject memory context: {e}")
                inputs["memory_enhanced"] = False

        # Execute workflow with memory logging
        workflow_id = await self.orchestrator.execute_workflow(workflow_config, inputs)

        return workflow_id

    def _extract_query_terms(self, inputs: Dict[str, Any]) -> List[str]:
        """Extract relevant query terms from workflow inputs"""
        terms = []

        # Extract from common input fields
        for field in ["title", "description", "query", "research_questions"]:
            value = inputs.get(field)
            if isinstance(value, str):
                terms.extend(value.split())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        terms.extend(item.split())

        # Filter and clean terms
        filtered_terms = []
        for term in terms:
            cleaned = term.strip().lower()
            if len(cleaned) > 2 and cleaned.isalpha():  # Only alphabetic terms > 2 chars
                filtered_terms.append(cleaned)

        return list(set(filtered_terms))[:10]  # Unique terms, max 10

    async def log_tool_interaction(
        self,
        run_id: str,
        node_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        success: bool,
        execution_time: float
    ):
        """Log tool interactions to memory"""
        try:
            # Log tool call
            await self.memory_service.log_event(
                run_id=run_id,
                node_id=node_id,
                source=f"tool:{tool_name}",
                kind=MemoryEventKind.TOOL_CALL,
                body={
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "execution_time": execution_time
                },
                importance=0.5
            )

            # Log tool result
            result_importance = 0.7 if success else 0.8  # Failures are more important
            await self.memory_service.log_event(
                run_id=run_id,
                node_id=node_id,
                source=f"tool:{tool_name}",
                kind=MemoryEventKind.TOOL_RESULT,
                body={
                    "tool_name": tool_name,
                    "success": success,
                    "result": result,
                    "execution_time": execution_time
                },
                importance=result_importance,
                maybe_artifact=True  # Tool results can be large
            )

        except Exception as e:
            logger.error(f"Failed to log tool interaction: {e}")

    async def log_decision(
        self,
        run_id: str,
        node_id: str,
        decision: str,
        reasoning: str,
        context: Dict[str, Any] = None,
        importance: float = 0.7
    ):
        """Log decision-making to memory"""
        try:
            await self.memory_service.log_event(
                run_id=run_id,
                node_id=node_id,
                source="agent",
                kind=MemoryEventKind.DECISION,
                body={
                    "decision": decision,
                    "reasoning": reasoning,
                    "context": context or {},
                    "timestamp": datetime.now().isoformat()
                },
                importance=importance
            )

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

    async def log_observation(
        self,
        run_id: str,
        node_id: str,
        observation: str,
        data: Dict[str, Any] = None,
        importance: float = 0.4
    ):
        """Log observations to memory"""
        try:
            await self.memory_service.log_event(
                run_id=run_id,
                node_id=node_id,
                source="agent",
                kind=MemoryEventKind.OBSERVATION,
                body={
                    "observation": observation,
                    "data": data or {},
                    "timestamp": datetime.now().isoformat()
                },
                importance=importance
            )

        except Exception as e:
            logger.error(f"Failed to log observation: {e}")

    async def log_error(
        self,
        run_id: str,
        node_id: str,
        error: str,
        stack_trace: str = None,
        context: Dict[str, Any] = None
    ):
        """Log errors to memory with high importance"""
        try:
            await self.memory_service.log_event(
                run_id=run_id,
                node_id=node_id,
                source="system",
                kind=MemoryEventKind.ERROR,
                body={
                    "error": error,
                    "stack_trace": stack_trace,
                    "context": context or {},
                    "timestamp": datetime.now().isoformat()
                },
                importance=0.9  # Errors are very important
            )

        except Exception as e:
            logger.error(f"Failed to log error to memory: {e}")

    async def create_milestone_summary(
        self,
        run_id: str,
        node_id: str,
        milestone_name: str,
        achievements: List[str],
        next_steps: List[str]
    ) -> int:
        """Create a milestone summary snapshot"""
        try:
            # Get recent events for summarization
            events = await self.memory_service.get_events_since_last_snapshot(run_id, node_id)

            if not events:
                logger.warning(f"No events found for milestone summary: {run_id}/{node_id}")
                return -1

            # Create milestone summary
            summary_id = await self.memory_service.create_rolling_summary(
                run_id, node_id, MemorySnapshotType.MILESTONE
            )

            # Promote key achievements to semantic memory
            for achievement in achievements:
                await self.memory_service.promote_to_semantic_memory(
                    text=achievement,
                    title=f"Achievement: {milestone_name}",
                    meta={
                        "type": "achievement",
                        "milestone": milestone_name,
                        "run_id": run_id,
                        "node_id": node_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            logger.info(f"Created milestone summary {summary_id} for {milestone_name}")
            return summary_id

        except Exception as e:
            logger.error(f"Failed to create milestone summary: {e}")
            return -1

    async def create_postmortem(
        self,
        run_id: str,
        node_id: str,
        final_results: Dict[str, Any],
        lessons_learned: List[str]
    ) -> int:
        """Create a comprehensive postmortem summary"""
        try:
            # Create postmortem summary
            summary_id = await self.memory_service.create_rolling_summary(
                run_id, node_id, MemorySnapshotType.POSTMORTEM
            )

            # Promote lessons learned to semantic memory
            for lesson in lessons_learned:
                await self.memory_service.promote_to_semantic_memory(
                    text=lesson,
                    title=f"Lesson Learned: {run_id}",
                    meta={
                        "type": "lesson_learned",
                        "run_id": run_id,
                        "node_id": node_id,
                        "final_success": final_results.get("success", False),
                        "final_confidence": final_results.get("confidence", 0.0),
                        "timestamp": datetime.now().isoformat()
                    }
                )

            # Promote successful patterns to semantic memory
            if final_results.get("success", False):
                success_pattern = self._extract_success_pattern(final_results)
                if success_pattern:
                    await self.memory_service.promote_to_semantic_memory(
                        text=success_pattern,
                        title=f"Success Pattern: {run_id}",
                        meta={
                            "type": "success_pattern",
                            "run_id": run_id,
                            "node_id": node_id,
                            "confidence": final_results.get("confidence", 0.0),
                            "timestamp": datetime.now().isoformat()
                        }
                    )

            logger.info(f"Created postmortem summary {summary_id} for {run_id}/{node_id}")
            return summary_id

        except Exception as e:
            logger.error(f"Failed to create postmortem: {e}")
            return -1

    def _extract_success_pattern(self, final_results: Dict[str, Any]) -> Optional[str]:
        """Extract a reusable success pattern from final results"""
        try:
            success_factors = []

            # Extract methodology if available
            if "methodology" in final_results:
                success_factors.append(f"Methodology: {final_results['methodology']}")

            # Extract key tools used
            if "tools_used" in final_results:
                tools = final_results["tools_used"]
                if tools:
                    success_factors.append(f"Tools: {', '.join(tools)}")

            # Extract approach
            if "approach" in final_results:
                success_factors.append(f"Approach: {final_results['approach']}")

            # Extract key insights
            if "insights" in final_results:
                insights = final_results["insights"]
                if insights:
                    success_factors.append(f"Key insights: {'; '.join(insights[:3])}")

            if success_factors:
                return " | ".join(success_factors)

            return None

        except Exception as e:
            logger.error(f"Failed to extract success pattern: {e}")
            return None

    async def get_memory_enhanced_context(
        self,
        run_id: str,
        node_id: str,
        query: str = "",
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """Get memory-enhanced context for LLM prompts"""
        try:
            query_terms = query.split() if query else []

            from .memory_service import ContextConfig
            config = ContextConfig(max_tokens=max_tokens)

            context = await self.memory_service.assemble_context(
                run_id=run_id,
                node_id=node_id,
                query_terms=query_terms,
                config=config
            )

            return context

        except Exception as e:
            logger.error(f"Failed to get memory-enhanced context: {e}")
            return {"error": str(e), "components": {}}

    async def search_similar_experiences(
        self,
        query: str,
        k: int = 5,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar past experiences in semantic memory"""
        try:
            query_terms = query.split()

            results = await self.memory_service.search_semantic_memory(
                query_terms=query_terms,
                k=k,
                use_mmr=True,
                mmr_lambda=0.3  # Prioritize diversity for experience search
            )

            # Filter by success and confidence if available
            filtered_results = []
            for result in results:
                meta = result.get("meta", {})
                confidence = meta.get("final_confidence", meta.get("confidence", 1.0))

                if confidence >= min_confidence:
                    filtered_results.append(result)

            return filtered_results

        except Exception as e:
            logger.error(f"Failed to search similar experiences: {e}")
            return []

    async def get_memory_system_status(self) -> Dict[str, Any]:
        """Get comprehensive memory system status"""
        try:
            # Get memory metrics
            memory_metrics = await self.memory_service.get_memory_system_metrics()

            # Get semantic engine stats
            semantic_stats = self.memory_service.semantic_engine.get_embedding_stats()

            # Get orchestrator status
            orchestrator_status = await self.orchestrator.get_system_status()

            return {
                "memory_service": memory_metrics,
                "semantic_engine": semantic_stats,
                "orchestrator": orchestrator_status,
                "integration_status": "active",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get memory system status: {e}")
            return {"error": str(e)}

    # Delegate orchestrator methods
    async def execute_workflow(self, workflow_config, inputs=None):
        """Execute workflow with memory awareness"""
        return await self.execute_memory_enhanced_workflow(workflow_config, inputs)

    async def wait_for_completion(self, workflow_id, timeout=None):
        """Wait for workflow completion"""
        return await self.orchestrator.wait_for_completion(workflow_id, timeout)

    async def get_workflow_status(self, workflow_id):
        """Get workflow status"""
        return await self.orchestrator.get_workflow_status(workflow_id)

    async def list_active_workflows(self):
        """List active workflows"""
        return await self.orchestrator.list_active_workflows()

    async def cancel_workflow(self, workflow_id):
        """Cancel workflow"""
        return await self.orchestrator.cancel_workflow(workflow_id)