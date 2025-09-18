"""
Memory Summarization Module for UAgent
Implements LLM-based rolling summarization with structured output
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .llm_client import QwenLLMClient
from .memory_service import MemoryEvent, MemorySnapshotType

logger = logging.getLogger(__name__)


@dataclass
class SummaryContext:
    """Context for generating summaries"""
    task_description: str
    success_criteria: List[str]
    constraints: Dict[str, Any]
    current_state: str
    open_questions: List[str]


@dataclass
class StructuredSummary:
    """Structured summary output"""
    context: str
    state: str
    evidence: List[Dict[str, str]]  # citations with URLs/IDs
    actions: List[str]
    key_insights: List[str]
    token_count: int
    compression_ratio: float


class RollingSummarizer:
    """
    Handles rolling summarization of memory events using LLM
    """

    def __init__(self, llm_client: Optional[QwenLLMClient] = None):
        self.llm_client = llm_client or QwenLLMClient()

        # Summarization templates
        self.summary_prompt_template = """
You are a research assistant creating concise, structured summaries of experimental work.

CONTEXT:
{context}

EVENTS TO SUMMARIZE ({event_count} events, {total_tokens} tokens):
{events_text}

Create a structured summary with exactly these sections:

**CONTEXT**: Brief task description and constraints
**STATE**: Current hypotheses, decisions, and progress
**EVIDENCE**: Top 3 key findings with specific citations
**ACTIONS**: Next logical steps or recommendations
**INSIGHTS**: 2-3 key learnings or patterns discovered

Keep total length under {max_tokens} tokens. Focus on actionable information and maintain research continuity.

Summary:"""

        self.postmortem_prompt_template = """
You are creating a final postmortem summary for a completed research task.

TASK OVERVIEW:
{context}

COMPLETE EVENT HISTORY:
{events_text}

FINAL RESULTS:
{final_results}

Create a comprehensive postmortem with these sections:

**OBJECTIVE**: What was the research goal and success criteria
**METHODOLOGY**: Approach taken and tools used
**KEY FINDINGS**: Most important discoveries and insights
**CHALLENGES**: Problems encountered and how they were resolved
**LESSONS LEARNED**: Reusable patterns and best practices
**RECOMMENDATIONS**: Suggestions for future similar work

Focus on creating reusable knowledge for future research tasks.

Postmortem:"""

    async def create_rolling_summary(
        self,
        events: List[Dict[str, Any]],
        context: Optional[SummaryContext] = None,
        max_tokens: int = 800
    ) -> StructuredSummary:
        """
        Create a rolling summary from memory events
        """
        try:
            if not events:
                return StructuredSummary(
                    context="No events to summarize",
                    state="Initial state",
                    evidence=[],
                    actions=[],
                    key_insights=[],
                    token_count=0,
                    compression_ratio=0.0
                )

            # Prepare context
            context_str = self._format_context(context) if context else "Research task in progress"

            # Format events for summarization
            events_text = self._format_events_for_llm(events)

            # Calculate input tokens for compression ratio
            input_tokens = self._estimate_tokens(events_text)

            # Build prompt
            prompt = self.summary_prompt_template.format(
                context=context_str,
                event_count=len(events),
                total_tokens=input_tokens,
                events_text=events_text,
                max_tokens=max_tokens
            )

            # Generate summary using LLM
            try:
                response = await self.llm_client.generate_response(prompt, max_tokens=max_tokens)
                summary_text = response.get("content", "")
            except Exception as e:
                logger.warning(f"LLM summarization failed, using fallback: {e}")
                summary_text = self._create_fallback_summary(events, context)

            # Parse structured summary
            structured = self._parse_structured_summary(summary_text)

            # Calculate metrics
            output_tokens = self._estimate_tokens(summary_text)
            compression_ratio = output_tokens / max(input_tokens, 1)

            structured.token_count = output_tokens
            structured.compression_ratio = compression_ratio

            logger.info(f"Created rolling summary: {len(events)} events -> {output_tokens} tokens (compression: {compression_ratio:.2f})")

            return structured

        except Exception as e:
            logger.error(f"Failed to create rolling summary: {e}")
            return self._create_error_summary(str(e))

    async def create_postmortem_summary(
        self,
        events: List[Dict[str, Any]],
        final_results: Dict[str, Any],
        context: Optional[SummaryContext] = None
    ) -> StructuredSummary:
        """
        Create a comprehensive postmortem summary
        """
        try:
            # Prepare context and results
            context_str = self._format_context(context) if context else "Research task completed"
            events_text = self._format_events_for_llm(events)
            results_text = json.dumps(final_results, indent=2)[:2000]  # Limit size

            # Build postmortem prompt
            prompt = self.postmortem_prompt_template.format(
                context=context_str,
                events_text=events_text,
                final_results=results_text
            )

            # Generate postmortem using LLM
            try:
                response = await self.llm_client.generate_response(prompt, max_tokens=1500)
                summary_text = response.get("content", "")
            except Exception as e:
                logger.warning(f"LLM postmortem failed, using fallback: {e}")
                summary_text = self._create_fallback_postmortem(events, final_results)

            # Parse structured summary
            structured = self._parse_structured_summary(summary_text, is_postmortem=True)

            # Calculate metrics
            input_tokens = self._estimate_tokens(events_text + results_text)
            output_tokens = self._estimate_tokens(summary_text)
            compression_ratio = output_tokens / max(input_tokens, 1)

            structured.token_count = output_tokens
            structured.compression_ratio = compression_ratio

            logger.info(f"Created postmortem summary: {output_tokens} tokens")

            return structured

        except Exception as e:
            logger.error(f"Failed to create postmortem summary: {e}")
            return self._create_error_summary(str(e))

    def _format_context(self, context: SummaryContext) -> str:
        """Format context for LLM prompt"""
        parts = [
            f"Task: {context.task_description}",
        ]

        if context.success_criteria:
            parts.append(f"Success Criteria: {', '.join(context.success_criteria)}")

        if context.constraints:
            parts.append(f"Constraints: {json.dumps(context.constraints)}")

        if context.current_state:
            parts.append(f"Current State: {context.current_state}")

        if context.open_questions:
            parts.append(f"Open Questions: {', '.join(context.open_questions)}")

        return "\n".join(parts)

    def _format_events_for_llm(self, events: List[Dict[str, Any]]) -> str:
        """Format events in a structured way for LLM processing"""
        formatted_events = []

        for i, event in enumerate(events, 1):
            timestamp = event.get("ts", "")
            source = event.get("source", "unknown")
            kind = event.get("kind", "")

            # Parse body JSON
            try:
                body = json.loads(event.get("body_json", "{}"))
            except:
                body = {"raw": event.get("body_json", "")}

            # Extract key information based on event type
            content = self._extract_event_content(kind, body, source)

            formatted_events.append(f"{i}. [{timestamp}] {source}:{kind} - {content}")

        return "\n".join(formatted_events)

    def _extract_event_content(self, kind: str, body: Dict[str, Any], source: str) -> str:
        """Extract meaningful content from event body"""
        if kind == "tool_call":
            tool_name = body.get("tool_name", "unknown")
            params = body.get("parameters", {})
            return f"Called {tool_name} with {len(params)} parameters"

        elif kind == "tool_result":
            success = body.get("success", False)
            result_summary = str(body.get("result", ""))[:100]
            return f"Tool {'succeeded' if success else 'failed'}: {result_summary}"

        elif kind == "decision":
            decision = body.get("decision", "")
            reasoning = body.get("reasoning", "")[:100]
            return f"Decided: {decision} (Reason: {reasoning})"

        elif kind == "observation":
            observation = body.get("observation", "")
            return f"Observed: {observation[:150]}"

        elif kind == "error":
            error_msg = body.get("error", "")
            return f"Error: {error_msg[:100]}"

        elif kind == "reflection":
            reflection = body.get("reflection", "")
            return f"Reflected: {reflection[:150]}"

        else:
            # Generic content extraction
            if "text" in body:
                return body["text"][:150]
            elif "message" in body:
                return body["message"][:150]
            else:
                return str(body)[:150]

    def _parse_structured_summary(self, summary_text: str, is_postmortem: bool = False) -> StructuredSummary:
        """Parse LLM output into structured summary"""
        try:
            # Define section headers to look for
            if is_postmortem:
                sections = {
                    "OBJECTIVE": "context",
                    "METHODOLOGY": "state",
                    "KEY FINDINGS": "evidence",
                    "CHALLENGES": "actions",
                    "LESSONS LEARNED": "key_insights"
                }
            else:
                sections = {
                    "CONTEXT": "context",
                    "STATE": "state",
                    "EVIDENCE": "evidence",
                    "ACTIONS": "actions",
                    "INSIGHTS": "key_insights"
                }

            parsed = {
                "context": "",
                "state": "",
                "evidence": [],
                "actions": [],
                "key_insights": []
            }

            current_section = None
            current_content = []

            lines = summary_text.split('\n')

            for line in lines:
                line = line.strip()

                # Check if this line starts a new section
                section_found = None
                for header, field in sections.items():
                    if line.upper().startswith(f"**{header}"):
                        section_found = field
                        break

                if section_found:
                    # Save previous section
                    if current_section and current_content:
                        content = '\n'.join(current_content).strip()
                        if current_section in ["evidence", "actions", "key_insights"]:
                            # Parse list items
                            items = [item.strip("- ").strip() for item in content.split('\n') if item.strip()]
                            if current_section == "evidence":
                                # Convert to citation format
                                parsed[current_section] = [{"text": item, "source": "memory"} for item in items[:3]]
                            else:
                                parsed[current_section] = items
                        else:
                            parsed[current_section] = content

                    current_section = section_found
                    current_content = []

                    # Check if content is on the same line after the header
                    if ':' in line:
                        remaining = line.split(':', 1)[1].strip()
                        if remaining:
                            current_content.append(remaining)

                elif current_section and line:
                    current_content.append(line)

            # Handle last section
            if current_section and current_content:
                content = '\n'.join(current_content).strip()
                if current_section in ["evidence", "actions", "key_insights"]:
                    items = [item.strip("- ").strip() for item in content.split('\n') if item.strip()]
                    if current_section == "evidence":
                        parsed[current_section] = [{"text": item, "source": "memory"} for item in items[:3]]
                    else:
                        parsed[current_section] = items
                else:
                    parsed[current_section] = content

            return StructuredSummary(
                context=parsed["context"] or "No context available",
                state=parsed["state"] or "No state information",
                evidence=parsed["evidence"] or [],
                actions=parsed["actions"] or [],
                key_insights=parsed["key_insights"] or [],
                token_count=0,  # Will be set by caller
                compression_ratio=0.0  # Will be set by caller
            )

        except Exception as e:
            logger.error(f"Failed to parse structured summary: {e}")
            return StructuredSummary(
                context=summary_text[:200],
                state="Parse error occurred",
                evidence=[],
                actions=[],
                key_insights=[],
                token_count=0,
                compression_ratio=0.0
            )

    def _create_fallback_summary(self, events: List[Dict[str, Any]], context: Optional[SummaryContext]) -> str:
        """Create a simple fallback summary when LLM is unavailable"""
        event_types = {}
        tools_used = set()
        errors = []

        for event in events:
            kind = event.get("kind", "unknown")
            event_types[kind] = event_types.get(kind, 0) + 1

            if event.get("source", "").startswith("tool:"):
                tool_name = event["source"].split(":", 1)[1]
                tools_used.add(tool_name)

            if kind == "error":
                try:
                    body = json.loads(event.get("body_json", "{}"))
                    errors.append(body.get("error", "Unknown error"))
                except:
                    pass

        summary_parts = [
            "**CONTEXT**: Research task in progress",
            f"**STATE**: Processed {len(events)} events of types: {dict(event_types)}",
            "**EVIDENCE**: [Fallback mode - limited analysis available]",
            f"**ACTIONS**: Continue research using tools: {list(tools_used)}",
            f"**INSIGHTS**: {len(errors)} errors encountered during execution"
        ]

        return "\n".join(summary_parts)

    def _create_fallback_postmortem(self, events: List[Dict[str, Any]], final_results: Dict[str, Any]) -> str:
        """Create a simple fallback postmortem"""
        success = final_results.get("success", False)
        confidence = final_results.get("confidence", 0.0)

        return f"""
**OBJECTIVE**: Research task completion
**METHODOLOGY**: Processed {len(events)} events using automated tools
**KEY FINDINGS**: Task {'completed successfully' if success else 'encountered issues'} with {confidence:.1%} confidence
**CHALLENGES**: [Fallback mode - detailed analysis unavailable]
**LESSONS LEARNED**: Task execution generated significant event history
**RECOMMENDATIONS**: Review event log for detailed insights
"""

    def _create_error_summary(self, error_msg: str) -> StructuredSummary:
        """Create an error summary"""
        return StructuredSummary(
            context=f"Error occurred during summarization: {error_msg}",
            state="Summary generation failed",
            evidence=[],
            actions=["Review summarization system", "Check LLM connectivity"],
            key_insights=["Summarization system needs attention"],
            token_count=0,
            compression_ratio=0.0
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)"""
        return len(text) // 4