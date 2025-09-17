"""
LLM Client for qwen3-max-review via DashScope
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role

logger = logging.getLogger(__name__)

# Import the node tracker
try:
    from .node_llm_tracker import llm_message_listener, node_llm_tracker
    NODE_TRACKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Node tracker not available: {e}")
    NODE_TRACKER_AVAILABLE = False

    async def llm_message_listener(event_type: str, data: dict):
        pass


class QwenLLMClient:
    """Client for qwen3-max-review model via DashScope"""

    def __init__(self):
        self.model_name = "qwen-max-latest"
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.api_available = bool(self.api_key)
        self.message_listeners: List[Callable] = []  # For real-time monitoring

        # Set up node tracker listener
        self._setup_node_tracker()

    def _setup_node_tracker(self):
        """Set up the node-specific LLM tracker"""
        self.add_message_listener(llm_message_listener)

    def add_message_listener(self, listener: Callable):
        """Add a listener for real-time LLM communication monitoring"""
        self.message_listeners.append(listener)

    def remove_message_listener(self, listener: Callable):
        """Remove a message listener"""
        if listener in self.message_listeners:
            self.message_listeners.remove(listener)

    async def _emit_message_event(self, event_type: str, data: Dict[str, Any]):
        """Emit message event to all listeners"""
        for listener in self.message_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event_type, data)
                else:
                    listener(event_type, data)
            except Exception as e:
                logger.error(f"Message listener error: {e}")

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        node_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using qwen3-max-review"""

        if not self.api_available:
            error_response = {
                "success": False,
                "error": "DASHSCOPE_API_KEY not configured. Please set the environment variable.",
                "model": self.model_name
            }
            await self._emit_message_event("error", {
                "timestamp": datetime.now().isoformat(),
                "error": error_response["error"],
                "model": self.model_name
            })
            return error_response

        messages = []
        if system_prompt:
            messages.append({
                'role': Role.SYSTEM,
                'content': system_prompt
            })

        messages.append({
            'role': Role.USER,
            'content': prompt
        })

        # Emit request event
        await self._emit_message_event("request", {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "node_id": node_id,
            "context": context,
            "messages": [{"role": msg['role'], "content": msg['content']} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        })

        try:
            # Run the synchronous dashscope call in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: Generation.call(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    result_format='message'
                )
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                success_response = {
                    "success": True,
                    "content": content,
                    "model": self.model_name,
                    "usage": response.usage if hasattr(response, 'usage') else None
                }

                # Emit successful response event
                await self._emit_message_event("response", {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name,
                    "node_id": node_id,
                    "context": context,
                    "content": content,
                    "usage": response.usage if hasattr(response, 'usage') else None,
                    "success": True
                })

                return success_response
            else:
                error_response = {
                    "success": False,
                    "error": f"API error: {response.code} - {response.message}",
                    "model": self.model_name
                }

                # Emit error response event
                await self._emit_message_event("response", {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name,
                    "node_id": node_id,
                    "context": context,
                    "error": error_response["error"],
                    "success": False
                })

                return error_response

        except Exception as e:
            exception_response = {
                "success": False,
                "error": f"Exception: {str(e)}",
                "model": self.model_name
            }

            # Emit exception event
            await self._emit_message_event("response", {
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "node_id": node_id,
                "context": context,
                "error": str(e),
                "success": False
            })

            return exception_response

    async def analyze_literature(self, goal: str, papers: List[Dict]) -> Dict[str, Any]:
        """Analyze literature for research goal"""
        system_prompt = """You are a research assistant specializing in literature analysis.
        Analyze the provided papers and extract key insights relevant to the research goal."""

        papers_text = "\n".join([
            f"Title: {paper.get('title', 'N/A')}\nAbstract: {paper.get('abstract', 'N/A')}\n"
            for paper in papers[:5]  # Limit to avoid token limits
        ])

        prompt = f"""Research Goal: {goal}

Papers to analyze:
{papers_text}

Please provide:
1. Key findings relevant to the research goal
2. Research gaps identified
3. Methodological insights
4. Potential research directions

Format your response as a structured analysis."""

        return await self.generate_response(prompt, system_prompt)

    async def generate_hypothesis(self, goal: str, context: str = "") -> Dict[str, Any]:
        """Generate research hypothesis"""
        system_prompt = """You are a research scientist who generates testable hypotheses.
        Create specific, measurable hypotheses that advance scientific understanding."""

        prompt = f"""Research Goal: {goal}

Context: {context}

Generate 3-5 specific, testable hypotheses that could advance this research goal.
For each hypothesis, provide:
1. The hypothesis statement
2. Rationale/background
3. Suggested methodology to test it
4. Expected outcomes

Format as a structured list."""

        return await self.generate_response(prompt, system_prompt)

    async def analyze_code(self, goal: str, code_snippets: List[Dict]) -> Dict[str, Any]:
        """Analyze code repositories for research insights"""
        system_prompt = """You are a research engineer who analyzes code to extract scientific insights.
        Focus on methodologies, algorithms, and experimental approaches."""

        code_text = "\n".join([
            f"Repository: {snippet.get('repo', 'N/A')}\nCode: {snippet.get('content', 'N/A')}\n"
            for snippet in code_snippets[:3]  # Limit to avoid token limits
        ])

        prompt = f"""Research Goal: {goal}

Code to analyze:
{code_text}

Please extract:
1. Key algorithms and methodologies
2. Experimental approaches used
3. Implementation insights
4. Potential improvements or extensions
5. Relevance to the research goal

Provide a technical analysis focused on research applications."""

        return await self.generate_response(prompt, system_prompt)


# Global client instance
llm_client = QwenLLMClient()