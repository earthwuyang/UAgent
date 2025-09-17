"""
LLM Client for qwen3-max-review via DashScope
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role


class QwenLLMClient:
    """Client for qwen3-max-review model via DashScope"""

    def __init__(self):
        self.model_name = "qwen-max-latest"
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.api_available = bool(self.api_key)

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate response using qwen3-max-review"""

        if not self.api_available:
            return {
                "success": False,
                "error": "DASHSCOPE_API_KEY not configured. Please set the environment variable.",
                "model": self.model_name
            }

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
                return {
                    "success": True,
                    "content": content,
                    "model": self.model_name,
                    "usage": response.usage if hasattr(response, 'usage') else None
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.code} - {response.message}",
                    "model": self.model_name
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Exception: {str(e)}",
                "model": self.model_name
            }

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