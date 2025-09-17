"""
AI Generation API for auto-generating research titles and success criteria
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import asyncio

from ..core.llm_client import llm_client

router = APIRouter(prefix="/ai-generation", tags=["ai-generation"])


class ResearchDescriptionRequest(BaseModel):
    description: str
    domain: str = "AI/ML Research"


class GeneratedResearchContent(BaseModel):
    title: str
    success_criteria: List[str]
    suggested_improvements: List[str]
    confidence_score: float


@router.post("/generate-research-content", response_model=GeneratedResearchContent)
async def generate_research_content(request: ResearchDescriptionRequest):
    """
    Generate research title and success criteria from description using qwen3-max-preview
    """
    try:
        # Construct prompt for LLM
        prompt = f"""
You are an AI research assistant tasked with generating a concise research title and specific success criteria based on a research description.

Research Domain: {request.domain}
Research Description: {request.description}

Please analyze the description and generate:

1. A concise, descriptive research title (max 80 characters)
2. 3-5 specific, measurable success criteria
3. 2-3 suggestions for improving the research description
4. A confidence score (0.0-1.0) for the quality of the generated content

Guidelines:
- Title should be professional and clearly indicate the research focus
- Success criteria should be specific, measurable, and achievable
- Each criterion should be actionable and verifiable
- Suggestions should help make the research more focused or comprehensive

Format your response as JSON:
{{
    "title": "Generated research title",
    "success_criteria": [
        "First specific success criterion",
        "Second specific success criterion",
        "Third specific success criterion"
    ],
    "suggested_improvements": [
        "First suggestion for improvement",
        "Second suggestion for improvement"
    ],
    "confidence_score": 0.85
}}

Ensure the JSON is valid and complete.
"""

        # Call LLM
        response = await llm_client.generate_response(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )

        if not response["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"LLM generation failed: {response.get('error', 'Unknown error')}"
            )

        # Parse LLM response
        try:
            import json
            llm_content = response["content"]

            # Extract JSON from response if it's wrapped in markdown
            if "```json" in llm_content:
                start = llm_content.find("```json") + 7
                end = llm_content.find("```", start)
                json_content = llm_content[start:end].strip()
            else:
                # Try to find JSON-like content
                start = llm_content.find("{")
                end = llm_content.rfind("}") + 1
                if start != -1 and end != 0:
                    json_content = llm_content[start:end]
                else:
                    json_content = llm_content

            parsed_response = json.loads(json_content)

            # Validate required fields
            if not all(key in parsed_response for key in ["title", "success_criteria", "suggested_improvements", "confidence_score"]):
                raise ValueError("Missing required fields in LLM response")

            return GeneratedResearchContent(
                title=parsed_response["title"][:80],  # Ensure max length
                success_criteria=parsed_response["success_criteria"][:5],  # Max 5 criteria
                suggested_improvements=parsed_response["suggested_improvements"][:3],  # Max 3 suggestions
                confidence_score=min(max(parsed_response["confidence_score"], 0.0), 1.0)  # Clamp 0-1
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: create reasonable defaults if JSON parsing fails
            fallback_title = f"Research on {request.description[:50]}..."
            if len(fallback_title) > 80:
                fallback_title = fallback_title[:77] + "..."

            return GeneratedResearchContent(
                title=fallback_title,
                success_criteria=[
                    "Conduct comprehensive literature review of the research area",
                    "Develop and validate proposed methodology or approach",
                    "Demonstrate measurable improvements over existing methods",
                    "Provide reproducible results and implementation"
                ],
                suggested_improvements=[
                    "Consider adding more specific details about the research methodology",
                    "Define clearer metrics for measuring success"
                ],
                confidence_score=0.5
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate research content: {str(e)}"
        )


@router.post("/improve-description")
async def improve_description(request: ResearchDescriptionRequest):
    """
    Suggest improvements to research description
    """
    try:
        prompt = f"""
You are an AI research assistant. Analyze the following research description and provide specific suggestions for improvement.

Research Domain: {request.domain}
Current Description: {request.description}

Please provide:
1. 3-5 specific suggestions to make the description more focused and comprehensive
2. Identify any missing elements (methodology, objectives, scope, etc.)
3. Suggest more precise technical terms if applicable

Format as JSON:
{{
    "improvements": [
        "Specific improvement suggestion 1",
        "Specific improvement suggestion 2"
    ],
    "missing_elements": [
        "Missing element 1",
        "Missing element 2"
    ],
    "technical_suggestions": [
        "Technical term or concept 1",
        "Technical term or concept 2"
    ]
}}
"""

        response = await llm_client.generate_response(
            prompt=prompt,
            temperature=0.7,
            max_tokens=800
        )

        if not response["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"LLM analysis failed: {response.get('error', 'Unknown error')}"
            )

        try:
            import json
            llm_content = response["content"]

            if "```json" in llm_content:
                start = llm_content.find("```json") + 7
                end = llm_content.find("```", start)
                json_content = llm_content[start:end].strip()
            else:
                start = llm_content.find("{")
                end = llm_content.rfind("}") + 1
                json_content = llm_content[start:end] if start != -1 and end != 0 else llm_content

            return json.loads(json_content)

        except (json.JSONDecodeError, ValueError):
            return {
                "improvements": [
                    "Consider adding more specific research objectives",
                    "Define the scope and limitations of the research",
                    "Specify the target outcomes and applications"
                ],
                "missing_elements": [
                    "Research methodology",
                    "Expected timeline",
                    "Success metrics"
                ],
                "technical_suggestions": [
                    "Consider domain-specific terminology",
                    "Reference relevant theoretical frameworks"
                ]
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to improve description: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai-generation"}