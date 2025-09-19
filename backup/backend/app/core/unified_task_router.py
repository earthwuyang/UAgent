"""
Unified Task Router for UAgent
Intelligently routes user requests to appropriate specialized systems
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from .llm_client import llm_client

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Available task types for routing"""
    RESEARCH = "research"           # Complex hierarchical research tasks
    SEARCH = "search"              # Information retrieval and search tasks
    DATA_ANALYSIS = "data_analysis" # Data processing and analysis tasks
    IMPLEMENTATION = "implementation" # Direct coding/deployment tasks
    QUESTION_ANSWER = "qa"         # Simple Q&A that doesn't need complex systems


@dataclass
class TaskClassification:
    """Result of task classification"""
    task_type: TaskType
    confidence: float
    reasoning: str
    suggested_system: str
    parameters: Dict[str, Any]


class UnifiedTaskRouter:
    """
    Unified routing layer that determines the best system to handle a user request
    """

    def __init__(self):
        self.classification_cache = {}  # Cache recent classifications

    async def classify_and_route(self,
                                title: str,
                                description: str,
                                success_criteria: List[str] = None) -> TaskClassification:
        """
        Classify the task and determine which system should handle it
        """

        # Create cache key
        cache_key = f"{title}|{description}"
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]

        # Use LLM to classify the task
        classification = await self._llm_classify_task(title, description, success_criteria or [])

        # Cache the result
        self.classification_cache[cache_key] = classification

        return classification

    async def _llm_classify_task(self,
                                title: str,
                                description: str,
                                success_criteria: List[str]) -> TaskClassification:
        """Use LLM to intelligently classify the task type"""

        system_prompt = """You are a task classification expert. Analyze user requests and determine the best system to handle them.

Available task types:
1. IMPLEMENTATION - Direct coding, deployment, setup, building things
   - Examples: "Setup MongoDB", "Build a web scraper", "Deploy a service", "Create hello world app"
   - Characteristics: Concrete deliverables, clear implementation steps, practical tasks
   - Best system: Research Tree (focused computational experiments)

2. SEARCH - Information retrieval, finding specific information
   - Examples: "Find papers about X", "Search for Y implementations", "Look up Z documentation"
   - Characteristics: Information gathering, specific queries, finding existing resources
   - Best system: Search System (to be implemented)

3. DATA_ANALYSIS - Processing, analyzing, or working with datasets
   - Examples: "Analyze this data", "Process CSV files", "Generate insights from data"
   - Characteristics: Data manipulation, statistical analysis, visualization
   - Best system: Data Analysis System (to be implemented)

4. RESEARCH - Complex research requiring literature review, multiple approaches, hypothesis testing
   - Examples: "Research the state of AI agents", "Comprehensive study of X", "Investigate novel approaches"
   - Characteristics: Academic depth, multiple experiment types, literature review needed
   - Best system: Research Tree (full hierarchical research)

5. QUESTION_ANSWER - Simple questions that can be answered directly
   - Examples: "What is X?", "How does Y work?", "Explain Z"
   - Characteristics: Direct answers, no complex workflows needed
   - Best system: Direct LLM response

IMPORTANT: Most practical tasks (setup, build, deploy, implement) should be IMPLEMENTATION, not RESEARCH.
Only use RESEARCH for complex academic/scientific investigations requiring multiple approaches.

Respond with JSON format:
{
  "task_type": "IMPLEMENTATION",
  "confidence": 0.9,
  "reasoning": "This is a deployment task requiring practical implementation",
  "suggested_system": "research_tree_computational",
  "parameters": {"focus": "implementation", "complexity": "medium"}
}"""

        prompt = f"""Task Title: {title}
Description: {description}
Success Criteria: {success_criteria}

Analyze this task and classify it. Consider:
1. Is this asking to BUILD/SETUP/DEPLOY something? → IMPLEMENTATION
2. Is this asking to FIND/SEARCH for information? → SEARCH
3. Is this asking to ANALYZE data? → DATA_ANALYSIS
4. Is this asking for complex academic RESEARCH? → RESEARCH
5. Is this a simple question? → QUESTION_ANSWER

Be practical: most "research" requests are actually implementation tasks in disguise.
"""

        try:
            llm_response = await llm_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,  # Low temperature for consistent classification
                max_tokens=300
            )

            if llm_response.get("success"):
                content = llm_response.get("content", "")

                # Try to parse JSON response
                import json
                import re

                # Try multiple JSON extraction patterns
                json_patterns = [
                    r'\{[\s\S]*?\}',  # Original pattern
                    r'```json\s*(\{[\s\S]*?\})\s*```',  # Code block pattern
                    r'```\s*(\{[\s\S]*?\})\s*```',  # Generic code block
                ]

                result = None
                for pattern in json_patterns:
                    json_match = re.search(pattern, content)
                    if json_match:
                        try:
                            json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
                            # Clean up common JSON issues
                            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                            result = json.loads(json_str)
                            break
                        except (json.JSONDecodeError, ValueError):
                            continue

                if result:
                    try:
                        task_type = TaskType(result.get("task_type", "implementation").lower())
                        confidence = float(result.get("confidence", 0.8))
                        reasoning = result.get("reasoning", "LLM classification")
                        suggested_system = result.get("suggested_system", "research_tree")
                        parameters = result.get("parameters", {})

                        logger.info(f"LLM classified task as {task_type.value} with confidence {confidence}")

                        return TaskClassification(
                            task_type=task_type,
                            confidence=confidence,
                            reasoning=reasoning,
                            suggested_system=suggested_system,
                            parameters=parameters
                        )

                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse LLM classification result: {e}")

                # Fallback: text analysis
                return self._fallback_classify_task(title, description, content)

        except Exception as e:
            logger.error(f"LLM task classification failed: {e}")

        # Final fallback
        return self._fallback_classify_task(title, description, "")

    def _fallback_classify_task(self, title: str, description: str, llm_content: str = "") -> TaskClassification:
        """Fallback heuristic classification when LLM fails"""

        combined_text = f"{title} {description}".lower()

        # Implementation task indicators (most common)
        implementation_keywords = [
            'setup', 'build', 'create', 'deploy', 'implement', 'develop', 'install',
            'configure', 'make', 'code', 'program', 'script', 'app', 'service',
            'mongodb', 'database', 'server', 'api', 'website', 'system'
        ]

        # Search task indicators
        search_keywords = [
            'find', 'search', 'look up', 'locate', 'discover', 'get information',
            'research papers', 'documentation', 'examples of', 'implementations of'
        ]

        # Data analysis indicators
        analysis_keywords = [
            'analyze', 'process data', 'csv', 'dataset', 'statistics', 'insights',
            'visualization', 'charts', 'graphs', 'metrics', 'trends'
        ]

        # Research task indicators (academic/complex)
        research_keywords = [
            'comprehensive study', 'literature review', 'state of the art',
            'research paper', 'academic', 'novel approach', 'hypothesis',
            'investigation', 'survey of', 'comparative analysis'
        ]

        # Question/answer indicators
        qa_keywords = [
            'what is', 'how does', 'explain', 'define', 'why', 'when',
            'what are the differences', 'how to', 'can you tell me'
        ]

        # Score each type
        scores = {
            TaskType.IMPLEMENTATION: sum(1 for kw in implementation_keywords if kw in combined_text),
            TaskType.SEARCH: sum(1 for kw in search_keywords if kw in combined_text),
            TaskType.DATA_ANALYSIS: sum(1 for kw in analysis_keywords if kw in combined_text),
            TaskType.RESEARCH: sum(1 for kw in research_keywords if kw in combined_text),
            TaskType.QUESTION_ANSWER: sum(1 for kw in qa_keywords if kw in combined_text)
        }

        # Default to implementation if no clear indicators
        if all(score == 0 for score in scores.values()):
            scores[TaskType.IMPLEMENTATION] = 1

        # Find the highest scoring type
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(0.9, scores[best_type] / 5.0 + 0.4)  # Scale confidence

        # Map to systems
        system_mapping = {
            TaskType.IMPLEMENTATION: "research_tree_computational",
            TaskType.SEARCH: "search_system",  # To be implemented
            TaskType.DATA_ANALYSIS: "data_analysis_system",  # To be implemented
            TaskType.RESEARCH: "research_tree_full",
            TaskType.QUESTION_ANSWER: "direct_llm"
        }

        logger.info(f"Fallback classified task as {best_type.value} with confidence {confidence}")

        return TaskClassification(
            task_type=best_type,
            confidence=confidence,
            reasoning=f"Heuristic classification based on keywords (score: {scores[best_type]})",
            suggested_system=system_mapping[best_type],
            parameters={"classification_method": "heuristic", "keyword_scores": {k.value: v for k, v in scores.items()}}
        )

    async def route_task(self, classification: TaskClassification, **task_data) -> Dict[str, Any]:
        """
        Route the task to the appropriate system based on classification
        """

        if classification.task_type == TaskType.IMPLEMENTATION:
            return await self._route_to_implementation_system(classification, **task_data)
        elif classification.task_type == TaskType.RESEARCH:
            return await self._route_to_research_system(classification, **task_data)
        elif classification.task_type == TaskType.SEARCH:
            return await self._route_to_search_system(classification, **task_data)
        elif classification.task_type == TaskType.DATA_ANALYSIS:
            return await self._route_to_analysis_system(classification, **task_data)
        elif classification.task_type == TaskType.QUESTION_ANSWER:
            return await self._route_to_qa_system(classification, **task_data)
        else:
            # Default to implementation
            return await self._route_to_implementation_system(classification, **task_data)

    async def _route_to_implementation_system(self, classification: TaskClassification, **task_data) -> Dict[str, Any]:
        """Route to research tree with computational focus"""
        # Import here to avoid circular imports
        from .research_tree import HierarchicalResearchSystem

        research_system = HierarchicalResearchSystem()

        # Add routing metadata to task
        task_data['routing_info'] = {
            'classification': classification.task_type.value,
            'confidence': classification.confidence,
            'reasoning': classification.reasoning,
            'focus': 'implementation',
            'experiment_preference': 'computational_only'
        }

        # Start the research goal with implementation focus
        goal_id = await research_system.start_hierarchical_research(
            title=task_data.get('title', ''),
            description=task_data.get('description', ''),
            success_criteria=task_data.get('success_criteria', []),
            constraints=task_data.get('constraints', {}),
            max_depth=task_data.get('max_depth', 4),  # Shallower for implementation
            max_experiments=task_data.get('max_experiments', 50)  # Fewer experiments
        )

        return {
            'goal_id': goal_id,
            'system': 'research_tree',
            'mode': 'implementation',
            'classification': classification.task_type.value
        }

    async def _route_to_research_system(self, classification: TaskClassification, **task_data) -> Dict[str, Any]:
        """Route to full research tree system"""
        from .research_tree import HierarchicalResearchSystem

        research_system = HierarchicalResearchSystem()

        task_data['routing_info'] = {
            'classification': classification.task_type.value,
            'confidence': classification.confidence,
            'reasoning': classification.reasoning,
            'focus': 'research',
            'experiment_preference': 'full_hierarchy'
        }

        # Start full research with all experiment types
        goal_id = await research_system.start_hierarchical_research(
            title=task_data.get('title', ''),
            description=task_data.get('description', ''),
            success_criteria=task_data.get('success_criteria', []),
            constraints=task_data.get('constraints', {}),
            max_depth=task_data.get('max_depth', 6),  # Deeper for research
            max_experiments=task_data.get('max_experiments', 150)  # More experiments
        )

        return {
            'goal_id': goal_id,
            'system': 'research_tree',
            'mode': 'research',
            'classification': classification.task_type.value
        }

    async def _route_to_search_system(self, classification: TaskClassification, **task_data) -> Dict[str, Any]:
        """Route to search system (to be implemented)"""
        logger.info("Search system not yet implemented, routing to research tree")
        return await self._route_to_implementation_system(classification, **task_data)

    async def _route_to_analysis_system(self, classification: TaskClassification, **task_data) -> Dict[str, Any]:
        """Route to data analysis system (to be implemented)"""
        logger.info("Data analysis system not yet implemented, routing to research tree")
        return await self._route_to_implementation_system(classification, **task_data)

    async def _route_to_qa_system(self, classification: TaskClassification, **task_data) -> Dict[str, Any]:
        """Route to direct Q&A system"""
        # For simple questions, just use LLM directly
        question = f"{task_data.get('title', '')} {task_data.get('description', '')}"

        try:
            llm_response = await llm_client.generate_response(
                prompt=question,
                system_prompt="You are a helpful assistant. Provide clear, concise answers to questions.",
                temperature=0.3,
                max_tokens=1000
            )

            return {
                'system': 'direct_llm',
                'response': llm_response.get('content', 'Unable to generate response'),
                'classification': classification.task_type.value
            }

        except Exception as e:
            logger.error(f"Direct QA failed: {e}")
            # Fallback to research system
            return await self._route_to_implementation_system(classification, **task_data)


# Global router instance
task_router = UnifiedTaskRouter()