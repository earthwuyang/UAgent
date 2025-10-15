"""
Idea Generation Service

This service bridges the tree orchestrator and research engines to provide
intelligent, LLM-based node generation for the research tree.
"""

import logging
import uuid
from typing import List, Optional

from ..uagent_research.engines.scientific_research_original import (
    ScientificResearchEngine,
)
from ..uagent_research.models.research_tree import (
    NodeStatus,
    NodeType,
    ResearchNode,
)

logger = logging.getLogger(__name__)


class IdeaGenerationService:
    """
    Service for generating research tree nodes using LLM-based engines.
    
    This service wraps the ScientificResearchEngine and provides a clean interface
    for the orchestrator to generate ideas, hypotheses, and experiments.
    """

    def __init__(self, llm, config: Optional[dict] = None):
        """
        Initialize the IdeaGenerationService.
        
        Args:
            llm: OpenHands LLM instance for making LLM calls
            config: Optional configuration dict with keys:
                - max_ideas: Maximum ideas to generate (default: 3)
                - max_hypotheses: Maximum hypotheses per idea (default: 2)
                - max_experiments: Maximum experiments per hypothesis (default: 1)
                - retry_count: Number of retries for failed LLM calls (default: 2)
        """
        self.llm = llm
        self.config = config or {}
        
        # Initialize the research engine
        try:
            self.engine = ScientificResearchEngine(llm=llm)
            logger.info("IdeaGenerationService initialized with ScientificResearchEngine")
        except Exception as e:
            logger.error(f"Failed to initialize ScientificResearchEngine: {e}")
            self.engine = None
        
        # Configuration
        self.max_ideas = self.config.get('max_ideas', 3)
        self.max_hypotheses = self.config.get('max_hypotheses', 2)
        self.max_experiments = self.config.get('max_experiments', 1)
        self.retry_count = self.config.get('retry_count', 2)

    def _extract_llm_text(self, response) -> str:
        """
        Extract text from LLM response, handling different response structures.
        
        Args:
            response: LLM response object
        
        Returns:
            Extracted text content
        """
        try:
            # Try OpenAI-style response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message'):
                    return response.choices[0].message.content
                elif hasattr(response.choices[0], 'text'):
                    return response.choices[0].text
            
            # Try direct content attribute
            if hasattr(response, 'content'):
                return response.content
            
            # Try string conversion
            return str(response)
        except Exception as e:
            logger.error(f"Failed to extract text from LLM response: {e}")
            return ""

    def _extract_json_from_text(self, text: str):
        """
        Extract and parse JSON from text, handling code fences and malformed responses.
        
        Args:
            text: Text potentially containing JSON
        
        Returns:
            Parsed JSON object or None
        """
        import json
        import re
        
        if not text:
            return None
        
        # Remove code fences if present
        text = re.sub(r'^```(?:json)?\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON array with regex
        json_array_pattern = r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]'
        matches = re.findall(json_array_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to extract JSON object
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_object_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                # If it looks like it might be part of an array, wrap it
                if isinstance(parsed, dict):
                    return [parsed]
                return parsed
            except json.JSONDecodeError:
                continue
        
        logger.warning(f"Could not extract valid JSON from text: {text[:200]}...")
        return None

    async def generate_ideas(
        self, 
        goal: str, 
        context: Optional[str] = None, 
        max_ideas: Optional[int] = None
    ) -> List[ResearchNode]:
        """
        Generate research ideas for a given goal.
        
        Args:
            goal: The research goal/question
            context: Optional additional context
            max_ideas: Maximum number of ideas to generate (overrides config)
        
        Returns:
            List of ResearchNode objects with type=IDEA
        """
        max_ideas = max_ideas or self.max_ideas
        
        if not self.engine:
            logger.warning("No research engine available, returning empty list")
            return []
        
        logger.info(f"Generating up to {max_ideas} ideas for goal: {goal[:100]}...")
        
        for attempt in range(self.retry_count + 1):
            try:
                # Call the engine to generate ideas
                ideas = await self.engine.generate_research_ideas(
                    goal=goal,
                    context=context or "",
                    max_ideas=max_ideas
                )
                
                # Transform engine ideas to ResearchNode objects
                nodes = []
                for idx, idea in enumerate(ideas[:max_ideas]):
                    node = ResearchNode(
                        id=f"idea-{uuid.uuid4().hex[:8]}",
                        type=NodeType.IDEA,
                        title=getattr(idea, 'title', f"Research Idea {idx + 1}"),
                        content=getattr(idea, 'summary', getattr(idea, 'objective', str(idea))),
                        status=NodeStatus.PENDING,
                        prior=self._calculate_prior_from_idea(idea),
                        parent_id=None
                    )
                    nodes.append(node)
                
                logger.info(f"Successfully generated {len(nodes)} ideas")
                return nodes
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry_count + 1} failed: {e}")
                if attempt == self.retry_count:
                    logger.error(f"Failed to generate ideas after {self.retry_count + 1} attempts", exc_info=True)
                    return []
        
        return []

    async def generate_hypotheses(
        self,
        idea_content: str,
        parent_node: ResearchNode,
        max_hypotheses: Optional[int] = None
    ) -> List[ResearchNode]:
        """
        Generate hypotheses for a research idea.
        
        Args:
            idea_content: The content of the parent idea
            parent_node: The parent ResearchNode (IDEA type)
            max_hypotheses: Maximum number of hypotheses (overrides config)
        
        Returns:
            List of ResearchNode objects with type=HYPOTHESIS
        """
        max_hypotheses = max_hypotheses or self.max_hypotheses
        
        if not self.engine:
            logger.warning("No research engine available, returning empty list")
            return []
        
        logger.info(f"Generating up to {max_hypotheses} hypotheses for idea: {parent_node.title[:50]}...")
        
        for attempt in range(self.retry_count + 1):
            try:
                # Create prompt for hypothesis generation
                prompt = f"""Given the research idea: "{idea_content}"

Generate {max_hypotheses} specific, testable hypotheses that could be investigated.
Each hypothesis should be:
1. Specific and measurable
2. Testable with available methods
3. Relevant to the research idea
4. Novel and insightful

Return a JSON array of objects with:
- title: Brief title of the hypothesis
- description: Detailed description
- confidence: Confidence score (0.0-1.0)

Example format:
[{{"title": "Hypothesis 1", "description": "Detailed description...", "confidence": 0.75}}]
"""
                
                # Call LLM
                response = await self.llm.completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                
                # Parse response with robust JSON extraction
                response_text = self._extract_llm_text(response)
                hypotheses_data = self._extract_json_from_text(response_text)
                
                if not hypotheses_data or not isinstance(hypotheses_data, list):
                    logger.warning("Failed to extract valid JSON array from LLM response")
                    hypotheses_data = []
                
                # Transform to ResearchNode objects
                nodes = []
                for idx, hyp_data in enumerate(hypotheses_data[:max_hypotheses]):
                    node = ResearchNode(
                        id=f"hypothesis-{uuid.uuid4().hex[:8]}",
                        type=NodeType.HYPOTHESIS,
                        title=hyp_data.get('title', f"Hypothesis {idx + 1}"),
                        content=hyp_data.get('description', ''),
                        status=NodeStatus.PENDING,
                        prior=float(hyp_data.get('confidence', 0.7)),
                        parent_id=parent_node.id
                    )
                    nodes.append(node)
                
                logger.info(f"Successfully generated {len(nodes)} hypotheses")
                return nodes
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry_count + 1} failed: {e}")
                if attempt == self.retry_count:
                    logger.error(f"Failed to generate hypotheses after {self.retry_count + 1} attempts", exc_info=True)
                    return []
        
        return []

    async def generate_experiments(
        self,
        hypothesis_content: str,
        parent_node: ResearchNode
    ) -> List[ResearchNode]:
        """
        Generate experiments for a hypothesis.
        
        Args:
            hypothesis_content: The content of the parent hypothesis
            parent_node: The parent ResearchNode (HYPOTHESIS type)
        
        Returns:
            List of ResearchNode objects with type=EXPERIMENT
        """
        if not self.engine:
            logger.warning("No research engine available, returning empty list")
            return []
        
        logger.info(f"Generating experiments for hypothesis: {parent_node.title[:50]}...")
        
        for attempt in range(self.retry_count + 1):
            try:
                # Create prompt for experiment generation
                prompt = f"""Given the hypothesis: "{hypothesis_content}"

Design {self.max_experiments} concrete experiment(s) to test this hypothesis.
Each experiment should include:
1. Clear methodology
2. Required data/resources
3. Expected outcomes
4. Success criteria

Return a JSON array of objects with:
- title: Brief title of the experiment
- methodology: Detailed experimental approach
- expected_outcome: What results would support/refute the hypothesis
- confidence: Confidence in the experimental design (0.0-1.0)

Example format:
[{{"title": "Experiment 1", "methodology": "Step-by-step approach...", "expected_outcome": "If hypothesis is true...", "confidence": 0.65}}]
"""
                
                # Call LLM
                response = await self.llm.completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                
                # Parse response with robust JSON extraction
                response_text = self._extract_llm_text(response)
                experiments_data = self._extract_json_from_text(response_text)
                
                if not experiments_data or not isinstance(experiments_data, list):
                    logger.warning("Failed to extract valid JSON array from LLM response")
                    experiments_data = []
                
                # Transform to ResearchNode objects
                nodes = []
                for idx, exp_data in enumerate(experiments_data[:self.max_experiments]):
                    content = f"""Methodology: {exp_data.get('methodology', '')}

Expected Outcome: {exp_data.get('expected_outcome', '')}
"""
                    node = ResearchNode(
                        id=f"experiment-{uuid.uuid4().hex[:8]}",
                        type=NodeType.EXPERIMENT,
                        title=exp_data.get('title', f"Experiment {idx + 1}"),
                        content=content,
                        status=NodeStatus.PENDING,
                        prior=float(exp_data.get('confidence', 0.6)),
                        parent_id=parent_node.id
                    )
                    nodes.append(node)
                
                logger.info(f"Successfully generated {len(nodes)} experiments")
                return nodes
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry_count + 1} failed: {e}")
                if attempt == self.retry_count:
                    logger.error(f"Failed to generate experiments after {self.retry_count + 1} attempts", exc_info=True)
                    return []
        
        return []

    async def generate_experiments_for_idea(
        self,
        idea_content: str,
        parent_node: ResearchNode,
        hypotheses: List[ResearchNode],
        max_experiments: Optional[int] = None
    ) -> List[ResearchNode]:
        """
        Generate experiments for an IDEA node (new sibling structure).
        
        Each experiment tests ALL hypotheses of the parent IDEA in parallel.
        
        Args:
            idea_content: The content of the parent IDEA
            parent_node: The parent ResearchNode (IDEA type)
            hypotheses: List of hypothesis nodes that are siblings
            max_experiments: Number of experiments to generate (overrides config)
        
        Returns:
            List of ResearchNode objects with type=EXPERIMENT, each with metadata
            containing all parent hypothesis IDs
        """
        if not self.engine:
            logger.warning("No research engine available, returning empty list")
            return []
        
        max_experiments = max_experiments or self.max_experiments
        
        logger.info(f"Generating {max_experiments} experiments for IDEA: {parent_node.title[:50]}...")
        logger.info(f"  Experiments will test {len(hypotheses)} hypotheses in parallel")
        
        # Build hypotheses summary for prompt
        hypotheses_summary = "\\n".join([
            f"{i+1}. {h.title}: {h.content[:100]}"
            for i, h in enumerate(hypotheses)
        ])
        
        for attempt in range(self.retry_count + 1):
            try:
                # Create prompt for experiment generation
                prompt = f"""Given the research idea: "{idea_content}"

And the following hypotheses to test:
{hypotheses_summary}

Design {max_experiments} concrete experiment(s) that can test ALL of these hypotheses simultaneously.
Each experiment should:
1. Have a clear methodology that addresses all hypotheses
2. Specify required data/resources
3. Define expected outcomes for each hypothesis
4. Include success criteria

Return a JSON array of objects with:
- title: Brief title of the experiment
- methodology: Detailed experimental approach
- expected_outcomes: What results would support/refute each hypothesis
- confidence: Confidence in the experimental design (0.0-1.0)

Example format:
[{{"title": "Experiment 1", "methodology": "Step-by-step approach...", "expected_outcomes": "For hypothesis 1..., For hypothesis 2...", "confidence": 0.70}}]
"""
                
                # Call LLM
                response = await self.llm.completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                
                # Parse response with robust JSON extraction
                response_text = self._extract_llm_text(response)
                experiments_data = self._extract_json_from_text(response_text)
                
                if not experiments_data or not isinstance(experiments_data, list):
                    logger.warning("Failed to extract valid JSON array from LLM response")
                    experiments_data = []
                
                # Transform to ResearchNode objects
                nodes = []
                hypothesis_ids = [h.id for h in hypotheses]
                
                for idx, exp_data in enumerate(experiments_data[:max_experiments]):
                    content = f"""Methodology: {exp_data.get('methodology', '')}

Expected Outcomes: {exp_data.get('expected_outcomes', '')}

This experiment tests ALL {len(hypotheses)} hypotheses of the parent idea.
"""
                    # Initialize metadata with parent hypotheses
                    metadata = {
                        'parent_hypotheses': hypothesis_ids,
                        'parent_idea': parent_node.id,
                        'num_hypotheses': len(hypotheses)
                    }
                    
                    node = ResearchNode(
                        id=f"experiment-{uuid.uuid4().hex[:8]}",
                        type=NodeType.EXPERIMENT,
                        title=exp_data.get('title', f"Experiment {idx + 1} for {parent_node.title[:30]}"),
                        content=content,
                        status=NodeStatus.PENDING,
                        prior=float(exp_data.get('confidence', 0.6)),
                        parent_id=parent_node.id,
                        metadata=metadata
                    )
                    nodes.append(node)
                
                logger.info(f"Successfully generated {len(nodes)} experiments for IDEA node")
                for node in nodes:
                    logger.info(f"  - {node.id}: tests {len(hypothesis_ids)} hypotheses")
                
                return nodes
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry_count + 1} failed: {e}")
                if attempt == self.retry_count:
                    logger.error(f"Failed to generate experiments for IDEA after {self.retry_count + 1} attempts", exc_info=True)
                    return []
        
        return []

    def _calculate_prior_from_idea(self, idea) -> float:
        """
        Calculate prior probability from an idea object.
        
        Args:
            idea: Research idea object from engine
        
        Returns:
            Prior probability between 0.0 and 1.0
        """
        # Try to extract confidence/score from idea
        if hasattr(idea, 'confidence'):
            return float(idea.confidence)
        elif hasattr(idea, 'score'):
            return float(idea.score)
        
        # Default to 0.8 for ideas (high prior as they're LLM-generated)
        return 0.8
