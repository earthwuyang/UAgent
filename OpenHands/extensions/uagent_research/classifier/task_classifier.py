"""
LLM-based Task Classifier for Research Mode Detection

Uses LLM to intelligently determine if a user's task should trigger research mode.
This replaces the regex-based approach with natural language understanding.
"""

import os
import re
from enum import Enum
from typing import Dict, List, Tuple
import json

import logging

from ..utils.security import mask_secret

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be classified"""
    COMPLEX_RESEARCH = "complex_research"
    HYBRID = "hybrid"
    SIMPLE = "simple"


class TaskClassifier:
    """
    LLM-based classifier to determine if a task should trigger research mode.
    
    Uses the same LLM configured for OpenHands to analyze user intent.
    """
    
    def __init__(self):
        """Initialize the LLM-based task classifier"""
        self.llm = None
        self.api_key = None
        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM client using OpenHands' LLM configuration"""
        try:
            from litellm import completion
            self.llm_completion = completion
            
            # Try to get LLM configuration from OpenHands' config system
            try:
                from openhands.core.config import load_app_config
                from openhands.core.logger import openhands_logger as config_logger
                
                config = load_app_config()
                
                # Get the default LLM config
                llm_config = config.get_llm_config()
                
                self.model = llm_config.model
                self.api_key = llm_config.api_key
                self.base_url = llm_config.base_url
                
                masked_key = mask_secret(self.api_key) if self.api_key else 'None'
                logger.info("TaskClassifier initialized with OpenHands LLM config: model=%s, base_url=%s, key=%s", 
                           self.model, self.base_url, masked_key)
            except Exception as config_error:
                # Fallback to environment variables if OpenHands config fails
                logger.warning("Failed to load OpenHands LLM config, falling back to environment: %s", str(config_error))
                self.model = os.getenv('LLM_MODEL', 'openai/qwen3-coder-plus')
                self.api_key = os.getenv('LLM_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
                self.base_url = os.getenv('LLM_BASE_URL')
                
                masked_key = mask_secret(self.api_key) if self.api_key else 'None'
                logger.info("TaskClassifier initialized with env vars: model=%s, base_url=%s, key=%s", 
                           self.model, self.base_url, masked_key)
            
            if not self.api_key:
                logger.warning("No API key found for TaskClassifier, LLM classification will be disabled")
                self.llm_completion = None
                
        except Exception as e:
            logger.error(
                "Failed to initialize LLM for task classification: %s",
                self._sanitize_message(str(e)),
            )
            self.llm_completion = None

    def classify(self, user_message: str) -> Tuple[TaskType, float, Dict]:
        """
        Classify a user message using LLM.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Tuple of (task_type, confidence, reasoning)
        """
        if not self.llm_completion:
            # Fallback to heuristic if LLM not available
            return self._fallback_classify(user_message)
        
        try:
            # Create classification prompt
            system_prompt = """You are a task classifier for a research-oriented AI agent system.

Your job is to determine if a user's task should trigger "research mode" - an advanced mode that uses tree search, parallel exploration, and systematic experimentation.

TRIGGER RESEARCH MODE if the task involves:
1. **Research & Investigation**: Comparing approaches, benchmarking, literature review, evaluating methods
2. **Complex Multi-Stage Work**: Tasks with 3+ distinct phases (e.g., "first download, then extract, then train, then embed")
3. **Experimental Systems**: Building ML models, collecting data, running experiments, performance comparisons
4. **Source Code Modification**: Modifying database/system internals, kernel-level changes, embedding models into C/C++ code
5. **Systematic Comparison**: Comparing multiple baselines, threshold methods, different approaches

DO NOT TRIGGER for:
- Simple bug fixes or single-file edits
- Running existing scripts
- Basic CRUD operations
- Simple refactoring
- Installing packages

Respond ONLY with valid JSON in this exact format:
{
  "task_type": "complex_research" | "hybrid" | "simple",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "indicators": ["key indicator 1", "key indicator 2", ...]
}"""

            user_prompt = f"""Classify this task:

"{user_message}"

Respond with JSON only."""

            # Call LLM
            response = self.llm_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.3,  # Lower temperature for more consistent classification
                max_tokens=500,
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            result = json.loads(content)
            
            # Map to TaskType enum
            task_type_str = result.get('task_type', 'simple').lower()
            if 'complex' in task_type_str or 'research' in task_type_str:
                task_type = TaskType.COMPLEX_RESEARCH
            elif 'hybrid' in task_type_str:
                task_type = TaskType.HYBRID
            else:
                task_type = TaskType.SIMPLE
            
            confidence = float(result.get('confidence', 0.5))
            reasoning = {
                'decision': result.get('reasoning', 'LLM classification'),
                'indicators': result.get('indicators', []),
                'llm_response': content[:200],
                'method': 'llm'
            }
            
            logger.info(f"LLM classified task as {task_type.value} (confidence: {confidence:.2f})")
            logger.debug(f"LLM reasoning: {reasoning}")
            
            return task_type, confidence, reasoning
            
        except Exception as e:
            logger.error(
                "LLM classification failed: %s",
                self._sanitize_message(str(e)),
                exc_info=True,
            )
            logger.info("Falling back to heuristic classification")
            return self._fallback_classify(user_message)

    def _sanitize_message(self, message: str) -> str:
        """Mask any configured API keys from log output."""
        if not message:
            return message

        secrets = {
            secret
            for secret in (
                self.api_key,
                os.getenv('LLM_API_KEY'),
                os.getenv('DASHSCOPE_API_KEY'),
            )
            if secret
        }

        sanitized = message
        for secret in secrets:
            sanitized = sanitized.replace(secret, mask_secret(secret))
        return sanitized
    
    def _fallback_classify(self, user_message: str) -> Tuple[TaskType, float, Dict]:
        """
        Fallback heuristic classification when LLM is unavailable.
        
        This is a simplified version of the original regex-based classifier.
        """
        msg_lower = user_message.lower()
        
        # Research keywords
        research_indicators = [
            'research', 'compare', 'benchmark', 'evaluate', 
            'experiment', 'investigate', 'analyze', 'study'
        ]
        
        # Complexity indicators
        complexity_indicators = [
            'train', 'model', 'machine learning', 'ml', 'deep learning',
            'collect data', 'extract features', 'embed', 'modify source',
            'kernel', 'database source', 'implement baseline'
        ]
        
        # Multi-stage indicators
        has_multi_stage = any([
            'first' in msg_lower and 'then' in msg_lower,
            msg_lower.count(',') > 3,
            len(user_message) > 500,
            msg_lower.count(' and ') > 3,
        ])
        
        # Count matches
        research_score = sum(1 for kw in research_indicators if kw in msg_lower)
        complexity_score = sum(1 for kw in complexity_indicators if kw in msg_lower)
        
        # Decision logic
        if research_score >= 2 or (research_score >= 1 and complexity_score >= 2):
            task_type = TaskType.COMPLEX_RESEARCH
            confidence = min(0.85, 0.6 + research_score * 0.1 + complexity_score * 0.05)
            decision = "Strong research indicators (fallback heuristic)"
        elif has_multi_stage and (research_score >= 1 or complexity_score >= 3):
            task_type = TaskType.COMPLEX_RESEARCH
            confidence = min(0.80, 0.5 + research_score * 0.1 + complexity_score * 0.1)
            decision = "Multi-stage complex task (fallback heuristic)"
        elif complexity_score >= 4:
            task_type = TaskType.COMPLEX_RESEARCH
            confidence = min(0.75, 0.5 + complexity_score * 0.1)
            decision = "High complexity (fallback heuristic)"
        else:
            task_type = TaskType.SIMPLE
            confidence = 0.7
            decision = "Simple task (fallback heuristic)"
        
        reasoning = {
            'decision': decision,
            'research_score': research_score,
            'complexity_score': complexity_score,
            'has_multi_stage': has_multi_stage,
            'method': 'fallback_heuristic'
        }
        
        logger.info(f"Fallback classified task as {task_type.value} (confidence: {confidence:.2f})")
        
        return task_type, confidence, reasoning
    
    def should_trigger_research(
        self,
        user_message: str,
        confidence_threshold: float = 0.7
    ) -> Tuple[bool, TaskType, float, Dict]:
        """
        Determine if research mode should be triggered.
        
        Args:
            user_message: User's input message
            confidence_threshold: Minimum confidence to trigger research (default: 0.7)
            
        Returns:
            Tuple of (should_trigger, task_type, confidence, reasoning)
        """
        task_type, confidence, reasoning = self.classify(user_message)
        
        should_trigger = (
            task_type == TaskType.COMPLEX_RESEARCH and
            confidence >= confidence_threshold
        )
        
        return should_trigger, task_type, confidence, reasoning


# Global classifier instance
task_classifier = TaskClassifier()


# Example usage and testing
def test_classifier():
    """Test the classifier with example queries"""
    test_cases = [
        # Research tasks (should trigger)
        "Research neural architecture search methods and implement the best approach",
        "Compare different sorting algorithms and benchmark their performance",
        "Investigate the latest advances in transformer models",
        
        # Complex tasks (should trigger)
        "Download postgres source code, extract features, train a model, and embed it",
        "Modify postgres and pg_duckdb source code, extract pre-opt features from postgres kernel and log to files, collect dual-execution data and train ML model",
        
        # Simple tasks (should NOT trigger)
        "Fix the bug in the login function",
        "Add a new method to calculate sum",
        "Install numpy and run the script",
    ]
    
    classifier = TaskClassifier()
    
    for message in test_cases:
        should_trigger, task_type, confidence, reasoning = classifier.should_trigger_research(message)
        print(f"\nMessage: {message[:80]}...")
        print(f"  Type: {task_type.value}, Trigger: {should_trigger}, Confidence: {confidence:.2f}")
        print(f"  Reasoning: {reasoning.get('decision', 'N/A')}")


if __name__ == "__main__":
    test_classifier()
