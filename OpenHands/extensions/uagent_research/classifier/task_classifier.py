"""
Task Complexity Classifier

Detects whether a user query should trigger research mode based on complexity indicators.
"""

import logging
import re
from enum import Enum
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Classification of task types"""
    SIMPLE = "simple"  # Direct implementation task
    COMPLEX_RESEARCH = "complex_research"  # Multi-stage research required
    HYBRID = "hybrid"  # Research + implementation


class TaskClassifier:
    """
    Classifies user queries to determine if they require research mode.

    Classification Criteria:
    - Complex Research: Multi-source exploration, hypothesis testing, comparative studies
    - Simple: Well-defined implementation, single-step tasks
    - Hybrid: Research followed by implementation
    """

    def __init__(self):
        # Research indicators (strong signals for research mode)
        self.research_keywords = [
            r'\b(research|investigate|explore|study|analyze|survey|review)\b',
            r'\b(find out|discover|learn about|understand)\b',
            r'\b(compare|benchmark|evaluate|assess)\b',
            r'\b(state[- ]of[- ]the[- ]art|sota|latest|recent)\b',
            r'\b(papers|literature|publications|articles)\b',
            r'\b(approaches|methods|techniques|strategies)\b',
            r'\b(experimental|hypothesis|test)\b',
        ]

        # Complexity indicators (signals for multi-stage work)
        self.complexity_keywords = [
            r'\b(multi[- ]stage|multiple steps|end[- ]to[- ]end)\b',
            r'\b(train|training|model|machine learning|ml|ai)\b',
            r'\b(optimize|optimization|performance)\b',
            r'\b(collect|gather|extract)\s+(data|features|metrics)\b',
            r'\b(experiment|experiments|experimental)\b',
            r'\b(baseline|comparison|comparative)\b',
            r'\b(source code|download|clone)\s+(from|repo|repository)\b',
        ]

        # Implementation indicators (signals for direct execution)
        self.simple_keywords = [
            r'\b(fix|debug|patch|repair)\s+(bug|error|issue)\b',
            r'\b(add|create|write)\s+(function|class|method)\b',
            r'\b(refactor|clean up|format)\b',
            r'\b(install|setup|configure)\b',
            r'\b(run|execute)\s+(test|script|command)\b',
        ]

        # Compile patterns
        self.research_patterns = [re.compile(p, re.IGNORECASE) for p in self.research_keywords]
        self.complexity_patterns = [re.compile(p, re.IGNORECASE) for p in self.complexity_keywords]
        self.simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.simple_keywords]

    def classify(self, user_message: str) -> Tuple[TaskType, float, Dict[str, any]]:
        """
        Classify a user message.

        Args:
            user_message: User's input message

        Returns:
            Tuple of (task_type, confidence, reasoning)
            - task_type: TaskType enum
            - confidence: 0-1 confidence score
            - reasoning: Dict with classification details
        """
        # Calculate scores
        research_score = self._count_matches(user_message, self.research_patterns)
        complexity_score = self._count_matches(user_message, self.complexity_patterns)
        simple_score = self._count_matches(user_message, self.simple_patterns)

        # Additional heuristics
        has_multi_stage = any([
            'first' in user_message.lower() and 'then' in user_message.lower(),
            'step 1' in user_message.lower() or 'step 2' in user_message.lower(),
            user_message.count(',') > 3,  # Multiple comma-separated tasks
            len(user_message) > 500,  # Very long detailed request
            user_message.count(' and ') > 3,  # Multiple 'and' connectors
            sum(1 for word in ['download', 'modify', 'extract', 'collect', 'train', 'embed', 'implement'] if word in user_message.lower()) >= 4,  # Many action verbs
        ])

        has_research_goal = any([
            'research' in user_message.lower(),
            'compare' in user_message.lower(),
            'benchmark' in user_message.lower(),
            'evaluate' in user_message.lower(),
            'experiment' in user_message.lower() and 'run' in user_message.lower(),
        ])

        # Decision logic
        reasoning = {
            'research_score': research_score,
            'complexity_score': complexity_score,
            'simple_score': simple_score,
            'has_multi_stage': has_multi_stage,
            'has_research_goal': has_research_goal,
            'message_length': len(user_message),
        }

        # Classification thresholds
        if has_research_goal or (research_score >= 2 and complexity_score >= 1):
            # Strong research indicators
            task_type = TaskType.COMPLEX_RESEARCH
            confidence = min(0.95, 0.6 + research_score * 0.15 + complexity_score * 0.1)
            reasoning['decision'] = 'Strong research indicators detected'

        elif has_multi_stage and (research_score >= 1 or complexity_score >= 2):
            # Multi-stage with some research/complexity
            task_type = TaskType.COMPLEX_RESEARCH
            confidence = min(0.9, 0.5 + research_score * 0.15 + complexity_score * 0.15)
            reasoning['decision'] = 'Multi-stage complex task detected'

        elif complexity_score >= 3:
            # High complexity even without explicit research keywords
            task_type = TaskType.COMPLEX_RESEARCH
            confidence = min(0.85, 0.5 + complexity_score * 0.15)
            reasoning['decision'] = 'High complexity detected'

        elif simple_score > research_score and simple_score > complexity_score:
            # Clear simple task
            task_type = TaskType.SIMPLE
            confidence = min(0.9, 0.6 + simple_score * 0.1)
            reasoning['decision'] = 'Simple implementation task'

        elif research_score == 0 and complexity_score <= 1:
            # No research indicators, low complexity
            task_type = TaskType.SIMPLE
            confidence = 0.7
            reasoning['decision'] = 'No research indicators, treating as simple'

        else:
            # Ambiguous - default to simple unless clear research intent
            if research_score > 0:
                task_type = TaskType.HYBRID
                confidence = 0.6
                reasoning['decision'] = 'Hybrid task - some research + implementation'
            else:
                task_type = TaskType.SIMPLE
                confidence = 0.65
                reasoning['decision'] = 'Ambiguous, defaulting to simple'

        logger.info(f"Task classified as {task_type.value} (confidence: {confidence:.2f})")
        logger.debug(f"Classification reasoning: {reasoning}")

        return task_type, confidence, reasoning

    def _count_matches(self, text: str, patterns: List[re.Pattern]) -> int:
        """Count how many patterns match in the text"""
        return sum(1 for pattern in patterns if pattern.search(text))

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
        "Find papers on reinforcement learning and summarize key findings",

        # Complex tasks (should trigger)
        "Download postgres source code, extract features, train a model, and embed it",
        "First collect data from API, then train model, finally deploy to production",
        "Modify postgres and pg_duckdb source code, extract pre-opt features from postgres kernel and log to files, collect dual-execution data and train ML model",

        # Simple tasks (should NOT trigger)
        "Fix the bug in the login function",
        "Add a new method to calculate sum",
        "Refactor the database connection code",
        "Install numpy and run the script",
        "Create a function that returns hello world",
    ]

    classifier = TaskClassifier()

    for message in test_cases:
        should_trigger, task_type, confidence, reasoning = classifier.should_trigger_research(message)
        print(f"\nMessage: {message[:80]}...")
        print(f"  Type: {task_type.value}, Trigger: {should_trigger}, Confidence: {confidence:.2f}")
        print(f"  Reasoning: {reasoning['decision']}")


if __name__ == "__main__":
    test_classifier()
