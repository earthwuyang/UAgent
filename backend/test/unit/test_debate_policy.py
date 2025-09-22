from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.debate.policy import DebatePolicy, should_debate


def test_should_debate_triggers_on_low_confidence():
    policy = DebatePolicy(min_confidence=0.7)
    assert should_debate({"confidence": 0.4}, policy) is True
    assert should_debate({"confidence": 0.9}, policy) is False


def test_should_debate_triggers_on_disagreement():
    policy = DebatePolicy(min_confidence=0.9)
    assert should_debate({"confidence": 0.95, "disagreement_score": 0.5}, policy) is True
