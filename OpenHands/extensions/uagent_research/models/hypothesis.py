"""Hypothesis data model"""

from sqlalchemy import Column, String, Text, Float, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy import JSON
from .base import Base


class Hypothesis(Base):
    """
    Research hypothesis model for testable hypotheses.

    Stores hypotheses generated from ideas with experimental design.
    """
    __tablename__ = "hypotheses"

    # Primary key
    id = Column(String(64), primary_key=True)

    # Foreign keys
    idea_id = Column(String(64), nullable=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)

    # Content
    statement = Column(Text, nullable=False)
    null_hypothesis = Column(Text, nullable=True)

    # Experimental design
    testability_score = Column(Float, nullable=True)
    expected_outcome = Column(Text, nullable=True)
    experimental_design = Column(JSON, nullable=True)

    # Testing
    tested = Column(Boolean, default=False)
    experiment_id = Column(String(64), nullable=True)

    # Results
    result = Column(String(32), nullable=True)  # supported, rejected, inconclusive
    confidence = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    tested_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Hypothesis {self.id}: {self.statement[:50]}>"

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'idea_id': self.idea_id,
            'session_id': self.session_id,
            'statement': self.statement,
            'null_hypothesis': self.null_hypothesis,
            'testability_score': self.testability_score,
            'expected_outcome': self.expected_outcome,
            'experimental_design': self.experimental_design,
            'tested': self.tested,
            'experiment_id': self.experiment_id,
            'result': self.result,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'tested_at': self.tested_at.isoformat() if self.tested_at else None,
        }
