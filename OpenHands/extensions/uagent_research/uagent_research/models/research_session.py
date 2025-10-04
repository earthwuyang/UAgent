"""Research session data model"""

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.sql import func
import enum
from .base import Base


class SessionMode(str, enum.Enum):
    """Research session mode"""
    CHAT = "chat"
    RESEARCH = "research"
    HYBRID = "hybrid"


class ResearchSession(Base):
    """
    Research session model for tracking user research sessions.

    A session contains multiple experiments and maintains research state.
    """
    __tablename__ = "research_sessions"

    # Primary key
    id = Column(String(64), primary_key=True)

    # User info
    user_id = Column(String(64), nullable=True, index=True)
    mode = Column(SQLEnum(SessionMode), default=SessionMode.RESEARCH)

    # Metadata
    title = Column(String(256), nullable=True)
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # Array of strings

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_activity_at = Column(DateTime, default=func.now())

    # Session state
    state = Column(JSON, nullable=True)
    research_tree = Column(JSON, nullable=True)  # ROMA tree structure
    context = Column(JSON, nullable=True)

    # Settings
    settings = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<ResearchSession {self.id} ({self.mode})>"

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'mode': self.mode.value,
            'title': self.title,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_activity_at': self.last_activity_at.isoformat() if self.last_activity_at else None,
            'state': self.state,
            'research_tree': self.research_tree,
            'context': self.context,
            'settings': self.settings,
        }
