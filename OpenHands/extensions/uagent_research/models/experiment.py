"""Experiment data model"""

from sqlalchemy import Column, String, Text, Integer, Float, DateTime, JSON, Enum as SQLEnum, Index
from sqlalchemy.sql import func
from datetime import datetime
import enum
from .base import Base


class ExperimentStatus(str, enum.Enum):
    """Experiment execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ExperimentType(str, enum.Enum):
    """Type of research experiment"""
    SCIENTIFIC = "scientific"
    CODE = "code"
    ROMA = "roma"


class Experiment(Base):
    """
    Experiment model for tracking research experiments.

    Stores experiment configuration, progress, results, and metadata.
    """
    __tablename__ = "experiments"

    # Primary key
    id = Column(String(64), primary_key=True)

    # Foreign keys
    session_id = Column(String(64), nullable=False, index=True)
    parent_experiment_id = Column(String(64), nullable=True, index=True)

    # Metadata
    experiment_type = Column(SQLEnum(ExperimentType), nullable=False)
    goal = Column(Text, nullable=False)
    status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.PENDING, index=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Progress tracking
    progress_percentage = Column(Float, default=0.0)
    current_step = Column(String(256), nullable=True)
    total_steps = Column(Integer, nullable=True)
    steps_completed = Column(Integer, default=0)

    # Configuration
    config = Column(JSON, nullable=True)
    workspace_path = Column(String(512), nullable=True)

    # Results
    results = Column(JSON, nullable=True)
    artifacts = Column(JSON, nullable=True)
    logs = Column(JSON, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_type = Column(String(128), nullable=True)
    error_traceback = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Resource usage
    execution_time_seconds = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    api_calls = Column(Integer, default=0)

    # Indexes for common queries
    __table_args__ = (
        Index('idx_session_status', 'session_id', 'status'),
        Index('idx_created_at', 'created_at'),
        Index('idx_type_status', 'experiment_type', 'status'),
    )

    def __repr__(self):
        return f"<Experiment {self.id} ({self.status})>"

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'parent_experiment_id': self.parent_experiment_id,
            'experiment_type': self.experiment_type.value,
            'goal': self.goal,
            'status': self.status.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': {
                'percentage': self.progress_percentage,
                'current_step': self.current_step,
                'steps_completed': self.steps_completed,
                'total_steps': self.total_steps,
            },
            'config': self.config,
            'workspace_path': self.workspace_path,
            'results': self.results,
            'artifacts': self.artifacts,
            'error': {
                'message': self.error_message,
                'type': self.error_type,
                'traceback': self.error_traceback,
            } if self.error_message else None,
            'metrics': {
                'execution_time_seconds': self.execution_time_seconds,
                'memory_usage_mb': self.memory_usage_mb,
                'tokens_used': self.tokens_used,
                'api_calls': self.api_calls,
            }
        }
