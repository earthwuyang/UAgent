"""
UAgent Research Extension for OpenHands

Provides advanced research capabilities including:
- Scientific research with experiment execution
- Code repository analysis (RepoMaster)
- ROMA tree-based research orchestration
- Idea and hypothesis generation (AI Scientist style)
"""

__version__ = "0.1.0"
__author__ = "UAgent Team"
__license__ = "MIT"

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Extension metadata
EXTENSION_NAME = "uagent_research"
EXTENSION_DESCRIPTION = "Advanced research capabilities for scientific experiments"
EXTENSION_VERSION = __version__

# Will be populated by extension initialization
_research_engines: Dict[str, Any] = {}
_active_experiments: Dict[str, Any] = {}

def get_research_engine(engine_type: str):
    """Get research engine instance"""
    return _research_engines.get(engine_type)

def register_research_engine(engine_type: str, engine):
    """Register research engine"""
    _research_engines[engine_type] = engine
    logger.info(f"Registered research engine: {engine_type}")

def get_active_experiment(experiment_id: str):
    """Get active experiment"""
    return _active_experiments.get(experiment_id)

def register_active_experiment(experiment_id: str, experiment):
    """Register active experiment"""
    _active_experiments[experiment_id] = experiment
    logger.info(f"Registered active experiment: {experiment_id}")

def unregister_active_experiment(experiment_id: str):
    """Unregister active experiment"""
    if experiment_id in _active_experiments:
        del _active_experiments[experiment_id]
        logger.info(f"Unregistered experiment: {experiment_id}")
