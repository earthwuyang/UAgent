"""
OpenHands Core Module.

This module contains core functionality for the OpenHands system including
dependency analysis, optimistic concurrency control validation, and other
foundational components.
"""

# Import existing core components (if any)
try:
    from openhands.core.config import *
except ImportError:
    pass

try:
    from openhands.core.exceptions import *
except ImportError:
    pass

# Import OCC Validator components
from openhands.core.occ_validator import (
    OCCValidator,
    ReadWriteTracker,
    ASTRegionSignature,
    ValidationResult,
    ConflictDetail,
    ConflictType,
    Region,
)

from openhands.core.occ_validator_integration import (
    AgentReadWriteTracker,
    OCCValidatorManager,
    FileTrackingContext,
    OCCConfig,
    install_occ_hooks,
)

# Import dependency analyzer if available
try:
    from openhands.core.dependency_analyzer import (
        DependencyAnalyzer,
        DependencyAnalyzerConfig,
    )
except ImportError:
    pass

__all__ = [
    # OCC Validator core classes
    'OCCValidator',
    'ReadWriteTracker',
    'ASTRegionSignature',
    'ValidationResult',
    'ConflictDetail',
    'ConflictType',
    'Region',
    
    # OCC Validator integration classes
    'AgentReadWriteTracker',
    'OCCValidatorManager',
    'FileTrackingContext',
    'OCCConfig',
    'install_occ_hooks',
    
    # Dependency analyzer (if available)
    'DependencyAnalyzer',
    'DependencyAnalyzerConfig',
]
