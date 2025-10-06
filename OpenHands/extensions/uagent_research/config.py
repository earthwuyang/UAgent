"""
UAgent Research Configuration

Configuration for automatic research triggering.
"""

import os

# Research Middleware Configuration
# Override with environment variable or modify directly below
ENABLE_AUTO_RESEARCH_TRIGGER = os.getenv('ENABLE_AUTO_RESEARCH_TRIGGER', 'true').lower() == 'true'  # ENABLED by default
RESEARCH_CONFIDENCE_THRESHOLD = float(os.getenv('RESEARCH_CONFIDENCE_THRESHOLD', '0.7'))
RESEARCH_MAX_ITERATIONS = int(os.getenv('RESEARCH_MAX_ITERATIONS', '50'))
RESEARCH_MAX_COST = float(os.getenv('RESEARCH_MAX_COST', '10.0'))
RESEARCH_MAX_PARALLEL = int(os.getenv('RESEARCH_MAX_PARALLEL', '3'))

# To disable automatic research triggering:
# export ENABLE_AUTO_RESEARCH_TRIGGER=false

# Research auto-trigger is now ENABLED by default
