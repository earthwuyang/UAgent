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

# Agent coordination configuration
# Coordinators should stay responsive while research runs in background
ENABLE_AGENT_COORDINATION = os.getenv('ENABLE_AGENT_COORDINATION', 'true').lower() == 'true'
RESEARCH_POLL_INTERVAL = int(os.getenv('RESEARCH_POLL_INTERVAL', '10'))
PROGRESS_CACHE_TTL = float(os.getenv('PROGRESS_CACHE_TTL', '2.0'))

# Node Generation Configuration
# Configuration for intelligent node expansion using LLM

# Enable LLM-based node generation (vs hardcoded placeholders)
ENABLE_INTELLIGENT_EXPANSION = os.getenv('RESEARCH_ENABLE_INTELLIGENT_EXPANSION', 'true').lower() == 'true'

# Maximum number of ideas to generate from root
MAX_RESEARCH_IDEAS = int(os.getenv('RESEARCH_MAX_IDEAS', '3'))

# Maximum hypotheses per idea
MAX_HYPOTHESES_PER_IDEA = int(os.getenv('RESEARCH_MAX_HYPOTHESES', '2'))

# Maximum experiments per hypothesis
MAX_EXPERIMENTS_PER_HYPOTHESIS = int(os.getenv('RESEARCH_MAX_EXPERIMENTS', '1'))

# Number of retries for failed LLM calls
IDEA_GENERATION_RETRY_COUNT = int(os.getenv('RESEARCH_IDEA_RETRY_COUNT', '2'))

# To disable automatic research triggering:
# export ENABLE_AUTO_RESEARCH_TRIGGER=false

# To disable intelligent expansion:
# export RESEARCH_ENABLE_INTELLIGENT_EXPANSION=false

# Research auto-trigger is now ENABLED by default

# Logging configuration
# Enable verbose debug logging for research flow
RESEARCH_DEBUG_LOGGING = os.getenv('RESEARCH_DEBUG_LOGGING', 'false').lower() == 'true'

# Log when tree state is updated and published
# Set to 'true' to see when tree snapshots are created and sent to API/WebSocket
# Useful for debugging "disconnected" issues in frontend
RESEARCH_LOG_TREE_UPDATES = os.getenv('RESEARCH_LOG_TREE_UPDATES', 'true').lower() == 'true'

# Log WebSocket connection and broadcast events
# Set to 'true' to see WebSocket client connections and message broadcasts
# Useful for debugging real-time update issues
RESEARCH_LOG_WEBSOCKET = os.getenv('RESEARCH_LOG_WEBSOCKET', 'true').lower() == 'true'

# Log configuration on import (only if debug logging enabled)
if RESEARCH_DEBUG_LOGGING:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("🔧 Research Extension Configuration:")
    logger.info(f"  Auto-trigger: {ENABLE_AUTO_RESEARCH_TRIGGER}")
    logger.info(f"  Confidence threshold: {RESEARCH_CONFIDENCE_THRESHOLD}")
    logger.info(f"  Max iterations: {RESEARCH_MAX_ITERATIONS}")
    logger.info(f"  Max cost: ${RESEARCH_MAX_COST}")
    logger.info(f"  Max parallel: {RESEARCH_MAX_PARALLEL}")
    logger.info(f"  Debug logging: {RESEARCH_DEBUG_LOGGING}")
    logger.info(f"  Log tree updates: {RESEARCH_LOG_TREE_UPDATES}")
    logger.info(f"  Log WebSocket: {RESEARCH_LOG_WEBSOCKET}")
