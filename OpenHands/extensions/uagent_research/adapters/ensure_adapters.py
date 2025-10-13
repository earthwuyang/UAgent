"""
Centralized adapter registration utility to avoid duplication.

This module provides a single place to ensure research adapters are registered
with the adapter registry. Used by both middleware and coordinator.
"""

import logging

logger = logging.getLogger(__name__)

_adapters_registered = False


def ensure_research_adapters_registered() -> bool:
    """
    Ensure all research adapters are registered with the adapter registry.
    
    This function is idempotent - it checks if adapters are already registered
    and only registers them once to avoid duplication.
    
    Registers:
    - DeepResearchAdapter: Web research and information gathering
    - RepoMasterAdapter: Code repository search and analysis
    - CodeActAdapter: Code execution and validation
    
    Returns:
        True if adapters were registered or already registered, False on error
    """
    global _adapters_registered
    
    # Check current registry state
    from .base.agent_adapter import adapter_registry
    
    try:
        current_adapters = list(adapter_registry.get_all_adapters())
        logger.info(f"[ADAPTER_REGISTRY] Current state: {len(current_adapters)} adapters")
        
        if len(current_adapters) >= 3:
            logger.info(f"[ADAPTER_REGISTRY] Adapters already registered: {[a.name for a in current_adapters]}")
            return True
    except Exception as e:
        logger.warning(f"[ADAPTER_REGISTRY] Error checking registry: {e}")
    
    logger.info(f"[ADAPTER_REGISTRY] ensure_research_adapters_registered() called")
    logger.info(f"[ADAPTER_REGISTRY] Already registered: {_adapters_registered}")
    
    if _adapters_registered:
        logger.debug("Research adapters already registered, skipping")
        return True
    
    try:
        from .base.agent_adapter import adapter_registry
        from .deepresearch.adapter import DeepResearchAdapter
        from .repomaster.adapter import RepoMasterAdapter
        
        # Try to import CodeActAdapter, but handle gracefully if it fails
        CodeActAdapter = None
        try:
            from .codeact.adapter import CodeActAdapter
        except ImportError as e:
            logger.warning(f"[ADAPTER_REGISTRY] CodeActAdapter import failed (requires full OpenHands environment): {e}")

        # Check if adapters are already in registry
        existing = set()
        if hasattr(adapter_registry, '_adapters'):
            existing = set(adapter_registry._adapters.keys())
        elif hasattr(adapter_registry, 'adapters'):
            existing = set(adapter_registry.adapters.keys())
        
        # Only register if not already present
        if 'deepresearch' not in existing:
            adapter_registry.register(DeepResearchAdapter(config={}))
            logger.info(f"[ADAPTER_REGISTRY] DeepResearchAdapter registered successfully")
        if 'repomaster' not in existing:
            adapter_registry.register(RepoMasterAdapter(config={}))
            logger.info(f"[ADAPTER_REGISTRY] RepoMasterAdapter registered successfully")
        if 'codeact' not in existing and CodeActAdapter is not None:
            adapter_registry.register(CodeActAdapter(config={}))
            logger.info(f"[ADAPTER_REGISTRY] CodeActAdapter registered successfully")

        _adapters_registered = True
        
        # Log final state
        final_adapters = set()
        if hasattr(adapter_registry, '_adapters'):
            final_adapters = set(adapter_registry._adapters.keys())
        logger.info(f"[ADAPTER_REGISTRY] Registration complete. Final adapters: {final_adapters}")
        logger.info("Research adapters registered successfully: deepresearch, repomaster, codeact")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register research adapters: {e}", exc_info=True)
        return False

