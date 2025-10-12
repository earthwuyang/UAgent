"""
Configuration utilities for UAgent Research extension.

Provides helper functions to load configuration from environment variables,
ensuring consistency with main OpenHands configuration loading.
"""

import os
import logging
from typing import Optional
from pydantic import SecretStr
from dotenv import load_dotenv

from openhands.core.config import LLMConfig

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger(__name__)


def load_llm_config_from_env() -> LLMConfig:
    """
    Load LLM configuration from environment variables.
    
    Reads configuration from environment variables following the same pattern
    as main OpenHands:
    - LLM_MODEL: Model identifier (e.g., "openai/gpt-4o", "qwen3-coder-plus")
    - LLM_API_KEY: API key for the LLM provider
    - LLM_BASE_URL: Base URL for the LLM API (optional)
    - LLM_NUM_RETRIES: Number of retries (optional)
    - LLM_RETRY_MIN_WAIT: Minimum wait time between retries (optional)
    - LLM_RETRY_MAX_WAIT: Maximum wait time between retries (optional)
    - LLM_TIMEOUT: Request timeout in seconds (optional)
    - LLM_TEMPERATURE: Temperature for sampling (optional)
    
    Returns:
        LLMConfig: Configured LLM settings
        
    Example:
        >>> config = load_llm_config_from_env()
        >>> print(config.model)
        'openai/qwen3-coder-plus'
    """
    # Read from environment variables with fallbacks
    model = os.getenv('LLM_MODEL', 'claude-sonnet-4-20250514')
    api_key_str = os.getenv('LLM_API_KEY')
    base_url = os.getenv('LLM_BASE_URL')
    
    # Optional numeric parameters
    num_retries = int(os.getenv('LLM_NUM_RETRIES', '8'))
    retry_min_wait = int(os.getenv('LLM_RETRY_MIN_WAIT', '15'))
    retry_max_wait = int(os.getenv('LLM_RETRY_MAX_WAIT', '120'))
    timeout = int(os.getenv('LLM_TIMEOUT', '600')) if os.getenv('LLM_TIMEOUT') else None
    temperature = float(os.getenv('LLM_TEMPERATURE', '0.0'))
    
    # Convert API key to SecretStr if provided
    api_key = SecretStr(api_key_str) if api_key_str else None
    
    # Log configuration (without exposing sensitive data)
    logger.info(f"Loading LLM config from environment:")
    logger.info(f"  Model: {model}")
    logger.info(f"  Base URL: {base_url if base_url else 'default'}")
    logger.info(f"  API Key: {'set' if api_key else 'not set'}")
    logger.info(f"  Retries: {num_retries}")
    logger.info(f"  Timeout: {timeout}")
    
    # Create LLMConfig with environment values
    config = LLMConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        num_retries=num_retries,
        retry_min_wait=retry_min_wait,
        retry_max_wait=retry_max_wait,
        timeout=timeout,
        temperature=temperature,
    )
    
    return config


def get_llm_config_for_research(custom_config: Optional[LLMConfig] = None) -> LLMConfig:
    """
    Get LLM configuration for research agents.
    
    If a custom config is provided, use it. Otherwise, load from environment.
    
    Args:
        custom_config: Optional custom LLMConfig to use
        
    Returns:
        LLMConfig: The LLM configuration to use
        
    Example:
        >>> # Use environment config
        >>> config = get_llm_config_for_research()
        >>> 
        >>> # Use custom config
        >>> custom = LLMConfig(model="gpt-4o", api_key="...")
        >>> config = get_llm_config_for_research(custom)
    """
    if custom_config:
        logger.info("Using provided custom LLM config")
        return custom_config
    
    logger.info("Loading LLM config from environment variables")
    return load_llm_config_from_env()
