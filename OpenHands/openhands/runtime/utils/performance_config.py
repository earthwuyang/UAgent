"""Performance configuration for OpenHands runtime optimizations."""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PerformanceConfig:
    """Configuration for runtime performance optimizations."""

    # Bash session polling and timeout settings
    bash_poll_interval: float = 1.0  # Reduced from 0.5s
    bash_no_change_timeout: int = 15  # Reduced from 30s
    bash_adaptive_polling: bool = True

    # APT lock management settings
    apt_max_retries: int = 10
    apt_retry_delay: int = 5
    apt_lock_timeout: int = 300
    apt_force_unlock_threshold: int = 3  # Retries before considering force unlock

    # Logging levels for different components
    bash_debug_level: str = "INFO"  # Reduced from DEBUG to reduce log noise
    apt_debug_level: str = "INFO"

    # Timeout overrides for common operations
    default_command_timeout: int = 120
    apt_command_timeout: int = 600  # 10 minutes for package operations
    build_command_timeout: int = 1800  # 30 minutes for builds

    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Create configuration from environment variables."""
        return cls(
            bash_poll_interval=float(os.getenv('OPENHANDS_BASH_POLL_INTERVAL', cls.bash_poll_interval)),
            bash_no_change_timeout=int(os.getenv('OPENHANDS_BASH_NO_CHANGE_TIMEOUT', cls.bash_no_change_timeout)),
            bash_adaptive_polling=os.getenv('OPENHANDS_BASH_ADAPTIVE_POLLING', 'true').lower() == 'true',

            apt_max_retries=int(os.getenv('OPENHANDS_APT_MAX_RETRIES', cls.apt_max_retries)),
            apt_retry_delay=int(os.getenv('OPENHANDS_APT_RETRY_DELAY', cls.apt_retry_delay)),
            apt_lock_timeout=int(os.getenv('OPENHANDS_APT_LOCK_TIMEOUT', cls.apt_lock_timeout)),
            apt_force_unlock_threshold=int(os.getenv('OPENHANDS_APT_FORCE_UNLOCK_THRESHOLD', cls.apt_force_unlock_threshold)),

            bash_debug_level=os.getenv('OPENHANDS_BASH_DEBUG_LEVEL', cls.bash_debug_level),
            apt_debug_level=os.getenv('OPENHANDS_APT_DEBUG_LEVEL', cls.apt_debug_level),

            default_command_timeout=int(os.getenv('OPENHANDS_DEFAULT_COMMAND_TIMEOUT', cls.default_command_timeout)),
            apt_command_timeout=int(os.getenv('OPENHANDS_APT_COMMAND_TIMEOUT', cls.apt_command_timeout)),
            build_command_timeout=int(os.getenv('OPENHANDS_BUILD_COMMAND_TIMEOUT', cls.build_command_timeout)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'bash_poll_interval': self.bash_poll_interval,
            'bash_no_change_timeout': self.bash_no_change_timeout,
            'bash_adaptive_polling': self.bash_adaptive_polling,
            'apt_max_retries': self.apt_max_retries,
            'apt_retry_delay': self.apt_retry_delay,
            'apt_lock_timeout': self.apt_lock_timeout,
            'apt_force_unlock_threshold': self.apt_force_unlock_threshold,
            'bash_debug_level': self.bash_debug_level,
            'apt_debug_level': self.apt_debug_level,
            'default_command_timeout': self.default_command_timeout,
            'apt_command_timeout': self.apt_command_timeout,
            'build_command_timeout': self.build_command_timeout,
        }

    def get_timeout_for_command(self, command: str) -> int:
        """Get appropriate timeout for a command based on its type."""
        command_lower = command.lower()

        # APT and package management commands
        if any(cmd in command_lower for cmd in ['apt', 'dpkg', 'pip install', 'npm install']):
            return self.apt_command_timeout

        # Build commands
        if any(cmd in command_lower for cmd in ['make', 'cmake', 'gcc', 'g++', 'mvn', 'gradle', 'cargo build']):
            return self.build_command_timeout

        # Default timeout
        return self.default_command_timeout


# Global performance configuration instance
perf_config = PerformanceConfig.from_env()