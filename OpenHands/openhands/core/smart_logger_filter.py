"""Smart logging filter to reduce excessive log noise while keeping important information."""

import logging
import re
import time
from collections import defaultdict
from typing import Dict, Set, Pattern


class SmartLoggingFilter(logging.Filter):
    """
    Intelligent logging filter that suppresses repetitive and verbose messages
    while preserving important information.
    """

    def __init__(self, name: str = ""):
        super().__init__(name)

        # Rate limiting for repetitive messages
        self.message_counts: Dict[str, int] = defaultdict(int)
        self.last_message_time: Dict[str, float] = defaultdict(float)
        self.suppressed_counts: Dict[str, int] = defaultdict(int)

        # Configuration
        self.max_repetitions = 3  # Allow max 3 repetitions before suppressing
        self.rate_limit_window = 60  # Reset counts every 60 seconds
        self.burst_limit = 10  # Max messages per second for any pattern

        # Patterns to suppress (regex patterns)
        self.verbose_patterns: Set[Pattern] = {
            # Bash session polling and pane content
            re.compile(r'PANE CONTENT GOT after \d+\.\d+ seconds'),
            re.compile(r'BEGIN OF PANE CONTENT:'),
            re.compile(r'END OF PANE CONTENT:'),
            re.compile(r'SLEEPING for \d+\.?\d* seconds for next poll'),
            re.compile(r'GETTING PANE CONTENT at \d+\.\d+'),
            re.compile(r'CONTENT UPDATED DETECTED at \d+\.\d+'),
            re.compile(r'CHECKING NO CHANGE TIMEOUT'),
            re.compile(r'CHECKING HARD TIMEOUT'),

            # Agent controller verbose steps
            re.compile(r'LEVEL \d+ LOCAL STEP -?\d+ GLOBAL STEP \d+'),
            re.compile(r'Processing \d+ events from a total of \d+ events'),
            re.compile(r'Stepping agent after event:'),
            re.compile(r'Agent not stepping because of pending action'),

            # Server log forwarding (duplicate messages)
            re.compile(r'^server: '),

            # Memory and conversation repetitive logs
            re.compile(r'Visual browsing: (True|False)'),
            re.compile(r'Microagent knowledge recall'),

            # State management verbose logs
            re.compile(r'Saving state to session'),
            re.compile(r'shutdown_signal:\d+'),

            # LLM interaction repetitive patterns
            re.compile(r'Logging to .*/logs/llm/.*/.*\.log'),
        }

        # Important patterns to ALWAYS keep (never suppress)
        self.important_patterns: Set[Pattern] = {
            re.compile(r'ERROR|CRITICAL|FATAL', re.IGNORECASE),
            re.compile(r'Exception|Traceback', re.IGNORECASE),
            re.compile(r'APT lock|Lock timeout', re.IGNORECASE),
            re.compile(r'Command failed|Command timeout', re.IGNORECASE),
            re.compile(r'Agent.*completed|Agent.*failed', re.IGNORECASE),
            re.compile(r'Starting|Stopping|Initializing', re.IGNORECASE),
            re.compile(r'Authentication|Permission denied', re.IGNORECASE),
            re.compile(r'Connection.*lost|Network.*error', re.IGNORECASE),
        }

        # Specific loggers to reduce verbosity
        self.verbose_loggers = {
            'openhands.runtime.utils.bash',
            'openhands.controller.agent_controller',
            'openhands.runtime.impl.local.local_runtime',
            'openhands.memory',
            'openhands.agenthub.codeact_agent.codeact_agent',
        }

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records based on importance and repetition patterns.

        Returns:
            True if the record should be logged, False if it should be suppressed
        """
        message = record.getMessage()
        logger_name = record.name
        current_time = time.time()

        # Always log important messages (errors, exceptions, etc.)
        if self._is_important_message(message):
            return True

        # Always log if not from a verbose logger and not DEBUG level
        if logger_name not in self.verbose_loggers and record.levelno > logging.DEBUG:
            return True

        # Suppress DEBUG messages from verbose loggers unless they're important
        if record.levelno == logging.DEBUG and logger_name in self.verbose_loggers:
            if not self._is_important_message(message):
                return False

        # Check for verbose patterns that should be suppressed
        if self._is_verbose_message(message):
            # Use rate limiting for verbose messages
            message_key = self._get_message_key(message)

            # Reset counters if enough time has passed
            if current_time - self.last_message_time[message_key] > self.rate_limit_window:
                self.message_counts[message_key] = 0
                self.suppressed_counts[message_key] = 0

            self.last_message_time[message_key] = current_time
            self.message_counts[message_key] += 1

            # Allow the first few occurrences, then start suppressing
            if self.message_counts[message_key] <= self.max_repetitions:
                return True
            else:
                self.suppressed_counts[message_key] += 1

                # Every 50 suppressed messages, log a summary
                if self.suppressed_counts[message_key] % 50 == 0:
                    summary_record = logging.LogRecord(
                        name=record.name,
                        level=logging.INFO,
                        pathname=record.pathname,
                        lineno=record.lineno,
                        msg=f"[SUPPRESSED] {self.suppressed_counts[message_key]} similar messages in the last {self.rate_limit_window}s",
                        args=(),
                        exc_info=None
                    )
                    # Log the summary through the original logger
                    record.name = "openhands.core.smart_filter"
                    record.msg = summary_record.msg
                    record.args = ()
                    return True

                return False

        # Log everything else
        return True

    def _is_important_message(self, message: str) -> bool:
        """Check if a message matches important patterns."""
        return any(pattern.search(message) for pattern in self.important_patterns)

    def _is_verbose_message(self, message: str) -> bool:
        """Check if a message matches verbose patterns."""
        return any(pattern.search(message) for pattern in self.verbose_patterns)

    def _get_message_key(self, message: str) -> str:
        """Generate a key for rate limiting similar messages."""
        # Normalize the message by removing timestamps, numbers, and paths
        normalized = re.sub(r'\d+\.?\d*', 'N', message)  # Replace numbers
        normalized = re.sub(r'/[^\s]+', '/PATH', normalized)  # Replace paths
        normalized = re.sub(r'session_\w+', 'session_ID', normalized)  # Replace session IDs
        return normalized[:100]  # Limit key length


class ImportantOnlyFilter(logging.Filter):
    """
    Ultra-strict filter that only allows truly important messages.
    Use this for production environments where log noise must be minimized.
    """

    def __init__(self, name: str = ""):
        super().__init__(name)

        # Only these message types are allowed
        self.allowed_patterns = {
            re.compile(r'ERROR|CRITICAL|FATAL', re.IGNORECASE),
            re.compile(r'Exception|Traceback', re.IGNORECASE),
            re.compile(r'Command completed|Command failed', re.IGNORECASE),
            re.compile(r'Agent.*completed|Agent.*failed', re.IGNORECASE),
            re.compile(r'Starting.*agent|Stopping.*agent', re.IGNORECASE),
            re.compile(r'APT.*failed|Installation.*failed', re.IGNORECASE),
            re.compile(r'Authentication.*failed|Permission denied', re.IGNORECASE),
        }

        # Command execution results (important for users)
        self.command_patterns = {
            re.compile(r'COMMAND OUTPUT:'),
            re.compile(r'COMBINED OUTPUT:'),
        }

    def filter(self, record: logging.LogRecord) -> bool:
        """Only allow truly important messages."""
        message = record.getMessage()

        # Always allow ERROR and above
        if record.levelno >= logging.ERROR:
            return True

        # Allow specific important patterns
        if any(pattern.search(message) for pattern in self.allowed_patterns):
            return True

        # Allow command outputs (user needs to see results)
        if record.levelno >= logging.INFO and any(pattern.search(message) for pattern in self.command_patterns):
            return True

        # Suppress everything else
        return False


class ComponentLogLevelManager:
    """
    Manages log levels for different OpenHands components based on their verbosity.
    """

    # Component-specific log level configuration
    COMPONENT_LOG_LEVELS = {
        # Very verbose components - reduce to WARNING
        'openhands.runtime.utils.bash': logging.WARNING,
        'openhands.controller.agent_controller': logging.INFO,
        'openhands.runtime.impl.local.local_runtime': logging.WARNING,

        # Moderately verbose - reduce to INFO
        'openhands.memory': logging.INFO,
        'openhands.agenthub.codeact_agent': logging.INFO,
        'openhands.events': logging.INFO,

        # Keep at current level for important components
        'openhands.core': logging.INFO,
        'openhands.runtime.action_execution_server': logging.INFO,

        # Third-party libraries
        'litellm': logging.WARNING,
        'engineio': logging.WARNING,
        'socketio': logging.WARNING,
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
    }

    @classmethod
    def apply_component_levels(cls):
        """Apply optimized log levels to all components."""
        for component, level in cls.COMPONENT_LOG_LEVELS.items():
            logger = logging.getLogger(component)
            logger.setLevel(level)


# Filter configuration options
FILTER_MODES = {
    'smart': SmartLoggingFilter,      # Intelligent filtering with rate limiting
    'important': ImportantOnlyFilter,  # Only critical messages
    'normal': None,                   # No additional filtering
}


def setup_smart_logging(mode: str = 'smart'):
    """
    Setup smart logging filter for the OpenHands logger.

    Args:
        mode: 'smart', 'important', or 'normal'
    """
    from openhands.core.logger import openhands_logger

    # Remove existing filters
    openhands_logger.filters = [f for f in openhands_logger.filters
                               if not isinstance(f, (SmartLoggingFilter, ImportantOnlyFilter))]

    # Add the appropriate filter
    if mode in FILTER_MODES and FILTER_MODES[mode] is not None:
        filter_class = FILTER_MODES[mode]
        smart_filter = filter_class()
        openhands_logger.addFilter(smart_filter)

        # Also apply to handlers
        for handler in openhands_logger.handlers:
            # Remove existing smart filters
            handler.filters = [f for f in handler.filters
                             if not isinstance(f, (SmartLoggingFilter, ImportantOnlyFilter))]
            handler.addFilter(filter_class())

    # Apply component-specific log levels
    ComponentLogLevelManager.apply_component_levels()

    openhands_logger.info(f"Smart logging configured with mode: {mode}")