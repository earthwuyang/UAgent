from dotenv import load_dotenv

load_dotenv()


from openhands.agenthub import (  # noqa: E402
    codeact_agent,
    dummy_agent,
    loc_agent,
    readonly_agent,
)

# Conditionally import browsing agents if dependencies are available
try:
    from openhands.agenthub import browsing_agent, visualbrowsing_agent
    _browsing_available = True
except ImportError:
    browsing_agent = None
    visualbrowsing_agent = None
    _browsing_available = False
from openhands.controller.agent import Agent  # noqa: E402

__all__ = [
    'Agent',
    'codeact_agent',
    'dummy_agent',
    'readonly_agent',
    'loc_agent',
]

# Add browsing agents to __all__ only if available
if _browsing_available:
    __all__.extend(['browsing_agent', 'visualbrowsing_agent'])
