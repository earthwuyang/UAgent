"""Global application state management"""

from typing import Dict, Any


# Global application state
app_state: Dict[str, Any] = {}


def get_app_state() -> Dict[str, Any]:
    """Get application state"""
    return app_state


def set_app_state(state: Dict[str, Any]) -> None:
    """Set application state"""
    global app_state
    app_state.update(state)


def clear_app_state() -> None:
    """Clear application state"""
    global app_state
    app_state.clear()