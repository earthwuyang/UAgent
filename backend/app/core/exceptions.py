"""Core exceptions for UAgent application"""


class UAgentException(Exception):
    """Base exception for all UAgent errors"""
    pass


class ConfigurationError(UAgentException):
    """Configuration related errors"""
    pass


class RouterException(UAgentException):
    """Base exception for router errors"""
    pass


class ClassificationError(RouterException):
    """Error during classification process"""
    pass


class InvalidRequestError(RouterException):
    """Invalid request format or content"""
    pass


class ThresholdError(RouterException):
    """Classification confidence below threshold"""
    pass


class EngineError(UAgentException):
    """Base exception for research engine errors"""
    pass


class OpenHandsError(UAgentException):
    """OpenHands integration errors"""
    pass


class WorkspaceError(OpenHandsError):
    """Workspace management errors"""
    pass


class ExecutionError(OpenHandsError):
    """Code execution errors"""
    pass