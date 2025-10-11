"""Custom exceptions for the UAgent research extension."""


class ResearchError(Exception):
    """Base exception for research extension errors."""


class ConfigurationError(ResearchError):
    """Raised when configuration values are invalid."""


class ValidationError(ResearchError):
    """Raised for validation failures."""


class TreeIntegrityError(ValidationError):
    """Raised when the research tree structure is invalid."""


class ExternalServiceError(ResearchError):
    """Raised for failures communicating with external services."""
