"""
Unit tests for custom exceptions
"""

import pytest
from app.core.exceptions import (
    UAgentException,
    ConfigurationError,
    RouterException,
    ClassificationError,
    InvalidRequestError,
    ThresholdError,
    EngineError,
    OpenHandsError,
    WorkspaceError,
    ExecutionError
)


class TestExceptions:
    """Test custom exception classes"""

    def test_uagent_exception_creation(self):
        """Test UAgentException creation and inheritance"""
        error = UAgentException("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
        assert error.args == ("Test error message",)

    def test_configuration_error_creation(self):
        """Test ConfigurationError creation"""
        error = ConfigurationError("Invalid configuration")

        assert str(error) == "Invalid configuration"
        assert isinstance(error, UAgentException)

    def test_router_exception_creation(self):
        """Test RouterException creation"""
        error = RouterException("Router error")

        assert str(error) == "Router error"
        assert isinstance(error, UAgentException)

    def test_classification_error_creation(self):
        """Test ClassificationError creation"""
        error = ClassificationError("Classification failed")

        assert str(error) == "Classification failed"
        assert isinstance(error, RouterException)
        assert isinstance(error, UAgentException)

    def test_invalid_request_error_creation(self):
        """Test InvalidRequestError creation"""
        error = InvalidRequestError("Invalid request format")

        assert str(error) == "Invalid request format"
        assert isinstance(error, RouterException)
        assert isinstance(error, UAgentException)

    def test_threshold_error_creation(self):
        """Test ThresholdError creation"""
        error = ThresholdError("Confidence below threshold")

        assert str(error) == "Confidence below threshold"
        assert isinstance(error, RouterException)
        assert isinstance(error, UAgentException)

    def test_engine_error_creation(self):
        """Test EngineError creation"""
        error = EngineError("Engine processing failed")

        assert str(error) == "Engine processing failed"
        assert isinstance(error, UAgentException)

    def test_openhands_error_creation(self):
        """Test OpenHandsError creation"""
        error = OpenHandsError("OpenHands integration failed")

        assert str(error) == "OpenHands integration failed"
        assert isinstance(error, UAgentException)

    def test_workspace_error_creation(self):
        """Test WorkspaceError creation"""
        error = WorkspaceError("Workspace creation failed")

        assert str(error) == "Workspace creation failed"
        assert isinstance(error, OpenHandsError)
        assert isinstance(error, UAgentException)

    def test_execution_error_creation(self):
        """Test ExecutionError creation"""
        error = ExecutionError("Code execution failed")

        assert str(error) == "Code execution failed"
        assert isinstance(error, OpenHandsError)
        assert isinstance(error, UAgentException)

    def test_error_inheritance_hierarchy(self):
        """Test that all errors inherit properly"""
        # Router hierarchy
        classification_error = ClassificationError("test")
        assert isinstance(classification_error, RouterException)
        assert isinstance(classification_error, UAgentException)

        invalid_request_error = InvalidRequestError("test")
        assert isinstance(invalid_request_error, RouterException)
        assert isinstance(invalid_request_error, UAgentException)

        threshold_error = ThresholdError("test")
        assert isinstance(threshold_error, RouterException)
        assert isinstance(threshold_error, UAgentException)

        # OpenHands hierarchy
        workspace_error = WorkspaceError("test")
        assert isinstance(workspace_error, OpenHandsError)
        assert isinstance(workspace_error, UAgentException)

        execution_error = ExecutionError("test")
        assert isinstance(execution_error, OpenHandsError)
        assert isinstance(execution_error, UAgentException)

    def test_error_chaining(self):
        """Test error chaining functionality"""
        original_error = ValueError("Original error")
        try:
            raise UAgentException("Wrapped error") from original_error
        except UAgentException as e:
            assert str(e) == "Wrapped error"
            assert e.__cause__ == original_error

    def test_all_exceptions_are_subclasses(self):
        """Test that all custom exceptions are subclasses of appropriate base classes"""
        exceptions_to_test = [
            (ConfigurationError, UAgentException),
            (RouterException, UAgentException),
            (ClassificationError, RouterException),
            (InvalidRequestError, RouterException),
            (ThresholdError, RouterException),
            (EngineError, UAgentException),
            (OpenHandsError, UAgentException),
            (WorkspaceError, OpenHandsError),
            (ExecutionError, OpenHandsError)
        ]

        for exception_class, base_class in exceptions_to_test:
            assert issubclass(exception_class, base_class)
            assert issubclass(exception_class, UAgentException)
            assert issubclass(exception_class, Exception)

    def test_exception_messages(self):
        """Test exception message handling"""
        test_message = "Test error message"

        exceptions = [
            UAgentException(test_message),
            ConfigurationError(test_message),
            RouterException(test_message),
            ClassificationError(test_message),
            InvalidRequestError(test_message),
            ThresholdError(test_message),
            EngineError(test_message),
            OpenHandsError(test_message),
            WorkspaceError(test_message),
            ExecutionError(test_message)
        ]

        for exception in exceptions:
            assert str(exception) == test_message
            assert exception.args == (test_message,)

    def test_empty_message_handling(self):
        """Test handling of empty error messages"""
        error = UAgentException("")
        assert str(error) == ""
        assert error.args == ("",)

    def test_no_message_handling(self):
        """Test handling when no message is provided"""
        error = UAgentException()
        assert str(error) == ""
        assert error.args == ()