"""
Unit Tests for Research Middleware

Comprehensive tests for progress query and control intent features.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import middleware
try:
    from extensions.uagent_research.middleware.research_middleware import ResearchMiddleware
    from extensions.uagent_research.services.research_session_manager import ExperimentStatus
    from extensions.uagent_research.control.control_bus import ControlMessage
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytest.skip("Middleware not available", allow_module_level=True)


# ============================================================================
# Progress Query Detection Tests
# ============================================================================

class TestProgressQueryDetection:
    """Test detect_progress_query() with various phrasings."""
    
    def test_standard_progress_queries(self):
        """Test standard progress query patterns."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        # Standard patterns
        assert middleware.detect_progress_query("how's progress") is True
        assert middleware.detect_progress_query("how's progress?") is True
        assert middleware.detect_progress_query("what's the status") is True
        assert middleware.detect_progress_query("what's happening") is True
        assert middleware.detect_progress_query("how is it going") is True
        assert middleware.detect_progress_query("show me progress") is True
        assert middleware.detect_progress_query("check status") is True
        assert middleware.detect_progress_query("research status") is True
        assert middleware.detect_progress_query("update me") is True
    
    def test_case_insensitivity(self):
        """Test case insensitivity in detection."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        assert middleware.detect_progress_query("HOW'S PROGRESS") is True
        assert middleware.detect_progress_query("What's The Status") is True
        assert middleware.detect_progress_query("SHOW ME PROGRESS") is True
        assert middleware.detect_progress_query("Research Status") is True
    
    def test_apostrophe_variations(self):
        """Test with and without apostrophes."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        assert middleware.detect_progress_query("hows progress") is True
        assert middleware.detect_progress_query("how's progress") is True
        assert middleware.detect_progress_query("whats happening") is True
        assert middleware.detect_progress_query("what's happening") is True
    
    def test_negative_cases(self):
        """Test non-progress queries return False."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        assert middleware.detect_progress_query("hello") is False
        assert middleware.detect_progress_query("explain this code") is False
        assert middleware.detect_progress_query("write a function") is False
        assert middleware.detect_progress_query("") is False
        assert middleware.detect_progress_query("research quantum computing") is False


# ============================================================================
# Control Intent Detection Tests
# ============================================================================

class TestControlIntentDetection:
    """Test detect_control_intent() for each action type."""
    
    def test_pause_detection(self):
        """Test pause intent detection."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        result = middleware.detect_control_intent("pause research")
        assert result is not None
        action, target = result
        assert action == "pause"
        assert target == {}
        
        # Variations
        assert middleware.detect_control_intent("stop research")[0] == "pause"
        assert middleware.detect_control_intent("halt research")[0] == "pause"
        assert middleware.detect_control_intent("hold on")[0] == "pause"
    
    def test_resume_detection(self):
        """Test resume intent detection."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        result = middleware.detect_control_intent("resume research")
        assert result is not None
        action, target = result
        assert action == "resume"
        assert target == {}
        
        # Variations
        assert middleware.detect_control_intent("continue research")[0] == "resume"
        assert middleware.detect_control_intent("restart research")[0] == "resume"
        assert middleware.detect_control_intent("unpause")[0] == "resume"
    
    def test_cancel_detection(self):
        """Test cancel intent detection."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        result = middleware.detect_control_intent("cancel research")
        assert result is not None
        action, target = result
        assert action == "cancel"
        assert target == {}
        
        # Variations
        assert middleware.detect_control_intent("abort research")[0] == "cancel"
        assert middleware.detect_control_intent("kill research")[0] == "cancel"
        assert middleware.detect_control_intent("stop everything")[0] == "cancel"
    
    def test_cancel_node_detection(self):
        """Test cancel_node intent detection with entity extraction."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        # Standard format
        result = middleware.detect_control_intent("cancel node idea-2")
        assert result is not None
        action, target = result
        assert action == "cancel_node"
        assert target == {"node_id": "idea-2"}
        
        # Variations
        result = middleware.detect_control_intent("stop node-123")
        assert result[1] == {"node_id": "node-123"}
        
        result = middleware.detect_control_intent("cancel idea-2")
        assert result[1] == {"node_id": "idea-2"}
        
        result = middleware.detect_control_intent("cancel node idea-test-branch")
        assert result[1] == {"node_id": "idea-test-branch"}
    
    def test_case_insensitivity_control(self):
        """Test case insensitivity in control detection."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        assert middleware.detect_control_intent("PAUSE RESEARCH")[0] == "pause"
        assert middleware.detect_control_intent("Resume Research")[0] == "resume"
        assert middleware.detect_control_intent("CANCEL NODE IDEA-2")[0] == "cancel_node"
    
    def test_negative_cases_control(self):
        """Test non-control messages return None."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        assert middleware.detect_control_intent("hello") is None
        assert middleware.detect_control_intent("research quantum computing") is None
        assert middleware.detect_control_intent("what's happening") is None
        assert middleware.detect_control_intent("") is None


# ============================================================================
# Status Formatting Tests
# ============================================================================

class TestStatusFormatting:
    """Test _format_progress_summary() with various status dictionaries."""
    
    def test_complete_status(self, sample_status_data):
        """Test formatting with all fields populated."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        status = sample_status_data()
        
        summary = middleware._format_progress_summary(status)
        
        # Verify key information is present
        assert "Research Status" in summary
        assert "running" in summary.lower()
        assert "10" in summary  # total nodes
        assert "5" in summary   # completed
        assert "$0.25" in summary or "0.25" in summary  # cost
    
    def test_minimal_status(self):
        """Test formatting with minimal fields."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        status = {
            "experiment_id": "exp-test",
            "status": "running",
            "stats": {
                "total_nodes": 3,
                "completed": 1,
                "failed": 0,
                "running": 2,
                "pending": 0,
            }
        }
        
        summary = middleware._format_progress_summary(status)
        assert "Research Status" in summary
        assert "3" in summary  # total nodes
    
    def test_multiple_active_branches(self, sample_status_data):
        """Test formatting with multiple active branches."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        status = sample_status_data(
            active_branches=[
                {"branch_id": "idea-0", "title": "Test A", "status": "running"},
                {"branch_id": "idea-1", "title": "Test B", "status": "running"},
                {"branch_id": "idea-2", "title": "Test C", "status": "running"},
            ]
        )
        
        summary = middleware._format_progress_summary(status)
        assert "Active Branches" in summary or "branches" in summary.lower()
    
    def test_adapter_information(self, sample_status_data):
        """Test formatting with adapter information."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        status = sample_status_data()
        
        summary = middleware._format_progress_summary(status)
        # Should include adapter names or info
        assert "deepresearch" in summary.lower() or "codeact" in summary.lower() or "adapter" in summary.lower()
    
    def test_markdown_formatting(self, sample_status_data):
        """Test markdown formatting in output."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        status = sample_status_data()
        
        summary = middleware._format_progress_summary(status)
        # Should contain markdown elements
        assert "**" in summary or "#" in summary or "- " in summary
    
    def test_edge_cases(self):
        """Test edge cases (zero nodes, no branches, etc.)."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        # Zero nodes
        status = {
            "experiment_id": "exp-test",
            "status": "running",
            "stats": {"total_nodes": 0, "completed": 0, "failed": 0, "running": 0, "pending": 0},
        }
        summary = middleware._format_progress_summary(status)
        assert "0" in summary
        
        # No active branches
        status = {
            "experiment_id": "exp-test",
            "status": "running",
            "stats": {"total_nodes": 5, "completed": 5, "failed": 0, "running": 0, "pending": 0},
            "active_branches": [],
        }
        summary = middleware._format_progress_summary(status)
        assert summary is not None


# ============================================================================
# Progress Query Handler Tests
# ============================================================================

class TestProgressQueryHandler:
    """Test handle_progress_query() with mock session manager."""
    
    @pytest.mark.asyncio
    async def test_success_case(self, mock_middleware, sample_status_data):
        """Test success case with active experiment."""
        # Setup mock session manager
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(return_value=sample_status_data())
        
        result = await mock_middleware.handle_progress_query("test-session")
        
        assert result["status"] == "success"
        assert "summary" in result
        assert result["experiment_id"] == "exp-123"
    
    @pytest.mark.asyncio
    async def test_no_active_research(self, mock_middleware):
        """Test when no active research exists."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value=None)
        
        result = await mock_middleware.handle_progress_query("test-session")
        
        assert result["status"] == "error"
        assert "no active research" in result["message"].lower()
    
    @pytest.mark.asyncio
    async def test_session_manager_not_initialized(self, mock_middleware):
        """Test when session manager is not initialized."""
        mock_middleware._session_manager = None
        
        result = await mock_middleware.handle_progress_query("test-session")
        
        assert result["status"] == "error"
        assert "not initialized" in result["message"].lower() or "unavailable" in result["message"].lower()
    
    @pytest.mark.asyncio
    async def test_experiment_not_tracked(self, mock_middleware):
        """Test when experiment is not tracked by session manager."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(side_effect=KeyError("Not tracked"))
        
        result = await mock_middleware.handle_progress_query("test-session")
        
        assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_middleware):
        """Test error handling."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(side_effect=Exception("Test error"))
        
        result = await mock_middleware.handle_progress_query("test-session")
        
        assert result["status"] == "error"
        assert "error" in result["message"].lower()
    
    @pytest.mark.asyncio
    async def test_cache_behavior(self, mock_middleware, sample_status_data):
        """Test progress cache TTL and cache hits."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(return_value=sample_status_data())
        mock_middleware.progress_cache_ttl = 2.0
        
        # First query - cache miss
        result1 = await mock_middleware.handle_progress_query("test-session")
        assert result1["status"] == "success"
        call_count_1 = mock_middleware._session_manager.get_status.call_count
        
        # Second query within TTL - should use cache
        result2 = await mock_middleware.handle_progress_query("test-session")
        assert result2["status"] == "success"
        call_count_2 = mock_middleware._session_manager.get_status.call_count
        
        # Verify cache was used (no additional call)
        assert call_count_2 == call_count_1


# ============================================================================
# Control Intent Handler Tests
# ============================================================================

class TestControlIntentHandler:
    """Test handle_control_intent() with mock control bus."""
    
    @pytest.mark.asyncio
    async def test_pause_command(self, mock_middleware):
        """Test pause command success."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.send_control = AsyncMock()
        
        result = await mock_middleware.handle_control_intent("pause", {}, "test-session")
        
        assert result["status"] == "success"
        assert "pause" in result["message"].lower()
        mock_middleware._session_manager.send_control.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_node_with_target(self, mock_middleware):
        """Test cancel_node with target extraction."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.send_control = AsyncMock()
        
        result = await mock_middleware.handle_control_intent(
            "cancel_node",
            {"node_id": "idea-2"},
            "test-session"
        )
        
        assert result["status"] == "success"
        assert "idea-2" in result["message"]
        
        # Verify ControlMessage has correct target
        call_args = mock_middleware._session_manager.send_control.call_args
        control_msg = call_args[0][1]
        assert control_msg.target["node_id"] == "idea-2"
    
    @pytest.mark.asyncio
    async def test_no_active_research_control(self, mock_middleware):
        """Test control command when no active research."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value=None)
        
        result = await mock_middleware.handle_control_intent("pause", {}, "test-session")
        
        assert result["status"] == "error"
        assert "no active research" in result["message"].lower()
    
    @pytest.mark.asyncio
    async def test_control_system_unavailable(self, mock_middleware):
        """Test when control system is unavailable."""
        mock_middleware._session_manager = None
        
        result = await mock_middleware.handle_control_intent("pause", {}, "test-session")
        
        assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_cleanup_on_cancel(self, mock_middleware):
        """Test cleanup for terminal actions."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.send_control = AsyncMock()
        mock_middleware.active_orchestrators = {"exp-123": Mock()}
        
        result = await mock_middleware.handle_control_intent("cancel", {}, "test-session")
        
        assert result["status"] == "success"
        # Verify cleanup occurred
        assert "exp-123" not in mock_middleware.active_orchestrators


# ============================================================================
# Process Message Integration Tests
# ============================================================================

class TestProcessMessageIntegration:
    """Test process_message() with different message types."""
    
    @pytest.mark.asyncio
    async def test_progress_query_mode(self, mock_middleware, sample_status_data):
        """Test process_message with progress query."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(return_value=sample_status_data())
        
        result = await mock_middleware.process_message("how's progress?", "test-session")
        
        assert result["mode"] == "progress_query"
        assert "progress_data" in result
        assert result["should_trigger_research"] is False
        assert result["progress_data"]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_control_intent_mode(self, mock_middleware):
        """Test process_message with control intent."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.send_control = AsyncMock()
        
        result = await mock_middleware.process_message("pause research", "test-session")
        
        assert result["mode"] == "control_intent"
        assert "control_result" in result
        assert result["should_trigger_research"] is False
        assert result["control_result"]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_normal_mode(self, mock_middleware):
        """Test process_message with normal message."""
        result = await mock_middleware.process_message("explain this code", "test-session")
        
        assert result["mode"] == "normal"
        # With auto-trigger disabled and single-goal mode, should not trigger
        assert result["should_trigger_research"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
