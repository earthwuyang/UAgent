"""
Performance Tests for Research Middleware

Ensures real-time interaction requirements are met:
- Progress query response: < 300ms
- Control command latency: < 500ms
- No blocking of main chat during research
"""

import pytest
import asyncio
import time
import statistics
from typing import List
from unittest.mock import Mock, AsyncMock

try:
    from extensions.uagent_research.middleware.research_middleware import ResearchMiddleware
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytest.skip("Middleware not available", allow_module_level=True)


# ============================================================================
# Helper Functions
# ============================================================================

async def measure_execution_time(func, *args, **kwargs) -> float:
    """Measure execution time of async function in milliseconds."""
    start = time.perf_counter()
    result = await func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000  # Convert to ms


def measure_percentile(times: List[float], percentile: int) -> float:
    """Calculate percentile of execution times."""
    if not times:
        return 0.0
    sorted_times = sorted(times)
    index = int(len(sorted_times) * percentile / 100)
    return sorted_times[min(index, len(sorted_times) - 1)]


async def run_concurrent_queries(
    middleware,
    session_id: str,
    count: int,
    interval_ms: int
) -> List[float]:
    """Run concurrent progress queries and measure response times."""
    times = []
    for _ in range(count):
        start = time.perf_counter()
        await middleware.handle_progress_query(session_id)
        end = time.perf_counter()
        times.append((end - start) * 1000)
        await asyncio.sleep(interval_ms / 1000)
    return times


# ============================================================================
# Performance Tests
# ============================================================================

class TestProgressQueryPerformance:
    """Test progress query response time requirements."""
    
    @pytest.mark.asyncio
    async def test_progress_query_response_time(self, mock_middleware, sample_status_data):
        """Test progress query completes in < 300ms."""
        # Setup
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(return_value=sample_status_data())
        
        # Measure response time
        elapsed = await measure_execution_time(
            mock_middleware.handle_progress_query,
            "test-session"
        )
        
        # Assert meets requirement (< 300ms)
        assert elapsed < 300, f"Progress query took {elapsed:.2f}ms, exceeds 300ms limit"
    
    @pytest.mark.asyncio
    async def test_complex_status_performance(self, mock_middleware, sample_status_data):
        """Test with complex status (many nodes, branches, adapters)."""
        # Create complex status
        complex_status = sample_status_data(
            total_nodes=100,
            completed=50,
            failed=10,
            running=40,
            active_branches=[
                {"branch_id": f"idea-{i}", "title": f"Branch {i}", "status": "running"}
                for i in range(20)
            ]
        )
        
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(return_value=complex_status)
        
        elapsed = await measure_execution_time(
            mock_middleware.handle_progress_query,
            "test-session"
        )
        
        assert elapsed < 300, f"Complex status query took {elapsed:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, mock_middleware, sample_status_data):
        """Test cache hit is significantly faster."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(return_value=sample_status_data())
        mock_middleware.progress_cache_ttl = 10.0  # Long TTL
        
        # Cache miss
        time_miss = await measure_execution_time(
            mock_middleware.handle_progress_query,
            "test-session"
        )
        
        # Cache hit
        time_hit = await measure_execution_time(
            mock_middleware.handle_progress_query,
            "test-session"
        )
        
        # Cache hit should be much faster (< 10ms)
        assert time_hit < 10, f"Cache hit took {time_hit:.2f}ms, should be < 10ms"
        assert time_hit < time_miss, "Cache hit should be faster than miss"
    
    @pytest.mark.asyncio
    async def test_95th_percentile_performance(self, mock_middleware, sample_status_data):
        """Test 95th percentile of 100 iterations < 300ms."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(return_value=sample_status_data())
        
        times = []
        for _ in range(100):
            # Clear cache for each iteration
            mock_middleware._progress_cache.clear()
            
            elapsed = await measure_execution_time(
                mock_middleware.handle_progress_query,
                "test-session"
            )
            times.append(elapsed)
        
        p95 = measure_percentile(times, 95)
        assert p95 < 300, f"95th percentile is {p95:.2f}ms, exceeds 300ms"


class TestControlCommandPerformance:
    """Test control command latency requirements."""
    
    @pytest.mark.asyncio
    async def test_control_command_latency(self, mock_middleware):
        """Test control command completes in < 500ms."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.send_control = AsyncMock()
        
        elapsed = await measure_execution_time(
            mock_middleware.handle_control_intent,
            "pause",
            {},
            "test-session"
        )
        
        assert elapsed < 500, f"Control command took {elapsed:.2f}ms, exceeds 500ms limit"
    
    @pytest.mark.asyncio
    async def test_all_control_actions_latency(self, mock_middleware):
        """Test all control actions meet latency requirements."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.send_control = AsyncMock()
        mock_middleware.active_orchestrators = {"exp-123": Mock()}
        
        actions = [
            ("pause", {}),
            ("resume", {}),
            ("cancel", {}),
            ("cancel_node", {"node_id": "idea-2"}),
        ]
        
        for action, target in actions:
            # Reset for cancel test
            if action == "cancel":
                mock_middleware.active_orchestrators = {"exp-123": Mock()}
            
            elapsed = await measure_execution_time(
                mock_middleware.handle_control_intent,
                action,
                target,
                "test-session"
            )
            
            assert elapsed < 500, f"{action} took {elapsed:.2f}ms, exceeds 500ms"


class TestConcurrentPerformance:
    """Test no blocking during concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_queries_no_blocking(self, mock_middleware, sample_status_data):
        """Test concurrent progress queries don't block each other."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(return_value=sample_status_data())
        
        # Run 10 concurrent queries
        times = await run_concurrent_queries(
            mock_middleware,
            "test-session",
            count=10,
            interval_ms=100
        )
        
        # All queries should complete within limit
        for i, elapsed in enumerate(times):
            assert elapsed < 300, f"Query {i} took {elapsed:.2f}ms"
        
        # Average should be reasonable
        avg_time = statistics.mean(times)
        assert avg_time < 200, f"Average query time {avg_time:.2f}ms too high"
    
    @pytest.mark.asyncio
    async def test_background_research_no_blocking(self, mock_middleware, sample_status_data):
        """Test progress queries don't block during background research."""
        mock_middleware._session_manager = Mock()
        mock_middleware._session_manager.get_experiment_for_session = Mock(return_value="exp-123")
        mock_middleware._session_manager.get_status = Mock(return_value=sample_status_data())
        
        # Simulate background work
        async def background_work():
            for _ in range(10):
                await asyncio.sleep(0.1)
        
        # Start background task
        bg_task = asyncio.create_task(background_work())
        
        # Run queries while background work is happening
        times = []
        for _ in range(5):
            elapsed = await measure_execution_time(
                mock_middleware.handle_progress_query,
                "test-session"
            )
            times.append(elapsed)
            await asyncio.sleep(0.2)
        
        await bg_task
        
        # Queries should not be affected by background work
        for elapsed in times:
            assert elapsed < 300, f"Query during background work took {elapsed:.2f}ms"


class TestEntityExtractionPerformance:
    """Test entity extraction performance."""
    
    def test_entity_extraction_speed(self):
        """Test regex matching completes in < 1ms."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        
        test_inputs = [
            "cancel node idea-2",
            "stop node-123",
            "cancel idea-test-branch",
            "pause research",
            "resume research now",
        ]
        
        times = []
        for input_text in test_inputs * 200:  # 1000 total tests
            start = time.perf_counter()
            middleware.detect_control_intent(input_text)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        assert avg_time < 1, f"Average extraction time {avg_time:.3f}ms exceeds 1ms"
        assert max_time < 5, f"Max extraction time {max_time:.3f}ms too high"


class TestStatusFormattingPerformance:
    """Test status formatting performance."""
    
    def test_formatting_small_status(self, sample_status_data):
        """Test formatting small status."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        status = sample_status_data(total_nodes=1, active_branches=[])
        
        start = time.perf_counter()
        summary = middleware._format_progress_summary(status)
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 50, f"Small status formatting took {elapsed:.2f}ms"
    
    def test_formatting_medium_status(self, sample_status_data):
        """Test formatting medium status."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        status = sample_status_data(
            total_nodes=10,
            active_branches=[
                {"branch_id": f"branch-{i}", "title": f"Test {i}", "status": "running"}
                for i in range(10)
            ]
        )
        
        start = time.perf_counter()
        summary = middleware._format_progress_summary(status)
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 50, f"Medium status formatting took {elapsed:.2f}ms"
    
    def test_formatting_large_status(self, sample_status_data):
        """Test formatting large status."""
        middleware = ResearchMiddleware(confidence_threshold=0.7, enable_auto_trigger=False)
        status = sample_status_data(
            total_nodes=100,
            active_branches=[
                {"branch_id": f"branch-{i}", "title": f"Test {i}", "status": "running"}
                for i in range(100)
            ],
            adapters={
                f"adapter-{i}": {
                    "status": "running",
                    "current_step": f"Step {i}",
                    "cost": 0.1
                }
                for i in range(10)
            }
        )
        
        start = time.perf_counter()
        summary = middleware._format_progress_summary(status)
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 50, f"Large status formatting took {elapsed:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
