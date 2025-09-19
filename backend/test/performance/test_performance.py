"""
Performance tests for UAgent system
"""

import asyncio
import time
import psutil
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, AsyncMock, patch

from app.core.smart_router import SmartRouter
from app.core.research_engines.deep_research import DeepResearchEngine
from app.core.openhands.workspace_manager import WorkspaceManager
from app.core.openhands.code_executor import CodeExecutor


class TestPerformance:
    """Performance tests for core components"""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client with realistic response times"""
        mock_client = MagicMock()

        async def mock_generate(prompt):
            # Simulate realistic LLM response time
            await asyncio.sleep(0.1)  # 100ms simulated response time
            return "Mock LLM response"

        mock_client.generate = mock_generate
        return mock_client

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache"""
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        return mock_cache

    @pytest.mark.asyncio
    async def test_smart_router_classification_performance(self, mock_llm_client, mock_cache):
        """Test smart router classification performance"""
        router = SmartRouter(mock_llm_client, mock_cache)

        requests = [
            "Find information about machine learning",
            "Analyze code repositories for best practices",
            "Design experiments to test hypothesis about neural networks",
            "Research quantum computing applications",
            "Implement a sorting algorithm"
        ]

        start_time = time.time()

        # Test concurrent classifications
        tasks = [router.classify_request(request) for request in requests]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 2.0  # Should complete within 2 seconds
        assert len(results) == len(requests)

        # Average time per classification should be reasonable
        avg_time = total_time / len(requests)
        assert avg_time < 0.5  # Less than 500ms per classification on average

    @pytest.mark.asyncio
    async def test_workspace_creation_performance(self):
        """Test workspace creation performance"""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            workspace_manager = WorkspaceManager(temp_dir)

            start_time = time.time()

            # Create multiple workspaces concurrently
            tasks = [
                workspace_manager.create_workspace(f"perf_test_{i}")
                for i in range(10)
            ]
            configs = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            # Performance assertions
            assert total_time < 5.0  # Should complete within 5 seconds
            assert len(configs) == 10

            # Average time per workspace creation
            avg_time = total_time / 10
            assert avg_time < 1.0  # Less than 1 second per workspace

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_concurrent_code_execution_performance(self):
        """Test concurrent code execution performance"""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            workspace_manager = WorkspaceManager(temp_dir)
            code_executor = CodeExecutor(workspace_manager)

            # Create a test workspace
            config = await workspace_manager.create_workspace("perf_test")
            workspace_id = config.workspace_id

            # Simple Python code for testing
            test_code = """
import time
print("Starting execution")
time.sleep(0.1)  # Small delay to simulate work
result = sum(range(100))
print(f"Result: {result}")
"""

            start_time = time.time()

            # Execute code multiple times concurrently
            tasks = [
                code_executor.execute_python_code(workspace_id, test_code, f"test_{i}.py")
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            # Performance assertions
            assert total_time < 3.0  # Should complete within 3 seconds
            assert all(result.success for result in results)

            # Should be faster than sequential execution
            assert total_time < 5 * 0.5  # Faster than 5 * 0.5 seconds

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_memory_usage_during_operations(self):
        """Test memory usage during intensive operations"""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            workspace_manager = WorkspaceManager(temp_dir)

            # Create multiple workspaces
            workspace_ids = []
            for i in range(20):
                config = await workspace_manager.create_workspace(f"memory_test_{i}")
                workspace_ids.append(config.workspace_id)

            # Check memory usage after workspace creation
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB for 20 workspaces)
            assert memory_increase < 100

            # Cleanup workspaces
            await workspace_manager.cleanup_all_workspaces()

            # Memory should be mostly freed after cleanup
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_after_cleanup = final_memory - initial_memory

            # Memory after cleanup should be close to initial
            assert memory_after_cleanup < memory_increase * 0.5

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_file_operations_performance(self):
        """Test file operation performance"""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            workspace_manager = WorkspaceManager(temp_dir)
            config = await workspace_manager.create_workspace("file_perf_test")
            workspace_id = config.workspace_id

            # Test file write performance
            file_content = "x" * 1024  # 1KB content

            start_time = time.time()

            # Write multiple files concurrently
            write_tasks = [
                workspace_manager.write_file(workspace_id, f"test_file_{i}.txt", file_content)
                for i in range(100)
            ]
            write_results = await asyncio.gather(*write_tasks)

            write_time = time.time() - start_time

            # Test file read performance
            start_time = time.time()

            read_tasks = [
                workspace_manager.read_file(workspace_id, f"test_file_{i}.txt")
                for i in range(100)
            ]
            read_results = await asyncio.gather(*read_tasks)

            read_time = time.time() - start_time

            # Performance assertions
            assert write_time < 2.0  # 100 file writes in less than 2 seconds
            assert read_time < 1.0   # 100 file reads in less than 1 second
            assert all(write_results)  # All writes successful
            assert all(content == file_content for content in read_results)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_classification_caching_performance(self, mock_llm_client):
        """Test classification caching performance improvement"""
        from app.core.cache import MemoryCache

        cache = MemoryCache()
        router = SmartRouter(mock_llm_client, cache)

        test_request = "Analyze machine learning algorithms"

        # First classification (cache miss)
        start_time = time.time()
        result1 = await router.classify_request(test_request)
        first_time = time.time() - start_time

        # Second classification (cache hit)
        start_time = time.time()
        result2 = await router.classify_request(test_request)
        second_time = time.time() - start_time

        # Cache hit should be significantly faster
        assert second_time < first_time * 0.1  # At least 10x faster
        assert result1["engine"] == result2["engine"]

    def test_cpu_usage_during_operations(self):
        """Test CPU usage during intensive operations"""
        import threading

        cpu_percentages = []

        def monitor_cpu():
            for _ in range(10):
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_percentages.append(cpu_percent)

        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        # Perform CPU-intensive operation
        start_time = time.time()
        result = sum(i * i for i in range(100000))
        end_time = time.time()

        monitor_thread.join()

        # CPU usage should be reasonable
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
        max_cpu = max(cpu_percentages)

        # Should not max out CPU for simple operations
        assert avg_cpu < 80
        assert max_cpu < 95
        assert result > 0  # Operation completed successfully

    @pytest.mark.asyncio
    async def test_error_handling_performance(self, mock_llm_client, mock_cache):
        """Test performance impact of error handling"""
        router = SmartRouter(mock_llm_client, mock_cache)

        # Mock LLM client to raise exceptions
        async def failing_generate(prompt):
            await asyncio.sleep(0.05)  # Small delay
            raise Exception("Simulated LLM failure")

        mock_llm_client.generate = failing_generate

        start_time = time.time()

        # Multiple failing requests
        tasks = [
            router.classify_request(f"Test request {i}")
            for i in range(10)
        ]

        # Should handle all errors gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Error handling should not significantly slow down the system
        assert total_time < 2.0  # Should complete within 2 seconds

        # All requests should have been handled (even if they failed)
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_large_data_handling_performance(self):
        """Test performance with large data sets"""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            workspace_manager = WorkspaceManager(temp_dir)
            config = await workspace_manager.create_workspace("large_data_test")
            workspace_id = config.workspace_id

            # Large file content (1MB)
            large_content = "x" * (1024 * 1024)

            start_time = time.time()

            # Write large file
            success = await workspace_manager.write_file(
                workspace_id, "large_file.txt", large_content
            )

            write_time = time.time() - start_time

            # Read large file
            start_time = time.time()
            read_content = await workspace_manager.read_file(
                workspace_id, "large_file.txt"
            )
            read_time = time.time() - start_time

            # Performance assertions for large files
            assert write_time < 1.0  # 1MB write in less than 1 second
            assert read_time < 0.5   # 1MB read in less than 0.5 seconds
            assert success is True
            assert read_content == large_content

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_concurrent_session_performance(self):
        """Test performance with multiple concurrent sessions"""
        from app.core.openhands.client import OpenHandsClient
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            client = OpenHandsClient(temp_dir)

            start_time = time.time()

            # Create multiple sessions concurrently
            session_tasks = [
                client.create_session("scientific_research", f"session_{i}")
                for i in range(10)
            ]
            configs = await asyncio.gather(*session_tasks)

            creation_time = time.time() - start_time

            # Test concurrent operations on sessions
            start_time = time.time()

            status_tasks = [
                client.get_session_state(config.session_id)
                for config in configs
            ]
            states = await asyncio.gather(*status_tasks)

            status_time = time.time() - start_time

            # Performance assertions
            assert creation_time < 3.0  # 10 sessions in less than 3 seconds
            assert status_time < 1.0    # Status checks in less than 1 second
            assert len(configs) == 10
            assert all(state is not None for state in states)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)