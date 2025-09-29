"""Docker Container Manager for UAgent System

Manages Docker containers created by UAgent and ensures proper cleanup on shutdown.
"""

import logging
import docker
from typing import Set, Optional
import threading
import atexit

logger = logging.getLogger(__name__)


class DockerContainerManager:
    """Manages Docker containers for UAgent system with automatic cleanup"""

    _instance: Optional['DockerContainerManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'DockerContainerManager':
        """Singleton pattern to ensure one instance across the application"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Docker container manager"""
        # Only initialize once (singleton pattern)
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.docker_client = docker.from_env()
        self.managed_containers: Set[str] = set()
        self._lock = threading.Lock()

        # Register cleanup on process exit
        atexit.register(self.cleanup_all_containers)

        logger.info("Docker Container Manager initialized")

    def register_container(self, container_id: str) -> None:
        """Register a container for management and cleanup"""
        with self._lock:
            self.managed_containers.add(container_id)
            logger.info(f"Registered container for management: {container_id}")

    def unregister_container(self, container_id: str) -> None:
        """Unregister a container from management"""
        with self._lock:
            self.managed_containers.discard(container_id)
            logger.info(f"Unregistered container from management: {container_id}")

    def cleanup_container(self, container_id: str) -> bool:
        """Clean up a specific container"""
        try:
            container = self.docker_client.containers.get(container_id)

            # Stop the container if it's running
            if container.status in ['running', 'paused']:
                logger.info(f"Stopping container: {container_id}")
                container.stop(timeout=10)

            # Remove the container
            logger.info(f"Removing container: {container_id}")
            container.remove(force=True)

            # Unregister from management
            self.unregister_container(container_id)

            logger.info(f"Successfully cleaned up container: {container_id}")
            return True

        except docker.errors.NotFound:
            logger.info(f"Container not found (already removed): {container_id}")
            self.unregister_container(container_id)
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup container {container_id}: {e}")
            return False

    def cleanup_all_containers(self) -> None:
        """Clean up all managed containers"""
        if not hasattr(self, 'managed_containers'):
            return

        logger.info("Starting cleanup of all managed Docker containers...")

        # Make a copy to avoid modification during iteration
        with self._lock:
            containers_to_cleanup = self.managed_containers.copy()

        if not containers_to_cleanup:
            logger.info("No managed containers to cleanup")
            return

        cleaned_count = 0
        for container_id in containers_to_cleanup:
            if self.cleanup_container(container_id):
                cleaned_count += 1

        logger.info(f"Cleanup complete: {cleaned_count}/{len(containers_to_cleanup)} containers cleaned up")

    def cleanup_openhands_containers(self) -> None:
        """Clean up all OpenHands-related containers (regardless of management status)"""
        logger.info("Cleaning up all OpenHands-related containers...")

        try:
            # Find all containers with OpenHands-related images
            openhands_filters = [
                "docker.all-hands.dev/all-hands-ai/runtime:",
                "all-hands-ai/runtime:",
                "openhands-runtime:",
                "openhands:",
            ]

            all_containers = self.docker_client.containers.list(all=True)
            openhands_containers = []

            for container in all_containers:
                if container.image and container.image.tags:
                    image_name = str(container.image.tags[0]) if container.image.tags else ""
                    if any(filter_name in image_name for filter_name in openhands_filters):
                        openhands_containers.append(container)

            if not openhands_containers:
                logger.info("No OpenHands containers found to cleanup")
                return

            logger.info(f"Found {len(openhands_containers)} OpenHands containers to cleanup")

            cleaned_count = 0
            for container in openhands_containers:
                try:
                    container_id = container.id
                    logger.info(f"Cleaning up OpenHands container: {container_id}")

                    # Stop if running
                    if container.status in ['running', 'paused']:
                        container.stop(timeout=10)

                    # Remove container
                    container.remove(force=True)
                    cleaned_count += 1

                    logger.info(f"Successfully cleaned up OpenHands container: {container_id}")

                except Exception as e:
                    logger.error(f"Failed to cleanup OpenHands container {container.id}: {e}")

            logger.info(f"OpenHands cleanup complete: {cleaned_count}/{len(openhands_containers)} containers cleaned up")

        except Exception as e:
            logger.error(f"Error during OpenHands container cleanup: {e}")

    def get_managed_containers(self) -> Set[str]:
        """Get list of currently managed containers"""
        with self._lock:
            return self.managed_containers.copy()

    def shutdown(self) -> None:
        """Shutdown the container manager and cleanup all containers"""
        logger.info("Shutting down Docker Container Manager...")

        # Clean up all OpenHands containers (broader cleanup)
        self.cleanup_openhands_containers()

        # Clean up any remaining managed containers
        self.cleanup_all_containers()

        logger.info("Docker Container Manager shutdown complete")


# Global instance
_container_manager: Optional[DockerContainerManager] = None


def get_container_manager() -> DockerContainerManager:
    """Get the global Docker container manager instance"""
    global _container_manager
    if _container_manager is None:
        _container_manager = DockerContainerManager()
    return _container_manager