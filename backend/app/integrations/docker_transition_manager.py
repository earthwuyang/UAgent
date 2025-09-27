"""Docker Transition Manager for smooth migration to Docker runtime"""

import logging
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import docker

logger = logging.getLogger(__name__)


class DockerTransitionManager:
    """Manages smooth transition to Docker runtime with fallback strategies"""

    def __init__(self):
        self.docker_client = None
        self._docker_available = False
        self._image_verified = False
        self._verified_image = "ghcr.io/all-hands-ai/runtime:latest"

    def check_docker_availability(self) -> bool:
        """Check if Docker is available and running"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            self._docker_available = True
            logger.info("Docker is available and responsive")
            return True
        except Exception as e:
            logger.error(f"Docker not available: {e}")
            self._docker_available = False
            return False

    def verify_openhands_image(self) -> bool:
        """Verify OpenHands runtime image is available"""
        if not self._docker_available:
            return False

        try:
            # Try multiple image options (prefer older stable version)
            image_options = [
                "ghcr.io/all-hands-ai/runtime:latest",  # 13 months old, more stable
                "ghcr.io/all-hands-ai/runtime:0.56-nikolaik"  # Recent but has micromamba issues
            ]

            for image_name in image_options:
                # Check if image exists locally
                try:
                    self.docker_client.images.get(image_name)
                    logger.info(f"OpenHands runtime image {image_name} found locally")
                    self._image_verified = True
                    self._verified_image = image_name
                    return True
                except docker.errors.ImageNotFound:
                    continue

            # If no local images found, try to pull latest
            image_name = "ghcr.io/all-hands-ai/runtime:latest"
            logger.info(f"OpenHands runtime image not found locally, pulling {image_name}...")

            try:
                self.docker_client.images.pull(image_name)
                logger.info(f"Successfully pulled OpenHands runtime image {image_name}")
                self._image_verified = True
                self._verified_image = image_name
                return True
            except Exception as e:
                logger.error(f"Failed to pull OpenHands runtime image: {e}")
                return False

        except Exception as e:
            logger.error(f"Error verifying OpenHands image: {e}")
            return False

    def get_docker_config(self) -> Dict[str, str]:
        """Get optimized Docker configuration"""
        config = {
            "RUNTIME": "docker",
            "ENABLE_BROWSER": "false",
        }

        # Use standard Docker runtime with pre-built images
        config.update({
            # Use pre-built images instead of rebuilding
            "SANDBOX_FORCE_REBUILD_RUNTIME": "false",

            # Use original 0.57 image with Poetry environment
            "SANDBOX_BASE_CONTAINER_IMAGE": "docker.all-hands.dev/all-hands-ai/runtime:0.57-nikolaik",

            # UAgent: Use base image directly without custom building
            "SANDBOX_USE_BASE_IMAGE_DIRECTLY": "true",

            # Use host network for Docker containers
            "SANDBOX_USE_HOST_NETWORK": "true",

            # Connection and timeout settings
            "SANDBOX_TIMEOUT": "600",
            "SANDBOX_CONNECTION_TIMEOUT": "120",
            "SANDBOX_CONNECTION_RETRIES": "5",

            # Keep runtime alive for performance
            "SANDBOX_KEEP_RUNTIME_ALIVE": "true",
            "SANDBOX_PAUSE_CLOSED_RUNTIMES": "false",

            # Use Poetry python environment completely isolated from micromamba
            "SANDBOX_PYTHON_EXECUTABLE": "/openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python",
            # Completely override PYTHONPATH to avoid micromamba interference
            "PYTHONPATH": "/openhands/code",
            # Override environment to isolate Poetry from micromamba
            "SANDBOX_RUNTIME_STARTUP_ENV_VARS": '{"PYTHONPATH": "/openhands/code", "CONDA_PREFIX": "", "CONDA_DEFAULT_ENV": "", "MAMBA_EXE": ""}',
        })

        # Map LLM configuration from .env file to OpenHands format
        if "LITELLM_MODEL" in os.environ:
            config["LLM_MODEL"] = os.environ["LITELLM_MODEL"]
        if "LITELLM_API_KEY" in os.environ:
            config["LLM_API_KEY"] = os.environ["LITELLM_API_KEY"]
        if "LITELLM_API_BASE" in os.environ:
            config["LLM_BASE_URL"] = os.environ["LITELLM_API_BASE"]

        # Also copy any additional LiteLLM options
        if "LITELLM_EXTRA_OPTIONS" in os.environ:
            config["LLM_EXTRA_OPTIONS"] = os.environ["LITELLM_EXTRA_OPTIONS"]

        return config

    def create_error_report(self, error: Exception, workspace: Path) -> Dict[str, Any]:
        """Create a standardized error report for Docker issues"""
        error_report = {
            "success": False,
            "data": {},
            "analysis": {
                "error_type": "docker_transition_error",
                "error_message": str(error),
                "docker_available": self._docker_available,
                "image_verified": self._image_verified,
            },
            "conclusions": [
                "Docker runtime initialization failed",
                "Hardware isolation could not be established",
                "System requires Docker to be properly configured"
            ],
            "errors": [str(error)]
        }

        # Write error report to workspace
        try:
            results_dir = workspace / "experiments" / "docker_transition" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            final_json_path = results_dir / "final.json"
            with open(final_json_path, 'w') as f:
                json.dump(error_report, f, indent=2)

            logger.info(f"Error report written to {final_json_path}")
        except Exception as write_error:
            logger.error(f"Failed to write error report: {write_error}")

        return error_report

    def ensure_docker_ready(self) -> bool:
        """Ensure Docker is ready for OpenHands operations"""
        logger.info("Ensuring Docker is ready for OpenHands operations...")

        # Step 1: Check Docker availability
        if not self.check_docker_availability():
            logger.error("Docker is not available - cannot proceed with hardware isolation")
            return False

        # Step 2: Verify OpenHands image
        if not self.verify_openhands_image():
            logger.error("OpenHands Docker image is not available")
            return False

        logger.info("Docker environment is ready for OpenHands operations")
        return True

    def get_transition_status(self) -> Dict[str, Any]:
        """Get current transition status"""
        return {
            "docker_available": self._docker_available,
            "image_verified": self._image_verified,
            "ready_for_operations": self._docker_available and self._image_verified,
            "recommended_action": self._get_recommended_action()
        }

    def _get_recommended_action(self) -> str:
        """Get recommended action based on current status"""
        if not self._docker_available:
            return "Install and start Docker service"
        elif not self._image_verified:
            return "Pull OpenHands runtime image: docker pull ghcr.io/all-hands-ai/runtime:latest"
        else:
            return "System ready for Docker operations"


# Global instance for easy access
docker_transition_manager = DockerTransitionManager()