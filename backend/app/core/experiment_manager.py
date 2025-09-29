"""Experiment Directory Manager - Organizes and preserves experiments with meaningful names"""

import asyncio
import json
import os
import re
import shutil
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment"""
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CLEANUP_PENDING = "cleanup_pending"


@dataclass
class ExperimentInfo:
    """Information about an experiment"""
    session_id: str
    original_query: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    workspace_path: Path = None
    arxiv_path: Optional[Path] = None
    readable_name: Optional[str] = None
    final_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ExperimentManager:
    """Manages experiment directories with intelligent naming and organization"""

    def __init__(self, workspace_dir: str):
        """Initialize experiment manager

        Args:
            workspace_dir: Base workspace directory (UAGENT_WORKSPACE_DIR)
        """
        self.workspace_dir = Path(workspace_dir)
        self.arxiv_dir = self.workspace_dir / "arxiv"
        self.active_experiments: Dict[str, ExperimentInfo] = {}
        self.cleanup_on_exit = True

        # Create arxiv directory structure
        self.arxiv_dir.mkdir(exist_ok=True)
        (self.arxiv_dir / "successful").mkdir(exist_ok=True)
        (self.arxiv_dir / "failed").mkdir(exist_ok=True)
        (self.arxiv_dir / "interrupted").mkdir(exist_ok=True)

        # Register signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info(f"ExperimentManager initialized with workspace: {workspace_dir}")
        logger.info(f"Arxiv directory: {self.arxiv_dir}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful experiment preservation"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, preserving experiments...")

            # Check if we have experiments to preserve
            if not self.active_experiments:
                logger.info("No active experiments to preserve")
                # Exit gracefully when no experiments to preserve
                os._exit(0)
            else:
                # Create async task to preserve experiments and then exit
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No running loop, create new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    # Schedule the task and exit
                    task = asyncio.create_task(self._preserve_and_exit(status=ExperimentStatus.INTERRUPTED))
                else:
                    # Run the preservation synchronously
                    loop.run_until_complete(self._preserve_and_exit(status=ExperimentStatus.INTERRUPTED))

        # Handle Ctrl+C and other termination signals
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def register_experiment(self, session_id: str, query: str, workspace_path: Path) -> ExperimentInfo:
        """Register a new experiment

        Args:
            session_id: Unique session identifier
            query: Original user query/goal
            workspace_path: Path to experiment workspace

        Returns:
            ExperimentInfo: Registered experiment information
        """
        experiment = ExperimentInfo(
            session_id=session_id,
            original_query=query,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now(),
            workspace_path=workspace_path
        )

        self.active_experiments[session_id] = experiment
        logger.info(f"Registered experiment: {session_id}")
        return experiment

    def _generate_readable_name(self, experiment: ExperimentInfo) -> str:
        """Generate a readable name for an experiment

        Args:
            experiment: Experiment information

        Returns:
            str: Human-readable experiment name
        """
        query = experiment.original_query

        # Extract key terms from the query
        # Remove common words and focus on meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'can', 'could', 'should', 'would', 'will',
            'may', 'might', 'must', 'shall', 'do', 'does', 'did', 'is', 'are', 'was', 'were',
            'be', 'being', 'been', 'have', 'has', 'had', 'having', 'please', 'help', 'me',
            'i', 'you', 'we', 'they', 'this', 'that', 'these', 'those', 'what', 'how'
        }

        # Clean and tokenize the query
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = [word for word in clean_query.split() if word not in stop_words and len(word) > 2]

        # Take first 4-6 meaningful words
        key_words = words[:6] if len(words) >= 6 else words[:4] if len(words) >= 4 else words

        # Create readable name
        if key_words:
            readable_name = "_".join(key_words)
        else:
            # Fallback to session ID if no meaningful words found
            readable_name = f"experiment_{experiment.session_id}"

        # Add timestamp for uniqueness
        timestamp = experiment.start_time.strftime("%Y%m%d_%H%M%S")

        # Limit length and clean up
        readable_name = readable_name[:50]  # Limit length
        readable_name = re.sub(r'[^\w_]', '_', readable_name)  # Clean special chars
        readable_name = re.sub(r'_+', '_', readable_name)  # Remove multiple underscores
        readable_name = readable_name.strip('_')  # Remove leading/trailing underscores

        return f"{timestamp}_{readable_name}"

    async def complete_experiment(self, session_id: str, success: bool,
                                final_result: Optional[Dict[str, Any]] = None,
                                error_message: Optional[str] = None) -> Optional[Path]:
        """Complete an experiment and move it to arxiv

        Args:
            session_id: Session identifier
            success: Whether the experiment succeeded
            final_result: Final result data if successful
            error_message: Error message if failed

        Returns:
            Path: Final arxiv path or None if experiment not found
        """
        if session_id not in self.active_experiments:
            logger.warning(f"Unknown experiment session: {session_id}")
            return None

        experiment = self.active_experiments[session_id]
        experiment.end_time = datetime.now()
        experiment.final_result = final_result
        experiment.error_message = error_message
        experiment.status = ExperimentStatus.SUCCESS if success else ExperimentStatus.FAILED

        # Generate readable name
        readable_name = self._generate_readable_name(experiment)

        if not success and not readable_name.startswith("fail_"):
            readable_name = f"fail_{readable_name}"

        experiment.readable_name = readable_name

        # Determine arxiv subdirectory
        subdir = "successful" if success else "failed"
        arxiv_path = self.arxiv_dir / subdir / readable_name

        # Move experiment to arxiv
        try:
            if experiment.workspace_path and experiment.workspace_path.exists():
                # Ensure arxiv path doesn't exist
                counter = 1
                original_arxiv_path = arxiv_path
                while arxiv_path.exists():
                    arxiv_path = original_arxiv_path.parent / f"{original_arxiv_path.name}_{counter}"
                    counter += 1

                # Move the entire experiment directory
                shutil.move(str(experiment.workspace_path), str(arxiv_path))
                experiment.arxiv_path = arxiv_path

                # Create experiment metadata
                await self._create_experiment_metadata(experiment)

                logger.info(f"Experiment {session_id} archived to: {arxiv_path}")
            else:
                logger.warning(f"Workspace not found for experiment {session_id}")

        except Exception as e:
            logger.error(f"Failed to archive experiment {session_id}: {e}")
            return None

        # Remove from active experiments
        del self.active_experiments[session_id]

        # Ensure any OpenHands containers are stopped/removed after experiment ends
        try:
            from .docker_container_manager import get_container_manager
            get_container_manager().cleanup_openhands_containers()
        except Exception as exc:
            logger.warning(f"Container cleanup after experiment {session_id} failed: {exc}")

        return experiment.arxiv_path

    async def _create_experiment_metadata(self, experiment: ExperimentInfo):
        """Create metadata file for an archived experiment

        Args:
            experiment: Experiment information
        """
        if not experiment.arxiv_path:
            return

        metadata = {
            "session_id": experiment.session_id,
            "original_query": experiment.original_query,
            "status": experiment.status.value,
            "start_time": experiment.start_time.isoformat(),
            "end_time": experiment.end_time.isoformat() if experiment.end_time else None,
            "duration_seconds": (experiment.end_time - experiment.start_time).total_seconds() if experiment.end_time else None,
            "readable_name": experiment.readable_name,
            "success": experiment.status == ExperimentStatus.SUCCESS,
            "final_result": experiment.final_result,
            "error_message": experiment.error_message,
            "archived_at": datetime.now().isoformat()
        }

        metadata_file = experiment.arxiv_path / "experiment_metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to create metadata for {experiment.session_id}: {e}")

    async def preserve_all_experiments(self, status: ExperimentStatus = ExperimentStatus.INTERRUPTED):
        """Preserve all active experiments (called on Ctrl+C)

        Args:
            status: Status to assign to preserved experiments
        """
        if not self.active_experiments:
            logger.info("No active experiments to preserve")
            return

        logger.info(f"Preserving {len(self.active_experiments)} active experiments...")

        preserved_count = 0
        for session_id, experiment in list(self.active_experiments.items()):
            try:
                experiment.status = status
                experiment.end_time = datetime.now()

                # Generate readable name for interrupted experiments
                readable_name = self._generate_readable_name(experiment)
                readable_name = f"interrupted_{readable_name}"
                experiment.readable_name = readable_name

                # Move to interrupted directory
                arxiv_path = self.arxiv_dir / "interrupted" / readable_name

                if experiment.workspace_path and experiment.workspace_path.exists():
                    # Ensure unique path
                    counter = 1
                    original_arxiv_path = arxiv_path
                    while arxiv_path.exists():
                        arxiv_path = original_arxiv_path.parent / f"{original_arxiv_path.name}_{counter}"
                        counter += 1

                    shutil.move(str(experiment.workspace_path), str(arxiv_path))
                    experiment.arxiv_path = arxiv_path

                    # Create metadata
                    await self._create_experiment_metadata(experiment)

                    preserved_count += 1
                    logger.info(f"Preserved experiment {session_id} to: {arxiv_path}")

            except Exception as e:
                logger.error(f"Failed to preserve experiment {session_id}: {e}")

        # Clear active experiments
        self.active_experiments.clear()

        # Stop and remove any OpenHands containers when preserving on interrupt
        try:
            from .docker_container_manager import get_container_manager
            get_container_manager().cleanup_openhands_containers()
        except Exception as exc:
            logger.warning(f"Container cleanup during preservation failed: {exc}")
        logger.info(f"Successfully preserved {preserved_count} experiments")

    async def _preserve_and_exit(self, status: ExperimentStatus = ExperimentStatus.INTERRUPTED):
        """Preserve all experiments and then exit the application

        Args:
            status: Status to assign to preserved experiments
        """
        try:
            await self.preserve_all_experiments(status)
        except Exception as e:
            logger.error(f"Error preserving experiments: {e}")
        finally:
            # Exit the application after preservation
            logger.info("Exiting application after experiment preservation")
            os._exit(0)

    def list_archived_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List archived experiments

        Args:
            status_filter: Filter by status ('successful', 'failed', 'interrupted')

        Returns:
            List of experiment metadata
        """
        experiments = []

        subdirs = ["successful", "failed", "interrupted"]
        if status_filter:
            subdirs = [status_filter] if status_filter in subdirs else []

        for subdir in subdirs:
            subdir_path = self.arxiv_dir / subdir
            if not subdir_path.exists():
                continue

            for exp_dir in subdir_path.iterdir():
                if exp_dir.is_dir():
                    metadata_file = exp_dir / "experiment_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                metadata["arxiv_path"] = str(exp_dir)
                                experiments.append(metadata)
                        except Exception as e:
                            logger.error(f"Failed to read metadata for {exp_dir}: {e}")

        # Sort by start time (newest first)
        experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return experiments

    def get_experiment_path(self, session_id: str) -> Optional[Path]:
        """Get the current path for an active experiment

        Args:
            session_id: Session identifier

        Returns:
            Path to experiment or None if not found
        """
        if session_id in self.active_experiments:
            return self.active_experiments[session_id].workspace_path
        return None

    def is_experiment_active(self, session_id: str) -> bool:
        """Check if an experiment is currently active

        Args:
            session_id: Session identifier

        Returns:
            bool: True if experiment is active
        """
        return session_id in self.active_experiments

    async def cleanup_old_experiments(self, days_old: int = 30):
        """Clean up old experiments (optional maintenance)

        Args:
            days_old: Remove experiments older than this many days
        """
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        removed_count = 0

        for subdir in ["successful", "failed", "interrupted"]:
            subdir_path = self.arxiv_dir / subdir
            if not subdir_path.exists():
                continue

            for exp_dir in subdir_path.iterdir():
                if exp_dir.is_dir():
                    try:
                        # Check creation time
                        if exp_dir.stat().st_ctime < cutoff_time:
                            shutil.rmtree(exp_dir)
                            removed_count += 1
                            logger.info(f"Cleaned up old experiment: {exp_dir.name}")
                    except Exception as e:
                        logger.error(f"Failed to cleanup {exp_dir}: {e}")

        logger.info(f"Cleaned up {removed_count} old experiments")


# Global experiment manager instance
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> Optional[ExperimentManager]:
    """Get the global experiment manager instance"""
    return _experiment_manager


def initialize_experiment_manager(workspace_dir: str) -> ExperimentManager:
    """Initialize the global experiment manager

    Args:
        workspace_dir: Base workspace directory

    Returns:
        ExperimentManager: Initialized experiment manager
    """
    global _experiment_manager
    _experiment_manager = ExperimentManager(workspace_dir)
    return _experiment_manager


async def shutdown_experiment_manager():
    """Shutdown the experiment manager and preserve active experiments"""
    global _experiment_manager
    if _experiment_manager:
        await _experiment_manager.preserve_all_experiments()
        # Also ensure containers are stopped on shutdown
        try:
            from .docker_container_manager import get_container_manager
            get_container_manager().cleanup_openhands_containers()
        except Exception as exc:
            logger.warning(f"Container cleanup on experiment manager shutdown failed: {exc}")
        _experiment_manager = None
