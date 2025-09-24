"""Workspace Manager for isolated code execution environments"""

import asyncio
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceConfig:
    """Configuration for workspace creation"""
    workspace_id: str
    base_path: str
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_total_size: int = 100 * 1024 * 1024  # 100MB
    timeout: int = 300  # 5 minutes
    python_path: str = "/usr/bin/python3"
    allowed_commands: List[str] = None


@dataclass
class WorkspaceStatus:
    """Current status of a workspace"""
    workspace_id: str
    status: str  # 'created', 'active', 'stopped', 'error'
    created_at: str
    last_activity: str
    file_count: int
    total_size: int
    processes: List[Dict[str, Any]]


class WorkspaceManager:
    """Manages isolated workspaces for code execution"""

    _ALLOWED_CONFIG_KEYS = {
        "max_file_size",
        "max_total_size",
        "timeout",
        "python_path",
        "allowed_commands",
    }

    @classmethod
    def _sanitize_config(cls, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(config, dict):
            return {}
        return {key: config[key] for key in cls._ALLOWED_CONFIG_KEYS if key in config}

    def __init__(self, base_workspace_dir: str = None):
        """Initialize workspace manager

        Args:
            base_workspace_dir: Base directory for all workspaces
        """
        self.base_dir = Path(base_workspace_dir or tempfile.gettempdir()) / "uagent_workspaces"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces: Dict[str, WorkspaceConfig] = {}
        self.active_processes: Dict[str, List[asyncio.subprocess.Process]] = {}
        logger.info(f"WorkspaceManager initialized with base directory: {self.base_dir}")

    async def create_workspace(self, research_id: str = None, config: Dict[str, Any] = None) -> WorkspaceConfig:
        """Create a new isolated workspace

        Args:
            research_id: Optional research session ID
            config: Additional configuration options

        Returns:
            WorkspaceConfig: Configuration for the created workspace
        """
        workspace_id = research_id or f"ws_{uuid.uuid4().hex[:8]}"
        workspace_path = self.base_dir / workspace_id

        # Create (or reuse) workspace directory. Do NOT delete existing content.
        # Users may want to inspect artifacts after runs; we keep prior files.
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        (workspace_path / "code").mkdir(exist_ok=True)
        (workspace_path / "data").mkdir(exist_ok=True)
        (workspace_path / "output").mkdir(exist_ok=True)
        (workspace_path / "logs").mkdir(exist_ok=True)
        (workspace_path / "scripts").mkdir(exist_ok=True)

        # Create workspace configuration
        sanitized_config = self._sanitize_config(config)

        workspace_config = WorkspaceConfig(
            workspace_id=workspace_id,
            base_path=str(workspace_path),
            **sanitized_config
        )

        # Initialize workspace with basic files
        await self._initialize_workspace(workspace_config)

        self.workspaces[workspace_id] = workspace_config
        self.active_processes[workspace_id] = []

        logger.info(f"Created workspace: {workspace_id} at {workspace_path}")
        return workspace_config

    async def _initialize_workspace(self, config: WorkspaceConfig):
        """Initialize workspace with basic setup files"""
        workspace_path = Path(config.base_path)

        # Create requirements.txt (only if missing)
        requirements_content = """
# Basic scientific computing packages
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=1.0.0
jupyter>=1.0.0
ipython>=7.0.0

# Additional packages for research
requests>=2.25.0
beautifulsoup4>=4.9.0
seaborn>=0.11.0
plotly>=5.0.0
"""
        (workspace_path / "requirements.txt").write_text(requirements_content.strip())

        # Create .gitignore
        gitignore_content = """
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Research outputs
output/
logs/
*.log
"""
        req_path = workspace_path / "requirements.txt"
        if not req_path.exists():
            req_path.write_text(requirements_content.strip())

        gitignore_path = workspace_path / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(gitignore_content.strip())

        # Create workspace info file
        info_content = f"""# UAgent Workspace: {config.workspace_id}

This workspace was created for UAgent research tasks.

## Structure
- `code/`: Source code files
- `data/`: Input data and datasets
- `output/`: Generated outputs and results
- `logs/`: Execution logs

## Usage
This workspace is isolated and can be used for:
- Code generation and testing
- Data analysis and experiments
- Scientific research workflows
- Repository analysis and integration

Created: {asyncio.get_event_loop().time()}
"""
        readme_path = workspace_path / "README.md"
        if not readme_path.exists():
            readme_path.write_text(info_content.strip())

        # Add a proxy helper script that agents can source explicitly when needed.
        proxy_script = workspace_path / "scripts" / "net_proxy.sh"
        if not proxy_script.exists():
            proxy_script.write_text(
                (
                    "#!/usr/bin/env bash\n"
                    "# Opt-in proxy helper; source this only when network downloads require a proxy.\n"
                    "export http_proxy=${http_proxy:-http://127.0.0.1:7890}\n"
                    "export https_proxy=${https_proxy:-http://127.0.0.1:7890}\n"
                    "export HTTP_PROXY=$http_proxy\n"
                    "export HTTPS_PROXY=$https_proxy\n"
                    "export ALL_PROXY=${ALL_PROXY:-socks5h://127.0.0.1:7890}\n"
                    "git config --global http.proxy \"$http_proxy\" >/dev/null 2>&1 || true\n"
                    "git config --global https.proxy \"$https_proxy\" >/dev/null 2>&1 || true\n"
                    "echo \"[proxy] http=$http_proxy https=$https_proxy all=$ALL_PROXY\"\n"
                ).strip()
            )
            try:
                os.chmod(proxy_script, 0o755)
            except Exception:
                pass

    async def get_workspace_status(self, workspace_id: str) -> Optional[WorkspaceStatus]:
        """Get current status of a workspace

        Args:
            workspace_id: ID of the workspace

        Returns:
            WorkspaceStatus or None if workspace doesn't exist
        """
        if workspace_id not in self.workspaces:
            return None

        config = self.workspaces[workspace_id]
        workspace_path = Path(config.base_path)

        if not workspace_path.exists():
            return WorkspaceStatus(
                workspace_id=workspace_id,
                status="error",
                created_at="unknown",
                last_activity="unknown",
                file_count=0,
                total_size=0,
                processes=[]
            )

        # Calculate workspace statistics
        file_count = 0
        total_size = 0
        for file_path in workspace_path.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size

        # Get process information
        processes = []
        for proc in self.active_processes.get(workspace_id, []):
            if proc.returncode is None:  # Still running
                processes.append({
                    "pid": proc.pid,
                    "status": "running"
                })

        return WorkspaceStatus(
            workspace_id=workspace_id,
            status="active" if processes else "stopped",
            created_at=str(workspace_path.stat().st_ctime),
            last_activity=str(workspace_path.stat().st_mtime),
            file_count=file_count,
            total_size=total_size,
            processes=processes
        )

    async def write_file(self, workspace_id: str, file_path: str, content: str, overwrite: bool = True) -> bool:
        """Write content to a file in the workspace

        Args:
            workspace_id: ID of the workspace
            file_path: Relative path within workspace
            content: File content to write
            overwrite: Whether to overwrite existing files

        Returns:
            bool: Success status
        """
        if workspace_id not in self.workspaces:
            # Workspace may have been cleaned up after session completion; treat as non-fatal
            logger.warning(f"Workspace {workspace_id} not found (write_file)")
            return False

        config = self.workspaces[workspace_id]
        full_path = Path(config.base_path) / file_path

        # Security check: ensure file is within workspace
        try:
            full_path.resolve().relative_to(Path(config.base_path).resolve())
        except ValueError:
            logger.error(f"Path {file_path} is outside workspace {workspace_id}")
            return False

        # Check file size limits
        if len(content.encode('utf-8')) > config.max_file_size:
            logger.error(f"File {file_path} exceeds size limit")
            return False

        # Check if file exists and overwrite policy
        if full_path.exists() and not overwrite:
            logger.warning(f"File {file_path} exists and overwrite=False")
            return False

        try:
            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            full_path.write_text(content, encoding='utf-8')
            logger.info(f"Wrote file {file_path} in workspace {workspace_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return False

    async def read_file(self, workspace_id: str, file_path: str) -> Optional[str]:
        """Read content from a file in the workspace

        Args:
            workspace_id: ID of the workspace
            file_path: Relative path within workspace

        Returns:
            str: File content or None if not found
        """
        if workspace_id not in self.workspaces:
            logger.warning(f"Workspace {workspace_id} not found (read_file)")
            return None

        config = self.workspaces[workspace_id]
        full_path = Path(config.base_path) / file_path

        # Security check: ensure file is within workspace
        try:
            full_path.resolve().relative_to(Path(config.base_path).resolve())
        except ValueError:
            logger.error(f"Path {file_path} is outside workspace {workspace_id}")
            return None

        try:
            if not full_path.exists():
                return None
            return full_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None

    async def list_files(self, workspace_id: str, directory: str = ".") -> List[Dict[str, Any]]:
        """List files in a workspace directory

        Args:
            workspace_id: ID of the workspace
            directory: Directory to list (relative to workspace root)

        Returns:
            List of file information dictionaries
        """
        if workspace_id not in self.workspaces:
            return []

        config = self.workspaces[workspace_id]
        dir_path = Path(config.base_path) / directory

        # Security check
        try:
            dir_path.resolve().relative_to(Path(config.base_path).resolve())
        except ValueError:
            logger.error(f"Directory {directory} is outside workspace {workspace_id}")
            return []

        files = []
        try:
            if not dir_path.exists():
                return []

            for item in dir_path.iterdir():
                stat = item.stat()
                files.append({
                    "name": item.name,
                    "path": str(item.relative_to(config.base_path)),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else 0,
                    "modified": stat.st_mtime
                })

            return sorted(files, key=lambda x: (x["type"] == "file", x["name"]))

        except Exception as e:
            logger.error(f"Failed to list files in {directory}: {e}")
            return []

    async def cleanup_workspace(self, workspace_id: str) -> bool:
        """Clean up and remove a workspace

        Args:
            workspace_id: ID of the workspace to clean up

        Returns:
            bool: Success status
        """
        if workspace_id not in self.workspaces:
            return False

        try:
            # Terminate any running processes
            if workspace_id in self.active_processes:
                for proc in self.active_processes[workspace_id]:
                    if proc.returncode is None:
                        proc.terminate()
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            proc.kill()
                del self.active_processes[workspace_id]

            # Remove workspace directory unless retention is requested
            config = self.workspaces[workspace_id]
            workspace_path = Path(config.base_path)
            retain_env = os.getenv("UAGENT_RETAIN_WORKSPACES", "true").lower() == "true"
            preserve_openhands = os.getenv("UAGENT_PRESERVE_OPENHANDS_DIRS", "true").lower() == "true"
            is_openhands_like = workspace_id.startswith("openhands_") or "openhands" in workspace_id
            if workspace_path.exists():
                if retain_env or (preserve_openhands and is_openhands_like):
                    logger.info("Retention enabled; keeping workspace %s at %s", workspace_id, workspace_path)
                else:
                    shutil.rmtree(workspace_path)

            # Remove from tracking
            del self.workspaces[workspace_id]

            logger.info(f"Cleaned up workspace: {workspace_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup workspace {workspace_id}: {e}")
            return False

    async def cleanup_all_workspaces(self):
        """Clean up all workspaces"""
        workspace_ids = list(self.workspaces.keys())
        for workspace_id in workspace_ids:
            await self.cleanup_workspace(workspace_id)

        logger.info("Cleaned up all workspaces")

    def get_workspace_path(self, workspace_id: str) -> Optional[Path]:
        """Get the filesystem path for a workspace

        Args:
            workspace_id: ID of the workspace

        Returns:
            Path object or None if workspace doesn't exist
        """
        if workspace_id not in self.workspaces:
            return None
        return Path(self.workspaces[workspace_id].base_path)
