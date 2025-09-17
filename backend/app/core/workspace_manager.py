"""
Workspace Manager for Code Generation and Task Execution
Handles workspace creation, virtual environments, and Docker containers
"""

import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
import logging

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages workspaces for code generation and execution"""

    def __init__(self, base_workspace_dir: str = None):
        if base_workspace_dir is None:
            # Use relative path: go up from backend/app/core/ to project root, then to workspace
            import os
            current_dir = os.path.dirname(__file__)  # app/core/
            backend_dir = os.path.dirname(os.path.dirname(current_dir))  # backend/
            project_root = os.path.dirname(backend_dir)  # project root
            base_workspace_dir = os.path.join(project_root, "workspace")

        self.base_workspace_dir = Path(base_workspace_dir)
        self.base_workspace_dir.mkdir(exist_ok=True)

    def create_workspace(self, task_name: str, task_type: str = "general") -> Dict[str, str]:
        """Create a new workspace for a task"""

        # Create workspace name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = "".join(c for c in task_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name[:50].replace(' ', '_')  # Limit length and replace spaces
        workspace_name = f"{clean_name}_{timestamp}"

        workspace_path = self.base_workspace_dir / workspace_name
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        subdirs = ["src", "tests", "docs", "data", "output", "logs"]
        for subdir in subdirs:
            (workspace_path / subdir).mkdir(exist_ok=True)

        # Create workspace metadata
        metadata = {
            "workspace_name": workspace_name,
            "workspace_path": str(workspace_path.absolute()),
            "task_name": task_name,
            "task_type": task_type,
            "created_at": datetime.now().isoformat(),
            "src_dir": str((workspace_path / "src").absolute()),
            "tests_dir": str((workspace_path / "tests").absolute()),
            "docs_dir": str((workspace_path / "docs").absolute()),
            "data_dir": str((workspace_path / "data").absolute()),
            "output_dir": str((workspace_path / "output").absolute()),
            "logs_dir": str((workspace_path / "logs").absolute())
        }

        # Write metadata file
        metadata_file = workspace_path / "workspace_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create README
        readme_content = f"""# {task_name}

**Workspace Created:** {metadata['created_at']}
**Task Type:** {task_type}

## Directory Structure
- `src/`: Source code files
- `tests/`: Test files
- `docs/`: Documentation
- `data/`: Input data files
- `output/`: Generated output files
- `logs/`: Execution logs

## Usage
This workspace was automatically created for the task: {task_name}

Use this directory for all code generation and execution related to this task.
"""
        with open(workspace_path / "README.md", 'w') as f:
            f.write(readme_content)

        logger.info(f"Created workspace: {workspace_path}")
        return metadata

    def parse_and_save_comprehensive_output(self, workspace_path: str, llm_output: str) -> Dict[str, List[str]]:
        """Parse LLM output and save all generated files with proper structure"""
        workspace = Path(workspace_path)
        saved_files = {
            "code_files": [],
            "scripts": [],
            "docs": [],
            "configs": []
        }

        # Enhanced file parsing patterns
        import re

        # Pattern to match files with various formats
        file_patterns = [
            r"(?:###?\s*)?(?:File:|Filename:|Create|Save as:)?\s*[`\"']?([^`\"'\n]+\.(py|sh|yml|yaml|json|md|txt|dockerfile|env|conf))[`\"']?\s*\n(.*?)(?=\n(?:###?\s*)?(?:File:|Filename:|Create|Save as:)|$)",
            r"```(\w+)?\s*(?:#\s*([^`\n]+\.(py|sh|yml|yaml|json|md|txt|dockerfile|env|conf)))?\n(.*?)```",
        ]

        files_found = []
        for pattern in file_patterns:
            matches = re.findall(pattern, llm_output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match) >= 3:
                    if isinstance(match[0], str) and '.' in match[0]:
                        # First pattern match
                        filename = match[0].strip()
                        content = match[2].strip()
                    else:
                        # Second pattern match
                        filename = match[1] if match[1] else f"generated_code.{match[0] if match[0] else 'txt'}"
                        content = match[3].strip()

                    files_found.append((filename, content))

        # If no files found, create basic structure from content
        if not files_found:
            # Extract any code blocks
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", llm_output, re.DOTALL)
            for i, code in enumerate(code_blocks):
                if "FROM " in code or "RUN " in code:
                    files_found.append((f"Dockerfile", code.strip()))
                elif "version:" in code and "services:" in code:
                    files_found.append((f"docker-compose.yml", code.strip()))
                elif "#!/bin/bash" in code or "#!/bin/sh" in code:
                    files_found.append((f"script_{i+1}.sh", code.strip()))
                else:
                    files_found.append((f"generated_code_{i+1}.py", code.strip()))

        # Save files to appropriate directories
        for filename, content in files_found:
            try:
                # Determine the appropriate directory
                if filename.endswith(('.sh', '.bash')):
                    target_dir = workspace / "scripts"
                    target_dir.mkdir(exist_ok=True)
                    saved_files["scripts"].append(filename)
                elif filename.endswith(('.py', '.dockerfile', 'Dockerfile')):
                    target_dir = workspace / "src"
                    saved_files["code_files"].append(filename)
                elif filename.endswith(('.yml', '.yaml', '.json', '.env', '.conf')):
                    target_dir = workspace / "src"
                    saved_files["configs"].append(filename)
                elif filename.endswith(('.md', '.txt')):
                    target_dir = workspace / "docs"
                    saved_files["docs"].append(filename)
                else:
                    target_dir = workspace / "src"
                    saved_files["code_files"].append(filename)

                # Clean filename
                clean_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                file_path = target_dir / clean_filename

                # Write file with proper permissions
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # Make shell scripts executable
                if filename.endswith(('.sh', '.bash')):
                    import stat
                    file_path.chmod(file_path.stat().st_mode | stat.S_IEXEC)

                logger.info(f"Saved file: {file_path}")

            except Exception as e:
                logger.error(f"Failed to save file {filename}: {e}")

        # Generate comprehensive deployment instructions
        self._generate_deployment_instructions(workspace, saved_files, llm_output)
        return saved_files

    def _generate_deployment_instructions(self, workspace: Path, saved_files: Dict[str, List[str]], llm_output: str):
        """Generate comprehensive deployment instructions"""

        instructions = f"""# Deployment Instructions

Generated at: {datetime.now().isoformat()}

## Files Generated
"""

        for category, files in saved_files.items():
            if files:
                instructions += f"\n### {category.replace('_', ' ').title()}\n"
                for file in files:
                    instructions += f"- {file}\n"

        instructions += f"""

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ (if running locally)
- Git (for version control)

### Build and Deploy
```bash
# Navigate to the workspace
cd {workspace.name}

# Make scripts executable
chmod +x scripts/*.sh

# Build the application
./scripts/build.sh

# Run tests (if available)
./scripts/test.sh

# Deploy the application
./scripts/deploy.sh

# Run locally for development
./scripts/run.sh
```

### Verification Steps
1. Check that all containers are running: `docker ps`
2. Check application logs: `docker logs <container_name>`
3. Test endpoints: `curl http://localhost:<port>/health`
4. Monitor resource usage: `docker stats`

### Troubleshooting
- If port conflicts occur, check running services: `netstat -tulpn`
- For permission errors, ensure user is in docker group: `sudo usermod -aG docker $USER`
- For build failures, check Dockerfile syntax and dependencies
- For network issues, verify Docker network configuration

### Generated Content Analysis
"""

        # Add analysis of what was generated
        if "Dockerfile" in str(saved_files):
            instructions += "✅ Docker containerization detected\n"
        if any("compose" in f for f in saved_files.get("configs", [])):
            instructions += "✅ Docker Compose orchestration detected\n"
        if saved_files.get("scripts"):
            instructions += "✅ Automated deployment scripts generated\n"

        instructions += f"""
### Complete LLM Output
```
{llm_output}
```
"""

        # Save instructions
        with open(workspace / "DEPLOYMENT.md", 'w') as f:
            f.write(instructions)

    def create_python_venv(self, workspace_path: str, python_version: str = "python3") -> Dict[str, str]:
        """Create a Python virtual environment in the workspace"""

        workspace = Path(workspace_path)
        venv_path = workspace / "venv"

        try:
            # Create virtual environment
            result = subprocess.run([
                python_version, "-m", "venv", str(venv_path)
            ], capture_output=True, text=True, cwd=workspace)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to create virtual environment: {result.stderr}")

            # Determine activation script path
            if os.name == 'nt':  # Windows
                activate_script = venv_path / "Scripts" / "activate.bat"
                pip_path = venv_path / "Scripts" / "pip"
                python_path = venv_path / "Scripts" / "python.exe"
            else:  # Unix/Linux/macOS
                activate_script = venv_path / "bin" / "activate"
                pip_path = venv_path / "bin" / "pip"
                python_path = venv_path / "bin" / "python"

            venv_info = {
                "venv_path": str(venv_path.absolute()),
                "activate_script": str(activate_script.absolute()),
                "pip_path": str(pip_path.absolute()),
                "python_path": str(python_path.absolute()),
                "created_at": datetime.now().isoformat()
            }

            # Create activation helper script
            helper_script = workspace / "activate_venv.sh"
            helper_content = f"""#!/bin/bash
# Virtual Environment Activation Helper
source {activate_script}
echo "Virtual environment activated: {venv_path}"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
"""
            with open(helper_script, 'w') as f:
                f.write(helper_content)
            helper_script.chmod(0o755)

            logger.info(f"Created Python virtual environment: {venv_path}")
            return venv_info

        except Exception as e:
            logger.error(f"Failed to create virtual environment: {e}")
            raise

    def install_python_packages(self, workspace_path: str, packages: List[str]) -> Dict[str, str]:
        """Install Python packages in the workspace virtual environment"""

        workspace = Path(workspace_path)
        venv_path = workspace / "venv"

        if not venv_path.exists():
            raise RuntimeError("Virtual environment not found. Create it first with create_python_venv()")

        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = venv_path / "Scripts" / "pip"
        else:  # Unix/Linux/macOS
            pip_path = venv_path / "bin" / "pip"

        installation_log = []

        try:
            for package in packages:
                logger.info(f"Installing package: {package}")
                result = subprocess.run([
                    str(pip_path), "install", package
                ], capture_output=True, text=True, cwd=workspace)

                if result.returncode == 0:
                    installation_log.append(f"✅ {package}: Installed successfully")
                else:
                    installation_log.append(f"❌ {package}: Failed - {result.stderr}")

            # Save installation log
            log_file = workspace / "logs" / "package_installation.log"
            with open(log_file, 'w') as f:
                f.write("\n".join(installation_log))

            return {
                "packages_requested": packages,
                "installation_log": installation_log,
                "log_file": str(log_file.absolute()),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
            raise

    def create_dockerfile(self, workspace_path: str, base_image: str = "python:3.9-slim",
                         packages: Optional[List[str]] = None) -> str:
        """Create a Dockerfile for the workspace"""

        workspace = Path(workspace_path)
        packages = packages or []

        dockerfile_content = f"""# Auto-generated Dockerfile for workspace
FROM {base_image}

WORKDIR /app

# Copy workspace contents
COPY . /app/

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
{f"RUN pip install {' '.join(packages)}" if packages else "# No additional packages specified"}

# Create non-root user
RUN useradd -m -u 1000 workspace_user
USER workspace_user

# Default command
CMD ["python", "--version"]
"""

        dockerfile_path = workspace / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        # Create .dockerignore
        dockerignore_content = """venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
.gitignore
README.md
workspace_metadata.json
logs/
"""
        with open(workspace / ".dockerignore", 'w') as f:
            f.write(dockerignore_content)

        logger.info(f"Created Dockerfile: {dockerfile_path}")
        return str(dockerfile_path.absolute())

    def build_docker_image(self, workspace_path: str, image_name: Optional[str] = None) -> Dict[str, str]:
        """Build Docker image for the workspace"""

        workspace = Path(workspace_path)
        dockerfile = workspace / "Dockerfile"

        if not dockerfile.exists():
            raise RuntimeError("Dockerfile not found. Create it first with create_dockerfile()")

        if not image_name:
            workspace_name = workspace.name
            image_name = f"workspace-{workspace_name.lower()}"

        try:
            logger.info(f"Building Docker image: {image_name}")
            result = subprocess.run([
                "docker", "build", "-t", image_name, "."
            ], capture_output=True, text=True, cwd=workspace)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to build Docker image: {result.stderr}")

            return {
                "image_name": image_name,
                "workspace_path": str(workspace.absolute()),
                "build_output": result.stdout,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise

    def run_docker_container(self, image_name: str, command: str = None,
                           workspace_path: str = None) -> Dict[str, str]:
        """Run a Docker container from the built image"""

        container_name = f"workspace-{uuid.uuid4().hex[:8]}"

        docker_cmd = [
            "docker", "run", "--rm", "--name", container_name
        ]

        # Mount workspace if provided
        if workspace_path:
            workspace = Path(workspace_path)
            docker_cmd.extend(["-v", f"{workspace.absolute()}:/app/workspace"])

        docker_cmd.append(image_name)

        if command:
            docker_cmd.extend(command.split())

        try:
            logger.info(f"Running Docker container: {container_name}")
            result = subprocess.run(docker_cmd, capture_output=True, text=True)

            return {
                "container_name": container_name,
                "image_name": image_name,
                "command": command or "default",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to run Docker container: {e}")
            raise

    def cleanup_workspace(self, workspace_path: str) -> bool:
        """Clean up a workspace directory"""

        try:
            workspace = Path(workspace_path)
            if workspace.exists():
                shutil.rmtree(workspace)
                logger.info(f"Cleaned up workspace: {workspace}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to cleanup workspace: {e}")
            return False

    def list_workspaces(self) -> List[Dict[str, str]]:
        """List all existing workspaces"""

        workspaces = []

        for workspace_dir in self.base_workspace_dir.iterdir():
            if workspace_dir.is_dir():
                metadata_file = workspace_dir / "workspace_metadata.json"

                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        workspaces.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {workspace_dir}: {e}")
                else:
                    # Basic info for workspaces without metadata
                    workspaces.append({
                        "workspace_name": workspace_dir.name,
                        "workspace_path": str(workspace_dir.absolute()),
                        "created_at": "unknown",
                        "task_name": "unknown"
                    })

        return sorted(workspaces, key=lambda x: x.get('created_at', ''), reverse=True)

    def get_docker_hello_world_workspace(self, task_name: str = "Docker Hello World") -> Dict[str, str]:
        """Create a specialized workspace for Docker hello-world task"""

        workspace_info = self.create_workspace(task_name, "docker")
        workspace_path = workspace_info["workspace_path"]

        # Create simple hello-world execution script
        script_content = """#!/bin/bash
# Docker Hello World Execution Script

echo "🐳 Docker Hello World Task"
echo "=========================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker daemon."
    exit 1
fi

echo "✅ Docker daemon is running"

# Pull hello-world image
echo "📥 Pulling hello-world image..."
docker pull hello-world

# Run hello-world container
echo "🚀 Running hello-world container..."
docker run hello-world

echo ""
echo "✅ Docker hello-world task completed!"
"""

        script_path = Path(workspace_path) / "src" / "run_hello_world.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        # Create Python equivalent
        python_script = '''#!/usr/bin/env python3
"""
Docker Hello World Task - Python Version
"""

import subprocess
import sys

def check_docker():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', 'info'],
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def main():
    print("🐳 Docker Hello World Task (Python)")
    print("===================================")

    # Check Docker
    if not check_docker():
        print("❌ Docker is not running. Please start Docker daemon.")
        sys.exit(1)

    print("✅ Docker daemon is running")

    # Pull hello-world image
    print("📥 Pulling hello-world image...")
    result = subprocess.run(['docker', 'pull', 'hello-world'])
    if result.returncode != 0:
        print("❌ Failed to pull hello-world image")
        sys.exit(1)

    # Run hello-world container
    print("🚀 Running hello-world container...")
    result = subprocess.run(['docker', 'run', 'hello-world'])

    if result.returncode == 0:
        print("")
        print("✅ Docker hello-world task completed!")
    else:
        print("❌ Docker hello-world task failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        python_script_path = Path(workspace_path) / "src" / "run_hello_world.py"
        with open(python_script_path, 'w') as f:
            f.write(python_script)
        python_script_path.chmod(0o755)

        workspace_info.update({
            "bash_script": str(script_path.absolute()),
            "python_script": str(python_script_path.absolute()),
            "execution_instructions": [
                f"cd {workspace_path}",
                "./src/run_hello_world.sh",
                "# OR",
                "python src/run_hello_world.py"
            ]
        })

        return workspace_info