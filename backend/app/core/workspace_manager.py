"""
Workspace Manager for Code Generation and Task Execution
Handles workspace creation, virtual environments, and Docker containers
"""

import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
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

        written_paths = set()

        # Save files to appropriate directories
        for filename, content in files_found:
            try:
                destination = self._determine_file_destination(workspace, filename, content)
                if not destination:
                    continue

                target_dir = destination["directory"]
                target_dir.mkdir(exist_ok=True)
                clean_filename = destination["filename"]
                category = destination["category"]

                file_path = target_dir / clean_filename
                if file_path in written_paths:
                    base, ext = os.path.splitext(clean_filename)
                    suffix = 2
                    while (target_dir / f"{base}_{suffix}{ext}") in written_paths:
                        suffix += 1
                    file_path = target_dir / f"{base}_{suffix}{ext}"
                    clean_filename = file_path.name

                cleaned_content = self._clean_file_content(clean_filename, content)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)

                if file_path.suffix in {'.sh', '.bash'} or clean_filename.endswith('.sh'):
                    import stat
                    file_path.chmod(file_path.stat().st_mode | stat.S_IEXEC)

                logger.info(f"Saved file: {file_path}")
                written_paths.add(file_path)

                saved_files.setdefault(category, [])
                saved_files[category].append(clean_filename)

            except Exception as e:
                logger.error(f"Failed to save file {filename}: {e}")

        # Generate comprehensive deployment instructions
        self._generate_deployment_instructions(workspace, saved_files, llm_output)
        return saved_files

    def _determine_file_destination(self, workspace: Path, filename: str, content: str) -> Optional[Dict[str, Any]]:
        normalized = content.strip()
        if not normalized:
            return None

        lower_name = filename.lower()

        def root_file(name: str, category: str) -> Dict[str, Any]:
            return {"directory": workspace, "filename": name, "category": category}

        def scripts_file(name: str) -> Dict[str, Any]:
            return {"directory": workspace / "scripts", "filename": name, "category": "scripts"}

        def src_file(name: str, category: str = "code_files") -> Dict[str, Any]:
            return {"directory": workspace / "src", "filename": name, "category": category}

        if lower_name.endswith(('.md', '.txt')) and normalized.startswith('#'):
            return {"directory": workspace / "docs", "filename": re.sub(r'[<>:"/\\|?*]', '_', filename), "category": "docs"}

        if normalized.lower().startswith('from '):
            return root_file('Dockerfile', 'code_files')

        if normalized.startswith('[mysqld]'):
            return root_file('my.cnf', 'configs')

        if normalized.lower().startswith('version:'):
            return root_file('docker-compose.yml', 'configs')

        if normalized.startswith('#!/bin/bash') or lower_name.endswith(('.sh', '.bash')):
            script_name = self._infer_script_name(normalized, filename)
            return scripts_file(script_name)

        lowered = normalized.lower()
        if lowered.startswith('sudo ') or lowered.startswith('docker ') or lowered.startswith('chmod '):
            return None

        if lower_name.endswith('.ini'):
            return root_file('my.cnf', 'configs')

        if lower_name.endswith('.sql') or 'create database' in lowered:
            return src_file('init.sql', 'code_files')

        if lower_name.endswith('.yml') or lower_name.endswith('.yaml'):
            return root_file('docker-compose.yml', 'configs')

        return src_file(re.sub(r'[<>:"/\\|?*]', '_', filename))

    def _infer_script_name(self, content: str, original_name: str) -> str:
        sanitized_original = re.sub(r'[<>:"/\\|?*]', '_', Path(original_name).name)
        if sanitized_original.lower().endswith(('.sh', '.bash')) and not sanitized_original.lower().startswith('generated_code'):
            return sanitized_original

        meaningful_line = None
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#!'):
                continue
            meaningful_line = stripped.lstrip('#').strip()
            if meaningful_line:
                break

        if meaningful_line:
            slug = re.sub(r'[^a-zA-Z0-9]+', '-', meaningful_line.lower()).strip('-')
            if not slug:
                slug = 'script'
            return f"{slug}.sh"

        base = Path(original_name).stem
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', base.lower()).strip('-')
        if not slug or slug == 'generated_code':
            slug = 'script'
        return f"{slug}.sh"

    def _clean_file_content(self, filename: str, raw_content: str) -> str:
        """Strip markdown fences and extraneous prose from generated files."""
        content = raw_content.strip()

        # Extract primary code block if markdown fences present
        if '```' in content:
            import re
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
            if code_blocks:
                content = code_blocks[0].strip()
            else:
                content = content.replace('```', '').strip()

        # Remove lingering markdown headers or fences
        if content.startswith('```') and content.endswith('```'):
            content = content.strip('`').strip()

        lines = content.splitlines()
        if lines and lines[0].strip().lower() in {"dockerfile", "yaml", "ini", "bash", "sh"}:
            lines = lines[1:]
            content = "\n".join(lines)

        # For shell scripts ensure shebang present
        if filename.endswith(('.sh', '.bash')) and not content.startswith('#!'):
            content = f"#!/bin/bash\n{content}"

        # For docker-compose ensure YAML document clean (remove leading fenced markers etc handled above)
        return content.strip() + "\n"

    def _generate_deployment_instructions(self, workspace: Path, saved_files: Dict[str, List[str]], llm_output: str):
        """Generate comprehensive deployment instructions"""

        timestamp = datetime.now().isoformat()
        scripts = saved_files.get("scripts", [])
        configs = saved_files.get("configs", [])
        code_files = saved_files.get("code_files", [])

        def has_script(name: str) -> bool:
            return name in scripts

        def script_cmd(name: str) -> str:
            return f"./scripts/{name}" if has_script(name) else ""

        instructions = ["# Deployment Guide", "", f"Generated at: {timestamp}"]

        instructions.append("\n## Workspace Overview")
        if code_files:
            instructions.append("### Code & Configuration")
            for file_name in sorted(set(code_files + configs)):
                instructions.append(f"- `{file_name}`")
        if scripts:
            instructions.append("### Automation Scripts")
            for script_name in sorted(set(scripts)):
                instructions.append(f"- `{script_name}`")

        instructions.append("\n## Prerequisites")
        prerequisites = [
            "Docker (Engine) installed and running",
            "Docker Compose installed",
            "Internet access to pull the `mysql:8.0` image",
        ]
        for item in prerequisites:
            instructions.append(f"- {item}")

        instructions.append("\n## Setup Steps")
        instructions.append("```bash")
        instructions.append(f"cd {workspace.name}")
        if scripts:
            instructions.append("chmod +x scripts/*.sh")

        if has_script('build.sh'):
            instructions.append(script_cmd('build.sh'))
        elif 'Dockerfile' in code_files:
            instructions.append("docker build -t custom-mysql .")

        if has_script('run.sh'):
            instructions.append(script_cmd('run.sh'))
        elif 'docker-compose.yml' in configs:
            instructions.append("docker-compose up -d")

        if has_script('test.sh'):
            instructions.append(script_cmd('test.sh'))
        if has_script('deploy.sh'):
            instructions.append(script_cmd('deploy.sh'))

        instructions.append("```")

        instructions.append("\n## Verification")
        verify_steps = [
            "docker ps --filter name=mysql",  # list running mysql
            "docker logs -f mysql-container",  # tail logs
            "docker exec -it mysql-container mysql -u root -prootpassword -e \"SHOW DATABASES;\"",
        ]
        instructions.append("```bash")
        instructions.extend(verify_steps)
        instructions.append("```")

        instructions.append("\n## Troubleshooting")
        trouble = [
            "If `docker-compose build` fails with missing `my.cnf`, ensure the file exists and rerun the validation node.",
            "If the container exits immediately, check logs: `docker logs mysql-container`.",
            "For port conflicts, update the port mapping in `docker-compose.yml`."
        ]
        for item in trouble:
            instructions.append(f"- {item}")

        instructions.append("\n---\n")
        instructions.append("<details><summary>Complete LLM Output</summary>")
        instructions.append("\n\n```")
        instructions.append(llm_output)
        instructions.append("```\n</details>")

        (workspace / "DEPLOYMENT.md").write_text("\n".join(instructions), encoding="utf-8")

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
