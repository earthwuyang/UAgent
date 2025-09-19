"""Utilities for preparing repository-specific virtual environments."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import venv
from pathlib import Path
from typing import Iterable, List, Set


logger = logging.getLogger(__name__)


def ensure_repo_virtual_envs(repo_root: str) -> None:
    """Rebuild any bundled ``.venvs/persistent_venv`` directories if they are incomplete.

    When repositories are cloned from GitHub the virtual environment directory often
    ships without the ``lib`` contents, leaving only lightweight scripts and symlinks.
    Downstream tooling then hits ``FileNotFoundError`` when traversing the checkout.

    This routine detects those incomplete environments, removes them, recreates a fresh
    venv, and installs dependencies hinted by nearby README instructions or standard
    requirement files.
    """

    root_path = Path(repo_root)
    if not root_path.exists():
        logger.debug("Repository root missing when attempting env bootstrap: %s", repo_root)
        return

    for venv_path in root_path.glob("**/.venvs/persistent_venv"):
        if not venv_path.is_dir():
            continue
        if _is_valid_venv(venv_path):
            continue

        logger.info("Rebuilding incomplete virtual environment at %s", venv_path)
        _recreate_virtualenv(venv_path)

        requirement_paths = _collect_requirement_files(root_path, venv_path)
        if requirement_paths:
            _install_requirements(venv_path, requirement_paths, cwd=root_path)
        else:
            logger.info("No requirement files detected for %s", venv_path)


def _is_valid_venv(venv_path: Path) -> bool:
    """Heuristically determine whether a virtual environment looks usable."""

    if not venv_path.exists():
        return False

    # Linux/macOS layout
    unix_python = venv_path / "bin" / "python"
    unix_lib = venv_path / "lib"

    # Windows layout
    windows_python = venv_path / "Scripts" / "python.exe"
    windows_lib = venv_path / "Lib"

    python_executable_present = unix_python.exists() or windows_python.exists()
    if not python_executable_present:
        return False

    lib_dir = unix_lib if unix_lib.exists() else windows_lib
    if not lib_dir.exists():
        return False

    # Ensure there is at least one Python version directory under lib
    has_python_runtime = any(child.is_dir() and child.name.startswith("python") for child in lib_dir.iterdir())
    return has_python_runtime


def _recreate_virtualenv(venv_path: Path) -> None:
    """Remove the existing directory (if any) and create a fresh virtual environment."""

    shutil.rmtree(venv_path, ignore_errors=True)
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_path)


def _collect_requirement_files(repo_root: Path, venv_path: Path) -> List[Path]:
    """Assemble requirement files from README hints and conventional locations."""

    candidates: Set[Path] = set()

    # 1. Look for requirements next to the virtualenv's parent project directory.
    project_dir = venv_path.parent.parent
    default_requirements = project_dir / "requirements.txt"
    if default_requirements.exists():
        candidates.add(default_requirements.resolve())

    for pattern in ("requirements/*.txt", "requirements/**/requirements*.txt"):
        for match in project_dir.glob(pattern):
            if match.is_file():
                candidates.add(match.resolve())

    # 2. Parse repo README files for explicit pip install -r instructions.
    candidates.update(_extract_requirements_from_readme(repo_root))

    # Only keep files that still exist under the repository root.
    result = [path for path in candidates if path.exists() and _is_within_repo(repo_root, path)]
    return sorted(result)


def _extract_requirements_from_readme(repo_root: Path) -> Set[Path]:
    requirement_paths: Set[Path] = set()
    readme_names = [
        "README.md",
        "README.MD",
        "Readme.md",
        "README",
    ]

    pattern = re.compile(r"pip\s+install\s+-r\s+([\w./\\-]+)")

    for name in readme_names:
        readme_path = repo_root / name
        if not readme_path.exists():
            continue

        try:
            content = readme_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = readme_path.read_text(encoding="latin-1")

        for match in pattern.findall(content):
            cleaned = match.strip('"\'')
            candidate = (repo_root / cleaned).resolve()
            requirement_paths.add(candidate)

    return requirement_paths


def _is_within_repo(repo_root: Path, target: Path) -> bool:
    try:
        repo_root_resolved = repo_root.resolve()
        target_resolved = target.resolve()
    except FileNotFoundError:
        return False

    return repo_root_resolved in target_resolved.parents or repo_root_resolved == target_resolved


def _install_requirements(venv_path: Path, requirement_paths: Iterable[Path], cwd: Path) -> None:
    pip_executable = venv_path / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    if not pip_executable.exists():
        logger.warning("pip executable missing for %s", venv_path)
        return

    for requirement_file in requirement_paths:
        cmd = [str(pip_executable), "install", "-r", str(requirement_file)]
        logger.info("Installing requirements via %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=str(cwd))
        except subprocess.CalledProcessError as exc:
            logger.warning("Failed to install requirements from %s: %s", requirement_file, exc)

