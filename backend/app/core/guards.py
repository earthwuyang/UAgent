"""Generic acceptance gates to ensure real experiments (no simulation)."""

from __future__ import annotations

from pathlib import Path
import subprocess


def git_diff_nonempty(workspace: str) -> bool:
    """Check whether there are any changes in the workspace git repo.

    Falls back to checking for any files under code/ when git is absent.
    """
    try:
        out = subprocess.check_output(["git", "-C", workspace, "diff", "--name-only"], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        code_dir = Path(workspace) / "code"
        return any(code_dir.rglob("*"))


def artifact_min_size(path: str, min_kb: int = 1) -> bool:
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size >= min_kb * 1024


def rc_ok(rc: int) -> bool:
    return rc == 0

