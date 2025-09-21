from __future__ import annotations

from pathlib import Path
from glob import glob
import subprocess
import json


def file_exists(run_dir: Path, pattern: str, min_count: int = 1) -> bool:
    return len(glob(str(run_dir / pattern), recursive=True)) >= int(min_count)


def git_diff_nonempty(repo_dir: Path) -> bool:
    try:
        out = subprocess.run(
            ["bash", "-lc", "git status --porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def command_executed(exec_log: Path, step_id: str | None = None, expect_rc: int | None = None) -> bool:
    if not exec_log.exists():
        return False
    try:
        for line in exec_log.read_text().splitlines():
            try:
                evt = json.loads(line)
            except Exception:
                continue
            if evt.get("type") == "run":
                if step_id and evt.get("step_id") != step_id:
                    continue
                if expect_rc is not None and evt.get("rc") != expect_rc:
                    continue
                return True
        return False
    except Exception:
        return False


NO_MOCK_TOKENS = (
    "random.uniform",
    "np.random",
    "mock",
    "simulate",
    "simulation",
    "Generated placeholder",
)

