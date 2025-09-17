from __future__ import annotations

import os
from typing import Optional, Dict

from ..core.jobs import JobManager


AI_SCIENTIST_ROOT = os.getenv("AI_SCIENTIST_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ai_scientist_local")))


def start_ai_scientist_job(
    ideas_json: str = "ai_scientist/ideas/i_cant_believe_its_not_better.json",
    idea_idx: int = 0,
    attempt_id: int = 0,
    skip_writeup: bool = True,
    skip_review: bool = True,
    extra_env: Optional[Dict[str, str]] = None,
) -> str:
    """
    Launch AI-Scientist-v2 experiment pipeline as a background job.
    Returns a job_id that can be polled via the jobs API.

    Raises:
        ValueError: If AI_SCIENTIST_ROOT is not set or directory doesn't exist
    """
    if not AI_SCIENTIST_ROOT:
        raise ValueError(
            "AI_SCIENTIST_ROOT environment variable must be set. "
            "Please set it to the path of your AI-Scientist-v2 installation."
        )

    if not os.path.isdir(AI_SCIENTIST_ROOT):
        raise ValueError(
            f"AI-Scientist directory not found: {AI_SCIENTIST_ROOT}. "
            "Please check your AI_SCIENTIST_ROOT environment variable."
        )

    cwd = AI_SCIENTIST_ROOT
    py = os.environ.get("PYTHON", "python")
    cmd = [
        py,
        "launch_scientist_bfts.py",
        "--load_ideas",
        ideas_json,
        "--idea_idx",
        str(idea_idx),
        "--attempt_id",
        str(attempt_id),
    ]
    if skip_writeup:
        cmd.append("--skip_writeup")
    if skip_review:
        cmd.append("--skip_review")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    return JobManager.start_subprocess("ai-scientist", cmd, cwd=cwd, env=env)

