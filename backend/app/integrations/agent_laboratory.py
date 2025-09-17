from __future__ import annotations

import os
from typing import Optional, Dict

from ..core.jobs import JobManager


# Use local AgentLaboratory copy
AGENT_LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_laboratory_local"))


def start_agentlab_job(
    config_yaml: str = "experiment_configs/MATH_agentlab.yaml",
    extra_args: Optional[list[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> str:
    """
    Launch AgentLaboratory pipeline as a background job.
    Returns a job_id that can be polled via the jobs API.

    Raises:
        ValueError: If AgentLaboratory directory doesn't exist
    """
    if not os.path.isdir(AGENT_LAB_ROOT):
        raise ValueError(
            f"AgentLaboratory directory not found: {AGENT_LAB_ROOT}. "
            "This should contain the local AgentLaboratory installation."
        )

    cwd = AGENT_LAB_ROOT
    py = os.environ.get("PYTHON", "python")
    cmd = [py, "ai_lab_repo.py", "--yaml-location", config_yaml]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    return JobManager.start_subprocess("agent-laboratory", cmd, cwd=cwd, env=env)

