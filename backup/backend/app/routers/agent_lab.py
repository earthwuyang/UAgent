from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, List

from ..integrations.agent_laboratory import start_agentlab_job


router = APIRouter(prefix="/experiments/agent-lab", tags=["experiments-agent-lab"])


class AgentLabRun(BaseModel):
    config_yaml: str = "experiment_configs/MATH_agentlab.yaml"
    extra_args: Optional[List[str]] = None
    extra_env: Optional[Dict[str, str]] = None


@router.post("/start")
def start(run: AgentLabRun):
    job_id = start_agentlab_job(
        config_yaml=run.config_yaml,
        extra_args=run.extra_args,
        extra_env=run.extra_env,
    )
    return {"job_id": job_id}

