from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict

from ..integrations.ai_scientist import start_ai_scientist_job


router = APIRouter(prefix="/experiments/ai-scientist", tags=["experiments-ai-scientist"])


class AISciRun(BaseModel):
    ideas_json: str = "ai_scientist/ideas/i_cant_believe_its_not_better.json"
    idea_idx: int = 0
    attempt_id: int = 0
    skip_writeup: bool = True
    skip_review: bool = True
    extra_env: Optional[Dict[str, str]] = None


@router.post("/start")
def start(run: AISciRun):
    job_id = start_ai_scientist_job(
        ideas_json=run.ideas_json,
        idea_idx=run.idea_idx,
        attempt_id=run.attempt_id,
        skip_writeup=run.skip_writeup,
        skip_review=run.skip_review,
        extra_env=run.extra_env,
    )
    return {"job_id": job_id}

