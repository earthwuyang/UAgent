from fastapi import APIRouter, HTTPException
from ..core.jobs import JobManager


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(job_id: str):
    job = JobManager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "id": job.id,
        "name": job.name,
        "status": job.status,
        "returncode": job.returncode,
        "log_tail": JobManager.tail_log(job_id),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }

