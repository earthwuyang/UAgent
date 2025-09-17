from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import threading
import subprocess
import time
import os
import uuid


@dataclass
class Job:
    id: str
    name: str
    status: str = "queued"  # queued|running|succeeded|failed
    returncode: Optional[int] = None
    log_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobManager:
    jobs: Dict[str, Job] = {}
    processes: Dict[str, subprocess.Popen] = {}
    logs_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "..", "job_logs")

    @classmethod
    def ensure_logs_dir(cls) -> str:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "job_logs"))
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def start_subprocess(cls, name: str, cmd: list[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> str:
        cls.ensure_logs_dir()
        job_id = str(uuid.uuid4())
        log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "job_logs", f"{job_id}.log"))
        job = Job(id=job_id, name=name, status="running", log_path=log_path)
        cls.jobs[job_id] = job

        def _run():
            with open(log_path, "a", buffering=1) as log_file:
                try:
                    proc = subprocess.Popen(
                        cmd, cwd=cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT
                    )
                    cls.processes[job_id] = proc
                    rc = proc.wait()
                    job.returncode = rc
                    job.status = "succeeded" if rc == 0 else "failed"
                except Exception as e:
                    log_file.write(f"\n[ERROR] {e}\n")
                    job.status = "failed"
                    job.returncode = -1
                finally:
                    job.updated_at = time.time()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return job_id

    @classmethod
    def get(cls, job_id: str) -> Optional[Job]:
        return cls.jobs.get(job_id)

    @classmethod
    def tail_log(cls, job_id: str, n: int = 2000) -> str:
        job = cls.get(job_id)
        if not job or not job.log_path or not os.path.exists(job.log_path):
            return ""
        with open(job.log_path, "r", errors="ignore") as f:
            data = f.read()
        return data[-n:]

