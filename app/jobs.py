from dataclasses import asdict, dataclass
from time import time
from typing import Dict, Optional


@dataclass
class Job:
    id: str
    status: str = "PENDING"  # PENDING | PROCESSING | SUCCEEDED | FAILED
    progress: int = 0
    file_path: Optional[str] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = time()
    updated_at: float = time()

    def to_dict(self) -> dict:
        return asdict(self)


_JOBS: Dict[str, Job] = {}


def create_job(job: Job):
    _JOBS[job.id] = job


def get_job(job_id: str) -> Optional[Job]:
    return _JOBS.get(job_id)


def update_job(job_id: str, **fields):
    job = _JOBS[job_id]
    for k, v in fields.items():
        setattr(job, k, v)
    job.updated_at = time()
