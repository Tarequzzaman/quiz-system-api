import os
import re
import shutil
from pathlib import Path

UPLOAD_ROOT = Path(os.getenv("WORK_DIR", "/app/uploaded_files")).resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    return (SAFE_RE.sub("_", name) or "file")[:200]


def job_dir(job_id: str) -> Path:
    return UPLOAD_ROOT / job_id


def job_in_dir(job_id: str) -> Path:
    p = job_dir(job_id) / "in"
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_out_dir(job_id: str) -> Path:
    p = job_dir(job_id) / "out"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cleanup_job(job_id: str) -> None:
    d = job_dir(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def cleanup_job_in(job_id: str) -> None:
    d = job_dir(job_id) / "in"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
