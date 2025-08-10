import os
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .jobs import Job, create_job, get_job
from .storage import job_in_dir, safe_name
from .tasks import process_job

app = FastAPI(title="Async Upload API (BackgroundTasks)")


from typing import List

from fastapi import FastAPI, File
from pydantic import BaseModel

from .quiz import generate_quiz_from_chunks
from .rag import RAGIndex

app = FastAPI(title="Async Upload API (BackgroundTasks, multi-file)")


class QuizIn(BaseModel):
    jobId: str
    numQuestions: int = 12
    types: list[str] = ["mcq_single", "mcq_multi", "true_false"]
    topicHint: str | None = None


@app.post("/quizzes")
def create_quiz(body: QuizIn):
    work_dir = os.getenv("WORK_DIR", "/app/uploaded_files")
    idx = RAGIndex(work_dir=work_dir, collection=body.jobId)

    # pull all chunks for this jobId
    docs, metas = idx.get_all_for_docset(body.jobId)
    if not docs:
        raise HTTPException(
            status_code=404, detail="No indexed documents for this jobId"
        )

    types = body.types

    quiz = generate_quiz_from_chunks(
        docs=docs,
        metas=metas,
        num_questions=body.numQuestions,
        types=types,
        topic_hint=body.topicHint,
    )
    return quiz


@app.post("/uploads")
async def upload(
    files: List[UploadFile] = File(...), background: BackgroundTasks = None
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    job_id = str(uuid4())
    in_dir = job_in_dir(job_id)

    total_size = 0
    saved = 0
    MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024

    for f in files:
        if not f.filename:
            continue
        dest = in_dir / safe_name(f.filename)
        size = 0
        with dest.open("wb") as out:
            while chunk := await f.read(1_048_576):  # 1 MB chunks
                size += len(chunk)
                total_size += len(chunk)
                out.write(chunk)
        saved += 1
        await f.close()

        if total_size > MAX_BYTES:
            for p in in_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            raise HTTPException(413, detail="Total upload too large")

    if saved == 0:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    create_job(Job(id=job_id, status="PENDING", progress=0))
    # Hand off by job_id only; the worker will read everything under in/
    background.add_task(process_job, job_id)

    return JSONResponse(
        status_code=202,
        content={"jobId": job_id, "statusUrl": f"/jobs/{job_id}"},
        headers={"Location": f"/jobs/{job_id}"},
    )


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status": job.status,
        "progress": job.progress,
        "resultUrl": f"/jobs/{job_id}/result" if job.result_path else None,
        "error": job.error,
    }


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = get_job(job_id)
    if not job or job.status != "SUCCEEDED" or not job.result_path:
        raise HTTPException(404, "Result not available")
    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(410, "Result expired")
    return FileResponse(path)
