import os
import shutil
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import crud, database, schemas
from app.db.schemas import QuizIn
from app.services import auth
from app.services.jobs import Job, create_job, get_job
from app.services.quiz import evaluate_short_answer, generate_quiz_from_chunks
from app.services.rag import RAGIndex
from app.services.storage import job_in_dir, safe_name
from app.services.tasks import process_job

router = APIRouter()


@router.post("/quizzes", tags=["Quiz"])
def create_quiz(body: QuizIn):
    work_dir = os.getenv("WORK_DIR", "/app/uploaded_files")
    idx = RAGIndex(work_dir=work_dir, collection=body.jobId)
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


@router.post("/uploads", tags=["Quiz"])
async def upload(
    course_code: str = Form(...),  # Add course_code parameter
    week: str = Form(...),  # Add week parameter
    db: Session = Depends(database.get_db),
    current_user: schemas.UserOut = Depends(auth.get_current_active_user),
    files: List[UploadFile] = File(...),
    background: BackgroundTasks = None,
):
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")
    # if current_user.role != "admin":
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="You do not have appropriate permissions to upload files",
    #     )
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    job_id = str(uuid4())
    in_dir = job_in_dir(job_id)

    total_size = 0
    saved = 0
    MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
    uploaded_filenames = []

    for f in files:
        if not f.filename:
            continue
        safe_filename = safe_name(f.filename)
        dest = in_dir / safe_filename
        size = 0
        with dest.open("wb") as out:
            while chunk := await f.read(1_048_576):
                size += len(chunk)
                total_size += len(chunk)
                out.write(chunk)
        uploaded_filenames.append(safe_filename)
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
    background.add_task(process_job, job_id)

    try:
        created_upload = crud.create_upload_file(
            db=db,
            user_id=current_user.id,
            upload_id=job_id,
            filenames=uploaded_filenames,
            course_code=course_code,
            week=week,
        )
    except Exception:
        for p in in_dir.rglob("*"):
            if p.is_file():
                p.unlink()
        raise HTTPException(500, detail="Failed to save upload information")
    return JSONResponse(
        status_code=202,
        content={
            "id": created_upload.id,
            "job_id": job_id,
            "filenames": uploaded_filenames,
            "week": week,
            "course_code": course_code,
            "user_name": created_upload.user.first_name
            + " "
            + created_upload.user.last_name,
        },
        headers={"Location": f"/jobs/{job_id}"},
    )


@router.get("/jobs/{job_id}", tags=["Quiz"])
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


@router.get("/jobs/{job_id}/result", tags=["Quiz"])
def job_result(job_id: str):
    job = get_job(job_id)
    if not job or job.status != "SUCCEEDED" or not job.result_path:
        raise HTTPException(404, "Result not available")
    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(410, "Result expired")
    return FileResponse(path)


@router.get("/jobs/all/{job_id}/docksets", tags=["Quiz"])
def get_docsets(job_id: str):
    work_dir = os.getenv("WORK_DIR", "/app/uploaded_files")
    idx = RAGIndex(work_dir=work_dir, collection=job_id)
    docsets, metas = idx.get_all_for_docset(docset_id=job_id)

    if not docsets:
        raise HTTPException(404, "Docsets not found")
    return {"docsets": docsets, "metas": metas}


class EvaluateAnswerRequest(BaseModel):
    user_answer: str
    question: dict


@router.post("/evaluate_short_answer", tags=["Quiz"])
def evaluate_user_answer(body: EvaluateAnswerRequest):
    """
    Evaluate the user's answer against the correct answer(s) for a given question.
    """
    return evaluate_short_answer(user_answer=body.user_answer, question=body.question)


@router.post("/log_in", response_model=schemas.TokenWithUser, tags=["Login"])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(database.get_db),
):
    """Authenticate user and return access token along with user details."""

    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = auth.create_access_token(data={"sub": user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active,
        },
    }


@router.post("/users", response_model=schemas.UserOut, tags=["Login"])
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    try:
        db_user = crud.get_user_by_email(db, email=user.email)
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        crud.create_user(db=db, user=user)
        return JSONResponse(
            {"detail": "User successfully created"},
            status_code=201,
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Email already registered")


@router.get("/get_files/{job_id}", tags=["Quiz"])
def read_root(
    job_id: str,
    current_user: schemas.UserOut = Depends(auth.get_current_active_user),
    db: Session = Depends(database.get_db),
):
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")
    # if current_user.role != "admin":
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="You do not have appropriate permissions to upload files",
    #     )
    uploaded_files = crud.get_files_by_job_id(db, job_id=job_id)
    if not uploaded_files:
        raise HTTPException(status_code=404, detail="Files not found")

    data = {
        "job_id": uploaded_files.upload_id,
        "filenames": uploaded_files.filenames,
        "user_id": uploaded_files.user_id,
    }
    return data


@router.get("/user_uploads", tags=["Quiz"])
def get_user_uploads(
    current_user: schemas.UserOut = Depends(auth.get_current_active_user),
    db: Session = Depends(database.get_db),
):
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")

    user_uploads = None

    if current_user.role == "admin":
        user_uploads = crud.get_user_uploads(db)
    elif current_user.role == "tutor":
        user_uploads = crud.get_user_uploads(db, user_id=current_user.id)
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have appropriate permissions to upload files",
        )

    result = []
    for upload in user_uploads:
        result.append(
            {
                "id": upload.id,
                "course_code": upload.course_code,
                "week": upload.week,
                "job_id": str(upload.upload_id),
                "filenames": upload.filenames,
                "user_name": f"{upload.user.first_name} {upload.user.last_name}",
            }
        )

    return result


@router.get("/users", response_model=List[schemas.UserOut], tags=["Admin"])
def get_all_users(
    db: Session = Depends(database.get_db),
    current_user: schemas.UserOut = Depends(auth.get_current_active_user),
):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have admin permissions",
        )

    return crud.get_all_users(db)


@router.put("/users/{user_id}", response_model=schemas.UserOut, tags=["Admin"])
def update_user(
    user_id: int,
    user_update: schemas.UserUpdate,
    db: Session = Depends(database.get_db),
    current_user: schemas.UserOut = Depends(auth.get_current_active_user),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    user = crud.get_user(db=db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return crud.update_user(db, user, user_update)


@router.delete("/uploaded_files/{job_id}", tags=["Admin"], status_code=204)
def delete_uploaded_files(
    job_id: str,
    db: Session = Depends(database.get_db),
    current_user: schemas.UserOut = Depends(auth.get_current_active_user),
):
    """Delete uploaded job folder and its ChromaDB collection."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")

    work_dir = os.getenv("WORK_DIR", "/app/uploaded_files")
    job_dir = Path(work_dir) / job_id

    # 1️⃣ Remove the job's upload directory
    if job_dir.exists():
        shutil.rmtree(job_dir)

    # 2️⃣ Remove associated Chroma collection and its UUID folder
    try:
        idx = RAGIndex(work_dir=work_dir, collection=job_id)
        cleanup = idx.delete_collection()
        print(f"Chroma cleanup for {job_id}: {cleanup}")
    except Exception as e:
        print(f"Chroma deletion failed for {job_id}: {e}")

    # 3️⃣ Remove database record
    try:
        crud.delete_uploaded_files(db, job_id=job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB cleanup failed: {str(e)}")

    # 4️⃣ Return proper 204 (no content)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/random_upload_id", tags=["Admin"])
def get_random_upload_id(
    db: Session = Depends(database.get_db),
):
    return {"job_id": crud.get_random_upload_id(db)}
