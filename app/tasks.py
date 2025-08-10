import os

from .extract import walk_and_extract
from .jobs import update_job
from .rag import RAGIndex
from .storage import cleanup_job_in, job_dir


def process_job(job_id: str):
    try:
        update_job(job_id, status="PROCESSING", progress=0)
        in_root = job_dir(job_id) / "in"
        if not in_root.exists():
            update_job(job_id, status="FAILED", error="Input directory missing")
            return

        update_job(job_id, progress=10)

        extracted_files = walk_and_extract([in_root], ocr=False, return_per_file=True)
        if not extracted_files:
            update_job(job_id, status="FAILED", error="No text extracted from files")
            return

        update_job(job_id, progress=40)

        work_dir = os.getenv("WORK_DIR", "/app/uploaded_files")
        idx = RAGIndex(work_dir=work_dir, collection=job_id)

        total_chunks = 0
        for file_path, text in extracted_files:
            chunks = idx.add_document(
                docset_id=job_id, source=str(file_path), text=text
            )
            total_chunks += chunks

        update_job(
            job_id, status="SUCCEEDED", progress=100, chunks_indexed=total_chunks
        )

    except Exception as e:
        update_job(job_id, status="FAILED", error=str(e))
    finally:
        cleanup_job_in(job_id)
