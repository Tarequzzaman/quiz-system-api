from datetime import datetime
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import models, schemas
from app.services.password_helper import get_password_hash, verify_password


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_all_users(db: Session):
    return db.query(models.User).order_by(models.User.id.asc()).all()


def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(
        first_name=user.first_name,
        last_name=user.last_name,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        role="user",
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user: models.User, user_update: schemas.UserUpdate):
    user.first_name = user_update.first_name
    user.last_name = user_update.last_name
    user.role = user_update.role if user_update.role in ("admin", "user") else "user"
    db.commit()
    db.refresh(user)
    return user


def update_user_details(
    db: Session, user: models.User, user_update: schemas.UserUpdateDetails
):
    user.first_name = user_update.first_name
    user.last_name = user_update.last_name

    print(
        f"Updating {user.id} with first_name={user.first_name} and last_name={user.last_name}"
    )

    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, user: models.User):
    db.delete(user)
    db.commit()
    return {"detail": "User deleted successfully"}


def authenticate_user(db: Session, email: str, password: str) -> Optional[models.User]:
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def insert_log_in_code_forgot_password(
    db: Session, code: str, user_id: int, expiry_time
) -> None:
    reset_code_entry = models.PasswordResetCode(
        user_id=user_id, code=code, expiry_time=expiry_time, status="pending"
    )
    db.add(reset_code_entry)
    db.commit()


def delete_old_pending_code(db: Session, user_id: int) -> None:
    db.query(models.PasswordResetCode).filter(
        models.PasswordResetCode.user_id == user_id,
        models.PasswordResetCode.status == "pending",
    ).delete()
    db.commit()


def get_pending_code_by_user(db: Session, user_id: int):
    return (
        db.query(models.PasswordResetCode)
        .filter(
            models.PasswordResetCode.user_id == user_id,
            models.PasswordResetCode.status == "pending",
            models.PasswordResetCode.expiry_time > datetime.now(),
        )
        .first()
    )


def accept_reset_code(db: Session, reset_entry: models.PasswordResetCode):
    reset_entry.status = "accepted"
    db.commit()


def reseat_password(db: Session, user: models.User, password: str) -> None:
    hashed_password = get_password_hash(password)
    user.hashed_password = hashed_password
    db.commit()


def insert_log_in_code(
    db: Session, code: str, user_id: int, expiry_time: datetime, email: str = None
):
    new_entry = models.PendingVerificationCode(
        code=code,
        expiry_time=expiry_time,
        email=email,  # Store email temporarily for new users
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry


def get_pending_code_by_email(db: Session, email: str):
    return (
        db.query(models.PendingVerificationCode)
        .filter(models.PendingVerificationCode.email == email)
        .filter(models.PendingVerificationCode.accepted == False)
        .order_by(models.PendingVerificationCode.created_at.desc())
        .first()
    )


def accept_reset_code(db: Session, reset_entry: models.PendingVerificationCode):
    reset_entry.accepted = True
    db.commit()
    db.refresh(reset_entry)
    return reset_entry


def create_upload_file(
    db: Session,
    user_id: int,
    upload_id: str,
    filenames: List[str],
    course_code: str,  # Add course_code parameter
    week: str,
):
    """Create a new upload file record."""
    db_upload = models.UploadFile(
        user_id=user_id,
        upload_id=upload_id,
        filenames=filenames,
        course_code=course_code,
        week=week,
    )
    db.add(db_upload)
    db.commit()
    db.refresh(db_upload)
    return db_upload


def get_files_by_job_id(db: Session, job_id: str) -> Optional[models.UploadFile]:
    return (
        db.query(models.UploadFile)
        .filter(models.UploadFile.upload_id == job_id)
        .first()
    )


def get_user_uploads(db: Session, user_id: Optional[int] = None):
    """Get uploads with user information (first_name, last_name)"""
    if not user_id:
        return (
            db.query(models.UploadFile)
            .join(models.User)
            .order_by(models.UploadFile.id.desc())
            .all()
        )

    return (
        db.query(models.UploadFile)
        .join(models.User)
        .filter(
            models.User.id == user_id,
        )
        .order_by(models.UploadFile.id.desc())
        .all()
    )


def delete_uploaded_files(db: Session, job_id: str) -> None:
    db.query(models.UploadFile).filter(models.UploadFile.upload_id == job_id).delete()
    db.commit()


def get_random_upload_id(db: Session) -> Optional[str]:
    upload = db.query(models.UploadFile).order_by(func.random()).first()
    if upload:
        return upload.upload_id
    return None