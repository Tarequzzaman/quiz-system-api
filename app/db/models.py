import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import relationship

from app.db.database import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    tutor = "tutor"
    user = "user"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.user, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    upload_files = relationship(
        "UploadFile",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PasswordResetCode(Base):
    __tablename__ = "password_reset_codes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    code = Column(String(6), nullable=False)
    expiry_time = Column(DateTime, nullable=False)
    status = Column(String(20), nullable=False, default="pending")

    user = relationship("User")


class PendingVerificationCode(Base):
    __tablename__ = "pending_verification_codes"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(6), nullable=False)
    email = Column(String, nullable=False)
    expiry_time = Column(DateTime, nullable=False)
    accepted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class UploadFile(Base):
    __tablename__ = "upload_files"

    id = Column(Integer, primary_key=True, index=True)
    course_code = Column(String, nullable=False)
    week = Column(String, nullable=False)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    upload_id = Column(
        UUID(as_uuid=True), default=uuid4, nullable=False, unique=True, index=True
    )
    filenames = Column(
        ARRAY(String), nullable=False, server_default=text("ARRAY[]::text[]")
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    user = relationship("User", back_populates="upload_files", passive_deletes=True)
    __table_args__ = (
        UniqueConstraint("user_id", "upload_id", name="uix_user_upload"),
        Index("ix_upload_files_filenames_gin", "filenames", postgresql_using="gin"),
    )
