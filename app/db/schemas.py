from typing import Union

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    first_name: str
    last_name: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    first_name: str
    last_name: str
    role: str


class UserUpdateDetails(BaseModel):
    first_name: str
    last_name: str


class TokenWithUser(BaseModel):
    access_token: str
    token_type: str
    user: dict  # or


class LogInUser(BaseModel):
    email: str
    password: str


class TokenData(BaseModel):
    email: Union[str, None] = None


class CreatorInfo(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: str

    class Config:
        from_attributes = True


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class VerifyResetCodeRequest(BaseModel):
    email: EmailStr
    code: str


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    password: str


class QuizIn(BaseModel):
    jobId: str
    numQuestions: int = 12
    types: list[str] = [
        "mcq_single",
        "mcq_multi",
        "true_false",
        "answer_short_question",
    ]
    topicHint: str | None = None
