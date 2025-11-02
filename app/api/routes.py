from fastapi import APIRouter

from app.api import views

router = APIRouter()


router.include_router(views.router)
