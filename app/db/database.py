from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
POSTGRESS_DB = get_settings()
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRESS_DB.DB_USER}:{POSTGRESS_DB.DB_PASSWORD}@{'db'}/{POSTGRESS_DB.DB_NAME}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()