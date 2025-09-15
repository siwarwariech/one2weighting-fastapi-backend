from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

POSTGRES_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")

# Ajoutez ce paramètre pour SQLite uniquement
connect_args = {"check_same_thread": False} if "sqlite" in POSTGRES_URL else {}

engine = create_engine(
    POSTGRES_URL,
    echo=False,
    future=True,
    connect_args=connect_args  # Important pour SQLite
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# Créez cette fonction séparée pour les migrations
def create_tables():
    Base.metadata.create_all(bind=engine)

# Dependency pour FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()