from ensurepip import version
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from ..database.database import Base
from sqlalchemy.orm import relationship
from enum import Enum


class ProjectStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    status = Column(String(20), default=ProjectStatus.NOT_STARTED.value)
    current_step = Column(Integer, default=1)
    steps_completed = Column(JSON, default=[])
    selected_variables = Column(JSON, default=[])
    n_cases = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    user_id = Column(Integer, ForeignKey("users.id"))

    survey_file = Column(String)  # Étape 1
    selected_variables = Column(JSON)  # Étape 2
    target_files = Column(JSON)  # Étape 3
    weighting_results = Column(JSON)  # Étape 4
    report_file = Column(String)  # Étape 5
    
    owner = relationship("User", back_populates="projects")