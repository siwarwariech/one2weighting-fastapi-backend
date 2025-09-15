from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

class ProjectStatusUpdate(BaseModel):
    status: str  # Or use ProjectStatus enum if defined in models


# In schemas/project.py
class ProjectBase(BaseModel):
    name: str

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: str  # Changed from int
    created_at: datetime
    updated_at: Optional[datetime] = None
    status: str
    n_cases: int  # Backend uses n_cases
    user_id: str  # Changed from int
    owner_name: Optional[str] = None
    
    class Config:
        from_attributes = True


class ProjectOut(BaseModel):
    id: int
    user_id: int
    name: str
    created_at: datetime
    # NEW
    step: int
    status: str
    progress: Dict[str, Any]

    class Config:
        from_attributes = True



class ProjectIn(BaseModel):
    name: str

class ProjectOut(ProjectBase):
    id: int
    user_id: int
    status: str
    n_cases: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    current_step: int
    steps_completed: List[int]
    next_step: Optional[dict] = None
    
    class Config:
        from_attributes = True