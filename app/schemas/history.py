from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class HistoryItem(BaseModel):
    id: int
    user_id: int
    action: str
    details: Optional[str]
    ip_address: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True