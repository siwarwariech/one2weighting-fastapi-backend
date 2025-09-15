from sqlalchemy.orm import Session
from models.history import UserHistory

def create_history_entry(db: Session, user_id: int, action: str, details: str = ""):
    entry = UserHistory(user_id=user_id, action=action, details=details)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry