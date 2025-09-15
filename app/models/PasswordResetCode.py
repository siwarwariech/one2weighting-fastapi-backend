# models/PasswordResetCode.py
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from ..database.database import Base
from datetime import datetime, timedelta
import uuid

class PasswordResetCode(Base):
    __tablename__ = "password_reset_codes"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    code = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
    
    @staticmethod
    def generate_code():
        return str(uuid.uuid4())[:8].upper()
    
    def is_valid(self):
        return datetime.now() < self.expires_at and not self.used