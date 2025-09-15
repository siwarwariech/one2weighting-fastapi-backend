# models/user.py
from sqlalchemy import CheckConstraint, Column, String, Integer, DateTime, func, UniqueConstraint
from ..database.database import Base
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_user_email"),)
    __table_args__ = (
        CheckConstraint("email NOT LIKE '%demo%'", name="no_demo_accounts"),
    )

    id            = Column(Integer, primary_key=True)
    first_name    = Column(String(100), nullable=False)
    last_name     = Column(String(100),  nullable=False)
    company_name  = Column(String(150), nullable=False)
    email         = Column(String(256), nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    projects = relationship("Project", back_populates="owner")

