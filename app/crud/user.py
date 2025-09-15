from datetime import datetime, timedelta
from jose import jwt
from typing import Optional

# Configuration (à adapter)
SECRET_KEY = "0f87fed02f34b7a9d3d38a48a47cefa9f8ccde13bd0ec7005f8b2abb5e0897af"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Crée un token JWT sécurisé
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

from sqlalchemy.orm import Session
from ..models.user import User
from ..utils.security import verify_password

def verify_user_credentials(db: Session, email: str, password: str) -> User | None:
    """
    Vérifie strictement les credentials utilisateur
    - Retourne l'utilisateur si valide
    - Retourne None si invalide
    """
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        return None
    
    if not verify_password(password, user.password_hash):
        return None
    
    return user
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from ..models.user import User
from ..schemas.user import UserCreate  # Assurez-vous que ce schéma existe
from ..utils.security import hash_password as get_password_hash
# Dans crud/user.py

def delete_demo_users(db: Session):
    db.query(User).filter(User.email.contains("demo")).delete()
    db.commit()
    
def create_user(db: Session, user_data: UserCreate) -> User:
    """
    Crée un nouvel utilisateur en base de données
    """
    try:
        hashed_password = get_password_hash(user_data.password)
        db_user = User(
            email=user_data.email,
            password_hash=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            # Ajoutez d'autres champs si nécessaire
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email déjà enregistré"
        )

def get_user_by_email(db: Session, email: str) -> User | None:
    """
    Récupère un utilisateur par son email
    """
    return db.query(User).filter(User.email == email).first()