from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
from typing import Optional

# Configuration (à adapter)
SECRET_KEY = "0f87fed02f34b7a9d3d38a48a47cefa9f8ccde13bd0ec7005f8b2abb5e0897af"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# app/utils/security.py
from passlib.context import CryptContext

# Configuration du hachage de mot de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hache un mot de passe (alias pour get_password_hash)"""
    return pwd_context.hash(password)

# Ajoutez cette fonction pour corriger l'erreur d'importation
def get_password_hash(password: str) -> str:
    """Hache un mot de passe - fonction attendue par d'autres modules"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie si un mot de passe correspond à son hash"""
    return pwd_context.verify(plain_password, hashed_password)