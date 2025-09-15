# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.models.user import User
from app.models.PasswordResetCode import PasswordResetCode
from app.utils.email import send_password_reset_email
from datetime import datetime, timedelta
from app.utils.security import get_password_hash  # Correction du nom

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/forgot-password")
async def forgot_password(email: str, db: Session = Depends(get_db)):
    # Vérifier si l'utilisateur existe
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Pour des raisons de sécurité, ne pas révéler si l'email existe
        return {"message": "Si l'email existe, un code de réinitialisation a été envoyé"}
    
    # Générer un code de réinitialisation
    code = PasswordResetCode.generate_code()
    expires_at = datetime.now() + timedelta(hours=1)
    
    # Désactiver les anciens codes pour cet email
    db.query(PasswordResetCode).filter(PasswordResetCode.email == email).update({"used": True})
    
    # Créer un nouveau code
    reset_code = PasswordResetCode(
        email=email,
        code=code,
        expires_at=expires_at
    )
    
    db.add(reset_code)
    db.commit()
    
    # Envoyer l'email (simulé pour le moment)
    print(f"Code de réinitialisation pour {email}: {code}")
    # await send_password_reset_email(email, code)
    
    return {"message": "Si l'email existe, un code de réinitialisation a été envoyé"}

@router.post("/reset-password")
async def reset_password(email: str, code: str, new_password: str, db: Session = Depends(get_db)):
    # Trouver le code de réinitialisation
    reset_code = db.query(PasswordResetCode).filter(
        PasswordResetCode.email == email,
        PasswordResetCode.code == code
    ).first()
    
    if not reset_code or not reset_code.is_valid():
        raise HTTPException(status_code=400, detail="Code invalide ou expiré")
    
    # Marquer le code comme utilisé
    reset_code.used = True
    
    # Mettre à jour le mot de passe de l'utilisateur
    user = db.query(User).filter(User.email == email).first()
    if user:
        user.password_hash = get_password_hash(new_password)
    
    db.commit()
    
    return {"message": "Mot de passe réinitialisé avec succès"}