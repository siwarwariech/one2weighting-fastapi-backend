@echo off
chcp 65001 > nul
title One2Weighting Backend
echo [BACKEND] Démarrage du serveur FastAPI...

REM Activer l'environnement virtuel
if exist "venv" (
    call venv\Scripts\activate.bat
) else (
    echo Création de l'environnement virtuel...
    python -m venv venv
    call venv\Scripts\activate.bat
    if exist "requirements.txt" (
        pip install -r requirements.txt
    ) else (
        pip install fastapi uvicorn
    )
)

echo [BACKEND] Serveur démarré sur: http://localhost:8000
echo [BACKEND] Documentation API: http://localhost:8000/docs
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause