from datetime import datetime, timedelta
import json
from fastapi import FastAPI, Depends, HTTPException, Request, logger, status, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List, Dict, Optional
import os
import logging
import uuid  # Ajoutez cette ligne avec les autres imports
from app.models.PasswordResetCode import PasswordResetCode  # AJOUTEZ CET IMPORT
from transformers import pipeline


from app.crud.project import complete_project_step, get_project_with_next_step

from .utils.project_steps import PROJECT_STEPS
from .routers import project

from app.crud import user

logging.basicConfig(level=logging.DEBUG)
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from pydantic import BaseModel
from jose import jwt, JWTError
# Database & Models
from .database.database import Base, engine, get_db
from .models.user import User
from .models.history import UserHistory
from .models.project import Project

# Schemas
from .schemas.auth import SignUpRequest, SignInRequest, UserResponse, Token
from .schemas.history import HistoryItem
from .schemas.project import ProjectCreate, ProjectOut

# Auth utilities
from .auth.auth import create_access_token, get_current_user
from .crud.user import create_user, get_user_by_email
from .utils.security import get_password_hash, verify_password
from .utils.encoding import smart_csv
from passlib.context import CryptContext  # Pour le hachage des mots de passe
from fastapi.security import OAuth2PasswordRequestForm
import requests

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Initialisation de l'application
app = FastAPI()
Base.metadata.create_all(bind=engine)
app.include_router(project.router)

# Configuration CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # "*" for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des uploads
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)



import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

# Charger les variables d'environnement
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ Ajoute GROQ_API_KEY=... dans ton .env")


# Init client Groq
client = Groq(api_key=GROQ_API_KEY)

# --- Conversation Memory ---
class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a useful AI assistant."}
        ]
        self.active: bool = True

conversations: Dict[str, Conversation] = {}

def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]

# --- Input Schema ---
class UserInput(BaseModel):
    message: str
    role: str = "user"
    conversation_id: str

# --- Query Groq API ---
def query_groq_api(conversation: Conversation) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # modèle rapide
            messages=conversation.messages,
            temperature=0.7,
            max_tokens=512,
        )

        return completion.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(
            status_code=400,
            detail="The chat session has ended. Please start a new session."
        )

    # Ajouter message utilisateur
    conversation.messages.append({"role": input.role, "content": input.message})

    # Générer réponse
    response = query_groq_api(conversation)

    # Ajouter réponse assistant
    conversation.messages.append({"role": "assistant", "content": response})

    return {"reply": response}


# Fonctions utilitaires pour les calculs
# ==============================================

def load_df(path: Path, **kwargs) -> pd.DataFrame:
    """Lit un CSV ou XLSX de manière robuste"""
    if path.suffix.lower() == ".csv":
        return smart_csv(path, **kwargs)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def improved_rake(df: pd.DataFrame, variables: List[str], targets: Dict[str, pd.Series], max_iter: int = 30, tol: float = 1e-4) -> np.ndarray:
    """Implémentation améliorée du raking/ipf"""
    weights = np.ones(len(df))
    for _ in range(max_iter):
        max_diff = 0
        for var in variables:
            current = df.groupby(var)["weight"].sum() / weights.sum()
            ratios = targets[var] / current
            weights *= df[var].map(ratios).fillna(1).values
            diff = (current - targets[var]).abs().max()
            max_diff = max(max_diff, diff)
        if max_diff < tol:
            break
    return weights / weights.mean()

# ==============================================
# CRUD Operations
# ==============================================

def create_project(db: Session, project: ProjectCreate, user_id: int):
    db_project = Project(**project.dict(), user_id=user_id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

def get_projects(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(Project).filter(Project.user_id == user_id).offset(skip).limit(limit).all()

def get_project(db: Session, project_id: int, user_id: int):
    return db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()

def create_history_entry(db: Session, user_id: int, action: str, details: str = ""):
    entry = UserHistory(user_id=user_id, action=action, details=details)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry

# ==============================================
# Routes d'authentification
# ==============================================

@app.post("/auth/signup")
async def signup(payload: SignUpRequest, db: Session = Depends(get_db)):
    try:
        print("Received payload:", payload.dict())
        
        existing_user = db.query(User).filter(User.email == payload.email).first()
        if existing_user:
            return JSONResponse(
                status_code=400,
                content={"detail": "Email already registered"}
            )
        
        hashed_password = pwd_context.hash(payload.password)
        user = User(
            email=payload.email,
            password_hash=hashed_password,
            first_name=payload.first_name,
            last_name=payload.last_name,
            company_name=payload.company_name
        )
        
        db.add(user)
        db.commit()
        
        return {
            "message": "User created successfully",
            "user": {
                "email": user.email,
                "first_name": user.first_name
            }
        }
        
    except Exception as e:
        db.rollback()
        print("Error:", str(e))
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )
    
from fastapi import HTTPException

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm


from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

@app.post("/auth/signin")
async def signin(
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        # Handle both form-data and JSON
        content_type = request.headers.get('Content-Type')
        
        if content_type == "application/x-www-form-urlencoded":
            form_data = await request.form()
            email = form_data.get("username") or form_data.get("email")
            password = form_data.get("password")
        elif content_type == "application/json":
            json_data = await request.json()
            email = json_data.get("username") or json_data.get("email")
            password = json_data.get("password")
        else:
            raise HTTPException(status_code=400, detail={
                "error": "invalid_content_type",
                "message": "Content-Type must be application/json or application/x-www-form-urlencoded"
            })

        if not email or not password:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_credentials",
                    "message": "Email and password are required"
                }
            )

        user = get_user_by_email(db, email)
        if not user:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_email",
                    "message": "Email incorrect - Veuillez réessayer"
                }
            )

        if not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_password", 
                    "message": "Mot de passe incorrect - Veuillez réessayer"
                }
            )

        return {
            "access_token": create_access_token({"sub": user.email}),
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "first_name": user.first_name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"SignIn Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Internal server error"
            }
        )
    
@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Récupère les informations de l'utilisateur connecté"""
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )
    return current_user

@app.get("/api/projects")
def list_projects_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # Cette dépendance vérifie le token
):
    return get_projects(db, current_user.id) 


@app.get("/auth/history", response_model=List[HistoryItem])
def get_user_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 100
):
    """Récupère l'historique des actions de l'utilisateur"""
    history = db.query(UserHistory)\
               .filter(UserHistory.user_id == current_user.id)\
               .order_by(UserHistory.created_at.desc())\
               .limit(limit)\
               .all()
    return history


# ==============================================
# Routes des projets
# ==============================================
@app.get("/projects/my-projects")
async def get_user_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    projects = db.query(Project).filter(Project.owner_id == current_user.id).all()
    return projects

# in main.py -> create_project_endpoint
@app.post("/api/projects", response_model=ProjectOut, status_code=201)
def create_project_endpoint(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Crée un nouveau projet"""
    # NEW: per-user name uniqueness
    exists = db.query(Project).filter(
        Project.user_id == current_user.id,
        Project.name == project.name
    ).first()
    if exists:
        raise HTTPException(status_code=409, detail="A project with this name already exists")

    db_project = create_project(db, project, current_user.id)
    create_history_entry(db, current_user.id, "project_created", f"Project {db_project.name} created")
    return db_project


@app.get("/api/projects", response_model=List[ProjectOut])
def list_projects_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Liste tous les projets de l'utilisateur"""
    return get_projects(db, current_user.id)

@app.get("/api/projects/{proj_id}", response_model=ProjectOut)
def get_project_endpoint(
    proj_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Récupère un projet spécifique"""
    project = get_project(db, proj_id, current_user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

# ==============================================
# Routes d'upload et de calcul
# ==============================================

import shutil
from pathlib import Path
from fastapi import HTTPException, UploadFile, File
def update_project_progress(db: Session, project_id: int, step_id: int, user_id: int):
    """Update project progress when a step is completed"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == user_id
    ).first()
    
    if not project:
        return
    
    # Update completed steps
    if step_id not in project.steps_completed:
        project.steps_completed.append(step_id)
    PROJECT_STEPS
    # Update current step (next uncompleted step)
    all_step_ids = [step["id"] for step in PROJECT_STEPS]
    next_step = next(
        (step_id for step_id in all_step_ids 
         if step_id not in project.steps_completed),
        None
    )
    
    project.current_step = next_step if next_step else max(all_step_ids)
    
    # Update status
    if not project.steps_completed:
        project.status = "Not Started"
    elif len(project.steps_completed) == len(PROJECT_STEPS):
        project.status = "Completed"
    else:
        project.status = f"In Progress ({len(project.steps_completed)}/{len(PROJECT_STEPS)})"
    
    db.commit()



# Assurez-vous cette route existe
@app.get("/api/projects/{project_id}/current-step")
def get_current_step(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return {"current_step": project.current_step}

@app.get("/api/projects/{project_id}/next-step")
def get_next_step(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    all_steps = [s["id"] for s in PROJECT_STEPS]
    next_step = next((step for step in all_steps if step not in project.steps_completed), None)

    step_routes = {
        1: "upload",
        2: "variables",
        3: "targets",
        4: "weighting",
        5: "report",
    }

    return {
        "next_step": next_step,
        "redirect_to": step_routes.get(next_step, "results" if project.status == "completed" else "upload")
    }


@app.patch("/projects/{project_id}/steps")
def update_project_step(
    project_id: int,
    step_data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    step_id = step_data.get("step_id")
    completed = step_data.get("completed", False)

    if not completed or not step_id:
        raise HTTPException(status_code=400, detail="Invalid step data")

    # ✅ UTILISER LA LOGIQUE CENTRALE
    project = complete_project_step(db, project_id, step_id, current_user.id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    step_routes = {
    1: "upload",
    2: "variables",
    3: "targets",
    4: "weighting",
    5: "report",
   }

    return {
       "status": "success",
       "current_step": project.current_step,
       "steps_completed": project.steps_completed,
       "project_status": project.status,
       "redirect_to": step_routes.get(project.current_step, "results")
    }



@app.get("/api/projects/{proj_id}/progress")
def get_project_progress(
    proj_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    project = get_project_with_next_step(db, proj_id, current_user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return {
        "current_step": project.current_step,
        "completed_steps": project.steps_completed,
        "next_step": project.next_step,
        "all_steps": PROJECT_STEPS
    }

def read_csv_flexible(path: Path):
    """Essaye différents séparateurs et encodages pour lire un CSV"""
    for enc in ["utf-8", "latin1"]:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if len(df.columns) > 1:
                    print(f"✅ Lecture {path.name} avec enc={enc}, sep={repr(sep)}")
                    return df
            except Exception:
                continue
    print(f"❌ Impossible de lire {path.name} correctement")
    return pd.DataFrame()


@app.post("/api/projects/{proj_id}/upload-survey")
async def upload_survey(
    proj_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Validation de l'extension
        ext = Path(file.filename).suffix.lower()
        if ext not in {".csv", ".xlsx", ".xls"}:
            raise HTTPException(400, detail="Seuls les fichiers CSV ou Excel sont autorisés")

        # Création du dossier projet
        project_dir = UPLOAD_DIR / str(proj_id)
        project_dir.mkdir(exist_ok=True)

        # Sauvegarde du fichier
        dest = project_dir / f"survey{ext}"
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Mise à jour du projet
        project = complete_project_step(db, proj_id, 1, current_user.id)

        
        # Ajoutez cette partie pour calculer next_step
                # ✅ Determine next step based on steps_completed
        all_steps = [1, 2, 3, 4]
        next_step = next(
            (step for step in all_steps if step not in project.steps_completed),
            None
        )

        step_redirect_map = {
            1: "upload",
            2: "variables",
            3: "targets",
            4: "weighting"
        }

        return {
            "ok": True,
            "next_step": next_step,
            "redirect_to": step_redirect_map.get(next_step, "")
        }

        
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur: {str(e)}")
    
    

@app.post("/api/projects/{proj_id}/upload-targets")
async def upload_targets(
    proj_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    project_dir = UPLOAD_DIR / str(proj_id)
    project_dir.mkdir(exist_ok=True)
    results = {}

    for file in files:
        try:
            ext = Path(file.filename).suffix.lower()
            if ext not in {".csv", ".xlsx"}:
                results[file.filename] = "❌ invalid extension"
                continue

            var_name = Path(file.filename).stem.replace("target_", "")
            dest = project_dir / f"target_{var_name}{ext}"

            with dest.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            results[file.filename] = "✅ saved"
        except Exception as e:
            results[file.filename] = f"❌ error: {str(e)}"

    # Marquer l'étape 3 (targets) comme complétée
    project = complete_project_step(db, proj_id, 3, current_user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # ⚡ Forcer next_step vers weighting
    next_step = 4
    redirect_to = "weighting"

    return {
        "ok": True,
        "results": results,
        "message": "All target files processed → proceeding to weighting",
        "redirect_to": redirect_to,
        "next_step": next_step
    }

def post_stratification(df: pd.DataFrame, var: str, target: pd.Series) -> np.ndarray:
    """Pondération post-stratifiée avec conversion de type automatique."""
    # 1. Conversion des types pour harmonisation
    if target.index.inferred_type == 'string' and df[var].dtype.kind in 'biufc':
        # Cas où target est string et enquête est numérique
        df = df.copy()
        df[var] = df[var].astype(str)
    elif target.index.inferred_type in ('integer', 'numeric') and df[var].dtype == object:
        # Cas où target est numérique et enquête est string
        try:
            df = df.copy()
            df[var] = pd.to_numeric(df[var])
        except ValueError:
            pass  # On garde les types originaux si la conversion échoue
    
    # 2. Nettoyage des données
    df_clean = df.dropna(subset=[var])
    
    # 3. Calcul des fréquences dans l'enquête
    cur = df_clean[var].value_counts(normalize=True)
    
    # 4. Debug: affichage des types
    print(f"Types - Target: {target.index.dtype}, Enquête: {df_clean[var].dtype}")
    
    # 5. Alignement des catégories
    common_categories = target.index.intersection(cur.index)
    
    if len(common_categories) == 0:
        # Tentative de conversion croisée si l'intersection est vide
        try:
            target_index_converted = target.index.astype(cur.index.dtype)
            common_categories = target_index_converted.intersection(cur.index)
            target = pd.Series(target.values, index=target_index_converted)
        except (TypeError, ValueError):
            pass
    
    if len(common_categories) == 0:
        raise ValueError(
            f"Aucune catégorie commune entre 'target' et l'enquête.\n"
            f"Target categories: {target.index.tolist()}\n"
            f"Survey categories: {cur.index.tolist()}\n"
            f"Types - Target: {target.index.dtype}, Enquête: {cur.index.dtype}"
        )
    
    # 6. Calcul des ratios
    target_filtered = target[common_categories]
    cur_filtered = cur[common_categories]
    
    # 7. Application des poids
    weights = df_clean[var].map(target_filtered / cur_filtered).dropna()
    return (weights / weights.mean()).to_numpy()

def improved_rake(df: pd.DataFrame,
                 variables: List[str],
                 targets: Dict[str, pd.Series],
                 max_iter: int = 30,
                 tol: float = 1e-4) -> np.ndarray:
    """Raking amélioré pour 2-3 variables avec gestion robuste des types et convergence"""
    # 1. Initialisation
    w = np.ones(len(df))
    
    # 2. Préparation des données
    df = df.copy()
    for v in variables:
        # Harmonisation des types entre target et enquête
        if targets[v].index.inferred_type == 'string' and df[v].dtype.kind in 'biufc':
            df[v] = df[v].astype(str)
        elif targets[v].index.inferred_type in ('integer', 'numeric') and df[v].dtype == object:
            try:
                df[v] = pd.to_numeric(df[v])
            except ValueError:
                pass

    # 3. Itérations de raking
    for iteration in range(max_iter):
        max_diff = 0.0
        for v in variables:
            # Calcul des marges courantes (pondérées)
            current = df.groupby(v).apply(lambda x: np.sum(w[x.index])) / np.sum(w)
            
            # Alignement des index et calcul du ratio
            aligned_current = current.reindex(targets[v].index).fillna(0.0)
            ratio = targets[v] / aligned_current
            
            # Protection contre les divisions par zéro
            ratio = ratio.replace([np.inf, -np.inf], 1.0).fillna(1.0)
            
            # Application des poids
            w *= df[v].map(ratio).fillna(1.0).to_numpy()
            
            # Suivi de la convergence
            diff = (aligned_current - targets[v]).abs().max()
            max_diff = max(max_diff, diff)
        
        # Normalisation des poids
        w /= w.mean()
        
        # Critère de convergence
        if max_diff < tol:
            print(f"Convergence atteinte en {iteration+1} itérations")
            break
    
    # 4. Post-traitement
    # Détection des poids extrêmes
    if np.max(w) > 10 or np.min(w) < 0.1:
        print(f"Avertissement : Poids extrêmes détectés (min={np.min(w):.2f}, max={np.max(w):.2f})")
    
    return w

@app.post("/api/projects/{proj_id}/run-weighting")
def run_weighting(
    proj_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Exécute les calculs de pondération"""
    try:
        method = payload["method"]
        variables = payload["vars"]
        spec = payload.get("targets", {})

        # Charger l'enquête - UPDATED PATH
        project_dir = UPLOAD_DIR / str(proj_id)
        csv = project_dir / "survey.csv"
        xls = project_dir / "survey.xlsx"
        
        if not csv.exists() and not xls.exists():
            raise HTTPException(404, "Survey file not found. Please upload survey data first.")

        df = smart_csv(csv) if csv.exists() else pd.read_excel(xls)

        # Construire les cibles
        targets: Dict[str, pd.Series] = {}
        for var in variables:
            info = spec.get(var, {})
            mapping = info.get("mapping", [])
            
            if mapping:
                tgt_pct = {row[0]: row[3] for row in mapping if row[3] is not None}
            else:
                # UPDATED TARGET PATH
                t_csv = project_dir / f"target_{var}.csv"
                t_xls = project_dir / f"target_{var}.xlsx"
                
                if not t_csv.exists() and not t_xls.exists():
                    raise HTTPException(400, f"Target file for '{var}' not found")
                
                tgt_df = smart_csv(t_csv) if t_csv.exists() else pd.read_excel(t_xls)
                pct_col = info.get("pctCol")
                
                if not pct_col or pct_col not in tgt_df.columns:
                    raise HTTPException(400, f"Percentage column not specified for '{var}'")
                
                tgt_pct = (
                    tgt_df[[var, pct_col]]
                    .assign(**{pct_col: (
                        tgt_df[pct_col].astype(str)
                        .str.replace("%", "")
                        .str.replace(",", ".")
                        .astype(float)
                    )})
                    .set_index(var)[pct_col]
                    .to_dict()
                )

            total = sum(tgt_pct.values())
            if total == 0:
                raise HTTPException(400, f"{var}: sum of percentages = 0")
            targets[var] = pd.Series({k: v/total for k, v in tgt_pct.items()})

        # Calcul des poids
        if method == "post" and len(variables) == 1:
            weights = post_stratification(df, variables[0], targets[variables[0]])
        elif method == "rake" and 2 <= len(variables) <= 3:
            weights = improved_rake(df, variables, targets)
        else:
            raise HTTPException(400, "Invalid method/variable combination")

        # Export
        df["weight"] = weights
        df_clean = df.copy()
        df_clean["weight"] = df_clean["weight"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # UPDATED OUTPUT PATH
        out_xlsx = project_dir / "weighted.xlsx"
        df_clean.to_excel(out_xlsx, index=False)

        preview_data = []
        for _, row in df_clean.head().iterrows():
            clean_row = {}
            for col, val in row.items():
                if isinstance(val, (float, np.floating)):
                    if np.isinf(val) or np.isnan(val):
                        clean_row[col] = None
                    else:
                        clean_row[col] = float(val)
                else:
                    clean_row[col] = val
            preview_data.append(clean_row)

        create_history_entry(db, current_user.id, "weighting_calculated", f"Weighting calculated for project {proj_id}")
        complete_project_step(db, proj_id, 4, current_user.id)

        return {
            "preview": preview_data,
            "download_url": f"/static/{proj_id}/weighted.xlsx",
        }


    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in run_weighting: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"Error during weighting calculation: {str(e)}")
        
# ==============================================
# Routes supplémentaires
# ==============================================


@app.post("/api/projects/{proj_id}/save-columns")
def save_selected_columns(
    proj_id: int,
    selected_columns: List[str] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Sauvegarde les variables sélectionnées par l'utilisateur
    et marque l'étape 2 comme complétée.
    """
    # Tu peux sauvegarder ces colonnes dans une table ProjectVariables si besoin.
    # Pour l’instant, on va juste marquer l’étape 2 comme faite.

    project = complete_project_step(db, proj_id, 2, current_user.id)

    return {
        "ok": True,
        "message": "Variables enregistrées",
        "steps_completed": project.steps_completed
    }



@app.get("/api/projects/{proj_id}/columns")
def get_columns(proj_id: int):
    """Récupère les colonnes avec le nom de fichier standard"""
    project_dir = UPLOAD_DIR / str(proj_id)
    csv_path = project_dir / "survey.csv"
    xls_path = project_dir / "survey.xlsx"

    if not csv_path.exists() and not xls_path.exists():
        raise HTTPException(404, detail="Aucun fichier d'enquête trouvé")

    try:
        # Read the entire file, not just the first 100 rows
        if csv_path.exists():
            df = pd.read_csv(csv_path, keep_default_na=False)
        else:
            df = pd.read_excel(xls_path, engine='openpyxl', keep_default_na=False)
        
        columns_meta = []
        for col in df.columns:
            col_series = df[col]
            
            # Function to detect if a value is blank
            def is_blank_value(x):
                if pd.isna(x) or x is None:
                    return True
                if isinstance(x, (str, bytes)):
                    return str(x).strip() == ''
                if isinstance(x, (int, float)):
                    return False  # Numbers are never considered blank
                return False
            
            # Apply blank detection
            blank_mask = col_series.apply(is_blank_value)
            blank_count = int(blank_mask.sum())
            total_count = len(col_series)
            blank_ratio = float(blank_count / total_count) if total_count > 0 else 0.0
            
            # Get non-blank values
            non_blank_values = col_series[~blank_mask]
            
            # Calculate unique values properly
            if len(non_blank_values) > 0:
                unique_count = int(non_blank_values.nunique())
                # For sample, take first 5 unique values instead of first 5 rows
                unique_sample = non_blank_values.unique()[:5]
                sample_str = ", ".join(str(item) for item in unique_sample)
            else:
                unique_count = 0
                sample_str = "—"
            
            columns_meta.append({
                "name": col,
                "hasBlank": bool(blank_count > 0),
                "sample": sample_str,
                "blank_ratio": blank_ratio,
                "unique": unique_count
            })
        
        return columns_meta
        
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur de lecture: {str(e)}")


def read_csv_flexible(path: Path):
    """Essaye différents séparateurs et encodages pour lire un CSV"""
    for enc in ["utf-8", "latin1"]:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if len(df.columns) > 1:
                    print(f"✅ Lecture {path.name} avec enc={enc}, sep={repr(sep)}")
                    return df
            except Exception:
                continue
    print(f"❌ Impossible de lire {path.name} correctement")
    return pd.DataFrame()



@app.get("/api/projects/{proj_id}/target-data")
def get_target_data(proj_id: int):
    """Récupère les données cibles pour toutes les variables"""
    project_dir = UPLOAD_DIR / str(proj_id)
    target_data = {}

    target_files = list(project_dir.glob("target_*.*"))

    for target_file in target_files:
        var_name = target_file.stem.replace("target_", "")

        try:
            # Lecture CSV ou Excel
            if target_file.suffix.lower() == ".csv":
                df = read_csv_flexible(target_file)
            else:
                df = pd.read_excel(target_file, engine="openpyxl")

            # Vérifie au moins 2 colonnes
            if len(df.columns) >= 2:
                cat_col, pct_col = df.columns[:2]
                mapping = []

                for _, row in df.iterrows():
                    modality = str(row[cat_col]).strip()
                    pct_value = row[pct_col]

                    official_pct = None
                    if pd.notna(pct_value):
                        try:
                            # Convertir en float sans diviser par 100
                            official_pct = float(
                                str(pct_value).replace("%", "").replace(",", ".")
                            )
                        except Exception:
                            official_pct = None

                    if official_pct is not None:
                        mapping.append([modality, official_pct])

                target_data[var_name] = mapping
            else:
                print(f"⚠️ Fichier {target_file.name} ignoré : moins de 2 colonnes")

        except Exception as e:
            print(f"❌ Erreur lecture {target_file.name} : {str(e)}")
            continue

    return target_data



# Dans main.py - ajoutez ces routes
@app.get("/api/projects/{proj_id}/download-weighted")
def download_weighted(proj_id: int, current_user: User = Depends(get_current_user)):
    """Télécharge les données pondérées"""
    project_dir = UPLOAD_DIR / str(proj_id)
    file = project_dir / "weighted.xlsx"
    
    if not file.exists():
        raise HTTPException(status_code=404, detail="Weighted file not found")
    
    return FileResponse(file, filename=f"weighted_data_{proj_id}.xlsx")

@app.get("/api/projects/{proj_id}/download-rapport")
def download_rapport(proj_id: int, current_user: User = Depends(get_current_user)):
    """Télécharge le rapport"""
    project_dir = UPLOAD_DIR / str(proj_id)
    rapport_file = project_dir / "rapport.pdf"  # ou .xlsx selon votre format
    
    if not rapport_file.exists():
        raise HTTPException(status_code=404, detail="Rapport not found")
    
    return FileResponse(rapport_file, filename=f"rapport_{proj_id}.pdf")

# Dans main.py - Ajoutez cette route
@app.patch("/api/projects/{proj_id}/status")
def update_project_status(
    proj_id: int,
    status_data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Met à jour le statut du projet"""
    project = db.query(Project).filter(
        Project.id == proj_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Mettre à jour le statut
    if 'status' in status_data:
        project.status = status_data['status']
    
    # Mettre à jour les étapes complétées
    if 'steps_completed' in status_data:
        project.steps_completed = status_data['steps_completed']
    
    # Si le statut est "finished", mettre current_step à 5
    if status_data.get('status') == 'finished':
        project.current_step = 5
    
    db.commit()
    
    return {
        "message": "Project status updated successfully",
        "status": project.status,
        "steps_completed": project.steps_completed,
        "current_step": project.current_step
    }

@app.get("/api/projects/{proj_id}/debug-targets")
def debug_targets(proj_id: int):
    """Route debug pour vérifier les fichiers cibles"""
    project_dir = UPLOAD_DIR / str(proj_id)
    
    if not project_dir.exists():
        return {"error": "Dossier projet non trouvé"}
    
    # Lister tous les fichiers
    all_files = [f.name for f in project_dir.glob("*")]
    
    # Fichiers cibles détectés
    target_files = [f.name for f in project_dir.glob("target_*")]
    
    return {
        "dossier": str(project_dir),
        "tous_les_fichiers": all_files,
        "fichiers_cibles_detectes": target_files
    }

@app.get("/api/projects/{proj_id}/check-uploads")
def check_uploads(proj_id: int):
    """Route debug pour vérifier les fichiers uploadés"""
    project_dir = UPLOAD_DIR / str(proj_id)
    
    if not project_dir.exists():
        return {"exists": False, "files": []}
    
    files = []
    for file_path in project_dir.glob("*"):
        files.append({
            "name": file_path.name,
            "size": file_path.stat().st_size,
            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        })
    
    return {"exists": True, "files": files}


@app.get("/api/projects/{proj_id}/modalities/{var_name}", response_model=List[str])
def get_modalities(proj_id: int, var_name: str):
    """Récupère les modalités d'une variable"""
    csv = UPLOAD_DIR / f"{proj_id}_survey.csv"
    xls = UPLOAD_DIR / f"{proj_id}_survey.xlsx"
    if not csv.exists() and not xls.exists():
        raise HTTPException(404, "Survey not uploaded")

    df = load_df(csv if csv.exists() else xls)
    if var_name not in df.columns:
        raise HTTPException(404, "Variable not found")

    return sorted(df[var_name].dropna().astype(str).unique().tolist())

@app.get("/api/projects/{proj_id}/survey-dist", response_model=Dict[str, Dict[str, float]])
def survey_dist(proj_id: int):
    project_dir = UPLOAD_DIR / str(proj_id)
    csv = project_dir / "survey.csv"
    xls = project_dir / "survey.xlsx"

    if not csv.exists() and not xls.exists():
        raise HTTPException(404, detail="Survey not uploaded")

    df = load_df(csv if csv.exists() else xls)

    out = {}
    for col in df.columns:
        vc = (
            df[col]
            .dropna()
            .astype(str)
            .value_counts(normalize=True)
            .round(4) * 100
        )
        out[col] = vc.to_dict()

    return out


@app.get("/api/projects/{proj_id}/download-weighted")
def download_weighted(
    proj_id: int,
    current_user: User = Depends(get_current_user)  # Add authentication
):
    """Télécharge les résultats pondérés"""
    # Updated path to match directory structure
    project_dir = UPLOAD_DIR / str(proj_id)
    file = project_dir / "weighted.xlsx"
    
    if not file.exists():
        raise HTTPException(
            status_code=404,
            detail="Weighted file not found. Please run weighting calculation first."
        )
    
    return FileResponse(
        file,
        filename=f"weighted_data_{proj_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.delete("/api/projects/{proj_id}", status_code=204)
def delete_project_endpoint(
    proj_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = db.query(Project).filter(Project.id == proj_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(project)
    db.commit()
    create_history_entry(db, current_user.id, "project_deleted", f"Project {proj_id} deleted")
    return



# --- AJOUTER / MODIFIER DANS main.py ---
from fastapi import Query
import math

def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
    return float(np.average(x, weights=w))

def _weighted_var(x: pd.Series, w: pd.Series) -> float:
    avg = np.average(x, weights=w)
    return float(np.average((x - avg)**2, weights=w))

def _weighted_std(x: pd.Series, w: pd.Series) -> float:
    return float(math.sqrt(_weighted_var(x, w)))

def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    sorter = np.argsort(x)
    x_sorted = x[sorter]
    w_sorted = w[sorter]
    cumw = np.cumsum(w_sorted)
    cutoff = q * cumw[-1]
    return float(x_sorted[np.searchsorted(cumw, cutoff)])

def _safe_numeric(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([np.nan] * len(s), index=s.index)

def _categorical_distribution(s: pd.Series, w: pd.Series):
    """Retourne toutes les modalités avec count pondéré + %"""
    dfc = pd.DataFrame({"val": s.astype(str), "w": w})
    vc = dfc.groupby("val", dropna=True)["w"].sum()
    total = vc.sum()
    if total == 0:
        return []
    return [
        {"label": idx, "count": float(c), "pct": float(round(c / total * 100, 4))}
        for idx, c in vc.sort_values(ascending=False).items()
    ]




# imports you need at top of file
from fastapi import Query, Request, HTTPException, Depends
from typing import List, Optional, Dict

# ... vos imports pandas/numpy/sqlalchemy etc.

def _value_distribution(x: pd.Series, w: pd.Series):
    """
    Distribution pondérée par valeur (chaque modalité/valeur devient une classe).
    Retourne [{label, count, pct}] trié par count décroissant.
    """
    s = pd.DataFrame({"val": x, "w": w}).dropna(subset=["val"])
    # ATTENTION: si val est float, on peut vouloir arrondir pour éviter des milliers de classes.
    # ici on garde tel quel comme demandé.
    vc = s.groupby("val", dropna=True)["w"].sum()
    total = float(vc.sum()) or 1.0
    rows = [
        {"label": str(idx), "count": float(c), "pct": float(c / total * 100.0)}
        for idx, c in vc.sort_values(ascending=False).items()
    ]
    return rows

@app.get("/api/projects/{proj_id}/weighted-stats")
def weighted_stats(
    proj_id: int,
    numeric_cols: Optional[List[str]] = Query(None, description="Repeated key, e.g. ?numeric_cols=Age&numeric_cols=Income"),
    categorical_cols: Optional[List[str]] = Query(None, description="Repeated key, e.g. ?categorical_cols=Sex&categorical_cols=Region"),
    request: Request = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # ---- Robust parsing (a=1&a=2 and a[]=1&a[]=2) ----
    if request is not None:
        q = request.query_params
        n1 = list(q.getlist("numeric_cols"))
        n2 = list(q.getlist("numeric_cols[]"))
        c1 = list(q.getlist("categorical_cols"))
        c2 = list(q.getlist("categorical_cols[]"))
        if n1 or n2:
            numeric_cols = n1 or n2
        if c1 or c2:
            categorical_cols = c1 or c2

    # ---- Load data ----
    project_dir = UPLOAD_DIR / str(proj_id)
    weighted_path = project_dir / "weighted.xlsx"
    survey_csv = project_dir / "survey.csv"
    survey_xls  = project_dir / "survey.xlsx"

    if weighted_path.exists():
        df = pd.read_excel(weighted_path, engine="openpyxl")
        if "weight" not in df.columns:
            raise HTTPException(500, detail="Weighted file present but 'weight' column missing.")
    elif survey_csv.exists() or survey_xls.exists():
        df = smart_csv(survey_csv) if survey_csv.exists() else pd.read_excel(survey_xls, engine="openpyxl")
        df["weight"] = 1.0
    else:
        raise HTTPException(404, detail="No survey or weighted data found. Upload data first.")

    # ---- weights ----
    w = df["weight"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0)

    # ---- KPIs ----
    sum_w = float(w.sum())
    sum_w2 = float((w**2).sum())
    n = int(len(df))
    n_eff = float((sum_w**2 / sum_w2) if sum_w2 > 0 else 0.0)
    w_min = float(np.min(w)) if len(w) else 0.0
    w_max = float(np.max(w)) if len(w) else 0.0

    # ---- Numeric stats => distribution par valeur (pas d'histogramme) ----
    numeric_out: List[Dict] = []
    if numeric_cols:
        for col in numeric_cols:
            if col not in df.columns:
                continue
            x = _safe_numeric(df[col])
            valid = x.notna() & w.notna()
            xv, wv = x[valid], w[valid]
            if len(xv) == 0 or float(wv.sum()) == 0:
                numeric_out.append({
                    "col": col,
                    "count": 0,
                    "mean": None, "std": None,
                    "min": None, "p25": None, "p50": None, "p75": None, "max": None,
                    "rows": [],            # <<< NOUVEAU: lignes par modalité
                    "hist": {"edges": [], "counts": [], "rows": []}  # pour compat
                })
                continue

            rows = _value_distribution(xv, wv)
            numeric_out.append({
                "col": col,
                "count": int(valid.sum()),
                "mean": round(_weighted_mean(xv, wv), 6),
                "std": round(_weighted_std(xv, wv), 6),
                "min": float(np.nanmin(xv)),
                "p25": round(_weighted_quantile(xv.to_numpy(float), wv.to_numpy(float), 0.25), 6),
                "p50": round(_weighted_quantile(xv.to_numpy(float), wv.to_numpy(float), 0.5), 6),
                "p75": round(_weighted_quantile(xv.to_numpy(float), wv.to_numpy(float), 0.75), 6),
                "max": float(np.nanmax(xv)),
                "rows": rows,            # <<< renvoyé au frontend
                "hist": {"edges": [], "counts": [], "rows": []}  # plus d'histogramme
            })

    # ---- Categorical stats (inchangé si tu veux les garder) ----
    categorical_out: List[Dict] = []
    if categorical_cols:
        for col in categorical_cols:
            if col not in df.columns:
                continue
            distrib = _categorical_distribution(df[col], w)
            categorical_out.append({"col": col, "rows": distrib})

    return {
        "weights": {"n_rows": n, "sum_w": round(sum_w, 6), "n_eff": round(n_eff, 6),
                    "w_min": round(w_min, 6), "w_max": round(w_max, 6)},
        "numeric": numeric_out,
        "categorical": categorical_out,
    }

# ==============================================
# Configuration des fichiers statiques
# ==============================================
# Inclusion des routeurs - CORRECTION ICI
from app.routers import auth as auth_router
from app.routers import project as project_router

app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(project_router.router, prefix="/projects", tags=["projects"])

# Route de test
@app.get("/")
async def root():
    return {"message": "API is working"}


from pydantic import BaseModel

# Ajoutez ce modèle Pydantic pour la validation
class ForgotPasswordRequest(BaseModel):
    email: str

@app.post("/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Endpoint pour mot de passe oublié - version indépendante de Gmail"""
    try:
        email = request.email
        
        # Vérifier si l'utilisateur existe
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Pour des raisons de sécurité, ne pas révéler si l'email existe
            return {"message": "Si l'email existe, un code de réinitialisation a été envoyé"}
        
        # Générer un code de réinitialisation
        code = str(uuid.uuid4())[:8].upper()
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
        
        # ✅ SOLUTION SANS GMAIL - Stockage local du code
        print("=" * 60)
        print("🔐 RÉINITIALISATION DE MOT DE PASSE")
        print("=" * 60)
        print(f"📧 Email: {email}")
        print(f"🔑 Code de réinitialisation: {code}")
        print(f"⏰ Expire à: {expires_at}")
        print("=" * 60)
        print("💡 Copiez ce code pour réinitialiser votre mot de passe")
        print("=" * 60)
        
        # Écriture dans un fichier pour faciliter la récupération
        with open("password_reset_codes.txt", "a") as f:
            f.write(f"{datetime.now()}: {email} - {code} (expire: {expires_at})\n")
        
        return {"message": "Si l'email existe, un code de réinitialisation a été envoyé"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    



@app.post("/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Endpoint pour mot de passe oublié - version indépendante de Gmail"""
    try:
        email = request.email
        
        # Vérifier si l'utilisateur existe
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Pour des raisons de sécurité, ne pas révéler si l'email existe
            return {"message": "Si l'email existe, un code de réinitialisation a été envoyé"}
        
        # Générer un code de réinitialisation
        code = str(uuid.uuid4())[:8].upper()
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
        
        # ✅ SOLUTION SANS GMAIL - Stockage local du code
        print("=" * 60)
        print("🔐 RÉINITIALISATION DE MOT DE PASSE")
        print("=" * 60)
        print(f"📧 Email: {email}")
        print(f"🔑 Code de réinitialisation: {code}")
        print(f"⏰ Expire à: {expires_at}")
        print("=" * 60)
        print("💡 Copiez ce code pour réinitialiser votre mot de passe")
        print("=" * 60)
        
        # Écriture dans un fichier pour faciliter la récupération
        with open("password_reset_codes.txt", "a") as f:
            f.write(f"{datetime.now()}: {email} - {code} (expire: {expires_at})\n")
        
        return {"message": "Si l'email existe, un code de réinitialisation a été envoyé"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")





class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str

@app.post("/auth/reset-password")
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Réinitialise le mot de passe avec le code"""
    try:
        # Trouver le code de réinitialisation
        reset_code = db.query(PasswordResetCode).filter(
            PasswordResetCode.email == request.email,
            PasswordResetCode.code == request.code
        ).first()
        
        if not reset_code or not reset_code.is_valid():
            raise HTTPException(status_code=400, detail="Code invalide ou expiré")
        
        # Marquer le code comme utilisé
        reset_code.used = True
        
        # Mettre à jour le mot de passe de l'utilisateur
        user = db.query(User).filter(User.email == request.email).first()
        if user:
            user.password_hash = get_password_hash(request.new_password)
            db.commit()
            
            print("=" * 50)
            print("✅ MOT DE PASSE RÉINITIALISÉ AVEC SUCCÈS")
            print("=" * 50)
            print(f"📧 Email: {request.email}")
            print(f"🔄 Mot de passe mis à jour")
            print("=" * 50)
            
            return {"message": "Mot de passe réinitialisé avec succès"}
        else:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
            
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
    
# Route de test CORS
@app.get("/test-cors")
def test():
    return {"ok": True}

# Configuration des fichiers statiques
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(static_dir / "favicon.ico")