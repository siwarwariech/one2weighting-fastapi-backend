from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.user import User
from app.crud import project as crud_project
from app.database.database import get_db
from app.auth.auth import get_current_user

router = APIRouter()

@router.get("/projects/{project_id}/next-step")
def get_next_step(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project = crud_project.get_project_with_next_step(db, project_id, current_user.id)


    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Map step ID to route slug
    step_routes = {
        1: "upload",
        2: "variables",
        3: "targets",
        4: "weighting",
        5: "rapport"
    }

    all_step_ids = list(step_routes.keys())

    # Get the first missing step
    next_step = next(
        (step_id for step_id in all_step_ids if step_id not in project.steps_completed),
        None
    )

    if next_step is None:
        # All steps are completed
        return {
            "next_step": None,
            "redirect_to": "results"  # ✅ redirect to summary page if available
        }

    return {
        "next_step": next_step,
        "redirect_to": step_routes[next_step]
    }

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.user import User
from app.auth.auth import get_current_user
from app.crud.project import complete_project_step
from app.database.database import get_db

# routers/project.py (ou ton fichier routes)
@router.patch("/projects/{project_id}/steps")
def update_project_step(
    project_id: int,
    step_data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    step_id = step_data.get("step_id")
    completed = step_data.get("completed", False)

    if not isinstance(step_id, int) or not completed:
        raise HTTPException(status_code=400, detail="Invalid step data: step_id and completed=true required.")

    project = complete_project_step(db, project_id, step_id, current_user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    step_routes = {1: "upload", 2: "variables", 3: "targets", 4: "weighting" , 5: "rapport"}
    all_steps = list(step_routes.keys())

    next_step = next((sid for sid in all_steps if sid not in project.steps_completed), None)

    return {
        "status": "success",
        "current_step": project.current_step,
        "steps_completed": project.steps_completed,
        "project_status": project.status,
        "next_step": next_step,
        "redirect_to": step_routes.get(next_step, "results")
    }

@router.post("/projects/{project_id}/complete-step/{step_id}")
def manually_complete_step(
    project_id: int,
    step_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    ⚠️ Route TEMPORAIRE pour forcer la complétion d'une étape
    Exemple : POST /projects/123/complete-step/2
    """
    project = complete_project_step(db, project_id, step_id, current_user.id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "ok": True,
        "steps_completed": project.steps_completed,
        "current_step": project.current_step,
        "status": project.status
    }
