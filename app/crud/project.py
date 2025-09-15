
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.utils.project_steps import PROJECT_STEPS, get_step_by_id
from ..models.project import Project, ProjectStatus
from ..schemas.project import ProjectCreate

def create_project(db: Session, project: ProjectCreate, user_id: int):
    db_project = Project(**project.dict(), user_id=user_id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

def get_projects(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    raw_projects = db.query(Project)\
                     .filter(Project.user_id == user_id)\
                     .offset(skip)\
                     .limit(limit)\
                     .all()

    enriched_projects = [
        get_project_with_next_step(db, p.id, user_id)
        for p in raw_projects
    ]
    return enriched_projects


def get_project(db: Session, project_id: int, user_id: int):
    return db.query(Project)\
             .filter(Project.id == project_id, Project.user_id == user_id)\
             .first()

def get_project_with_next_step(db: Session, project_id: int, user_id: int):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == user_id
    ).first()
    
    if project:
        # Calculer la prochaine étape
        all_step_ids = [step["id"] for step in PROJECT_STEPS]
        next_step_id = next(
            (step_id for step_id in all_step_ids 
             if step_id not in project.steps_completed),
            None
        )
        
        if next_step_id:
            next_step = get_step_by_id(next_step_id)
            project.next_step = next_step
        else:
            project.next_step = None
    
    return project


from sqlalchemy.orm.attributes import flag_modified

def complete_project_step(db: Session, project_id: int, step_id: int, user_id: int):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == user_id
    ).first()
    
    if not project:
        return None
    
    # Ajouter l'étape aux étapes complétées si elle n'y est pas déjà
    if step_id not in project.steps_completed:
        project.steps_completed.append(step_id)
    
    # Déterminer la prochaine étape
    all_step_ids = [1, 2, 3, 4, 5]  # IDs de toutes les étapes
    next_step = next((step for step in all_step_ids if step not in project.steps_completed), None)
    
    # Mettre à jour l'étape courante
    project.current_step = next_step if next_step else 5  # 5 = dernière étape
    
    # Mettre à jour le statut
    if len(project.steps_completed) == 0:
        project.status = "not_started"
    elif len(project.steps_completed) == len(all_step_ids):
        project.status = "finished"  # ✅ Changé de "completed" à "finished"
    else:
        project.status = f"in_progress_{project.current_step}"
    
    db.commit()
    return project