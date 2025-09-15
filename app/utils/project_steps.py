# Définition des étapes du projet
PROJECT_STEPS = [
    {"id": 1, "name": "upload", "display_name": "Data Uploaded", "route": "upload"},
    {"id": 2, "name": "variables", "display_name": "Variables Selected", "route": "variables"},
    {"id": 3, "name": "targets", "display_name": "Targets Uploaded", "route": "targets"},
    {"id": 4, "name": "weighting", "display_name": "Weighting Done", "route": "weighting"},
    {"id": 5, "name": "report", "display_name": "Rapport Generated", "route": "rapport-weighted"}
]

def get_step_route(step_id):
    """Retourne la route correspondante à une étape"""
    for step in PROJECT_STEPS:
        if step["id"] == step_id:
            return step["route"]
    return "upload"

def get_next_step(current_step):
    """Retourne la prochaine étape"""
    if current_step >= len(PROJECT_STEPS):
        return None
    return current_step + 1

def validate_step_completion(project, step_id):
    """Valide si une étape peut être marquée comme complétée"""
    if step_id == 1:  # Upload survey
        return bool(project.survey_file)
    elif step_id == 2:  # Variables selection
        return bool(project.selected_variables and len(project.selected_variables) > 0)
    elif step_id == 3:  # Targets upload
        return bool(project.target_files and len(project.target_files) > 0)
    elif step_id == 4:  # Weighting
        return bool(project.weighting_results)
    elif step_id == 5:  # Report
        return bool(project.report_file)
    return False

def get_step_by_id(step_id: int):
    return next((s for s in PROJECT_STEPS if s["id"] == step_id), None)