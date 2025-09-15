from app.database.database import Base, engine
from app.models import project  # Importez tous vos modèles

def migrate():
    print("Création des tables...")
    Base.metadata.create_all(bind=engine)
    print("Migration terminée avec succès!")

if __name__ == "__main__":
    migrate()