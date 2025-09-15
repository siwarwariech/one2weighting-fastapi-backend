from app.database.database import engine
from sqlalchemy import text

def upgrade():
    with engine.connect() as conn:
        # Pour SQLite qui ne supporte pas bien ALTER TABLE
        conn.execute(text("""
        CREATE TABLE projects_new (
            id INTEGER NOT NULL,
            name VARCHAR(100) NOT NULL,
            status VARCHAR(20),
            current_step INTEGER DEFAULT 0 NOT NULL,
            steps_completed TEXT DEFAULT '[]',
            n_cases INTEGER,
            created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
            updated_at DATETIME,
            user_id INTEGER,
            PRIMARY KEY (id),
            FOREIGN KEY(user_id) REFERENCES users (id)
        )
        """))
        
        # Copiez les données
        conn.execute(text("""
        INSERT INTO projects_new (id, name, status, n_cases, created_at, user_id)
        SELECT id, name, status, n_cases, created_at, user_id FROM projects
        """))
        
        # Remplacez l'ancienne table
        conn.execute(text("DROP TABLE projects"))
        conn.execute(text("ALTER TABLE projects_new RENAME TO projects"))
        conn.commit()

if __name__ == "__main__":
    upgrade()
    print("Migration manuelle réussie!")