
import sqlite3
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "wanas_memory.db"


def get_connection():
    """Create a connection to the WANAS SQLite database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """Create the memories table if it does not already exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                memory_text TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0
                    CHECK (verified IN (0, 1)),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_person_verified
            ON memories (person_id, verified)
            """
        )


if __name__ == "__main__":
    initialize_database()
    print("WANAS memory database initialized successfully.")
    print(f"Database location: {DATABASE_PATH}")
