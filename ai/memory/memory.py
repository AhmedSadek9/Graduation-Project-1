
from ai.memory.database import get_connection


def add_memory(person_id, memory_text, verified=False):
    """Save a memory for a person and return its new ID."""
    person_id = str(person_id or "").strip()
    memory_text = str(memory_text or "").strip()

    if not person_id:
        raise ValueError("person_id is required.")

    if not memory_text:
        raise ValueError("memory_text is required.")

    if not isinstance(verified, bool):
        raise ValueError("verified must be True or False.")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO memories (person_id, memory_text, verified)
            VALUES (?, ?, ?)
            """,
            (person_id, memory_text, int(verified)),
        )

        memory_id = cursor.lastrowid

    return memory_id


def get_memories(person_id, verified_only=False):
    """Return memories belonging to one person."""
    person_id = str(person_id or "").strip()

    if not person_id:
        raise ValueError("person_id is required.")

    query = """
        SELECT id, person_id, memory_text, verified, created_at
        FROM memories
        WHERE person_id = ?
    """
    parameters = [person_id]

    if verified_only:
        query += " AND verified = 1"

    query += " ORDER BY id"

    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]
