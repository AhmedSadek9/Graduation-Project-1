
from ai.memory.database import initialize_database, get_connection


SEED_MEMORIES = [
    {
        "person_id": "person-001",
        "memory_text": "Likes listening to music",
        "verified": 1,
    },
    {
        "person_id": "person-001",
        "memory_text": "Prefers tea in the morning",
        "verified": 0,
    },
    {
        "person_id": "person-002",
        "memory_text": "Enjoys reading books",
        "verified": 1,
    },
    {
        "person_id": "person-002",
        "memory_text": "Likes gardening",
        "verified": 0,
    },
    {
        "person_id": "person-003",
        "memory_text": "Enjoys walking in the park",
        "verified": 1,
    },
    {
        "person_id": "person-003",
        "memory_text": "Prefers coffee",
        "verified": 0,
    },
]


def seed_database():
    initialize_database()

    with get_connection() as connection:
        for memory in SEED_MEMORIES:
            connection.execute(
                """
                INSERT INTO memories (person_id, memory_text, verified)
                SELECT ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM memories
                    WHERE person_id = ? AND memory_text = ?
                )
                """,
                (
                    memory["person_id"],
                    memory["memory_text"],
                    memory["verified"],
                    memory["person_id"],
                    memory["memory_text"],
                ),
            )

    print("WANAS seed data inserted successfully.")


if __name__ == "__main__":
    seed_database()
