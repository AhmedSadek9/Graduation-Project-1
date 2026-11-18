import json
from pathlib import Path


class MemoryRetrievalSystem:
    def __init__(self, db_filepath="memory_db.json"):
        db_path = Path(db_filepath)
        if not db_path.is_absolute():
            provided_path = Path.cwd() / db_path
            module_path = Path(__file__).resolve().parent / db_path
            db_path = provided_path if provided_path.exists() else module_path

        self.db_filepath = db_path
        self.memory_data = self._load_database()

    def _load_database(self):
        if not self.db_filepath.exists():
            raise FileNotFoundError(
                f"Memory database was not found: {self.db_filepath}"
            )

        try:
            with self.db_filepath.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Memory database contains invalid JSON: {self.db_filepath}"
            ) from error

        if not isinstance(data, list):
            raise ValueError("Memory database must contain a JSON list of records.")

        seen_ids = set()
        for index, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"Record at index {index} must be a JSON object.")

            person_id = record.get("person_id")
            fact = record.get("fact")
            source = record.get("source")
            update_date = record.get("update_date")

            if not isinstance(person_id, str) or not person_id.strip():
                raise ValueError(f"Record at index {index} has an invalid person_id.")
            if person_id in seen_ids:
                raise ValueError(f"Duplicate person_id found in memory database: {person_id}")
            if not isinstance(fact, str) or not fact.strip():
                raise ValueError(f"Record for {person_id} has an empty or invalid fact.")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Record for {person_id} has an empty or invalid source.")
            if not isinstance(update_date, str) or not update_date.strip():
                raise ValueError(f"Record for {person_id} has an empty or invalid update_date.")

            seen_ids.add(person_id)

        return data

    def get_memory_by_person_id(self, person_id):
        """Return verified memory for one person, without mixing records."""
        if not isinstance(person_id, str) or not person_id.strip():
            return {
                "status": "no-verified-information",
                "person_id": person_id,
                "fact": None,
                "message": "A valid person ID was not provided.",
            }

        person_id = person_id.strip()
        if person_id.lower() == "unknown":
            return {
                "status": "no-verified-information",
                "person_id": person_id,
                "fact": None,
                "message": "No verified information found for an unknown person.",
            }

        for record in self.memory_data:
            if record["person_id"] == person_id:
                return {
                    "status": "success",
                    "person_id": person_id,
                    "fact": record["fact"],
                    "source": record["source"],
                    "update_date": record["update_date"],
                }

        return {
            "status": "no-verified-information",
            "person_id": person_id,
            "fact": None,
            "message": f"No verified information available for ID {person_id}.",
        }


if __name__ == "__main__":
    memory_system = MemoryRetrievalSystem()

    for person_id in ("P001", "P999", "unknown", None):
        print(f"Lookup {person_id!r}:", memory_system.get_memory_by_person_id(person_id))
