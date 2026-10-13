from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class SystemContext:
    """In-memory context for the current recognized person and latest request."""
    current_person: Optional[Dict[str, Any]] = None
    last_intent: Optional[str] = None
    last_transcript: Optional[str] = None
    updated_at: Optional[str] = None

    def update_person(self, face_result: Optional[Dict[str, Any]]) -> None:
        """
        Store identity only when Face Recognition confidently returns `known`.
        Unknown/low-confidence results clear the old identity to prevent leakage.
        """
        if (
            isinstance(face_result, dict)
            and face_result.get("status") == "known"
            and face_result.get("person_id")
        ):
            self.current_person = {
                "person_id": str(face_result["person_id"]),
                "name": face_result.get("name"),
                "relationship": face_result.get("relationship"),
                "confidence": float(face_result.get("confidence", 0.0)),
                "status": "known",
            }
        else:
            self.current_person = None

        self._touch()

    def update_request(
        self,
        transcript: Optional[str],
        intent: Optional[str]
    ) -> None:
        self.last_transcript = transcript
        self.last_intent = intent
        self._touch()

    def clear(self) -> None:
        self.current_person = None
        self.last_intent = None
        self.last_transcript = None
        self._touch()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()