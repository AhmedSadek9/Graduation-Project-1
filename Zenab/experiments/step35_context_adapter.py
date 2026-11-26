"""Step 35 — conservative, experimental pronoun resolution for WANAS fictional demo.

No production integration. Never recover identity from conversation history alone.
Context must be a dict containing referent_person_id, session_id, and fresh=True;
caller supplies the active session id explicitly. No face/stt model execution here.
"""
from __future__ import annotations
import math
import re
from typing import Any

_PRONOUN_PATTERNS = (
    re.compile(r"\s*(?:tell me about|who is)\s+(?:him|her|he|she)\s*[?.!]?\s*", re.I),
)
_SAFE_TEXT = "Please confirm who you are asking about before I share any personal memory."


def is_pronoun_question(question: Any) -> bool:
    return isinstance(question, str) and any(p.fullmatch(question) for p in _PRONOUN_PATTERNS)


def abstain(reason: str, question: Any) -> dict:
    return {
        "status": "context_clarification_required",
        "intent": "retrieve_memory",
        "person_id": None,
        "fact_id": None,
        "question": question,
        "spoken_text": _SAFE_TEXT,
        "can_speak_personal_memory": False,
        "abstention_reason": reason,
        "context_resolution": "blocked",
        "retrieval_engine": "step35_context_adapter_experimental",
    }


class SafeContextAdapter:
    """Wraps a retrieval bridge exposing `process(face_result, question)` (E5 demo).

    For a non-pronoun question, forward the text to underlying bridge unchanged;
    for a pronoun question, require independent current face validation, explicit
    same-session fresh context, and a caller-supplied verified profile-name lookup.
    The lookup is permitted only after matching the current recognized person ID.
    """

    def __init__(self, bridge, profile_name_lookup, min_face_score: float = 0.75):
        if isinstance(min_face_score, bool) or not isinstance(min_face_score, (int, float)) or not 0 <= min_face_score <= 1:
            raise ValueError("invalid face score threshold")
        self.bridge = bridge
        self.profile_name_lookup = profile_name_lookup
        self.min_face_score = float(min_face_score)

    def process(self, face_result: Any, question: Any, context: Any = None, session_id: Any = None):
        if not is_pronoun_question(question):
            # The wrapped bridge owns all verification and person filtering.
            return self.bridge.process(face_result, question)
        if not isinstance(face_result, dict) or face_result.get("status") != "recognized":
            return abstain("current_face_not_recognized", question)
        pid = face_result.get("person_id")
        score = face_result.get("recognition_confidence")
        if not isinstance(pid, str) or not pid.strip() or pid.lower().startswith("unknown"):
            return abstain("invalid_current_person_id", question)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 or score < self.min_face_score:
            return abstain("invalid_or_low_current_face_score", question)
        if not isinstance(session_id, str) or not session_id.strip():
            return abstain("missing_current_session_id", question)
        if not isinstance(context, dict) or context.get("fresh") is not True:
            return abstain("context_missing_or_stale", question)
        if context.get("session_id") != session_id:
            return abstain("context_session_mismatch", question)
        if context.get("referent_person_id") != pid:
            return abstain("context_person_mismatch", question)
        # This is an explicitly narrow query normalization, not a general anaphora solver.
        name = self.profile_name_lookup(pid)
        if not isinstance(name, str) or not name.strip():
            return abstain("no_verified_name_for_current_person", question)
        resolved = f"Who is {name.strip()}?"
        result = self.bridge.process(face_result, resolved)
        # Do not override a refusal from the wrapped bridge.
        return {**result, "original_question": question, "resolved_question": resolved,
                "context_resolution": "same_session_current_face_and_referent_matched",
                "context_adapter": "step35_experimental"}
