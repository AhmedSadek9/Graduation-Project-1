"""
WANAS + ARC — Personal Memory Retrieval Module (Part 2: Core Retrieval Engine)
Owner: Zinab (with Ahmed for Backend/API Integration)
Version: 2.4

This module loads `memory_database.json` and provides safe, deterministic
retrieval by `person_id` with strict safety gating for unknown identities,
low recognition confidence, and unanswerable/missing facts.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


SAFE_UNKNOWN_MESSAGE = "Unrecognized visitor — no saved profile."
SAFE_LOW_CONFIDENCE_MESSAGE = "Identity uncertain — please verify before retrieving personal memories."
SAFE_NO_MEMORY_MESSAGE = "No verified information available for this request."
SAFE_PERSON_MISMATCH_MESSAGE = "The person mentioned in the question does not match the recognized identity. Please verify the person."


# Production records must have both explicit caregiver verification and review approval.
PRODUCTION_VERIFICATION = "caregiver_verified"
PRODUCTION_REVIEW = "approved"
SAFE_NO_APPROVED_MEMORY_MESSAGE = "No caregiver-approved memory is available for this person."


class MemoryRetriever:
    def __init__(self, db_path: str = "memory_database.json", min_confidence: float = 0.75, data_mode: str = "demo"):
        self.db_path = Path(db_path)
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)) or not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be a finite number between 0 and 1")
        self.min_confidence = min_confidence
        # Backward-compatible demo default for existing synthetic test suite.
        # IMPORTANT: production callers MUST pass data_mode="production" explicitly.
        if data_mode not in ("demo", "production"):
            raise ValueError("data_mode must be 'demo' or 'production'")
        self.data_mode = data_mode
        self._persons_index: Dict[str, Dict[str, Any]] = {}
        self._load_database()

    def _load_database(self) -> None:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Memory database not found at: {self.db_path}")
        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for person in data.get("persons", []):
            pid = person.get("person_id")
            if pid:
                self._persons_index[pid] = person

    def _person_is_eligible(self, person: Dict[str, Any]) -> bool:
        if self.data_mode == "demo":
            return True
        # Never expose a synthetic/fictional person's identity in production.
        if person.get("source_type") == "synthetic_test_data" or person.get("status") == "fictional_for_testing":
            return False
        return True

    def _fact_is_eligible(self, fact: Dict[str, Any]) -> bool:
        if self.data_mode == "demo":
            return (fact.get("verification_status") == "verified_in_fictional_scenario"
                    and fact.get("source_type") == "synthetic_test_data")
        return (fact.get("verification_status") == PRODUCTION_VERIFICATION
                and fact.get("review_status") == PRODUCTION_REVIEW
                and fact.get("source_type") != "synthetic_test_data"
                and isinstance(fact.get("memory_fact"), str)
                and bool(fact["memory_fact"].strip()))

    def get_person_profile(
        self,
        person_id: Optional[str],
        recognition_confidence: float = 1.0,
        tier: Optional[int] = None,
        fact_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve verified memory facts for a given `person_id`.

        Safety Rules Enforced:
        1. If `person_id` is None, empty, or starts with 'unknown', abstain immediately.
        2. If `recognition_confidence` < `min_confidence`, abstain immediately.
        3. If `person_id` is not in the database, abstain safely without inventing facts.
        4. Never cross-query facts from any other `person_id`.
        """
        # 1. Check unknown or missing ID
        if not person_id or str(person_id).strip().lower().startswith("unknown"):
            return {
                "status": "unknown_identity_abstain",
                "person_id": person_id,
                "name": None,
                "relationship_to_wearer": None,
                "facts": [],
                "safe_message": SAFE_UNKNOWN_MESSAGE,
            }

        # 2. Check recognition confidence threshold
        if (
            isinstance(recognition_confidence, bool)
            or not isinstance(recognition_confidence, (int, float))
            or not math.isfinite(recognition_confidence)
            or not 0 <= recognition_confidence <= 1
        ):
            return {
                "status": "invalid_confidence_abstain",
                "person_id": person_id,
                "name": None,
                "relationship_to_wearer": None,
                "facts": [],
                "safe_message": SAFE_LOW_CONFIDENCE_MESSAGE,
            }
        if recognition_confidence < self.min_confidence:
            return {
                "status": "low_confidence_abstain",
                "person_id": person_id,
                "name": None,
                "relationship_to_wearer": None,
                "facts": [],
                "safe_message": SAFE_LOW_CONFIDENCE_MESSAGE,
            }

        # 3. Lookup exact person_id (strict isolation)
        person = self._persons_index.get(person_id)
        if not person:
            return {
                "status": "not_found_abstain",
                "person_id": person_id,
                "name": None,
                "relationship_to_wearer": None,
                "facts": [],
                "safe_message": SAFE_UNKNOWN_MESSAGE,
            }

        if not self._person_is_eligible(person):
            return {
                "status": "no_verified_facts", "person_id": person_id,
                "name": None, "relationship_to_wearer": None,
                "facts": [], "safe_message": SAFE_NO_APPROVED_MEMORY_MESSAGE,
            }

        # 4. Only allow eligible facts, then filter by tier/type.
        facts = [f for f in person.get("facts", []) if self._fact_is_eligible(f)]
        if tier is not None:
            facts = [f for f in facts if f.get("display_tier") == tier]
        if fact_type is not None:
            facts = [f for f in facts if f.get("fact_type") == fact_type]

        if not facts:
            return {
                "status": "no_verified_facts",
                "person_id": person["person_id"],
                "name": None,
                "relationship_to_wearer": None,
                "facts": [],
                "safe_message": SAFE_NO_APPROVED_MEMORY_MESSAGE,
            }

        return {
            "status": "success",
            "person_id": person["person_id"],
            "name": person["name"],
            "relationship_to_wearer": person["relationship_to_wearer"],
            "facts": facts,
            "safe_message": None,
        }

    def query_fact(
        self,
        person_id: Optional[str],
        question: str,
        recognition_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Retrieve the specific verified fact answering a user/assistant question
        for a given `person_id`. Returns `unanswerable` if the requested info
        (e.g., phone number, medical dose, address) is not stored in verified facts.
        """
        profile = self.get_person_profile(person_id, recognition_confidence=recognition_confidence)
        if profile["status"] != "success":
            return {
                "status": profile["status"],
                "person_id": person_id,
                "matched_fact": None,
                "safe_message": profile["safe_message"],
            }

        # Step 3: Cross-check explicit person references against the recognized ID.
        # This compares full names / explicit person IDs in questions against the
        # local database. It deliberately does NOT guess identity from first names.
        # Recognition and name resolution must be handled upstream; a mismatch
        # should never route a question to the currently recognized person's facts.
        if not isinstance(question, str) or not question.strip():
            return {
                "status": "unanswerable", "person_id": person_id,
                "matched_fact": None, "safe_message": SAFE_NO_MEMORY_MESSAGE,
            }
        q = question.lower().strip()
        mentioned_ids = self._mentioned_person_ids(q)
        if mentioned_ids and mentioned_ids != {person_id}:
            return {
                "status": "person_mismatch_abstain", "person_id": person_id,
                "matched_fact": None, "safe_message": SAFE_PERSON_MISMATCH_MESSAGE,
            }

        # Explicit check for unanswerable / out-of-scope private info (e.g. phone numbers, doses, passwords)
        unanswerable_keywords = [
            "phone", "number", "mobile", "address", "street number",
            "dose", "dosage", "medication", "medicine", "password", "pin", "bank", "email"
        ]
        if any(kw in q for kw in unanswerable_keywords):
            return {
                "status": "unanswerable",
                "person_id": person_id,
                "matched_fact": None,
                "safe_message": SAFE_NO_MEMORY_MESSAGE,
            }

        # Some attributes are absent from this schema and must never be guessed.
        # Check before broad intent rules (for example "who is" / "where is").
        unsupported_patterns = [
            r"\b(favou?rite|preferred)\s+(color|colour|food|movie|film|song|sport|number)\b",
            r"\b(birthday|birthdate|date of birth|age|how old|height|weight)\b",
            r"\b(where\s+(?:does|did)\s+.+\s+work|job title|occupation|salary)\b",
        ]
        if any(re.search(pattern, q) for pattern in unsupported_patterns):
            return {
                "status": "unanswerable",
                "person_id": person_id,
                "matched_fact": None,
                "safe_message": SAFE_NO_MEMORY_MESSAGE,
            }

        # Intent/keyword routing to the 6 verified fact_types
        target_type = self._classify_question_to_fact_type(q)
        facts = profile["facts"]

        if target_type:
            matched = [f for f in facts if f.get("fact_type") == target_type]
            if matched:
                return {
                    "status": "answerable",
                    "person_id": person_id,
                    "matched_fact": matched[0],
                    "safe_message": None,
                }

        # Safety-first fallback: do not guess from lexical overlap alone.
        # A name appearing in a question does not prove the requested fact exists.
        # Unsupported or ambiguous questions must abstain.

        return {
            "status": "unanswerable",
            "person_id": person_id,
            "matched_fact": None,
            "safe_message": SAFE_NO_MEMORY_MESSAGE,
        }

    def _mentioned_person_ids(self, question: str) -> set:
        """Find exact known full names and person IDs explicitly mentioned in text.

        Only treats full names as definitive person mentions, avoiding false
        mismatches from shared first names or common words. This is a conservative
        guard, not full NER, and is NOT authorization to query a different person.
        """
        mentioned = set()
        for pid, person in self._persons_index.items():
            name = person.get("name", "")
            if isinstance(name, str) and name.strip():
                parts = [re.escape(part) for part in name.casefold().split()]
                # Allow normal spaces between name components; require word edges.
                pattern = r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)"
                if re.search(pattern, question.casefold()):
                    mentioned.add(pid)
        # Explicit machine IDs can appear in debug / assistant prompts.
        for pid in re.findall(r"(?<![a-z0-9])person-\d{3}(?![a-z0-9])", question, re.I):
            mentioned.add(pid.lower())
        return mentioned

    @staticmethod
    def _classify_question_to_fact_type(q: str) -> Optional[str]:
        if any(p in q for p in ["who is", "who's", "relationship", "related"]):
            return "identity_relationship"
        if any(p in q for p in ["talk to", "talk about", "conversation", "ask him", "ask her", "ask about"]):
            return "conversation_starter"
        if any(p in q for p in ["where does", "where is", "which city", "live in", "lives in", "based"]):
            return "location_context"
        if any(p in q for p in ["how does", "help with", "help out", "help the", "help me"]):
            return "practical_support"
        if any(p in q for p in ["shared memory", "memory do i share", "together", "remember", "when we", "did we", "first meet", "last summer", "spring", "eid", "picnic", "gathering", "trip", "holiday"]):
            return "shared_memory"
        if ("outings" in q and "join" in q) or ("updates" in q and "share" in q):
            return "practical_support"
        if any(p in q for p in ["help", "bring", "brings", "share with", "shares", "organize", "organizes", "drop by", "arrange", "send", "keep", "keeps", "accompan"]):
            return "practical_support"
        if any(p in q for p in ["hobby", "hobbies", "interest", "enjoy", "enjoys", "like", "likes", "collect", "collection"]):
            return "personal_interest"
        return None


if __name__ == "__main__":
    # Quick standalone demonstration
    retriever = MemoryRetriever("memory_database.json")

    print("--- Test 1: Known Person (Tier 1 Immediate Cues) ---")
    print(json.dumps(retriever.get_person_profile("person-001", recognition_confidence=0.92, tier=1), indent=2))

    print("\n--- Test 2: Specific Question (Answerable) ---")
    print(json.dumps(retriever.query_fact("person-001", "Where does Omar live?", recognition_confidence=0.92), indent=2))

    print("\n--- Test 3: Specific Question (Unanswerable / Missing Info) ---")
    print(json.dumps(retriever.query_fact("person-001", "What is Omar Hassan's phone number?", recognition_confidence=0.92), indent=2))

    print("\n--- Test 4: Unknown Visitor (Safety Abstain) ---")
    print(json.dumps(retriever.get_person_profile("unknown-001", recognition_confidence=0.40), indent=2))
