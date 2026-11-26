"""
WANAS + ARC — Spoken Memory Cues & Assistant Context Formatter (Part 3)
Owner: Zinab (with Toqa for Voice/Assistant Flow & Ahmed for API Integration)
Version: 2.5 — mode-aware and fail-closed spoken cues

This module wraps `MemoryRetriever` (Part 2) and formats retrieved memory records
into concise, natural, TTS-ready spoken cues and HUD display strings for Toqa's
Voice/Assistant router.
"""

import json
from typing import Any, Dict, Optional
from memory_retrieval import (
    MemoryRetriever,
    SAFE_UNKNOWN_MESSAGE,
    SAFE_LOW_CONFIDENCE_MESSAGE,
    SAFE_NO_MEMORY_MESSAGE,
)


class SpokenMemoryFormatter:
    def __init__(self, db_path: str = "memory_database.json", min_confidence: float = 0.75, data_mode: str = "demo"):
        # Existing synthetic test suite uses demo; real assistant MUST set production.
        self.retriever = MemoryRetriever(db_path=db_path, min_confidence=min_confidence, data_mode=data_mode)

    def format_greeting_cue(
        self,
        person_id: Optional[str],
        recognition_confidence: Optional[float] = None,
        include_practical_support: bool = True,
    ) -> Dict[str, Any]:
        """
        Generates a concise Tier-1 spoken cue when a visitor is detected.
        Designed for immediate TTS readout without overwhelming the wearer.
        """
        profile = self.retriever.get_person_profile(
            person_id=person_id,
            recognition_confidence=recognition_confidence,
            tier=1,
        )

        if profile["status"] != "success":
            return {
                "intent": "identify_person",
                "status": profile["status"],
                "person_id": person_id,
                "spoken_text": profile["safe_message"],
                "hud_short_text": profile["safe_message"],
                "can_offer_more_context": False,
            }

        facts = profile["facts"]
        identity_fact = next(
            (f["memory_fact"] for f in facts if f.get("fact_type") == "identity_relationship"),
            None,
        )
        # A person's name/relationship fields aren't sufficient to invent a
        # spoken identity assertion when that memory fact is unapproved.
        if not identity_fact:
            return {
                "intent": "identify_person", "status": "no_verified_identity_fact",
                "person_id": person_id, "spoken_text": SAFE_NO_MEMORY_MESSAGE,
                "hud_short_text": SAFE_NO_MEMORY_MESSAGE,
                "can_offer_more_context": False,
            }
        support_fact = next(
            (f["memory_fact"] for f in facts if f.get("fact_type") == "practical_support"),
            None,
        )

        if include_practical_support and support_fact:
            spoken = f"{identity_fact} {support_fact}"
        else:
            spoken = identity_fact

        return {
            "intent": "identify_person",
            "status": "success",
            "person_id": profile["person_id"],
            "name": profile["name"],
            "relationship": profile["relationship_to_wearer"],
            "spoken_text": spoken,
            "hud_short_text": f"{profile['name']} ({profile['relationship_to_wearer']})",
            "can_offer_more_context": True,
        }

    def format_question_cue(
        self,
        person_id: Optional[str],
        question: str,
        recognition_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Answers a specific voice intent ("retrieve_memory") about the current visitor
        using concise, standalone TTS phrasing (explicit names, no ambiguous pronouns).
        """
        result = self.retriever.query_fact(
            person_id=person_id,
            question=question,
            recognition_confidence=recognition_confidence,
        )

        if result["status"] != "answerable" or not result["matched_fact"]:
            return {
                "intent": "retrieve_memory",
                "status": result["status"],
                "person_id": person_id,
                "question": question,
                "spoken_text": result["safe_message"],
                "fact_id": None,
                "fact_type": None,
            }

        fact = result["matched_fact"]
        return {
            "intent": "retrieve_memory",
            "status": "answerable",
            "person_id": person_id,
            "question": question,
            "spoken_text": fact["memory_fact"],
            "fact_id": fact["fact_id"],
            "fact_type": fact["fact_type"],
        }

    def format_conversation_starter(
        self,
        person_id: Optional[str],
        recognition_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves a Tier-2 conversation starter or shared memory cue when the wearer
        asks "What can I talk to them about?" or requests extra context.
        """
        profile = self.retriever.get_person_profile(
            person_id=person_id,
            recognition_confidence=recognition_confidence,
            fact_type="conversation_starter",
        )

        if profile["status"] != "success":
            return {
                "intent": "conversation_starter",
                "status": profile["status"],
                "person_id": person_id,
                "spoken_text": profile["safe_message"],
            }

        starter = profile["facts"][0]["memory_fact"]
        return {
            "intent": "conversation_starter",
            "status": "success",
            "person_id": profile["person_id"],
            "name": profile["name"],
            "spoken_text": starter,
        }


if __name__ == "__main__":
    formatter = SpokenMemoryFormatter("memory_database.json")

    print("--- 1. Immediate Greeting Cue (Tier 1 for Toqa's TTS) ---")
    print(json.dumps(formatter.format_greeting_cue("person-002", recognition_confidence=0.95), indent=2))

    print("\n--- 2. Voice Question Cue (Answerable) ---")
    print(json.dumps(formatter.format_question_cue("person-002", "Which city does Salma live in?", recognition_confidence=0.95), indent=2))

    print("\n--- 3. Conversation Starter Cue (Tier 2) ---")
    print(json.dumps(formatter.format_conversation_starter("person-002", recognition_confidence=0.95), indent=2))

    print("\n--- 4. Unanswerable Voice Question (Safe Abstain) ---")
    print(json.dumps(formatter.format_question_cue("person-002", "What is Salma's phone number?", recognition_confidence=0.95), indent=2))

    print("\n--- 5. Unknown Visitor Voice Cue (Safe Abstain) ---")
    print(json.dumps(formatter.format_greeting_cue("unknown-005", recognition_confidence=0.30), indent=2))
