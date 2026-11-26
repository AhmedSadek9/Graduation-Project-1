========================================================================
WANAS + ARC — Personal Memory Retrieval (Zinab's Task)
PART 2 HANDOVER: Core Memory Retrieval Prototype (memory_retrieval.py)
========================================================================

1. WHAT IS INCLUDED IN THIS HANDOVER:
- memory_retrieval.py      : Standalone Python module implementing `MemoryRetriever`.
- requirements_Part2.txt   : Dependencies list (Zero third-party libraries; standard Python 3.8+).
- README_Part2_Handover.txt: Setup, usage, input/output contracts, and safety rules.
(Note: Requires `memory_database.json` from Part 1 in the same working directory.)

2. WHO RECEIVES THIS PART & WHY:
- Ahmed (Backend/API & Face Recognition Lead):
  * Import `MemoryRetriever` directly into the local Python workflow or future FastAPI/Flask endpoint.
  * Pass the Face Recognition output (`person_id`, `recognition_confidence`) straight into `get_person_profile()` or `query_fact()`.

3. HOW TO RUN IT:
Make sure `memory_database.json` is in the same folder, then run:
    python memory_retrieval.py

4. HOW IT WORKS (FUNCTIONS & INPUT/OUTPUT CONTRACTS):
Initialize once:
    from memory_retrieval import MemoryRetriever
    retriever = MemoryRetriever(db_path="memory_database.json", min_confidence=0.75)

Function A: `retriever.get_person_profile(person_id, recognition_confidence=1.0, tier=None, fact_type=None)`
- Use when Face Recognition detects someone in front of the user.
- Pass `tier=1` to get immediate short cues (identity + practical support) for HUD/TTS.
- Pass `tier=None` to get all 6 verified facts.
- Returns JSON dict:
  {
    "status": "success" | "unknown_identity_abstain" | "low_confidence_abstain" | "not_found_abstain" | "no_verified_facts",
    "person_id": "person-001",
    "name": "Omar Hassan",
    "relationship_to_wearer": "son",
    "facts": [ ... ],
    "safe_message": null (or safe abstain message if status != "success")
  }

Function B: `retriever.query_fact(person_id, question, recognition_confidence=1.0)`
- Use when the user asks a specific question about the recognized person (e.g., "Where does Omar live?").
- Returns JSON dict:
  {
    "status": "answerable" | "unanswerable" | "unknown_identity_abstain" | "low_confidence_abstain",
    "person_id": "person-001",
    "matched_fact": { ... } or null,
    "safe_message": null (or "No verified information available for this request.")
  }

5. BUILT-IN SAFETY & ISOLATION GUARANTEES:
- Strict Person Isolation: Facts are indexed strictly by `person_id`. One person's facts can NEVER leak into another person's query.
- Confidence Gating: Rejects retrieval if `recognition_confidence < 0.75`.
- Unknown Abstention: Any `person_id` starting with `unknown` or missing from `memory_database.json` immediately returns `"Unrecognized visitor — no saved profile."` with an empty `facts` list.
- Out-of-Scope Rejection: Questions asking for unstored/sensitive data (phone numbers, addresses, medication dosages) return `"status": "unanswerable"` without guessing.
========================================================================
