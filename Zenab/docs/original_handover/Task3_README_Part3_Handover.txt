========================================================================
WANAS + ARC — Personal Memory Retrieval (Zinab's Task)
PART 3 HANDOVER: Spoken Memory Cues & Assistant Context (spoken_memory_cues.py)
========================================================================

1. WHAT IS INCLUDED IN THIS HANDOVER:
- spoken_memory_cues.py    : Python module formatting retrieved memories into concise, TTS-ready spoken cues and short HUD text.
- requirements_Part3.txt   : Dependencies list (Zero third-party libraries; standard Python 3.8+).
- README_Part3_Handover.txt: Integration guide for Toqa's Voice/Assistant router.
(Note: Requires `memory_database.json` from Part 1 and `memory_retrieval.py` from Part 2 in the same folder.)

2. WHO RECEIVES THIS PART & WHY:
- Toqa (Voice Interaction, Assistant Logic and Cross-Module Flow Lead):
  * Connect this module directly to your `identify_person` and `retrieve_memory` intents in the voice router.
  * Pass the returned `spoken_text` string straight into your Text-to-Speech (TTS) engine.
  * Use `hud_short_text` for brief visual overlay/display if needed.

3. HOW TO RUN IT:
Make sure `memory_database.json`, `memory_retrieval.py`, and `spoken_memory_cues.py` are in the same directory, then run:
    python spoken_memory_cues.py

4. HOW IT WORKS (FUNCTIONS & VOICE ROUTER CONTRACT):
Initialize once:
    from spoken_memory_cues import SpokenMemoryFormatter
    formatter = SpokenMemoryFormatter(db_path="memory_database.json", min_confidence=0.75)

Function A: `formatter.format_greeting_cue(person_id, recognition_confidence=1.0, include_practical_support=True)`
- Triggered when intent == "identify_person" (e.g., user asks "Who is this?" or camera detects someone).
- Returns concise Tier-1 cue (identity + practical support) with explicit names (no ambiguous pronouns).
- Example Output:
  {
    "intent": "identify_person",
    "status": "success",
    "person_id": "person-002",
    "name": "Salma Nabil",
    "relationship": "daughter",
    "spoken_text": "Salma Nabil is your daughter. Salma helps coordinate family visits.",
    "hud_short_text": "Salma Nabil (daughter)",
    "can_offer_more_context": true
  }

Function B: `formatter.format_question_cue(person_id, question, recognition_confidence=1.0)`
- Triggered when intent == "retrieve_memory" (e.g., user asks "Where does Salma live?").
- Returns single factual sentence in `spoken_text`, or safe abstain text if unanswerable.

Function C: `formatter.format_conversation_starter(person_id, recognition_confidence=1.0)`
- Triggered when user asks "What can I talk to them about?" or asks for extra context (`can_offer_more_context == True`).
- Returns Tier-2 `conversation_starter` prompt in `spoken_text`.

5. VOICE SAFETY RULES FOR TOQA'S ROUTER:
- If `status` is `"unknown_identity_abstain"` or `"low_confidence_abstain"`, read `spoken_text` ("Unrecognized visitor — no saved profile.") and DO NOT ask follow-up memory questions.
- If `status` is `"unanswerable"`, read `spoken_text` ("No verified information available for this request.") — never pass to an open-ended LLM to guess missing details.
========================================================================
