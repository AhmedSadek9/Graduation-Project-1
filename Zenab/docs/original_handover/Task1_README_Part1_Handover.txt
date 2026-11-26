========================================================================
WANAS + ARC — Personal Memory Retrieval (Zinab's Task)
PART 1 HANDOVER: Memory Schema & Sample JSON Database
========================================================================

1. WHAT IS INCLUDED IN THIS HANDOVER:
- memory_schema.json         : Official JSON Schema contract defining fields, types, and constraints.
- memory_database.json       : Full structured JSON database converted from WANAS_Fictional_Relationships_V2 (1).xlsx (20 known persons: person-001 to person-020, 120 verified fictional facts).
- unknown_test_cases.json    : 30 unknown identity test cases (unknown-001 to unknown-030) for safety/abstain testing.
- retrieval_test_queries.json: 160 evaluation queries (100 answerable, 20 unanswerable, 40 unknown_identity_abstain).
- requirements.txt           : Python dependencies list.

2. WHO RECEIVES THIS PART & WHY:
- Ahmed (Backend/API & Face Recognition Lead):
  * Use `memory_schema.json` to lock the shared Backend/Database data contract.
  * Align Face Recognition output `person_id` format (`person-001` .. `person-020`, and `unknown-XXX` / `unknown_identity`) with `memory_database.json`.
- Hanaa (Face Recognition Evaluation & Data Quality Lead):
  * Pair with Zinab to verify that the `person_id` values in your face validation set match `person-001` through `person-020` and unknown rejection IDs.

3. HOW THE DATA STRUCTURE WORKS:
- Each person record is keyed by `person_id` (e.g., "person-001") and contains:
  * `name`: Explicit full name (e.g., "Omar Hassan")
  * `relationship_to_wearer`: Role (e.g., "son", "daughter", "neighbor")
  * `facts`: Array of 6 facts per person (`F01` to `F06`), categorized by `fact_type`:
    - `identity_relationship` (Tier 1)
    - `practical_support` (Tier 1)
    - `personal_interest` (Tier 2)
    - `location_context` (Tier 2)
    - `shared_memory` (Tier 2)
    - `conversation_starter` (Tier 2)
- Display Tiers:
  * `display_tier: 1` -> Immediate short cues (identity + practical support) for quick HUD/TTS readout.
  * `display_tier: 2` -> Optional deeper context, shared memories, and conversation starters.

4. SAFETY & GOVERNANCE RULES (MANDATORY):
- Check Face Recognition confidence and identity BEFORE querying `memory_database.json`.
- If `person_id` starts with `unknown` or is not found in `persons`, return safe abstain response ("Unrecognized visitor — no saved profile.") and NEVER invent facts.
- All records are synthetic test data (`draft_for_team_review`) for development only.
========================================================================
