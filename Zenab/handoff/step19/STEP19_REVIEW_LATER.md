# Pending reciprocal handoff — do not claim received/sent

## Hanaa (هناء)
- Confirm 20 stable person IDs against actual face-enrollment manifest (ID, person, model dataset revision); identify synthetic IDs that have no real counterpart.
- Independently review all 40 AI-assisted draft questions, acceptable fact IDs, unanswerable cases and ambiguity; record reviewer name, actual approval and comments, rather than automatically marking `approved`.
- Check similar names, stale/conflicting facts, paraphrase grouping and whether proposed blind questions overlap training/evaluation examples.
- Tell Zinab what mapping fields and error labels Hanaa needs for integration checks.

## Ahmed (أحمد)
- Supply actual face predictions with session/image identifier, predicted person ID, status, raw score, score type, threshold, model version and timestamp.
- Clarify score calibration and whether current 0.75 recognition threshold is applicable; supply unknown/wrong-person/low-confidence examples.
- Agree on backend endpoint, identity access control, request logging, data storage and how upstream IDs are joined to question IDs.
- Receive from Zinab: `memory_integration.py` callable contract, allowed statuses and sample responses (not a deployed API).

## Toqa (تقى)
- Supply question IDs, reference transcripts, actual STT transcripts/errors, session/audio IDs, STT model/version and representative clear/unclear utterances.
- Confirm assistant intent routing and refusal wording; ensure TTS never speaks a personal fact when `can_speak_personal_memory=false`.
- Receive from Zinab: Step 19 five example responses, expected fact IDs and abstention status/reasons.
- Send back actual integration traces with both upstream decisions and downstream response.

## Zinab (زينب)
- Independent benchmark review remains pending; 120 synthetic facts intentionally retained.
- E5 is still an exploratory selected candidate; offline E5 ranking is not yet connected to `FaceMemoryBridge`.
- Freeze and verify executable configuration, collect genuinely untouched test results, then quantify loss with real face/STT outputs when available.
