# Team handoffs for Nov 30 evaluation (25 Nov 2026)

## Hanaa — data review + face-quality support
- Confirm master stable person IDs and relationship mapping against memory benchmark; return discrepancy list (missing, duplicate, mismatched) **or signed-off PASS**. Fictional IDs are not assumed to match face IDs.
- Review Step27 40 AI-assisted question drafts; distinguish reviewed drafts from a new untouched held-out set.
- Coordinate authentic poor-quality face captures per updated requirement: training 15 known+15 unknown, validation 5+5, independent test 15+15, with permissions and provenance. The existing 50-image experiment stays labeled validation.

## Ahmed — actual face decisions + backend
- Share joinable `image_id`, `case_id`, `decision` (unchanged raw status), predicted person ID, expected identity for scoring, score with definition/threshold, model/version. Include known, wrong-person, unknown, no_face, no_valid_face cases.
- Agree a single authorized ID map, session/context reset semantics, and how backend calls the in-process Python memory bridge. Do not infer from an example score of 0.95.
- Share results securely; do not commit real biometric images to GitHub.

## تقى (Toqa) — actual speech recognition
- For matching questions/cases, provide `utterance_id`, `question_id`, `case_id`/`session_id`, reference transcript, actual STT transcript, model/version, language, and missing-pair status.
- Confirm the TTS can speak a personal fact **only** when `can_speak_personal_memory` is true; never turn a refusal into a made-up answer.

## Zinab — evaluation owner
- Keep current 120 fictional facts and the baseline E5 threshold 0.8824205 untouched; 0.80 is an exploratory candidate only.
- Build a case manifest with paired image/question/utterance/context and acceptable fact IDs. Run A and B on validation once real face decisions arrive; C/D only after paired actual STT arrives. Do not score missing pairs as successful.
- On new unseen, frozen-condition pairs report supported answers, wrong-person answers, unsupported answers, correct abstentions, unnecessary abstentions, and A–D correct-outcome rate difference in percentage points, with numerator and denominator.
