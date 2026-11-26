# Step 29 — First new task: Case Manifest (blank template)

1. Use `CASE_MANIFEST_TEMPLATE.csv` to join one case to one image, question, utterance, stable session, expected person/status, context, and acceptable fact IDs.
2. Fill only **documented** case values from Hanaa/Ahmed/Toqa; do not invent image IDs or copy oracle IDs into `face_predicted_person_id`.
3. `expected_person_id` is ground truth used **only for evaluation**. `face_decision` and `face_predicted_person_id` must be the real output for Runs B/D; `actual_stt_transcript` must be real STT for C/D.
4. An unregistered face has `expected_identity_status=unknown` and must not retrieve remembered facts. `unknown`, `no_face`, and `no_valid_face` must not reuse previous-session person identity.
5. Keep the old 50-image experiment tagged `validation`; a new independent held-out set must be genuinely unseen. Leave missing output fields BLANK, mark `upstream_data_origin=pending` and `case_review_status=pending`.
6. Only use authorized data and store real predictions/biometric references outside public Git (`private_data/` is ignored).

This is a **template, not an actual experiment or completed A/B/C/D runner**. Next step: validate a filled case manifest and add frozen-policy scoring with upstream outputs.
