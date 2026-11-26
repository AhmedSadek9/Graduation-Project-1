# WANAS Memory Retrieval — Current status (25 November 2026)

**Scope:** Synthetic-data prototype and exploratory retrieval evaluation. Not a clinical or production system. Do not confuse test-suite PASS with real face/STT integration.

## Completed on synthetic fixtures

- 20 fictional person IDs and 120 fictional facts with provenance/verification-in-scenario metadata. This is **not caregiver-confirmed real-world verification**.
- Benchmark manifest contains **140 questions**: 78 train, 26 validation, 36 test. The original Step 1 suite separately has **160** queries; these are different evaluation collections, not additional independent observations.
- BM25, all-MiniLM-L6-v2 and multilingual-e5-small were actually compared; saved reports/predictions are under `experiments/results/`. Hybrid retrieval and context variants were explored.
- E5 on the previously inspected synthetic test: raw rank-one correct **18/20 answerable**, Recall@3 **1.0**, MRR@5 **0.95**; **3/20** accepted correct, **0** wrong accepted, **16/16** correct abstentions on unanswerable/unknown. The `top1: 0.15` field of semantic comparison refers to **accepted correct / answerable**, not raw ranking correctness.
- Frozen exploratory E5 threshold **0.8824205**. Candidate **0.80** was studied on prior validation examples only; **not independently approved**, do not deploy.
- Eight original regression suites pass on synthetic fixtures. Step 21 demo logic tests pass 7/7; Step 25 proposal rules pass 9/9. Step 28 document/file readiness check passed locally; **this does not mean real integration happened**.
- A Python `FaceMemoryBridge` exists in `memory_integration.py`; it is not an HTTP API. The original bridge is rule-based. The E5 demo adapter is separate and synthetic-demo only.
- Step 27 has **40 AI-assisted draft questions** (20 answerable and 20 unanswerable) awaiting human review. Reviewing them does not make them an untouched independent held-out set.

## Open issues / dependencies

1. **Hanaa + Ahmed:** verify approved person/relationship mapping, missing/duplicate IDs, and reconcile fictional IDs with real face-enrollment identities; no guessed ID mapping.
2. **Ahmed:** supply actual face `decision` and recognition status/score semantics with joinable `image_id`/`case_id` and reference identity, including unknown/no_face/no_valid_face cases.
3. **Toqa:** supply actual paired reference/STT transcripts with `question_id` and `utterance_id`; no simulated outputs labeled actual.
4. **Zinab:** create matched case manifest; implement and test A/B/C/D evaluation with frozen retrieval policy; report separate memory-module safety and user-visible correct-outcome metrics and paired A–D loss.
5. Independent held-out experiment needs new unseen approved cases. The existing 50-image experiment is **validation** per the updated task.
6. Historical music `not_found` question was not recovered: synthetic substitute did not reproduce the original bug. No proven historical repair.
7. Hybrid reranking overhead and any real end-to-end latency need explicit measurement; current saved numbers are not live upstream timings.

## Updated task scope

The current requirement uses the **existing 120 fictional facts**; 150–250 was an original collection target, **not a new blocker**. Hanaa's poor-quality real-face data collection is a team dependency: 30 training, 10 validation, 30 independent test images; do not fabricate captures or expose actual face images in a public repo.

`experiments/evaluate_upstream_actual.py` remains a **placeholder** and does not produce A/B/C/D metrics. Do not run it as if completed or report reliability loss until matched actual upstream outputs exist.
