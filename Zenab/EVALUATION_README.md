# WANAS — Zinab Retrieval Benchmark Extension (20 November 2026)

This is an **additive experiment bundle**, preserving the previously passing Task1–Task4 and integration bridge. All records are **fictional synthetic test data**, **not caregiver-verified patient memory**.

## What's completed now
- `benchmark/fact_manifest.csv` and `.json`: 120 unchanged source facts with stable IDs, source, fictional verification, date, and origin. The original workbook was inspected; these 120 facts map to its `Fictional Memory Facts` worksheet.
- `benchmark/question_manifest.csv` and `.json`: a curated **140-question subset** from the existing 160 (100 answerable, 20 unanswerable, 20 unknown identity), stable IDs, acceptable fact-ID lists, origin IDs, question-group IDs, and split IDs. Original 160 tests are unchanged. This is a structured **synthetic** benchmark, not a newly independently authored corpus.
- Train / validation / test splits grouped by person ID for known people. No person crosses splits. Unknown IDs are split by ID. This is intentionally stricter for generalization and prohibits paraphrase leakage across splits.
- `experiments/bm25_benchmark.py`: **actual** BM25 ranking after exact-person and demo-verification filtering, candidate/ranking traces, validation-only threshold selection, and held-out metrics.
- `experiments/semantic_comparison.py`: executable same-ID comparison of MiniLM / multilingual E5 **when sentence-transformers and model weights are available**. E5 prefixes: `query: ` for queries and `passage: ` for facts. No semantic results are claimed yet.
- `handoff/`: ID reconciliation sheet for Hanaa, reproducible Toqa demo examples, and upstream predictions template.

## Run
```powershell
python run_all_tests.py
python experiments/bm25_benchmark.py
python -m pip install sentence-transformers
python experiments/semantic_comparison.py
```

Model downloads may require Internet. CPU latency is hardware-dependent. Do not claim that pretrained models were compared unless result JSON files are produced.

## Provenance and caveats
- Source workbook: `WANAS_Fictional_Relationships_V2 (1)(1).xlsx`, user-provided, `Fictional Memory Facts`, `Known Persons`, `Retrieval Test Queries` sheets.
- Existing JSON source: `Task1/memory_database.json` and `Task1/retrieval_test_queries.json`.
- **120 facts only**; requirement for 150–250 remains **unmet**. Do not invent 30–130 new “verified” memories from this source. Need additional fictional facts reviewed by team, or real records only through consent and caregiver verification.
- Only 20 synthetic persons, and **Hanaa ID alignment remains unverified** without her identity manifest.
- Source lacks explicitly versioned *outdated* memories / conflicting temporal contexts. Add dated examples only with reviewed ground truth.
- 140 queries are selected from 160 existing, not evidence of 140 independently generated originals. Similar questions must share `original_question_id` / group.
- Train/validation/test selection is deterministic; generated metrics are synthetic. Never call them actual upstream reliability.
- BM25 alone is not evidence of model-selection win; need actual semantic checkpoints and measurements.
- The upstream error script refuses to score empty/simulated inputs. End-to-end reliability loss remains pending until actual face predictions and paired STT transcripts arrive.
- A high recognition similarity is not automatically a calibrated probability; Ahmad/Hanaa must supply the recognition contract and actual status/threshold.

## Real handoffs to request
**Hanaa (from Nov 20):** exact stable face person IDs, enrollment map, version, source of each ID, and mismatches with this 20-person fictional benchmark.
**Toqa (Nov 24–26):** paired reference and real STT transcripts, stable question IDs, timestamps, errors and STT version; output examples for verified answer / missing fact / unknown person.
**Ahmed:** actual face-model predictions per tested image, score type, threshold, recognized/unknown decision, model version, and join keys; no hardcoded confidence 0.95.
**Zinab:** add reviewed dated facts; create difficult near-name and stale-context questions, run models, inspect music-query filter/rank/threshold checkpoint, select frozen model/threshold using validation and report independent held-out person/question counts.

## Next experiments (not yet done)
- Three-model score report, error analysis and music query audit.
- Lexical + semantic fusion and reranking (only after three-model comparison).
- Question-only vs + current identity vs + last-request context; stale/unknown face conditions.
- Precision/coverage and consequential false answer examples, frozen held-out scoring.
- Actual upstream reliability drop; absence of actual predictions **must not** be labeled zero loss.
