# WANAS — Zinab Step 20 Final Audit and Handover Status

**Audit basis:** the uploaded `WANAS_Zinab_Evaluation_Updated (2).zip` (local copy, November 2026). This document describes an **internal synthetic-data prototype**, not a deployed or independently validated patient-facing system.

## Verified in this audit

- Original regression runner: **8/8 suites passed**. Raw execution retained in `reports/STEP20_TEST_EXECUTION.log`.
- Step 19 CSV-shape check: **5/5 records checked**. This is NOT live Toqa integration.
- Frozen-config SHA-256 checksums: **5/5 match** across manifests, E5 report, and guard/context scripts.
- Python file syntax: **24/24 parsed**.
- Existing benchmark: **120 synthetic facts**, **140 questions** (78 train, 26 validation, 36 previously inspected exploratory test).
- Step 16 draft: **40 proposed questions**, review approvals = **['pending']**; no independent blind score is claimed.
- E5 provisional config: `intfloat/multilingual-e5-small`, question-only, cosine ranking of normalized embeddings, `query: ` / `passage: ` prefixes, acceptance threshold **0.8824205**, fact-type guard described in its frozen config. This is **not** the active implementation in `memory_integration.py`.
- Music case: the diagnostic used a constructed example; **original `not_found` request and exact repair checkpoint remain unverified**.

## Important distinction

`Task2` and `memory_integration.py` still use the original rule-based memory retrieval. The E5 experimental scripts produce rankings and evaluation reports **offline**. There is **no verified production E5 adapter, live service, real STT handoff, actual face-module integration, or independent end-to-end reliability-loss measurement** in this ZIP. Synthetic-only experiments cannot establish clinical/patient safety.

## Results already recorded (not rerun by this Step 20 audit)

| Model | Accepted correct, exploratory test | Recall@3 | MRR@5 | Notes |
|---|---:|---:|---:|---|
| BM25 | 12/20 | 0.90 | 0.7975 | 4 unsupported answers reported |
| MiniLM | 2/20 | 0.75 | 0.71 | 1 wrong fact accepted; 16/16 unanswerable abstentions |
| E5-small | 3/20 | 1.00 | 0.95 | 0 wrong accepted; low coverage |
| Hybrid BM25+E5 | 3/20 | 1.00 | 0.95 | No observed gain over E5 |

Fact-type guard and context results are **retrospective / exploratory**, since previous test errors were inspected during development. See `reports/STEP17_RETRIEVAL_EVALUATION_REPORT.md` and `reports/STEP18_FAILURE_AND_MUSIC_AUDIT.md`.

## Pending decisions and work Zinab can do locally

1. Keep the 120-fact demo dataset; document deviation from the suggested 150–250-fact target rather than marking it complete.
2. Review the 40 AI-assisted questions as a **draft** and do not set independent-review approval without actual review. Create an untouched test later; consider leakage where the same fact/question family appears across splits.
3. Verify that every ranking trace contains candidate eligibility, verification state, original score, rank, chosen fact, and refusal reason; add missing records without rewriting historical outputs.
4. If E5 should power the eventual integration, build and test a separate E5-backed adapter with explicit `data_mode`, person restrictions, approved-fact filtering, score validation, and fail-closed behavior; leave existing stable adapter untouched until verified.
5. Benchmark incremental latency for ranking/guard/hybrid on the same hardware, separately from model download and cold start.

## Dependencies to request later — reciprocal handoffs

**هناء — Hanaa**
- Review the 40 draft questions against facts, acceptable IDs, ambiguity, unanswerable labels, similar names, stale facts, conflicting context, and paraphrase leakage; approve only individually reviewed rows.
- Provide stable face-enrollment person-ID manifest / ground-truth mapping and identify any demo IDs without real counterparts.
- **From Zinab:** `benchmark/`, `experiments/step16/blind_question_authoring_template.csv`, `Task1/memory_database.json`, and `handoff/step19/STEP19_MEMORY_API_CONTRACT.md`.

**أحمد — Ahmed**
- Provide real face predictions with per-session/image identifiers, `person_id`, raw similarity and score type, model version, recognition status and timestamp; agree on identity threshold/calibration and backend contract.
- **From Zinab:** memory response/status contract, sample I/O, person-ID constraints, mock bridge, rejection/error schema, and planned E5 adapter specification.

**تقى — Toqa**
- Provide reference/STT transcript pairs with question ID, audio/session ID, ASR version and real error examples; clarify last-request and assistant intent fields.
- **From Zinab:** `handoff/step19/TOQA_EXAMPLES_STEP19.csv`, example request/response JSON, `can_speak_personal_memory` gate and abstention reasons.

## Final evaluation to run when handoffs arrive

- Freeze executable code, model revision, hashes, accepted fact policy, and threshold **before** reviewing new labels/results.
- Run independent question set first with oracle person IDs/transcripts; repeat with actual face and STT outputs on matching IDs.
- Report wrong-person retrieval, unsupported responses, valid abstentions, answerable recall/coverage, precision, per-scenario counts and absolute reliability difference between oracle and upstream-real conditions.

## Reproducibility

```powershell
python -m pip install -r requirements.txt
python run_all_tests.py
python handoff/step19/STEP19_CHECK_RESULTS.py
```

For optional semantic experiments, also install a compatible `sentence-transformers`/PyTorch stack and use already cached checkpoints; these heavy dependencies are deliberately *not* required to run the core module. `experiments/step16_validate_blind_questions.py` will **correctly fail** while independent approvals remain pending. 

## ZIP clean-up

Removed only the redundant top-level `WANAS_Step18_Failure_Documentation/` copy, retaining the canonical file under `reports/`; removed transient `__pycache__`/`.pyc`. No source evaluation logic or benchmark facts were altered during this clean-up.
