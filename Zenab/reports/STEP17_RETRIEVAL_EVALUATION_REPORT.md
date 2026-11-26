# WANAS — Zinab | Step 17 Retrieval Evaluation Report

**Status:** Exploratory evaluation completed; independent final evaluation and actual upstream integration pending.  
**Data origin:** Entire benchmark is synthetic demo data, **not** caregiver-verified patient memory.  
**Report prepared:** 21 November 2026.  
**Evidence boundary:** BM25 and music diagnostics were inspected from the uploaded evaluation ZIP. MiniLM, E5, hybrid, context, and fact-type-guard figures below were transcribed from the terminal outputs supplied in this conversation, not independently re-executed in this report environment.

## 1. Dataset and splits

- 20 unique synthetic persons; 120 memory facts (6 per person).
- `benchmark/question_manifest.csv`: 140 questions; train 78, validation 26, test 36.
- Test split: 20 answerable and 16 unanswerable/unknown-identity cases.
- All fact and question IDs, source fields, verification statuses, date fields, and origin IDs should be retained in any downstream predictions.
- The separate Step 16 40-question authoring draft is **not** an approved independent test: independent review pending; do not use it to select thresholds or report final unbiased reliability.
- Hanaa must confirm person-ID compatibility with her face-recognition ground truth, and review the question/answer evidence and split leakage risk.

## 2. Retrieval comparison — test set (exploratory)

| Method | Answerable correct accepted /20 | Recall@3 | MRR@5 | Correct abstentions /16 | Unsupported answers /16 | Wrong accepted facts on answerable | Median latency, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 12/20 (60%) | 90% | 0.7975 | 12/16 | 4 | 3 | 0.0450 |
| all-MiniLM-L6-v2 | 2/20 (10%) | 75% | 0.7100 | 16/16 | 0 | 1 | 73.59 |
| multilingual-e5-small | 3/20 (15%) | 100% | 0.9500 | 16/16 | 0 | 0 | 23.33 |
| Hybrid BM25 + E5 (validation-selected RRF) | 3/20 (15%) | 100% | 0.9500 | 16/16 | 0 | 0 | not established from supplied hybrid output |

**Metric distinction:** `correct accepted /20` is not raw rank-1 accuracy. From the hybrid/context outputs, E5 ranked an acceptable fact first for **18/20** test-answerable questions, but accepted only **3/20** under the frozen score cutoff. These are separate metrics.

**Latency caveat:** BM25 and semantic-model timings cover different inference workloads and may vary by machine, warm-up, cache, and batching. The one-shot figures are observational, not robust hardware-normalized comparisons. The hybrid output supplied did not quantify additional latency.

### Validation-selected thresholds

- BM25: `0.657456` (from uploaded BM25 report).
- MiniLM: `0.7890985` (user terminal report).
- E5: `0.8824205` (user terminal report; frozen in Step 16 config).
- Hybrid: E5 weight `0.9`, acceptance threshold `0.882421` (user terminal report).

**Validation results for E5:** 4/20 correct accepted; 6/6 unanswerable correctly abstained; 0 unsupported answers.  
**Test results for E5:** 3/20 correct accepted; 16/16 correctly abstained; 0 unsupported answers.  
These thresholds were chosen on prior validation, but the test data and errors have since been repeatedly inspected. Treat subsequent improvements measured on this test set as exploratory.

## 3. Selected provisional retrieval configuration

`configuration_id = zinab-e5-question-only-typeguard-exploratory-v1`

- Checkpoint: `intfloat/multilingual-e5-small`.
- E5 prefixes: `query: ` and `passage: `.
- Candidate filter: **current person ID** + eligible verification status **before** scoring/ranking.
- Ranking: cosine similarity of normalized E5 embeddings.
- Context: `question_only`; do not append synthetic last-request text to the query for the provisional baseline.
- Acceptance: highest-ranked candidate with a permitted question-intent `fact_type` and its **own** score >= `0.8824205`; otherwise abstain.
- Source of guard logic: `experiments/fact_type_guard_step15.py`.
- Modes: `synthetic_test_data` is demo-only. Real/production memory requires caregiver verification and approval. No real production approval is claimed.
- This is a **configuration record**, not proof that the entire frozen pipeline has already been packaged or verified against fresh data. Recheck code versions/hashes and run a fresh independent evaluation before production use.

## 4. Hybrid comparison

Step 12 used validation to pick RRF-style fusion with `e5_weight=0.9`; on the already-inspected test, hybrid showed **no gain** over E5-only: 18/20 raw rank-1; Recall@3 1.00; MRR@5 0.95; 3/20 correct accepted; 0 wrong and 0 unsupported. Added-time and a separate candidate reranker were not evaluated. Keeping E5-only is simpler pending an independent comparison.

## 5. Context experiment (synthetic, prior test inspected)

| Context variant | Test correct accepted /20 | Wrong accepted facts | Unsupported answers | Correct abstentions /16 |
|---|---:|---:|---:|---:|
| Question only | 3 | 0 | 0 | 16 |
| Question + identity | 4 | 0 | 0 | 16 |
| Question + identity + synthetic last request | 14 | 1 | 1 | 15 |
| Stale last request guarded | 4 | 0 | 0 | 16 |
| Unknown face guarded | 0 | 0 | 0 | 16 |

The large apparent gain from last-request context uses constructed previous-request text, **not real conversational history**; cannot be extrapolated to STT or user dialogue.

**Step 15 fact-type guard, retrospective error correction:** In `question_identity_last_request`, validation went 16→18 correct accepted, 2→0 wrong facts, 2→0 unsupported; test went 14→15 correct accepted, 1→0 wrong, 1→0 unsupported. This guard was developed after examining known errors; those improvements are **not independent test performance**.

## 6. Consequential failure and investigation

**Failure example:** `Q-person-015-03`, “What is Fadi Younes’s phone number?” has no acceptable fact ID; the prior context experiment nevertheless selected `person-015-F01`, an identity/relationship fact. The type guard blocked this. Another failure was `Q-person-017-05` (shared meal), where a conversation-starter fact was initially returned instead of `person-017-F04` (shared memory), then reselected by the guard. Both are retrospective, exploratory cases.

**Music-query investigation:** Uploaded `music_diagnostic.json` uses the **constructed** query “Does Nada record music clips for the family?” for `person-020`, target `person-020-F05`. It survived person/verification filtering, ranked #1 with BM25 score 6.083245 above threshold 0.657456, and was accepted. But the original reported `not_found` question/trace was **not provided**; therefore its actual root cause and repair **remain unverified**. The original query and original checkpoint output must be retained before this item is marked fully complete.

## 7. Handoff & evaluation blockers

| Person | What Zinab needs | What Zinab will provide | Status |
|---|---|---|---|
| Hanaa | Stable person-ID mapping/face ground truth; genuine review of 40 Step 16 questions, answers, paraphrase groups, and leakage; approval evidence | `benchmark/fact_manifest.csv`, `benchmark/question_manifest.csv`, Step 16 question draft, intended IDs and review instructions | Pending, not yet requested |
| Ahmed | Actual face predictions: query/session ID, predicted person ID, score, decision/status, reference person ID, model/version | Memory request/response contract, identity-gating policy, abstention reason codes, reproducible oracle baseline | Pending actual outputs |
| تقى | Paired ground-truth/reference transcripts and real STT predictions linked to stable question IDs, timestamps/context | Verified-answer demo, missing-fact abstention, unresolved-person abstention, selected fact ID, response, abstention reason | Draft only; handoff not sent |

**End-to-end reliability loss:** Cannot be calculated from mock face/STT results. Evaluate correct upstream IDs/transcripts (oracle) and then actual upstream outputs on the same separately held-out questions. Report wrong-person retrieval, wrong fact, unsupported answer, correct abstention, and loss (percentage-point difference) separately.

## 8. Acceptance criteria and next experiment

1. Have Hanaa genuinely review the Step 16 40-question draft and inspect for answer correctness and question-family leakage; do not represent AI-assisted questions as independently authored.
2. Freeze code + model revision + dataset hashes before evaluation; verify all checksums.
3. Produce a **new, independently authored** evaluation set if feasible; do not tune on it. A reviewed AI-assisted draft may support supplementary testing but not a fully independent test claim.
4. Run E5 question-only + guard on that set; report raw rank-1, accepted correct, wrong fact, unsupported answer, correct abstention, answer coverage, precision, Recall@3, MRR@5 and latency with denominators and per-query traces.
5. When Ahmed and تقى provide actual predictions, compute oracle-vs-upstream degradation and document one success, one consequential failure and the next experiment with distinct-person/query counts.

**Recommendation:** Retain E5 question-only + fact-type guard as the **provisional evaluation candidate**, not production approved. Continue independent verification and actual-upstream testing before final acceptance.
