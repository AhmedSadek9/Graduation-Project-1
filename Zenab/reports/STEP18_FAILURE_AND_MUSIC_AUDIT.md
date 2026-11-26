# WANAS — Step 18: Music Query and Retrieval Failure Audit

**Scope:** Synthetic evaluation evidence only. This is a documentation step, not a new model run, independent holdout result, or production approval.

## 1. Music-query investigation

**Evidence source:** `experiments/results/music_diagnostic.json` in the existing evaluation project.

| Field | Observed value |
|---|---|
| Query text | Does Nada record music clips for the family? |
| Query provenance | Constructed diagnostic example; **not** the original failed question |
| Person ID | `person-020` |
| Relevant fact ID | `person-020-F05` |
| Fact text | Nada sometimes records short music clips for the family. |
| Survived person/verification filter | Yes |
| Candidates after filter | `person-020-F01` through `person-020-F06` (6 facts) |
| Rank of correct fact | 1 |
| BM25 score | 6.083245 (reported; rounded) |
| BM25 acceptance threshold | 0.657456 |
| Diagnostic example accepted | Yes |
| Original reported `not_found` cause/repair | **NOT VERIFIED** — original query and execution trace unavailable |

**Interpretation:** The constructed query passes filtering, retrieval ranking, and acceptance. It does **not** establish why the original query produced `not_found`, or that the original bug was fixed. Do not mark the full original-query debugging task complete until the original query or trace is supplied and replayed against a recorded checkpoint.

**When available, capture:** the exact original query, reference/person ID, filtering mode and candidate IDs, ranking positions and scores, acceptance threshold, final decision/reason, model/code version, and the before/after output that demonstrates a repair.

## 2. Consequential wrong-answer cases

Evidence: Step 14/15 context audit and console outputs provided in the project discussion. All outcomes below are **retrospective, exploratory**.

| Query ID | Question / expected | Before type guard | After type guard |
|---|---|---|---|
| `Q-person-015-03` | Fadi phone number; unanswerable | `person-015-F01` identity fact (unsupported) | Abstain |
| `Q-person-016-03` | Mariam phone number; unanswerable | `person-016-F06` conversation starter (unsupported) | Abstain |
| `Q-person-015-02` | Talk to Fadi; `person-015-F06` | `person-015-F01` (wrong fact) | `person-015-F06` |
| `Q-person-015-05` | Shared memory with Fadi; `person-015-F04` | `person-015-F01` (wrong fact) | `person-015-F04` in synthetic last-request scenario |
| `Q-person-017-03` | Ali phone number; unanswerable | `person-017-F01` identity fact (unsupported) | Abstain |
| `Q-person-017-05` | Meal cooked with Ali; `person-017-F04` | `person-017-F06` conversation starter | `person-017-F04` |

**Retrospective results, question + identity + synthetic last request:** Validation: accepted correct 16→18, wrong facts 2→0, unsupported 2→0, correct abstentions 4/6→6/6. Previously inspected test: accepted correct 14→15, wrong 1→0, unsupported 1→0, correct abstentions 15/16→16/16.

**Limits:** Fact-type rules are currently English-heavy; tests were inspected before guards were added; these numbers do not measure unseen-question accuracy. A newly authored/reviewed evaluation set and real face/STT errors remain pending.

## 3. Current decision and follow-ups

- **Provisional baseline only:** `intfloat/multilingual-e5-small`; `query: ` / `passage: ` prefixes; question-only; current person + verification filtering before ranking; normalized cosine similarity; threshold `0.8824205`; intent-to-fact-type guard; abstain if no eligible candidate above the threshold.
- **Demo-only evidence:** Synthetic facts with `verified_in_fictional_scenario` and draft review status must never be represented as approved real patient memory.
- **Do not claim:** music original-query repair, independent held-out validation, measured upstream error degradation, or production readiness.
- **Pending from Hanaa:** genuinely review question/fact correctness, person-ID mapping, similar-name/stale/conflicting cases, paraphrase grouping.
- **Pending from Ahmed:** actual face predictions with person ID, reference label, status, score, session/question link, model version.
- **Pending from تقى:** paired reference and predicted STT transcripts with stable question/session IDs and context/time metadata.

## 4. Next experiment

After independent question review, run the frozen E5 question-only + fact-type guard once on the new set, recording per-question candidate IDs after filtering, ranks/scores, selected fact, acceptance/abstention reason, correct/wrong/unsupported labels, and runtime. Keep any developmental exploratory runs separate. When actual upstream predictions arrive, compare oracle-ID/reference-transcript performance with actual-input performance and report the difference in percentage points with the same denominators.
