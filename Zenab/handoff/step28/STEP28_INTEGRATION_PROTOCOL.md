# WANAS — Step 28: Integration & Evaluation Readiness Protocol

**Status:** Protocol prepared; real upstream integration **NOT EXECUTED**. No production deployment authorized. This does not modify existing retrieval code, the 120 synthetic facts, 40 draft review questions, or frozen E5 threshold.

## A. Current verified boundaries

- `memory_integration.py` exposes `FaceMemoryBridge.process(face_result, intent, question)` as an **in-process Python function**, not an HTTP endpoint.
- Default production mode blocks synthetic memory records. Demo mode is synthetic only.
- Real Ahmed face outputs and real Toqa STT transcripts have **not been supplied**. Existing safety/unit tests do not estimate real-world identity or ASR failure rates.
- E5 Step 21 is an experimental separate adapter; the original bridge still uses rule-based retrieval.
- Frozen E5 threshold `0.8824205`. Candidate `0.80` is **exploratory only** (Step 27), not validated for deployment.
- All 40 Step 16 AI-assisted questions remain pending human review; no claim of an untouched blind evaluation.

## B. Required request/response handshake (mock contract)

Face model → coordinating backend → memory module:

```json
{"status":"recognized","person_id":"person-001","recognition_confidence":0.95}
```

`0.95` is a **dummy value**; the face team must specify whether confidence is a calibrated probability, embedding similarity, or something else. No assumption that every face model emits this shape. A stable person ID must match the approved memory identity mapping. Unknown, low-score, missing, nonfinite, boolean or mismatched identities must fail closed.

Memory call:

```python
bridge.process(face_result, intent="retrieve_memory", question="Who is Omar Hassan?")
```

For downstream speech, read `can_speak_personal_memory` first. Only `true` allows speaking the corresponding approved fact in `spoken_text`; if false, any spoken refusal must be generic and must not be reformulated by an LLM into invented personal information. Null fact ID on abstention. Request IDs, timestamps, STT versions and trace IDs should be attached by the backend because the current core does not implement those fields.

## C. Evaluation matrix after data arrives

| Test group | Input | Safety and behavior check | Measure |
|---|---|---|---|
| Face known / correctly matched | Real face result + reference person ID | Facts only from recognized, approved person | recognition correctness, memory top-1, coverage |
| Face wrong identity | Known misclassification with reference ID | Never disclose another person's memory | wrong-person disclosure count (target 0) |
| Face unknown / low confidence | Unknown or uncertain result | Abstain without personal details | safe-abstention rate |
| STT accurate | Audio + human reference transcript + STT output | Compare retrieval from transcript vs reference | WER/CER, accepted-correct rate |
| STT entity/name errors | Names, places, negation distorted | Avoid cross-person access and invented facts | unsafe disclosure count, missed answerable |
| No relevant memory | Valid person + unsupported question | Abstain, no unsupported fact | unsupported answers |
| Conflicting/stale/unapproved | Pending or contradictory memory | No disclosure until approved and current | unsafe disclosure count |
| Production with synthetic | Synthetic facts in production mode | Must block | blocked / total |
| Candidate E5 vs baseline | New unseen and reviewed question set | Equal filtering and evaluation conditions | accepted precision, coverage, recall@3, MRR@5, latency |

### Required metrics and definitions

- **Accepted precision:** accepted correct / all accepted non-abstentions (undefined if no accepted answers).
- **Answerable correct rate:** accepted correct / all answerable questions.
- **Correct abstention rate:** abstained unanswerable / all unanswerable.
- **Wrong-person disclosures:** count of emitted personal facts belonging to a person other than the approved recognized identity; must be **0** before further integration.
- **STT degradation:** difference between performance on verified reference transcripts and corresponding real STT transcripts for the **same audio items**; report paired counts, not just percentages.
- **Latency:** report component timings with machine/runtime and sample sizes; cached offline timings and live end-to-end timings are **not interchangeable**.

## D. Independent evaluation conditions

1. Hanaa reviews existing 40 draft questions for person/fact mapping, labels, ambiguity, stale or conflicting facts, and paraphrase overlap. Record corrections and reviewer identity/date **truthfully**.
2. Because these 40 were AI-assisted and design/evaluation decisions were made with knowledge of similar prior questions, **human approval does not automatically make them blind/untouched**. For independent performance estimates, create a fresh sealed set authored without viewing predictions, after policy lock; another person checks it.
3. Freeze policy/model/threshold and hash files before revealing new set. Do not choose threshold by inspecting that set's results.
4. Evaluate baseline and candidate once, publish count denominators and all error categories. Any subsequent tuning moves to a new development set; final figures remain tied to original frozen policy.

## E. Ownership and data requests (not yet sent)

**Hanaa — data quality & independent review:**
- Verify exact stable `person_id` ↔ name / face-enrollment mapping with Ahmed, duplicates, ambiguity and missing IDs.
- Review 40 Step 27 draft questions: expected fact ownership, answerability, unsupported topics, shared memories, provenance, leaks, stale/conflicting entries, wording changes, and record actual reviewer/decision/reason/date.
- Coordinate an actually new unseen evaluation set with separate author/reviewer if independent performance claim is needed. Return approved revisions as new versioned artifacts; do not overwrite old labels silently.

**Ahmed — face / integration:**
- Supply real recognition outputs for permitted test samples with stable person ID, predicted status, score and score definition, reference identity, model/version and sample/session identifiers (no unnecessary biometric files).
- Agree threshold calibration and unknown/mismatch mapping; normalize to bridge input. Provide backend caller mechanism and report failed identity cases with no raw face leakage.

**Toqa — voice/STT/assistant:**
- Supply privacy-safe audio references or permitted IDs, human reference transcripts, real STT text, language/dialect, STT model/version, sample/session identifiers, and key transcription failure examples.
- Use the same question pairs to quantify retrieval changes; do not voice personal facts unless `can_speak_personal_memory` true. Return voice behavior results for accepted/abstained outcomes.

**Zinab — retrieval:**
- Supply fact schema, person/fact IDs, frozen baseline, proposed policy, guard rules, example bridge responses, and metrics script; confirm safe mapping and run comparison only after approved inputs exist.
- Maintain current rule-based production bridge until E5 adapter is independently evaluated and integrated under explicit safety tests.

## F. Exit criteria

- Signed-off identity mapping + real face status/score semantics.
- STT paired transcripts and confirmed speech disclosure gate.
- No cross-person or synthetic-data disclosure in integration tests.
- Independent benchmark review and a genuinely unseen test set before making new accuracy claims.
- Explicit decision whether to adopt E5 candidate; **no automatic change** from this pack.
