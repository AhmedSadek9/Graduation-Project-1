# Step 19 — WANAS Memory → Voice / Backend Contract

**Status:** Internal synthetic demonstration only. **Not yet integrated with Toqa’s actual STT/TTS or Ahmed’s real face model.** The 5 examples in the CSV were executed against the existing `memory_integration.py` adapter.

## Existing callable API

```python
from memory_integration import FaceMemoryBridge
bridge = FaceMemoryBridge(data_mode="demo")  # synthetic tests only
response = bridge.process(face_result, intent="retrieve_memory", question="Who is Omar Hassan?")
```

`FaceMemoryBridge(data_mode="production")` is the default; it blocks the supplied synthetic records. Do **not** edit verification flags to bypass this.

### Face result input (agreed mock shape, **not a live face integration**)

```json
{"status":"recognized","person_id":"person-001","recognition_confidence":0.95}
```

- Allowed `status` for retrieval: `recognized`; else refuse.
- `person_id`: stable ID as a string, matched exactly; Ahmed and Hanaa must validate mapping.
- `recognition_confidence`: finite number in [0, 1]; score meaning and calibrated threshold remain unconfirmed. Do not blindly substitute raw cosine similarity or assume 0.95 is real.
- Supported `intent`: `identify_person`, `retrieve_memory`, `conversation_starter`.
- `question`: nonblank text required for `retrieve_memory`.
- This is an in-process **Python function**, **not** a REST endpoint. HTTP routes, request IDs, logging and authentication are *not implemented* here.

### Response fields for Toqa

- `status`: `answerable` or `success` only when a permitted memory can be spoken; otherwise an abstention status (such as `unanswerable`, `person_mismatch_abstain`, `face_not_recognized_abstain`, `no_verified_facts`).
- `spoken_text`: safe TTS candidate. On refusal it is a safe refusal message, **not** a personal fact.
- `can_speak_personal_memory`: **mandatory disclosure gate**. Toqa must never synthesize personal memory when false. A generic refusal may be voiced, respecting UX and consent.
- `fact_id`, `fact_type`, `question`: returned for `retrieve_memory` calls (fact ID null on refusal). `hud_short_text`, `name`, `relationship` are present on eligible greeting responses only; don't assume them for every intent.
- Current response does not include a stable upstream run ID, STT model/version, query ID or timestamp. These must be carried by the coordinating backend as metadata without changing the tested core unexpectedly.

### Toqa downstream logic (reference pseudocode)

```python
result = bridge.process(face_result, intent, question)
if result.get("can_speak_personal_memory") is True:
    tts_text = result["spoken_text"]
else:
    tts_text = result.get("spoken_text", "I cannot confirm that information.")
    # This is a refusal, NOT a verified memory; do not ask an LLM to invent a fact.
```

## Important distinction: retrieval implementations

The existing `memory_integration.py`/`SpokenMemoryFormatter` calls the **rule-based** `Task2/memory_retrieval.py` prototype. The later BM25/MiniLM/E5 scripts under `experiments/` are **offline evaluations**. The provisional E5 config in `experiments/step16/frozen_e5_config_step16.json` is **not yet wired into this bridge**. Do not report E5 as powering the live assistant.

## Open items and ownership

- **Zinab:** confirm desired query and abstention statuses; supply verified fact-ID examples, maintain schema and evaluation results; define adapter strategy for selected retrieval model after independent validation.
- **Toqa:** provide stable question IDs, reference transcripts, real STT transcripts/errors, STT version, session ID and audio reference; consume `spoken_text` only after checking the disclosure flag. Test voice flow after inputs arrive.
- **Ahmed:** normalize actual face outputs and score semantics into agreed contract; coordinate Python/HTTP integration and request/run metadata.
- **Hanaa:** validate identity mapping and review 40 AI-assisted draft questions and fact IDs; independent review must be genuine and documented.

No actual upstream model output, independent benchmark approval, or real-world reliability figures are claimed in this package.
