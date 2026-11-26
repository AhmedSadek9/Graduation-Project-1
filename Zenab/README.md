# WANAS + ARC — Personal Memory Retrieval Module

**Owner:** Zinab (Memory Retrieval). **Integration:** Ahmed (Face Recognition and Backend), Toqa (Assistant/Voice), Hanaa (recognition validation).  
**Status:** Internal, synthetic-data prototype handover; **not** approved for clinical or real-patient use.

## What is included

| Path | Purpose |
|---|---|
| `Task1/memory_database.json` | Synthetic profiles: 20 persons, 120 facts |
| `Task1/memory_schema.json` | JSON Schema for record structure |
| `Task1/retrieval_test_queries.json` | Synthetic evaluation questions |
| `Task1/unknown_test_cases.json` | Unknown-person test fixtures |
| `Task2/memory_retrieval.py` | Person-ID retrieval, query matching, refusal and data-mode checks |
| `Task3/spoken_memory_cues.py` | Status-aware text cues for the future assistant/TTS |
| `Task4/test_*.py` | Original and step 3–9 regression tests |
| `Task4/test_results_report.json` | Historical baseline report, **not** an automatically refreshed test report |
| `memory_integration.py` | Local face-result-to-memory adapter; **mock face input**, not real model connection |
| `run_all_tests.py` | Execute all eight test suites |
| `docs/original_handover/` | Original task-specific handover notes and requirements, preserved for reference |
| `docs/README_steps1_to8_original.md` | Previous progress README, retained as historical reference |

## Setup and run (Windows PowerShell)

From the project root:

```powershell
python -m pip install -r requirements.txt
python run_all_tests.py
python memory_integration.py
```

`python memory_integration.py` explicitly runs a **synthetic demo** and prints two example responses. The adapter **defaults to production** when imported; the older `MemoryRetriever` and `SpokenMemoryFormatter` constructors still default to `demo` for backwards compatibility. **Always pass `data_mode` explicitly** in application code. Do not pass fictional data to actual users.

## Integration contract

The adapter `FaceMemoryBridge.process(face_result, intent, question=None)` accepts only these intents: `identify_person`, `retrieve_memory`, `conversation_starter`.

A synthetic demonstration input:

```python
from memory_integration import FaceMemoryBridge

bridge = FaceMemoryBridge(data_mode="demo")  # synthetic demonstrations ONLY
face_result = {
    "status": "recognized",
    "person_id": "person-001",
    "recognition_confidence": 0.95,  # illustrative value, not a calibrated probability
}
result = bridge.process(face_result, intent="retrieve_memory", question="Who is Omar Hassan?")
print(result)
```

Use `result['can_speak_personal_memory']` as the downstream disclosure gate. Reject or safely handle any non-success status; do not replace a refusal with invented text. The identity score scale and threshold need agreement and testing with Ahmed/Hanaa; **raw cosine similarity is not necessarily calibrated recognition confidence**. User authentication, consent, authorization, secure storage, audit logging and real caregiver verification remain future work.

## Test interpretation

The eight suites include 160 retrieval questions, 10 name/ID mismatch tests, 8 verification tests, 11 spoken-cue tests, 24 additional safety tests, 17 data-integrity checks, 12 schema tests and 20 bridge checks. Some scenarios overlap. The 30 unknown-profile and 20 isolation checks are included in the original suite and should **not** be added again as independent counts. Passing these synthetic tests does **not** demonstrate real camera integration or clinical safety.

## Handover responsibilities / unresolved items

- **Zinab:** maintain memory schema, synthetic fixtures, retrieval and safety tests; support integration issues.
- **Ahmed:** map face model output into the agreed result contract, confirm score semantics/threshold, connect service and database.
- **Toqa:** use the safe `spoken_text`/status output to implement TTS and assistant flow.
- **Hanaa:** evaluate actual face results, incorrect identities, unknown subjects and handoff behavior.
- **Before real deployment:** default all public entry points to fail-closed production mode, replace synthetic fixtures with consented caregiver-verified data, and test privacy/authorization and real-world accuracy.

**Do not relabel synthetic facts as caregiver-approved to make a demo pass.**

## November 25 update / new integration evaluation

See [`PROGRESS_AND_GAPS.md`](PROGRESS_AND_GAPS.md) for current results (supersedes older historical documents), [`handoff/REQUESTS_FOR_TEAM.md`](handoff/REQUESTS_FOR_TEAM.md) for Ahmed/Hanaa/Toqa dependencies, and [`handoff/step29/STEP29_START_HERE_AR.md`](handoff/step29/STEP29_START_HERE_AR.md) for the new A/B/C/D case-manifest preparation. The raw current task requires real face `decision` and paired STT results; those are not included in this repo.
