# WANAS + ARC — Zinab Memory Module Handover (Steps 1–8)

## Scope and safety
This is a **prototype** for Personal Memory Retrieval. `Task1/memory_database.json` contains 20 fictional profiles / 120 fictional facts; **none are real caregiver-approved patient memories**. Never use this dataset as a real patient profile. Do not present successful synthetic-data tests as real-world recognition performance.

## Directory layout
```
Wanas31/
  Task1/
    memory_database.json
    memory_schema.json
    retrieval_test_queries.json
    unknown_test_cases.json
  Task2/memory_retrieval.py
  Task3/spoken_memory_cues.py
  Task4/
    test_memory_suite.py
    test_identity_mismatch_step3.py
    test_verification_step4.py
    test_spoken_cues_step5.py
    test_memory_safety_step6.py
    test_database_integrity_step7.py
    test_schema_validation_step8.py
```

## Running tests
Open PowerShell in the **Wanas31 root folder**. No need to move files into one directory.

```powershell
python Task4/test_memory_suite.py
python Task4/test_identity_mismatch_step3.py
python Task4/test_verification_step4.py
python Task4/test_spoken_cues_step5.py
python Task4/test_memory_safety_step6.py
python Task4/test_database_integrity_step7.py
python -m pip install "jsonschema>=4.18,<5"
python Task4/test_schema_validation_step8.py
```

The final script validates the real database and checks that malformed records are rejected by the hardened JSON Schema. The separate Step 7 script checks cross-record uniqueness and ID references, which JSON Schema alone cannot guarantee. Run **both**.

## API integration — Ahmed
Always pass the explicit recognition score and explicit mode. The score used for `recognition_confidence` must be mapped/calibrated from the Face Recognition model's similarity metric and the threshold validated on test data. Do not assume cosine similarity is a calibrated probability.

```python
import sys
sys.path.insert(0, 'Task2')
from memory_retrieval import MemoryRetriever

retriever = MemoryRetriever(
    db_path='Task1/memory_database.json',
    min_confidence=0.75,
    data_mode='demo',  # Synthetic test data only
)
print(retriever.query_fact('person-001', 'Who is Omar Hassan?', 0.95))
```

For a future real-data service, explicitly initialize `data_mode='production'` against a **separate caregiver-approved database**. Passing the fictional JSON database in production mode must refuse to disclose its facts. The **current constructor default is `demo` for legacy compatibility**: do not rely on this default for real deployments. Before production release, change the default to production/fail-closed and add authentication, authorization, encrypted storage, change audit logs, and consent/review workflows.

## Voice router integration — Toqa

```python
import sys
sys.path[:0] = ['Task2', 'Task3']
from spoken_memory_cues import SpokenMemoryFormatter

formatter = SpokenMemoryFormatter(
    db_path='Task1/memory_database.json', data_mode='demo'
)
response = formatter.format_greeting_cue(
    person_id='person-001', recognition_confidence=0.95
)
print(response['status'], response['spoken_text'])
```

Never speak an arbitrary LLM-generated memory in place of a safe abstention response. Treat anything other than `success` / `answerable` as **not authorized to disclose a memory**. Query references to *other known people* are rejected; ambiguous references, nickname resolution, first names, and language variants need additional tests/integration handling.

## Schema and data policy
`memory_schema.json` validates syntax, required metadata, field formats and categories. It cannot enforce database-wide uniqueness, fact ownership, or whether a caregiver genuinely approved a memory. The `MemoryRetriever` currently enforces:

- Demo: fictional facts with `source_type=synthetic_test_data` and `verification_status=verified_in_fictional_scenario`.
- Production: no fictional person or fact, and only facts marked `verification_status=caregiver_verified`, `review_status=approved`, with non-synthetic source. These markers are **not themselves proof of real authorization**.

**Do not change the 120 synthetic facts to `approved` just to make them pass.**

## Report status
Results confirmed by local terminal runs through Step 7: original 160/160, identity 10/10, verification 8/8, spoken cues 11/11, safety 24/24, data integrity 17/17. These test groups may overlap; they do not establish patient safety or end-to-end device integration. Step 8 must also be run on the team's machine.

## Next handover
Ahmed + Hanaa: agree identity/score contract and test face→person_id→retrieval. Toqa: connect status-aware voice cues. Zinab: keep the schema, run safety regression tests and document data provenance. Tasneem: verify abstention behavior in the full workflow.
