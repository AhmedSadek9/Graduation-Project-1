# WANAS — Personal Memory System & RAG Engine

This folder implements the Personal Memory System and retrieval component for WANAS.

## Responsibilities

- Store memories in a local SQLite database.
- Associate each memory with a `person_id`.
- Track whether a memory is verified.
- Retrieve relevant **verified** memories using a hybrid keyword and semantic similarity score.
- Return a fixed Arabic fallback when a suitable verified memory is not found.

## Module files

| File | Responsibility |
|---|---|
| `database.py` | Defines the SQLite database path, connection helper, table, and index. |
| `memory.py` | Adds memories and retrieves memories for a person. |
| `retriever.py` | Scores verified memories and returns the best match or fallback. |
| `seed_data.py` | Inserts illustrative seed records for three sample person IDs. |
| `tests/test_retriever.py` | Tests retrieval behavior and fallback cases. |

## Database

The database path is computed from the project root:

```text
data/wanas_memory.db
```

The `memories` table has these columns:

| Column | Type / behavior |
|---|---|
| `id` | Auto-incrementing integer primary key |
| `person_id` | Required text identifier |
| `memory_text` | Required memory text |
| `verified` | Integer restricted to `0` or `1`; defaults to `0` |
| `created_at` | Timestamp set by SQLite by default |

An index named `idx_memories_person_verified` covers `(person_id, verified)`.

### Initialize the database

Run these commands from the project root:

```powershell
python -m ai.memory.database
```

This creates the `data` directory if needed and initializes the table and index.

## Dependencies

The retriever imports `sentence-transformers` and `scikit-learn`. Install these packages in the project's active Python environment if they are not already installed:

```powershell
python -m pip install sentence-transformers scikit-learn
```

The SQLite database API is part of Python's standard library. This README does not claim pinned dependency versions because no `requirements.txt` was included with the files reviewed.

## Seed data

From the project root, run:

```powershell
python -m ai.memory.seed_data
```

The seed script calls database initialization and inserts sample records only when a matching `(person_id, memory_text)` record does not already exist. It is intended to be safe to run repeatedly without duplicating those exact records.

The current sample IDs are `person-001`, `person-002`, and `person-003` (hyphens). The seed text is illustrative test data and should not be treated as real personal information.

## Add and read memories

Run Python from the project root so imports such as `ai.memory.memory` resolve.

```python
from ai.memory.memory import add_memory, get_memories

memory_id = add_memory(
    person_id="person-001",
    memory_text="Likes listening to music",
    verified=True,
)

verified_memories = get_memories("person-001", verified_only=True)
all_memories = get_memories("person-001")
```

`add_memory(person_id, memory_text, verified=False)`:
- Returns the inserted row ID.
- Raises `ValueError` if `person_id` or `memory_text` is empty.
- Requires `verified` to be a Python boolean (`True` or `False`).

`get_memories(person_id, verified_only=False)`:
- Returns a list of dictionaries with `id`, `person_id`, `memory_text`, `verified`, and `created_at`.
- Raises `ValueError` if `person_id` is empty.
- Uses `verified_only=True` to return only rows with `verified = 1`.

## Retrieve a memory

```python
from ai.memory.retriever import retrieve_memory

result = retrieve_memory(
    person_id="person-001",
    query="What music do they like?",
)
print(result)
```

The function signature is:

```python
retrieve_memory(person_id, query, min_score=0.45, keyword_weight=0.3)
```

Retrieval behavior:
1. Rejects a missing/blank person ID with status `no_person`.
2. Rejects a missing/blank query with status `empty_query`.
3. Loads memories for the requested person with `verified_only=True`.
4. Scores each candidate using keyword overlap and semantic similarity.
5. Returns the highest-scoring memory if its hybrid score meets `min_score`; otherwise returns the fallback.

The keyword score is the proportion of distinct query words that also occur in the memory text. The semantic score uses sentence embeddings from `all-MiniLM-L6-v2`, with cosine similarity clipped at a minimum of zero. The default hybrid score is:

```text
0.3 * keyword_score + 0.7 * semantic_score
```

The embedding model is loaded lazily and cached for reuse. The first semantic retrieval may need to download model files, depending on the local environment and cache.

## Retriever response contract (current implementation)

A successful result has this shape:

```json
{
  "query": "What music do they like?",
  "memory_text": "Likes listening to music",
  "verified": true,
  "retrieval_score": 0.493,
  "status": "found"
}
```

When a suitable verified memory is not found, the `memory_text` is exactly:

```text
لا تتوفر لدي معلومات موثقة كافية
```

For `not_found`, the current implementation sets `verified` to `false`; `retrieval_score` is `0.0` if there are no verified memories for that person, or the best candidate score rounded to three decimal places if candidates were evaluated.

Possible status values currently returned by `retrieve_memory`:

| Status | Meaning |
|---|---|
| `found` | A verified memory met the score threshold. |
| `not_found` | No verified memories exist for the person, or no candidate met the threshold. |
| `no_person` | `person_id` is missing or blank. |
| `empty_query` | Query is missing or blank. |

**Integration note:** the project plan's example uses `status: "success"`, while this implementation uses `status: "found"` for a successful retrieval. The team should agree on one API contract with the backend/orchestrator owner before integration. Do not assume these values are interchangeable without coordinating the change.

## Tests

From the project root:

```powershell
python -m unittest tests.test_retriever -v
```

The test file supplied with this README defines 11 test methods. The tests use IDs prefixed with `test-` and remove only rows belonging to those test IDs during setup/teardown. Run the tests against a development/test database, not a database containing irreplaceable data.

The tests cover:
- Retrieving an exact or semantically similar verified memory.
- Excluding unverified memories.
- Returning fallback for unknown facts and unrelated questions.
- Handling a missing person ID.
- Checking the exact Arabic fallback text.

The README does not claim a test run passed in your current environment; run the command above and record the output when validating a release.

## Integration notes for the WANAS team

Before connecting this module to the face-recognition component and backend API, confirm:

1. **Person ID format:** the current seed data uses hyphens (`person-001`), while examples in the project plan use underscores (`person_001`). IDs must be normalized consistently across face recognition, memory storage, context management, and the orchestrator.
2. **Status field:** choose whether the shared API will expose `found`/`not_found` or a different agreed contract, and update the adapter/tests/documentation together.
3. **Fallback handling:** preserve the exact fallback text when no sufficiently similar verified memory is found.
4. **Verification policy:** only expose memories retrieved with `verified_only=True`; do not use unverified seed records as answers.
5. **Threshold evaluation:** `min_score=0.45` is the current default in code. Validate it against the team's evaluation set before treating it as a final quality threshold.

## Current limitations

- The retriever returns the stored memory text; it does not generate a natural-language answer.
- The score is a retrieval similarity score, not a calibrated probability that the answer is correct.
- The hybrid search combines word overlap and embedding similarity; it is not a guarantee against every incorrect match or hallucination.
- The current implementation does not define the final team-wide API contract.
- The current files do not include a `requirements.txt` with pinned package versions.
