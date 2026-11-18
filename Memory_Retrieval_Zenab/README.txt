========================================================================
WANAS - PERSONAL MEMORY RETRIEVAL MODULE
README, HANDOVER & TESTING GUIDE
========================================================================

1. MODULE SCOPE
------------------------------------------------------------------------
This module implements the Personal Memory Retrieval and Assistant Context
part of WANAS. It retrieves caregiver-recorded information using a person's
person_id and returns a structured response for the calling component.

Implemented files:
- memory_db.json: small local sample database for development/testing.
- memory_retrieval.py: MemoryRetrievalSystem and person_id-based lookup.
- test_memory.py: automated unit tests for normal and error cases.
- requirements.txt: third-party dependency list (currently none required).

Important: the included records are sample/demo data, not real patient data.
Only caregiver-verified information should be added to the database. Protect
real patient and caregiver information before using this prototype in practice.

2. MEMORY RECORD SCHEMA
------------------------------------------------------------------------
Each record in memory_db.json must be a JSON object with these fields:

{
  "person_id": "P001",
  "fact": "This is Ahmed, your elder son who lives in Cairo.",
  "source": "Caregiver (Fatma)",
  "update_date": "2026-10-25"
}

The database must be a JSON list. Each person_id must be unique. The fact,
source, and update_date fields must be non-empty strings. Invalid JSON,
missing database files, invalid records, or duplicate IDs raise clear errors
instead of being silently treated as an empty database.

3. REQUIREMENTS AND SETUP
------------------------------------------------------------------------
Prerequisite: Python 3.8 or newer.

This module uses only Python standard-library modules (json, pathlib,
tempfile, and unittest). No third-party packages are currently required.

From the project folder, run:

    python --version

Optional: install the requirements file (it contains no third-party packages):

    python -m pip install -r requirements.txt

4. RUN THE RETRIEVAL DEMO
------------------------------------------------------------------------
From the folder containing memory_retrieval.py and memory_db.json, run:

    python memory_retrieval.py

The demo looks up P001, P999, unknown, and None. Expected behavior:
- P001: status is "success" and the matching fact, source, and update_date
  are returned.
- P999: status is "no-verified-information" and fact is None.
- unknown: status is "no-verified-information" and fact is None.
- None: status is "no-verified-information" and fact is None.

The exact output includes the sample fact text and explanatory message.

5. RUN AUTOMATED TESTS
------------------------------------------------------------------------
Run the full test suite from the project folder:

    python -m unittest -v

The current suite contains 12 tests covering:
1. Valid person retrieval.
2. Cross-person isolation.
3. Missing person ID in the database.
4. Unknown person.
5. None as person_id.
6. Empty or whitespace-only person_id.
7. person_id with surrounding spaces.
8. Missing database file.
9. Invalid JSON.
10. Duplicate person IDs.
11. Empty fact.
12. Database root must be a list.

A successful run should finish with:

    Ran 12 tests
    OK

The test count can change as new tests are added; the command output is the
source of truth for the result on your machine.

6. RESPONSE CONTRACT FOR INTEGRATION
------------------------------------------------------------------------
Face Recognition / API caller:
    from memory_retrieval import MemoryRetrievalSystem
    memory_system = MemoryRetrievalSystem()
    response = memory_system.get_memory_by_person_id(person_id)

Input:
- person_id: a recognized ID such as "P001". Use "unknown" when no person
  has been confidently identified. Do not pass a person's name as the ID
  unless the database schema and calling system are deliberately changed.

Successful response:
{
  "status": "success",
  "person_id": "P001",
  "fact": "...",
  "source": "...",
  "update_date": "..."
}

Fallback response:
{
  "status": "no-verified-information",
  "person_id": "P999",
  "fact": None,
  "message": "No verified information available for ID P999."
}

Voice Assistant routing:
- If response["status"] == "success", use response["fact"] as the memory cue.
- If response["status"] == "no-verified-information", use a safe fallback
  such as: "I don't have verified information about this person."
- Do not speak unverified facts or infer relationships that are not recorded.

Integration status:
The retrieval module and its local unit tests have been tested independently.
Integration with Face Recognition, the API/backend, and the Voice Assistant
must be tested with the other team components; this README does not claim that
end-to-end integration has already been completed.
========================================================================
