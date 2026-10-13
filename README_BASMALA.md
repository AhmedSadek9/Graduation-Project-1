# WANAS — Backend Orchestrator & Context Manager

## Scope
- `core/context.py`: temporary context for the last known person and latest request.
- `core/orchestrator.py`: routes face recognition, STT, intent classification, and verified-memory retrieval.
- `mocks/mock_services.py`: deterministic mock implementations for integration tests.
- `tests/`: unit/integration-style tests.
- `docs/API_CONTRACTS.md`: internal interfaces and payload expectations.

## Setup (Windows PowerShell)
Run from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some AI packages (especially InsightFace/ONNX and faster-whisper) can require platform-specific setup and model downloads. Mock-based tests should be run in an environment where the project's imported dependencies are installed.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

Mock-based tests do not prove real camera/audio inference works end-to-end. Test the concrete face and speech services separately before demo/deployment.

## Integration notes
- `FaceRecognizer.recognize(image)` returns a list; the current orchestrator chooses the first face if multiple are present.
- Face `person_id` values must match memory database IDs exactly.
- The supplied speech module is configured for English; this does not provide Arabic STT.
- Only verified, found, non-empty memory text should be returned to the user.
- `CHECK_MEDICATION` and `HELP` are not connected to concrete service modules in this repository.
- The response dictionary and contracts are internal Python interfaces, not a public HTTP API. Confirm them with the team before freezing.
- Context is temporary and resets when the process restarts.


## Submission checklist

- Create and activate a project-local `.venv` using the setup commands above.
- Install dependencies from `requirements.txt` in that environment.
- Run the complete test suite with `python -m unittest discover -s tests -v`.
- The additional `tests/test_orchestrator_edge_cases.py` covers empty text,
  low-confidence intent, service exceptions, safe memory fallback, and explicit
  handling of an unimplemented intent.
- Context is session-only in this implementation. It is retained between calls
  on the same `WANASOrchestrator` instance and is cleared when the process restarts.
  Persistent-on-disk context is not implemented; confirm with the project team
  whether that is required before adding storage.
- The current multiple-face policy uses the first result from Face Recognition.
  This is documented behavior, not a guarantee that the first face is the intended
  person; confirm the policy with the team before the final demo.
- `requirements.txt` currently uses unpinned dependency names. Validate a clean
  install in the team's target Python version before producing a lock/pinned file.

- `test_follow_up_uses_most_recent_known_person` verifies that a follow-up request uses the latest recognized person ID rather than a stale one.
