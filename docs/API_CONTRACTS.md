# WANAS Backend Integration — API Contracts

These are the current **internal Python contracts** used by `WANASOrchestrator`; they are not HTTP endpoints. Confirm names/statuses with the team before treating them as a public API.

## 1. Face recognition

**Call:** `face_service.recognize(image)`

- `image`: image object in the format expected by the concrete recognizer (current face module expects an OpenCV BGR image).
- Returns: list of dictionaries; empty list means no face detected.
- A face result may contain `person_id`, `name`, `relationship`, `confidence`, `status`, `bbox`.
- Context identity is accepted only when `status == "known"` and `person_id` is non-empty.
- IDs must match memory database IDs exactly.

## 2. Speech-to-text

**Call:** `stt_service.transcribe(audio_path)`

- `audio_path`: path/string understood by the configured STT engine.
- Returns: transcript string. Empty/whitespace transcript should trigger clarification; service errors should be reported as errors.
- Current supplied speech service uses an English model/configuration; Arabic transcription is not promised by this contract.

## 3. Intent classification

**Call:** `intent_service.classify(text)`

- `text`: non-empty transcript/request string.
- Returns dictionary with at least `intent` and `confidence`; classifier may also include `text`.
- Orchestrator applies its minimum confidence threshold and routes supported intents.

## 4. Memory retrieval

**Call:** `memory_retriever(person_id, query)`

- `person_id`: current known person ID.
- `query`: current request text.
- Returns dictionary; expected fields include `status`, `verified`, `memory_text`, and optionally `retrieval_score`/`query`.
- Memory may be shown only when `status == "found"`, `verified is True`, and non-empty `memory_text` is present. Other statuses include `not_found`, `no_person`, and `empty_query`.

## 5. Orchestrator

**Call:** `WANASOrchestrator.process(image=None, audio_path=None, text=None, face_result=None)`

- Input may include an image, audio path, direct text, and/or a precomputed face result (used by tests/integration adapters).
- Returns dictionary with `status`, `message`, `transcript`, `intent`, `face`, `memory`, `context`, and `errors` (some values may be null/empty depending on path).
- `SystemContext` is in-memory temporary context; it does not persist across process restarts.
- Current multiple-face policy selects the first returned face; confirm this behavior with the team.

## 6. Mock service signatures

`MockFaceService.recognize(image) -> list[dict]`

`MockSTTService.transcribe(audio_path) -> str`

`MockIntentService.classify(text) -> dict`

`MockMemoryRetriever(person_id, query) -> dict` (callable object)

All mocks are deterministic and intended for tests only, not production identity or memory decisions.
