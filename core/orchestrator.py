
from typing import Any, Callable, Dict, Optional

from core.context import SystemContext
from ai.memory.memory import get_memories

FALLBACK_MESSAGE = "I don't have enough verified information to answer that."


class WANASOrchestrator:
    """
    Connects the existing WANAS modules using dependency injection.

    Expected interfaces:
      face_service.recognize(image) -> list of face-result dictionaries
      stt_service.transcribe(audio_path) -> str
      intent_service.classify(text) -> dict
      memory_retriever(person_id, query) -> dict
    """

    def __init__(
        self,
        face_service: Any = None,
        stt_service: Any = None,
        intent_service: Any = None,
        memory_retriever: Optional[Callable] = None,
        context: Optional[SystemContext] = None,
        min_intent_confidence: float = 0.35,
    ):
        self.face_service = face_service
        self.stt_service = stt_service
        self.intent_service = intent_service
        self.memory_retriever = memory_retriever
        self.context = context or SystemContext()
        self.min_intent_confidence = min_intent_confidence

    def process(
        self,
        image: Any = None,
        audio_path: Optional[str] = None,
        text: Optional[str] = None,
        face_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        errors = []

        # 1. Face Recognition
        if face_result is None and image is not None:
            if self.face_service is None:
                return self._response(
                    "error",
                    "Face Recognition service is not configured.",
                    text, None, None, errors
                )

            try:
                face_results = self.face_service.recognize(image)

                face_result = face_results[0] if face_results else {
                    "person_id": None,
                    "name": "Unknown",
                    "relationship": None,
                    "confidence": 0.0,
                    "status": "unknown",
                }

            except Exception as exc:
                errors.append(
                    f"face_recognition_error: {type(exc).__name__}"
                )

                face_result = {
                    "person_id": None,
                    "name": "Unknown",
                    "relationship": None,
                    "confidence": 0.0,
                    "status": "unknown",
                }

        # 2. Update Context
        if face_result is not None:
            self.context.update_person(face_result)

        # 3. Speech-to-Text
        if text is None and audio_path is not None:
            if self.stt_service is None:
                return self._response(
                    "error",
                    "Speech-to-text service is not configured.",
                    None, None, face_result, errors
                )

            try:
                text = self.stt_service.transcribe(audio_path)

            except Exception as exc:
                errors.append(f"stt_error: {type(exc).__name__}")

                return self._response(
                    "error",
                    "Speech transcription failed. Please try again.",
                    None, None, face_result, errors
                )

        # If no text was supplied, return the face result.
        if text is None:
            return self._response(
                "success",
                None,
                None,
                None,
                face_result,
                errors,
            )

        # 4. Validate the transcript
        if not str(text).strip():
            return self._response(
                "needs_clarification",
                "Sorry, I didn't understand. Please repeat.",
                text,
                {"intent": "UNKNOWN", "confidence": 0.0},
                face_result,
                errors,
            )

        # 5. Intent Classification
        if self.intent_service is None:
            return self._response(
                "error",
                "Intent classifier is not configured.",
                text, None, face_result, errors
            )

        try:
            intent_result = self.intent_service.classify(text)

        except Exception as exc:
            errors.append(f"intent_error: {type(exc).__name__}")

            return self._response(
                "error",
                "Intent classification failed. Please try again.",
                text, None, face_result, errors
            )

        intent = intent_result.get("intent", "UNKNOWN")
        confidence = float(intent_result.get("confidence", 0.0))

        self.context.update_request(text, intent)

        # 6. Handle unclear requests
        if (
            intent == "UNKNOWN"
            or confidence < self.min_intent_confidence
        ):
            return self._response(
                "needs_clarification",
                "Sorry, I didn't understand. Please repeat.",
                text,
                intent_result,
                face_result,
                errors,
            )

        # 7. Route IDENTIFY_PERSON
        if intent == "IDENTIFY_PERSON":
            person = self.context.current_person

            if person:
                message = (
                    f"This is {person.get('name') or 'a registered person'}."
                )
                status = "success"

            elif (
                face_result
                and face_result.get("status") == "low_confidence"
            ):
                message = "I couldn't identify this person confidently."
                status = "needs_clarification"

            else:
                message = "I don't recognize this person."
                status = "not_found"

            return self._response(
                status,
                message,
                text,
                intent_result,
                face_result,
                errors,
            )

        # 8. Route GET_MEMORY
        if intent == "GET_MEMORY":
            person = self.context.current_person

            if not person:
                memory_result = {
                    "query": text,
                    "memory_text": FALLBACK_MESSAGE,
                    "verified": False,
                    "retrieval_score": 0.0,
                    "status": "no_person",
                }

            elif self.memory_retriever is None:
                return self._response(
                    "error",
                    "Memory retriever is not configured.",
                    text, intent_result, face_result, errors
                )

            else:
                normalized_text = " ".join(str(text).lower().split())

                general_prompts = {
                    "tell me about him",
                    "tell me about her",
                    "tell me about this person",
                    "what do you know about him",
                    "what do you know about her",
                    "what do you remember about him",
                    "what do you remember about her",
                }

                is_general_prompt = normalized_text in general_prompts
                retrieval_failed = False

                # Always call the configured retriever first.
                try:
                    memory_result = self.memory_retriever(
                        person["person_id"],
                        text,
                    )

                except Exception as exc:
                    retrieval_failed = True
                    errors.append(
                        f"memory_error: {type(exc).__name__}"
                    )

                    memory_result = {
                        "query": text,
                        "memory_text": FALLBACK_MESSAGE,
                        "verified": False,
                        "retrieval_score": 0.0,
                        "status": "not_found",
                    }

                # For an explicit general prompt, list verified memories
                # only when normal retrieval safely returns not_found.
                # Never use this fallback to override an exception or
                # a result that contains unverified content.
                if (
                    is_general_prompt
                    and not retrieval_failed
                    and memory_result.get("status") == "not_found"
                ):
                    try:
                        memories = get_memories(
                            person["person_id"],
                            verified_only=True,
                        )

                        verified_texts = [
                            item["memory_text"].strip()
                            for item in memories
                            if item.get("verified") in (True, 1)
                            and isinstance(item.get("memory_text"), str)
                            and item["memory_text"].strip()
                        ]

                        if verified_texts:
                            memory_result = {
                                "query": text,
                                "memory_text": "\n".join(
                                    f"- {memory}"
                                    for memory in verified_texts
                                ),
                                "verified": True,
                                "retrieval_score": None,
                                "retrieval_method": "verified_memory_summary",
                                "status": "found",
                            }

                    except Exception as exc:
                        errors.append(
                            f"memory_summary_error: {type(exc).__name__}"
                        )

                        memory_result = {
                            "query": text,
                            "memory_text": FALLBACK_MESSAGE,
                            "verified": False,
                            "retrieval_score": 0.0,
                            "status": "not_found",
                        }

            # Only verified and found memories can be displayed.
            is_verified_result = (
                memory_result.get("status") == "found"
                and memory_result.get("verified") is True
                and isinstance(memory_result.get("memory_text"), str)
                and bool(memory_result.get("memory_text").strip())
            )

            if is_verified_result:
                message = memory_result["memory_text"]
                status = "success"
            else:
                message = FALLBACK_MESSAGE
                status = "not_found"

            return self._response(
                status,
                message,
                text,
                intent_result,
                face_result,
                errors,
                memory_result=memory_result,
            )

        # 9. Medication is not connected yet.
        if intent == "CHECK_MEDICATION":
            return self._response(
                "not_implemented",
                "Medication checking is not connected yet.",
                text, intent_result, face_result, errors
            )

        # 10. Caregiver alert is not connected yet.
        if intent == "HELP":
            return self._response(
                "not_implemented",
                "Caregiver alert service is not connected yet.",
                text, intent_result, face_result, errors
            )

        # 11. Unsupported intent
        return self._response(
            "not_implemented",
            "This request is not supported yet.",
            text, intent_result, face_result, errors
        )

    def _response(
        self,
        status,
        message,
        transcript,
        intent,
        face_result,
        errors,
        memory_result=None,
    ):
        return {
            "status": status,
            "message": message,
            "transcript": transcript,
            "intent": intent,
            "face": face_result,
            "memory": memory_result,
            "context": self.context.to_dict(),
            "errors": errors,
        }
