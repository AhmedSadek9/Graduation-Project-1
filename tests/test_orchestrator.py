import unittest

from core.context import SystemContext
from core.orchestrator import (
    WANASOrchestrator,
    FALLBACK_MESSAGE,
)
from mocks.mock_services import (
    MockFaceService,
    MockIntentService,
    MockMemoryRetriever,
)


KNOWN_FACE = {
    "person_id": "person-001",
    "name": "Ahmed",
    "relationship": "Son",
    "confidence": 0.91,
    "status": "known",
}


class OrchestratorTests(unittest.TestCase):

    def test_context_stores_known_person(self):
        context = SystemContext()

        context.update_person(KNOWN_FACE)

        self.assertEqual(
            context.current_person["person_id"],
            "person-001"
        )

    def test_context_clears_old_person_for_unknown(self):
        context = SystemContext()

        context.update_person(KNOWN_FACE)

        context.update_person({
            "person_id": None,
            "status": "unknown",
            "confidence": 0.2,
        })

        self.assertIsNone(context.current_person)

    def test_get_memory_calls_retriever_with_current_id(self):
        memory = MockMemoryRetriever()

        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY"),
            memory_retriever=memory,
        )

        result = app.process(
            text="Tell me about him",
            face_result=KNOWN_FACE,
        )

        self.assertEqual(
            memory.calls[0]["person_id"],
            "person-001"
        )

        self.assertEqual(
            result["message"],
            "Ahmed visits every Friday."
        )

        self.assertEqual(result["status"], "success")

    def test_unverified_memory_is_never_exposed(self):
        memory = MockMemoryRetriever({
            "query": "Tell me about him",
            "memory_text": "SECRET UNVERIFIED CONTENT",
            "verified": False,
            "retrieval_score": 0.99,
            "status": "found",
        })

        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY"),
            memory_retriever=memory,
        )

        result = app.process(
            text="Tell me about him",
            face_result=KNOWN_FACE,
        )

        self.assertEqual(
            result["message"],
            FALLBACK_MESSAGE
        )

        self.assertNotIn(
            "SECRET UNVERIFIED CONTENT",
            result["message"]
        )

    def test_no_person_returns_fallback(self):
        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY")
        )

        result = app.process(text="Tell me about him")

        self.assertEqual(result["status"], "not_found")

        self.assertEqual(
            result["message"],
            FALLBACK_MESSAGE
        )

    def test_low_confidence_person_does_not_get_memory(self):
        memory = MockMemoryRetriever()

        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY"),
            memory_retriever=memory,
        )

        low_face = {
            **KNOWN_FACE,
            "person_id": None,
            "name": "Unknown",
            "relationship": None,
            "status": "low_confidence",
        }

        result = app.process(
            text="Tell me about him",
            face_result=low_face,
        )

        self.assertEqual(
            result["message"],
            FALLBACK_MESSAGE
        )

        self.assertEqual(memory.calls, [])

    def test_context_person_persists_across_requests(self):
        memory = MockMemoryRetriever()

        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY"),
            memory_retriever=memory,
        )

        # First request: recognize Ahmed.
        first = app.process(
            text="Who is this?",
            face_result=KNOWN_FACE,
        )

        self.assertEqual(first["status"], "success")

        # Second request: no new face result.
        second = app.process(
            text="Tell me about him",
        )

        self.assertEqual(second["status"], "success")
        self.assertEqual(
            memory.calls[-1]["person_id"],
            "person-001",
        )
        self.assertEqual(
            second["context"]["current_person"]["name"],
            "Ahmed",
        )

    def test_unknown_face_clears_context_before_memory_lookup(self):
        memory = MockMemoryRetriever()

        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY"),
            memory_retriever=memory,
        )

        # First request: recognize Ahmed.
        app.process(
            text="Who is this?",
            face_result=KNOWN_FACE,
        )

        # Record calls made before the unknown face appears.
        calls_before_unknown = len(memory.calls)

        # Second request: an unknown person appears.
        result = app.process(
            text="Tell me about him",
            face_result={
                "person_id": None,
                "name": "Unknown",
                "relationship": None,
                "confidence": 0.2,
                "status": "unknown",
            },
        )

        self.assertEqual(result["status"], "not_found")
        self.assertIsNone(result["context"]["current_person"])

        # The unknown face must not trigger another memory lookup.
        self.assertEqual(
            len(memory.calls),
            calls_before_unknown,
        )

        self.assertEqual(result["message"], FALLBACK_MESSAGE)

    def test_unknown_intent_asks_for_repeat(self):
        app = WANASOrchestrator(
            intent_service=MockIntentService(
                "UNKNOWN",
                confidence=0.1,
            )
        )

        result = app.process(text="unclear request")

        self.assertEqual(
            result["status"],
            "needs_clarification"
        )

    def test_face_service_uses_first_detected_face(self):
        app = WANASOrchestrator(
            face_service=MockFaceService()
        )

        result = app.process(image=object())

        self.assertEqual(
            result["face"]["person_id"],
            "person-001"
        )


    def test_follow_up_uses_most_recent_known_person(self):
        """A follow-up must target the latest known person, not a stale identity."""
        memory = MockMemoryRetriever()
        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY"),
            memory_retriever=memory,
        )
        second_person = {
            "person_id": "person-002",
            "name": "Mona",
            "relationship": "Daughter",
            "confidence": 0.94,
            "status": "known",
        }

        app.process(text="Who is this?", face_result=KNOWN_FACE)
        result = app.process(
            text="Tell me about her",
            face_result=second_person,
        )

        self.assertEqual(memory.calls[-1]["person_id"], "person-002")
        self.assertEqual(
            result["context"]["current_person"]["name"],
            "Mona",
        )


if __name__ == "__main__":
    unittest.main()
