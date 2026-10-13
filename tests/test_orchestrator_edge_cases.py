"""Additional edge-case tests for the WANAS backend orchestrator."""

import unittest

from core.orchestrator import WANASOrchestrator, FALLBACK_MESSAGE
from mocks.mock_services import MockIntentService, MockMemoryRetriever


KNOWN_FACE = {
    "person_id": "person-001",
    "name": "Ahmed",
    "relationship": "Son",
    "confidence": 0.91,
    "status": "known",
}


class RaisingSTT:
    def transcribe(self, audio_path):
        raise RuntimeError("simulated STT failure")


class RaisingIntent:
    def classify(self, text):
        raise RuntimeError("simulated intent failure")


class RaisingMemory:
    def __call__(self, person_id, query):
        raise RuntimeError("simulated retrieval failure")


class OrchestratorEdgeCaseTests(unittest.TestCase):
    def test_empty_text_requests_clarification(self):
        app = WANASOrchestrator()
        result = app.process(text="   ")
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["intent"]["intent"], "UNKNOWN")

    def test_low_intent_confidence_requests_clarification(self):
        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY", confidence=0.1)
        )
        result = app.process(text="Tell me about him")
        self.assertEqual(result["status"], "needs_clarification")

    def test_stt_exception_returns_error(self):
        app = WANASOrchestrator(stt_service=RaisingSTT())
        result = app.process(audio_path="fake.wav")
        self.assertEqual(result["status"], "error")
        self.assertTrue(any("stt_error" in e for e in result["errors"]))

    def test_intent_exception_returns_error(self):
        app = WANASOrchestrator(intent_service=RaisingIntent())
        result = app.process(text="Tell me about him")
        self.assertEqual(result["status"], "error")
        self.assertTrue(any("intent_error" in e for e in result["errors"]))

    def test_memory_exception_returns_safe_fallback(self):
        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY"),
            memory_retriever=RaisingMemory(),
        )
        result = app.process(
            text="Tell me about him",
            face_result=KNOWN_FACE,
        )
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["message"], FALLBACK_MESSAGE)
        self.assertTrue(any("memory_error" in e for e in result["errors"]))

    def test_unimplemented_intent_is_explicit(self):
        app = WANASOrchestrator(
            intent_service=MockIntentService("HELP")
        )
        result = app.process(text="Help me")
        self.assertEqual(result["status"], "not_implemented")

    def test_unknown_person_does_not_call_memory_retriever(self):
        memory = MockMemoryRetriever()
        app = WANASOrchestrator(
            intent_service=MockIntentService("GET_MEMORY"),
            memory_retriever=memory,
        )
        result = app.process(
            text="Tell me about him",
            face_result={
                "person_id": None,
                "name": "Unknown",
                "relationship": None,
                "confidence": 0.1,
                "status": "unknown",
            },
        )
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["message"], FALLBACK_MESSAGE)
        self.assertEqual(memory.calls, [])


if __name__ == "__main__":
    unittest.main()
