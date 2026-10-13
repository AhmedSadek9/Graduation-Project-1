import unittest
from mocks.mock_services import (
    MockFaceService, MockSTTService, MockIntentService, MockMemoryRetriever,
)


class MockContractTests(unittest.TestCase):
    def test_face_mock_returns_list_of_results(self):
        result = MockFaceService().recognize(None)
        self.assertIsInstance(result, list)
        self.assertTrue(result)
        self.assertIn("person_id", result[0])
        self.assertIn("status", result[0])

    def test_stt_mock_returns_transcript(self):
        self.assertEqual(MockSTTService().transcribe("fake.wav"), "Tell me about him")

    def test_intent_mock_returns_required_fields(self):
        result = MockIntentService().classify("Tell me about him")
        self.assertEqual(result["intent"], "GET_MEMORY")
        self.assertIn("confidence", result)

    def test_memory_mock_records_person_and_query(self):
        retriever = MockMemoryRetriever()
        result = retriever("person-001", "Tell me about him")
        self.assertTrue(result["verified"])
        self.assertEqual(retriever.calls[0]["person_id"], "person-001")
        self.assertEqual(retriever.calls[0]["query"], "Tell me about him")


if __name__ == "__main__":
    unittest.main()
