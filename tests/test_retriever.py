
import unittest

from ai.memory.database import get_connection, initialize_database
from ai.memory.memory import add_memory
from ai.memory.retriever import retrieve_memory, FALLBACK_MESSAGE


class TestMemoryRetriever(unittest.TestCase):

    def setUp(self):
        initialize_database()

        # Remove only records belonging to these test IDs.
        with get_connection() as connection:
            connection.execute(
                "DELETE FROM memories WHERE person_id IN (?, ?)",
                ("test-person-001", "test-person-002"),
            )

        add_memory(
            "test-person-001",
            "Likes listening to music",
            verified=True,
        )
        add_memory(
            "test-person-001",
            "Prefers tea in the morning",
            verified=False,
        )
        add_memory(
            "test-person-002",
            "Enjoys reading books",
            verified=True,
        )

    def tearDown(self):
        # Keep the test database clean.
        with get_connection() as connection:
            connection.execute(
                "DELETE FROM memories WHERE person_id IN (?, ?)",
                ("test-person-001", "test-person-002"),
            )

    def test_finds_verified_memory(self):
        result = retrieve_memory("test-person-001", "music")

        self.assertEqual(result["status"], "found")
        self.assertTrue(result["verified"])
        self.assertEqual(
            result["memory_text"],
            "Likes listening to music",
        )

    def test_does_not_return_unverified_memory(self):
        result = retrieve_memory("test-person-001", "tea")

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["memory_text"], FALLBACK_MESSAGE)
        self.assertFalse(result["verified"])

    def test_unknown_fact_returns_fallback(self):
        result = retrieve_memory("test-person-001", "football")

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["memory_text"], FALLBACK_MESSAGE)

    def test_unknown_person_returns_fallback(self):
        result = retrieve_memory("test-person-002", "music")

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["memory_text"], FALLBACK_MESSAGE)

    def test_missing_person_id(self):
        result = retrieve_memory(None, "music")

        self.assertEqual(result["status"], "no_person")
        self.assertEqual(result["memory_text"], FALLBACK_MESSAGE)


    
    def test_semantic_search_matches_different_wording(self):
        result = retrieve_memory(
            "test-person-001",
            "Enjoys hearing songs",
        )

        self.assertEqual(result["status"], "found")
        self.assertTrue(result["verified"])
        self.assertEqual(
            result["memory_text"],
            "Likes listening to music",
        )

    
    def test_semantic_search_does_not_return_unverified_memory(self):
        result = retrieve_memory(
            "test-person-001",
            "Enjoys drinking tea early in the day",
        )

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["memory_text"], FALLBACK_MESSAGE)
        self.assertFalse(result["verified"])

    
    def test_does_not_infer_missing_music_preference(self):
        result = retrieve_memory(
            "test-person-001",
            "What is their favorite music genre?",
        )
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["memory_text"], FALLBACK_MESSAGE)
        self.assertFalse(result["verified"])

    
    def test_nonexistent_person_returns_fallback(self):
        result = retrieve_memory(
            "person-999",
            "Enjoys hearing songs",
        )
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["memory_text"], FALLBACK_MESSAGE)
        self.assertFalse(result["verified"])
        self.assertEqual(result["retrieval_score"], 0.0)

    
    def test_fallback_message_is_exact_arabic_text(self):
        self.assertEqual(
            FALLBACK_MESSAGE,
            "لا تتوفر لدي معلومات موثقة كافية",
        )

    def test_unrelated_question_returns_fallback(self):
        result = retrieve_memory(
            "test-person-001",
            "What is their home address?",
        )

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(
            result["memory_text"],
            FALLBACK_MESSAGE,
        )
        self.assertFalse(result["verified"])



if __name__ == "__main__":
    unittest.main()
