import json
import tempfile
import unittest
from pathlib import Path

from memory_retrieval import MemoryRetrievalSystem


class TestMemoryRetrieval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "memory_db.json"
        self.db_path.write_text(json.dumps([
            {
                "person_id": "P001",
                "fact": "This is Ahmed, your elder son who lives in Cairo.",
                "source": "Caregiver (Fatma)",
                "update_date": "2026-10-25"
            },
            {
                "person_id": "P002",
                "fact": "This is Mona, your sister who visits every Friday.",
                "source": "Caregiver (Fatma)",
                "update_date": "2026-10-26"
            }
        ]), encoding="utf-8")
        self.memory_sys = MemoryRetrievalSystem(str(self.db_path))

    def test_valid_person_retrieval(self):
        res = self.memory_sys.get_memory_by_person_id("P001")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["person_id"], "P001")
        self.assertIn("Ahmed", res["fact"])
        self.assertTrue(res["fact"])

    def test_cross_person_isolation(self):
        res_p1 = self.memory_sys.get_memory_by_person_id("P001")
        res_p2 = self.memory_sys.get_memory_by_person_id("P002")
        self.assertNotEqual(res_p1["fact"], res_p2["fact"])
        self.assertNotIn("Mona", res_p1["fact"])
        self.assertNotIn("Ahmed", res_p2["fact"])

    def test_missing_person(self):
        res = self.memory_sys.get_memory_by_person_id("P999")
        self.assertEqual(res["status"], "no-verified-information")
        self.assertIsNone(res["fact"])

    def test_unknown_person(self):
        res = self.memory_sys.get_memory_by_person_id("unknown")
        self.assertEqual(res["status"], "no-verified-information")
        self.assertIsNone(res["fact"])

    def test_none_person_id(self):
        res = self.memory_sys.get_memory_by_person_id(None)
        self.assertEqual(res["status"], "no-verified-information")
        self.assertIsNone(res["fact"])

    def test_empty_or_whitespace_person_id(self):
        for person_id in ("", "   "):
            with self.subTest(person_id=repr(person_id)):
                res = self.memory_sys.get_memory_by_person_id(person_id)
                self.assertEqual(res["status"], "no-verified-information")
                self.assertIsNone(res["fact"])

    def test_person_id_with_surrounding_spaces(self):
        res = self.memory_sys.get_memory_by_person_id(" P001 ")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["person_id"], "P001")

    def test_missing_database_file_raises_clear_error(self):
        missing_path = Path(self.temp_dir.name) / "does_not_exist.json"
        with self.assertRaisesRegex(FileNotFoundError, "Memory database was not found"):
            MemoryRetrievalSystem(str(missing_path))

    def test_invalid_json_raises_clear_error(self):
        self.db_path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            MemoryRetrievalSystem(str(self.db_path))

    def test_duplicate_person_ids_are_rejected(self):
        records = [
            {"person_id": "P001", "fact": "A", "source": "Caregiver", "update_date": "2026-10-25"},
            {"person_id": "P001", "fact": "B", "source": "Caregiver", "update_date": "2026-10-26"}
        ]
        self.db_path.write_text(json.dumps(records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate person_id"):
            MemoryRetrievalSystem(str(self.db_path))

    def test_empty_fact_is_rejected(self):
        records = [{"person_id": "P003", "fact": " ", "source": "Caregiver", "update_date": "2026-10-26"}]
        self.db_path.write_text(json.dumps(records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty or invalid fact"):
            MemoryRetrievalSystem(str(self.db_path))

    def test_database_must_be_a_list(self):
        self.db_path.write_text(json.dumps({"person_id": "P001"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must contain a JSON list"):
            MemoryRetrievalSystem(str(self.db_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
