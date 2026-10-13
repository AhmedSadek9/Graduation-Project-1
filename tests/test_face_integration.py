import unittest

from core.context import SystemContext
from core.orchestrator import WANASOrchestrator


KNOWN_FACE = {
    "person_id": "person-001",
    "name": "Ahmed",
    "relationship": "Son",
    "confidence": 0.91,
    "status": "known",
    "bbox": [10, 20, 100, 120],
}

UNKNOWN_FACE = {
    "person_id": None,
    "name": "Unknown",
    "relationship": None,
    "confidence": 0.2,
    "status": "unknown",
    "bbox": [10, 20, 100, 120],
}


class FakeFaceService:
    def __init__(self, results):
        self.results = results

    def recognize(self, image):
        return self.results


class FaceIntegrationTests(unittest.TestCase):

    def test_known_face_updates_context(self):
        context = SystemContext()
        face_service = FakeFaceService([KNOWN_FACE])

        app = WANASOrchestrator(
            face_service=face_service,
            context=context,
        )

        result = app.process(image=object())

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["context"]["current_person"]["person_id"],
            "person-001",
        )
        self.assertEqual(
            result["face"]["name"],
            "Ahmed",
        )

    def test_no_faces_clears_previous_identity(self):
        context = SystemContext()
        context.update_person(KNOWN_FACE)

        face_service = FakeFaceService([])

        app = WANASOrchestrator(
            face_service=face_service,
            context=context,
        )

        result = app.process(image=object())

        self.assertIsNone(result["context"]["current_person"])
        self.assertEqual(result["face"]["status"], "unknown")

    def test_unknown_face_clears_previous_identity(self):
        context = SystemContext()
        context.update_person(KNOWN_FACE)

        face_service = FakeFaceService([UNKNOWN_FACE])

        app = WANASOrchestrator(
            face_service=face_service,
            context=context,
        )

        result = app.process(image=object())

        self.assertIsNone(result["context"]["current_person"])
        self.assertIsNone(result["face"]["person_id"])
        self.assertEqual(result["face"]["status"], "unknown")

    def test_orchestrator_uses_first_face_result(self):
        second_face = {
            **KNOWN_FACE,
            "person_id": "person-002",
            "name": "Mona",
        }

        face_service = FakeFaceService([
            KNOWN_FACE,
            second_face,
        ])

        app = WANASOrchestrator(
            face_service=face_service,
        )

        result = app.process(image=object())

        self.assertEqual(
            result["face"]["person_id"],
            "person-001",
        )


if __name__ == "__main__":
    unittest.main()