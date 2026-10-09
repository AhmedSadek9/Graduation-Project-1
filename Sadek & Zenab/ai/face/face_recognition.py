
import json

import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from .config import (
    EMBEDDINGS_FILE,
    MODEL_NAME,
    MODEL_PROVIDER,
    SIMILARITY_THRESHOLD,
    MIN_FACE_SIZE,
    STATUS_KNOWN,
    STATUS_UNKNOWN,
    STATUS_LOW_CONFIDENCE,
)

# Provisional threshold for clearly low similarity.
# Validate this value against project evaluation data.
UNKNOWN_THRESHOLD = 0.50


class FaceRecognizer:
    """Detect faces, extract embeddings, and identify enrolled people."""

    def __init__(self):
        # Load CUDA and cuDNN libraries before creating model sessions.
        ort.preload_dlls(directory="")

        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=[MODEL_PROVIDER, "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.embeddings = {}
        self.load_embeddings()

    def load_embeddings(self):
        """Load enrolled face embeddings from JSON."""
        if not EMBEDDINGS_FILE.exists():
            self.embeddings = {}
            return

        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Invalid face embeddings JSON format.")

        self.embeddings = data

    def recognize(self, image):
        """
        Recognize faces in an image.

        Args:
            image: An image loaded with OpenCV (BGR format).

        Returns:
            A list of dictionaries containing identity information.
        """
        if image is None:
            raise ValueError("Image is empty or could not be loaded.")

        faces = self.app.get(image)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                continue

            embedding = face.normed_embedding.astype(np.float32)

            best_person = None
            best_similarity = -1.0

            for person_id, person_data in self.embeddings.items():
                person_embeddings = person_data.get("embeddings", [])

                for saved_embedding in person_embeddings:
                    saved = np.asarray(
                        saved_embedding,
                        dtype=np.float32,
                    )

                    if saved.shape != embedding.shape:
                        continue

                    similarity = float(np.dot(embedding, saved))

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_person = (person_id, person_data)

            if best_person is None:
                results.append({
                    "person_id": None,
                    "name": "Unknown",
                    "relationship": None,
                    "confidence": 0.0,
                    "status": STATUS_UNKNOWN,
                    "bbox": bbox.tolist(),
                })
                continue

            person_id, person_data = best_person

            if best_similarity >= SIMILARITY_THRESHOLD:
                status = STATUS_KNOWN
                name = person_data.get("name", "Unknown")
                relationship = person_data.get("relationship")
                returned_id = person_id

            elif best_similarity < UNKNOWN_THRESHOLD:
                status = STATUS_UNKNOWN
                name = "Unknown"
                relationship = None
                returned_id = None

            else:
                status = STATUS_LOW_CONFIDENCE
                name = "Unknown"
                relationship = None
                returned_id = None

            results.append({
                "person_id": returned_id,
                "name": name,
                "relationship": relationship,
                "confidence": max(0.0, best_similarity),
                "status": status,
                "bbox": bbox.tolist(),
            })

        return results
