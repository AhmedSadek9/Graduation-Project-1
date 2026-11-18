
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    BASE_DIR
    / "data"
    / "faces"
    / "enrollment_embeddings_candidate.npz"
)

MODEL_NAME = "buffalo_l"

DETECTION_SIZE = (640, 640)

SIMILARITY_THRESHOLD = 0.50

MIN_FACE_SIZE = 20

MODEL_PROVIDERS = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


# =========================================================
# Face Recognition Service
# =========================================================

class FaceRecognitionService:

    def __init__(self):

        self.face_app = None

        self.person_ids = None
        self.person_embeddings = None

        self.initialized = False

    # -----------------------------------------------------
    # Initialize model and load enrolled embeddings
    # -----------------------------------------------------

    def initialize(self):

        if self.initialized:
            return

        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {EMBEDDINGS_PATH}"
            )

        # Load CUDA libraries
        ort.preload_dlls()

        # Initialize InsightFace
        self.face_app = FaceAnalysis(
            name=MODEL_NAME,
            providers=MODEL_PROVIDERS
        )

        self.face_app.prepare(
            ctx_id=0,
            det_size=DETECTION_SIZE
        )

        # Load enrolled people
        data = np.load(EMBEDDINGS_PATH)

        self.person_ids = data["person_ids"].astype(str)

        self.person_embeddings = data["embeddings"].astype(
            np.float32
        )

        # Validate embeddings
        if self.person_embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if self.person_embeddings.shape[1] != 512:
            raise ValueError(
                "Expected embedding dimension: 512"
            )

        if len(self.person_ids) != len(self.person_embeddings):
            raise ValueError(
                "Person IDs and embeddings count mismatch."
            )

        if not np.isfinite(self.person_embeddings).all():
            raise ValueError(
                "Embeddings contain NaN or Inf values."
            )

        # Normalize embeddings
        norms = np.linalg.norm(
            self.person_embeddings,
            axis=1,
            keepdims=True
        )

        if np.any(norms <= 1e-12):
            raise ValueError(
                "One or more embeddings have zero norm."
            )

        self.person_embeddings = (
            self.person_embeddings / norms
        )

        self.initialized = True

        print("Face Recognition Service initialized.")
        print(f"Loaded persons: {len(self.person_ids)}")
        print(f"Model: {MODEL_NAME}")
        print(f"Available providers: {ort.get_available_providers()}")

    # -----------------------------------------------------
    # Select face with highest detection score
    # -----------------------------------------------------

    def select_best_face(self, faces: list) -> Any:

        if not faces:
            return None

        valid_faces = []

        for face in faces:

            bbox = face.bbox

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                continue

            valid_faces.append(face)

        if not valid_faces:
            return None

        # Highest detection confidence
        best_face = max(
            valid_faces,
            key=lambda face: float(face.det_score)
        )

        return best_face

    # -----------------------------------------------------
    # Extract normalized face embedding
    # -----------------------------------------------------

    @staticmethod
    def extract_embedding(face) -> np.ndarray:

        embedding = np.asarray(
            face.normed_embedding,
            dtype=np.float32
        )

        if embedding.shape != (512,):
            raise ValueError(
                "Invalid face embedding dimension."
            )

        if not np.isfinite(embedding).all():
            raise ValueError(
                "Face embedding contains NaN or Inf."
            )

        norm = np.linalg.norm(embedding)

        if norm <= 1e-12:
            raise ValueError(
                "Face embedding has zero norm."
            )

        return embedding / norm

    # -----------------------------------------------------
    # Calculate cosine similarity
    # -----------------------------------------------------

    def find_best_match(
        self,
        face_embedding: np.ndarray
    ) -> dict:

        similarities = np.dot(
            self.person_embeddings,
            face_embedding
        )

        best_index = int(
            np.argmax(similarities)
        )

        best_similarity = float(
            similarities[best_index]
        )

        matched_person = str(
            self.person_ids[best_index]
        )

        if best_similarity >= SIMILARITY_THRESHOLD:

            return {
                "person_id": matched_person,
                "similarity": best_similarity,
                "status": "recognized"
            }

        return {
            "person_id": None,
            "similarity": best_similarity,
            "status": "unknown"
        }

    # -----------------------------------------------------
    # Main recognition function
    # -----------------------------------------------------

    def recognize(self, image: np.ndarray) -> dict:

        if not self.initialized:
            self.initialize()

        if image is None or not isinstance(image, np.ndarray):
            raise ValueError(
                "Invalid image."
            )

        if image.size == 0:
            raise ValueError(
                "Empty image."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Expected a 3-channel BGR image."
            )

        if image.dtype != np.uint8:
            raise ValueError(
                "Expected uint8 image."
            )

        # Detect all faces
        faces = self.face_app.get(image)

        if not faces:

            return {
                "person_id": None,
                "similarity": None,
                "status": "no_face_detected",
                "detection_score": None,
                "faces_detected": 0
            }

        # Select highest detection score
        best_face = self.select_best_face(faces)

        if best_face is None:

            return {
                "person_id": None,
                "similarity": None,
                "status": "no_valid_face",
                "detection_score": None,
                "faces_detected": len(faces)
            }

        # Extract embedding
        embedding = self.extract_embedding(best_face)

        # Compare with enrolled persons
        result = self.find_best_match(embedding)

        result["detection_score"] = float(
            best_face.det_score
        )

        result["faces_detected"] = len(faces)

        return result


# =========================================================
# Global service instance
# =========================================================

face_service = FaceRecognitionService()
