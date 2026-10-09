
import json
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from .config import (
    DATA_DIR,
    FACES_DIR,
    EMBEDDINGS_FILE,
    MODEL_NAME,
    MODEL_PROVIDER,
    MIN_FACE_SIZE,
    IMAGE_EXTENSIONS,
)


class FaceEnrollment:
    """Register people and save their original images and embeddings."""

    def __init__(self):
        ort.preload_dlls(directory="")

        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=[MODEL_PROVIDER, "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        self.embeddings_file = EMBEDDINGS_FILE

    def _load_data(self):
        if not self.embeddings_file.exists():
            return {}

        with open(self.embeddings_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Invalid face embeddings JSON format.")

        return data

    def enroll(self, image, person_id, name, relationship):
        """
        Register one person from an image containing exactly one face.
        The person must consent to having their face image stored.
        """
        if image is None:
            raise ValueError("Image is empty or could not be loaded.")

        person_id = str(person_id or "").strip()
        name = str(name or "").strip()
        relationship = str(relationship or "").strip()

        if not person_id or not name or not relationship:
            raise ValueError(
                "person_id, name, and relationship are required."
            )

        # Restrict IDs to safe folder-name characters.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", person_id):
            raise ValueError(
                "person_id may contain only letters, numbers, _ and -."
            )

        faces = self.app.get(image)

        if len(faces) == 0:
            return {
                "success": False,
                "message": "No face detected in the image.",
                "person_id": person_id,
            }

        if len(faces) > 1:
            return {
                "success": False,
                "message": "Multiple faces detected; use a single-face image.",
                "person_id": person_id,
            }

        face = faces[0]
        bbox = face.bbox.astype(int)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            return {
                "success": False,
                "message": "Face is too small; use a clearer image.",
                "person_id": person_id,
            }

        embedding = face.normed_embedding.astype(np.float32).tolist()
        data = self._load_data()

        if person_id in data:
            person_data = data[person_id]

            if (
                person_data.get("name") != name
                or person_data.get("relationship") != relationship
            ):
                raise ValueError(
                    "This ID already belongs to different person details."
                )
        else:
            person_data = {
                "name": name,
                "relationship": relationship,
                "embeddings": [],
                "images": [],
            }
            data[person_id] = person_data

        # Prepare the person's image folder.
        person_dir = FACES_DIR / person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        image_number = len(person_data.get("images", [])) + 1
        image_path = person_dir / f"image_{image_number}.jpg"

        # Refuse to overwrite an existing image.
        while image_path.exists():
            image_number += 1
            image_path = person_dir / f"image_{image_number}.jpg"

        # Save the original image as JPEG.
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"Could not save image: {image_path}")

        person_data.setdefault("embeddings", []).append(embedding)
        person_data.setdefault("images", []).append(
            str(image_path.relative_to(DATA_DIR))
        )

        temporary_file = self.embeddings_file.with_suffix(".tmp")

        try:
            with open(temporary_file, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)

            temporary_file.replace(self.embeddings_file)

        except Exception:
            # Avoid leaving an orphaned image if saving metadata fails.
            if image_path.exists():
                image_path.unlink()
            raise

        return {
            "success": True,
            "message": "Person enrolled successfully.",
            "person_id": person_id,
            "name": name,
            "relationship": relationship,
            "image_path": str(image_path),
            "embeddings_count": len(person_data["embeddings"]),
        }
