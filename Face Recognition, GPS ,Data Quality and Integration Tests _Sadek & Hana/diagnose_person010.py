
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ==========================================
# 1. Paths
# ==========================================

ROOT = Path(__file__).resolve().parent

IMAGE_PATH = (
    ROOT / "data" / "faces" / "faces" /
    "person-010" / "person-010_test_lighting.jpg"
)

ENROLLMENT_FILE = (
    ROOT / "data" / "faces" / "enrollment_embeddings.npz"
)

THRESHOLD = 0.50


# ==========================================
# 2. Load Enrollment Database
# ==========================================

database = np.load(ENROLLMENT_FILE)

person_ids = database["person_ids"]
person_embeddings = database["embeddings"]


# ==========================================
# 3. Load InsightFace
# ==========================================

print("=" * 60)
print("WANAS - PERSON 010 MULTI-FACE DIAGNOSIS")
print("=" * 60)

ort.preload_dlls()

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("\nInsightFace loaded successfully!")


# ==========================================
# 4. Cosine Similarity
# ==========================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ==========================================
# 5. Read Image
# ==========================================

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise ValueError("Could not read image")

faces = app.get(image)

print("\nDetected faces:", len(faces))

if len(faces) == 0:
    raise ValueError("No faces detected")


# ==========================================
# 6. Analyze Every Face
# ==========================================

for index, face in enumerate(faces, start=1):

    embedding = np.asarray(
        face.embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        print(f"\nFace {index}: Invalid embedding")
        continue

    embedding /= norm

    similarities = [
        cosine_similarity(embedding, registered_embedding)
        for registered_embedding in person_embeddings
    ]

    best_index = int(np.argmax(similarities))

    best_person = str(person_ids[best_index])

    best_similarity = similarities[best_index]

    bbox = face.bbox.astype(int)

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1
    area = width * height

    print("\n" + "-" * 45)
    print(f"Face #{index}")

    print("Bounding Box:", bbox.tolist())
    print("Face Area:", area)
    print("Detection Score:", round(float(face.det_score), 4))

    print("Best Matching Identity:", best_person)
    print("Best Similarity:", round(best_similarity, 4))

    if best_similarity >= THRESHOLD:
        print("Decision: KNOWN")
    else:
        print("Decision: UNKNOWN")

    print("\nTop 3 Matches:")

    top_indices = np.argsort(similarities)[-3:][::-1]

    for rank, match_index in enumerate(top_indices, start=1):

        print(
            f"{rank}. {person_ids[match_index]} "
            f"-> {similarities[match_index]:.4f}"
        )


print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)
