
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ==========================================
# 1. Project Paths
# ==========================================

ROOT = Path(__file__).resolve().parent

DATASET_ROOT = ROOT / "data" / "november" / "faces_new" / "faces"

OUTPUT_DIR = ROOT / "data" / "november"

# Save candidate files without overwriting original enrollment
OUTPUT_FILE = OUTPUT_DIR / "enrollment_embeddings_candidate.npz"
SUMMARY_FILE = OUTPUT_DIR / "enrollment_summary_candidate.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load InsightFace Model
# ==========================================

print("=" * 55)
print("WANAS - FACE ENROLLMENT")
print("=" * 55)

ort.preload_dlls()

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("\nInsightFace model loaded successfully!")


# ==========================================
# 3. Helper Functions
# ==========================================

def normalize_embedding(embedding):
    """Normalize a face embedding to unit length."""

    embedding = np.asarray(embedding, dtype=np.float32)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Invalid zero embedding")

    return embedding / norm


def calculate_face_area(face):
    """Calculate bounding-box area."""

    x1, y1, x2, y2 = face.bbox

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def select_largest_face(faces):
    """Select the face with the largest bounding-box area."""

    return max(
        faces,
        key=calculate_face_area
    )


def select_face_for_image(faces, image_name):
    """
    Select the intended face.

    Special case:
    person-001_enroll_02.jpg -> Face #1

    All other images:
    Select the largest face.
    """

    if image_name == "person-001_enroll_02.jpg":

        selected_index = 0  # Face #1

        if selected_index >= len(faces):
            raise ValueError(
                "Expected Face #1 was not detected"
            )

        selected_face = faces[selected_index]

        print("Selection policy: October baseline override (first detection)")
        print("Selected Face: #1")
        print(
            "Selected Face Area:",
            calculate_face_area(selected_face)
        )

        return selected_face

    selected_face = select_largest_face(faces)

    print("Selection policy: Largest bounding-box area")
    print(
        "Selected Face Area:",
        calculate_face_area(selected_face)
    )

    return selected_face


# ==========================================
# 4. Enrollment Process
# ==========================================

person_ids = []
person_embeddings = []
raw_embeddings = []
summary = []

for person_number in range(1, 21):

    person_id = f"person-{person_number:03d}"

    person_folder = DATASET_ROOT / person_id

    print("\n" + "-" * 45)
    print("Enrolling:", person_id)

    current_embeddings = []
    successful_images = 0

    for image_number in range(1, 3):

        image_name = (
            f"{person_id}_enroll_{image_number:02d}.jpg"
        )

        image_path = person_folder / image_name

        print("\nProcessing:", image_name)

        try:

            if not image_path.is_file():
                raise FileNotFoundError("Image not found")

            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError("Cannot read image")

            faces = app.get(image)

            if len(faces) == 0:
                raise ValueError("No face detected")

            print("Faces detected:", len(faces))

            # Select intended face
            selected_face = select_face_for_image(
                faces,
                image_name
            )

            embedding = normalize_embedding(
                selected_face.embedding
            )

            current_embeddings.append(embedding)

            successful_images += 1

            print(
                "Selected face score:",
                float(selected_face.det_score)
            )

            print("Embedding shape:", embedding.shape)
            print("Enrollment image successful!")

        except Exception as error:

            print("ERROR:", error)

    # Register only if both enrollment images succeeded
    if successful_images == 2:

        # Average the two normalized embeddings
        average_embedding = np.mean(
            current_embeddings,
            axis=0
        )

        # Normalize final person representation
        average_embedding = normalize_embedding(
            average_embedding
        )

        person_ids.append(person_id)

        person_embeddings.append(
            average_embedding
        )

        raw_embeddings.append(
            current_embeddings
        )

        status = "SUCCESS"

        print("\n", person_id, "enrolled successfully!")

    else:

        status = "INCOMPLETE"

        print("\n", person_id, "enrollment incomplete!")

    summary.append({
        "person_id": person_id,
        "successful_images": successful_images,
        "required_images": 2,
        "status": status
    })


# ==========================================
# 5. Save Candidate Enrollment Embeddings
# ==========================================

if person_embeddings:

    prototypes = np.stack(person_embeddings)

    individual_embeddings = np.stack(raw_embeddings)

    np.savez_compressed(
        OUTPUT_FILE,
        person_ids=np.array(person_ids),
        embeddings=prototypes,
        individual_embeddings=individual_embeddings
    )

    print("\nCandidate embeddings saved successfully!")

else:

    print("\nNo valid enrollment records were generated!")


# ==========================================
# 6. Save Enrollment Summary
# ==========================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

print("\n" + "=" * 55)
print("ENROLLMENT SUMMARY")
print("=" * 55)

print(summary_df.to_string(index=False))

print("\nSuccessfully enrolled:", len(person_ids))
print("Total known persons:", 20)

print("\nCandidate files:")

if person_embeddings:
    print(OUTPUT_FILE)

print(SUMMARY_FILE)

print("\nEnrollment process completed!")
