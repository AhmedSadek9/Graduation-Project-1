
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

GROUND_TRUTH_PATH = (
    ROOT / "data" / "faces" / "ground_truth.xlsx"
)

ENROLLMENT_PATH = (
    ROOT / "data" / "faces" / "enrollment_embeddings_candidate.npz"
)

OUTPUT_PATH = (
    ROOT / "data" / "faces" / "recognition_results_candidate.csv"
)

THRESHOLD = 0.50


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_image_path(relative_path):

    relative_path = Path(
        str(relative_path).replace("/", "\\")
    )

    parts = relative_path.parts

    if (
        len(parts) >= 3
        and parts[0].lower() == "data"
        and parts[1].lower() == "faces"
    ):
        relative_path = Path(*parts[2:])

    return (
        ROOT
        / "data"
        / "faces"
        / "faces"
        / relative_path
    )


def normalize_embedding(embedding):

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def cosine_similarity(a, b):

    return float(np.dot(a, b))


def get_largest_face(faces):

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda face: (
            (face.bbox[2] - face.bbox[0])
            * (face.bbox[3] - face.bbox[1])
        )
    )


# ============================================================
# 3. LOAD INSIGHTFACE MODEL
# ============================================================

print("=" * 60)
print("LOADING INSIGHTFACE MODEL")
print("=" * 60)

ort.preload_dlls()

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace model loaded successfully.")


# ============================================================
# 4. LOAD CANDIDATE ENROLLMENT
# ============================================================

print("\n" + "=" * 60)
print("LOADING CANDIDATE ENROLLMENT")
print("=" * 60)

if not ENROLLMENT_PATH.exists():
    raise FileNotFoundError(
        f"Candidate enrollment file not found: {ENROLLMENT_PATH}"
    )

enrollment = np.load(ENROLLMENT_PATH)

person_ids = enrollment["person_ids"]
embeddings = enrollment["embeddings"]

embeddings = np.array(
    [normalize_embedding(e) for e in embeddings]
)

print(f"Number of enrolled persons: {len(person_ids)}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Threshold: {THRESHOLD}")


# ============================================================
# 5. LOAD GROUND TRUTH
# ============================================================

print("\n" + "=" * 60)
print("LOADING GROUND TRUTH")
print("=" * 60)

if not GROUND_TRUTH_PATH.exists():
    raise FileNotFoundError(
        f"Ground truth file not found: {GROUND_TRUTH_PATH}"
    )

df = pd.read_excel(GROUND_TRUTH_PATH)

# Correct column names from ground_truth.xlsx
required_columns = [
    "image_path",
    "true_person",
    "true_status",
    "condition"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in ground_truth.xlsx: {missing_columns}"
    )

# Exclude enrollment images
test_df = df[
    df["condition"].astype(str).str.lower() != "enrollment"
].copy()

print(f"Total test records: {len(test_df)}")


# ============================================================
# 6. FACE RECOGNITION
# ============================================================

print("\n" + "=" * 60)
print("STARTING CANDIDATE RECOGNITION TEST")
print("=" * 60)

results = []

for counter, (_, row) in enumerate(test_df.iterrows(), start=1):

    image_path = get_image_path(row["image_path"])

    # Correct Ground Truth column names
    true_person = str(row["true_person"]).strip()
    true_status = str(row["true_status"]).strip().lower()
    condition = str(row["condition"]).strip()

    result = {
        "true_person": true_person,
        "true_status": true_status,
        "condition": condition,
        "image_path": str(image_path),
        "file_exists": image_path.exists(),
        "face_detected": False,
        "number_of_faces": 0,
        "selected_face_area": None,
        "best_match_person": None,
        "similarity": None,
        "predicted_person": None,
        "decision": None,
        "error": None
    }

    print(
        f"\n[{counter}/{len(test_df)}] "
        f"Processing: {image_path.name}"
    )

    # --------------------------------------------------------
    # Check image existence
    # --------------------------------------------------------

    if not image_path.exists():

        result["error"] = "Image not found"

        print("ERROR: Image not found.")

        results.append(result)
        continue

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:

        result["error"] = "Could not read image"

        print("ERROR: Could not read image.")

        results.append(result)
        continue

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = app.get(image)

    result["number_of_faces"] = len(faces)

    if len(faces) == 0:

        result["error"] = "No face detected"

        print("ERROR: No face detected.")

        results.append(result)
        continue

    result["face_detected"] = True

    # --------------------------------------------------------
    # Select largest face (same baseline strategy)
    # --------------------------------------------------------

    selected_face = get_largest_face(faces)

    x1, y1, x2, y2 = selected_face.bbox

    selected_area = (
        (x2 - x1) * (y2 - y1)
    )

    result["selected_face_area"] = float(selected_area)

    face_embedding = normalize_embedding(
        selected_face.embedding
    )

    # --------------------------------------------------------
    # Compare with all enrolled persons
    # --------------------------------------------------------

    similarities = []

    for enrolled_embedding in embeddings:

        similarity = cosine_similarity(
            face_embedding,
            enrolled_embedding
        )

        similarities.append(similarity)

    best_index = int(np.argmax(similarities))

    best_similarity = float(similarities[best_index])

    best_match_person = str(person_ids[best_index])

    result["best_match_person"] = best_match_person
    result["similarity"] = best_similarity

    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    if best_similarity >= THRESHOLD:

        predicted_person = best_match_person

        result["predicted_person"] = predicted_person
        result["decision"] = "Known"

    else:

        predicted_person = "Unknown"

        result["predicted_person"] = predicted_person
        result["decision"] = "Unknown"

    print(f"Faces detected: {len(faces)}")
    print(f"Selected face area: {selected_area:.2f}")
    print(f"Best match: {best_match_person}")
    print(f"Similarity: {best_similarity:.4f}")
    print(f"Prediction: {predicted_person}")

    results.append(result)


# ============================================================
# 7. SAVE RECOGNITION RESULTS
# ============================================================

results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nResults saved to:")
print(OUTPUT_PATH)


# ============================================================
# 8. EVALUATION METRICS
# ============================================================

print("\n" + "=" * 60)
print("CANDIDATE RECOGNITION EVALUATION")
print("=" * 60)

valid_df = results_df[
    results_df["error"].isna()
].copy()

known_df = valid_df[
    valid_df["true_status"] == "known"
].copy()

unknown_df = valid_df[
    valid_df["true_status"] == "unknown"
].copy()


# ------------------------------------------------------------
# Known people metrics
# ------------------------------------------------------------

known_total = len(known_df)

known_correct = (
    (
        known_df["predicted_person"]
        == known_df["true_person"]
    )
    & (
        known_df["decision"] == "Known"
    )
).sum()

known_wrong_identity = (
    (
        known_df["decision"] == "Known"
    )
    & (
        known_df["predicted_person"]
        != known_df["true_person"]
    )
).sum()

known_rejected = (
    known_df["decision"] == "Unknown"
).sum()

known_accuracy = (
    known_correct / known_total * 100
    if known_total > 0 else 0
)

frr = (
    known_rejected / known_total * 100
    if known_total > 0 else 0
)


# ------------------------------------------------------------
# Unknown people metrics
# ------------------------------------------------------------

unknown_total = len(unknown_df)

false_acceptances = (
    unknown_df["decision"] == "Known"
).sum()

correct_unknown_rejections = (
    unknown_df["decision"] == "Unknown"
).sum()

unknown_rejection_rate = (
    correct_unknown_rejections / unknown_total * 100
    if unknown_total > 0 else 0
)

far = (
    false_acceptances / unknown_total * 100
    if unknown_total > 0 else 0
)


# ------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------

valid_total = len(valid_df)

overall_correct = (
    known_correct + correct_unknown_rejections
)

overall_accuracy = (
    overall_correct / valid_total * 100
    if valid_total > 0 else 0
)

error_count = len(results_df) - valid_total


# ============================================================
# 9. PRINT FINAL REPORT
# ============================================================

print("\nKNOWN PEOPLE")
print("-" * 40)
print(f"Valid known images: {known_total}")
print(f"Correct identifications: {known_correct}")
print(f"Wrong identity matches: {known_wrong_identity}")
print(f"False rejections: {known_rejected}")
print(f"Known identification accuracy: {known_accuracy:.2f}%")
print(f"False Rejection Rate (FRR): {frr:.2f}%")

print("\nUNKNOWN PEOPLE")
print("-" * 40)
print(f"Valid unknown images: {unknown_total}")
print(f"Correct unknown rejections: {correct_unknown_rejections}")
print(f"False acceptances: {false_acceptances}")
print(f"Unknown rejection rate: {unknown_rejection_rate:.2f}%")
print(f"False Acceptance Rate (FAR): {far:.2f}%")

print("\nOVERALL")
print("-" * 40)
print(f"Total test records: {len(results_df)}")
print(f"Valid processed records: {valid_total}")
print(f"Error records: {error_count}")
print(f"Overall correct predictions: {overall_correct}")
print(f"Overall accuracy: {overall_accuracy:.2f}%")

print("\n" + "=" * 60)
print("CANDIDATE EVALUATION COMPLETED")
print("=" * 60)
