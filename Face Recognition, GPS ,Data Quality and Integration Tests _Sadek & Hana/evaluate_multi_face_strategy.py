
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
    ROOT / "data" / "faces" / "multi_face_strategy_results.csv"
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


def calculate_similarity(face_embedding, prototypes):

    similarities = np.dot(
        prototypes,
        face_embedding
    )

    best_index = int(np.argmax(similarities))

    return (
        str(person_ids[best_index]),
        float(similarities[best_index])
    )


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("=" * 65)
print("LOADING INSIGHTFACE MODEL")
print("=" * 65)

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

print("Model loaded successfully.")


# ============================================================
# 4. LOAD CANDIDATE ENROLLMENT
# ============================================================

print("\n" + "=" * 65)
print("LOADING CANDIDATE ENROLLMENT")
print("=" * 65)

enrollment = np.load(ENROLLMENT_PATH)

person_ids = enrollment["person_ids"].astype(str)

prototypes = np.array([
    normalize_embedding(e)
    for e in enrollment["embeddings"]
])

print("Enrolled persons:", len(person_ids))
print("Prototype shape:", prototypes.shape)
print("Threshold:", THRESHOLD)


# ============================================================
# 5. LOAD TEST DATA
# ============================================================

print("\n" + "=" * 65)
print("LOADING GROUND TRUTH")
print("=" * 65)

df = pd.read_excel(GROUND_TRUTH_PATH)

test_df = df[
    df["condition"].astype(str).str.lower()
    != "enrollment"
].copy()

print("Test records:", len(test_df))


# ============================================================
# 6. PROCESS EVERY FACE IN EVERY IMAGE
# ============================================================

all_face_results = []
image_results = []

print("\n" + "=" * 65)
print("PROCESSING ALL DETECTED FACES")
print("=" * 65)

for counter, (_, row) in enumerate(
    test_df.iterrows(),
    start=1
):

    image_path = get_image_path(row["image_path"])

    true_person = str(row["true_person"]).strip()
    true_status = str(row["true_status"]).strip().lower()
    condition = str(row["condition"]).strip()

    print(
        f"\n[{counter}/{len(test_df)}] "
        f"{image_path.name}"
    )

    image_result = {
        "image_path": str(row["image_path"]),
        "true_person": true_person,
        "true_status": true_status,
        "condition": condition,
        "number_of_faces": 0,
        "largest_face_prediction": None,
        "largest_face_similarity": None,
        "highest_score_prediction": None,
        "highest_score_similarity": None,
        "any_face_matches_known_person": False,
        "any_face_accepted": False,
        "error": None
    }

    if not image_path.exists():

        image_result["error"] = "Image not found"

        image_results.append(image_result)
        print("Image not found.")
        continue

    image = cv2.imread(str(image_path))

    if image is None:

        image_result["error"] = "Could not read image"

        image_results.append(image_result)
        print("Could not read image.")
        continue

    faces = app.get(image)

    image_result["number_of_faces"] = len(faces)

    if len(faces) == 0:

        image_result["error"] = "No face detected"

        image_results.append(image_result)
        print("No face detected.")
        continue

    # --------------------------------------------------------
    # Evaluate every detected face
    # --------------------------------------------------------

    face_records = []

    for face_number, face in enumerate(faces, start=1):

        x1, y1, x2, y2 = face.bbox

        face_area = float(
            (x2 - x1) * (y2 - y1)
        )

        detection_score = float(face.det_score)

        face_embedding = normalize_embedding(
            face.embedding
        )

        best_match, best_similarity = calculate_similarity(
            face_embedding,
            prototypes
        )

        prediction = (
            best_match
            if best_similarity >= THRESHOLD
            else "Unknown"
        )

        correct_identity = (
            true_status == "known"
            and prediction == true_person
        )

        face_record = {
            "image_path": str(row["image_path"]),
            "true_person": true_person,
            "true_status": true_status,
            "condition": condition,
            "face_number": face_number,
            "face_area": face_area,
            "detection_score": detection_score,
            "best_match": best_match,
            "best_similarity": best_similarity,
            "prediction": prediction,
            "correct_identity": correct_identity
        }

        face_records.append(face_record)
        all_face_results.append(face_record)

        print(
            f"Face #{face_number} | "
            f"Area: {face_area:.0f} | "
            f"Score: {detection_score:.4f} | "
            f"Match: {best_match} | "
            f"Similarity: {best_similarity:.4f} | "
            f"Prediction: {prediction}"
        )

    # --------------------------------------------------------
    # Strategy A: Largest face
    # --------------------------------------------------------

    largest_face = max(
        face_records,
        key=lambda x: x["face_area"]
    )

    image_result["largest_face_prediction"] = (
        largest_face["prediction"]
    )

    image_result["largest_face_similarity"] = (
        largest_face["best_similarity"]
    )

    # --------------------------------------------------------
    # Strategy B: Highest detection score
    # --------------------------------------------------------

    highest_score_face = max(
        face_records,
        key=lambda x: x["detection_score"]
    )

    image_result["highest_score_prediction"] = (
        highest_score_face["prediction"]
    )

    image_result["highest_score_similarity"] = (
        highest_score_face["best_similarity"]
    )

    # --------------------------------------------------------
    # Strategy C: Evaluate all faces independently
    # --------------------------------------------------------

    image_result["any_face_accepted"] = any(
        face["prediction"] != "Unknown"
        for face in face_records
    )

    image_result["any_face_matches_known_person"] = any(
        face["prediction"] == true_person
        for face in face_records
    ) if true_status == "known" else False

    image_results.append(image_result)


# ============================================================
# 7. SAVE FACE-LEVEL AND IMAGE-LEVEL RESULTS
# ============================================================

face_df = pd.DataFrame(all_face_results)

image_df = pd.DataFrame(image_results)

face_output = (
    ROOT / "data" / "faces"
    / "multi_face_all_faces_results.csv"
)

face_df.to_csv(
    face_output,
    index=False,
    encoding="utf-8-sig"
)

image_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nFace-level results saved:", face_output)
print("Image-level results saved:", OUTPUT_PATH)


# ============================================================
# 8. IMAGE-LEVEL EVALUATION
# ============================================================

valid_df = image_df[
    image_df["error"].isna()
].copy()

known_df = valid_df[
    valid_df["true_status"] == "known"
].copy()

unknown_df = valid_df[
    valid_df["true_status"] == "unknown"
].copy()


def evaluate_single_prediction(data, prediction_column):

    if len(data) == 0:
        return 0, 0, 0

    correct = (
        data[prediction_column] == data["true_person"]
    ).sum()

    rejected = (
        data[prediction_column] == "Unknown"
    ).sum()

    wrong_identity = (
        (data[prediction_column] != "Unknown")
        & (data[prediction_column] != data["true_person"])
    ).sum()

    return int(correct), int(rejected), int(wrong_identity)


# ============================================================
# 9. COMPARE STRATEGIES
# ============================================================

print("\n" + "=" * 65)
print("STRATEGY COMPARISON")
print("=" * 65)

strategies = {
    "Largest Face": "largest_face_prediction",
    "Highest Detection Score": "highest_score_prediction"
}

for strategy_name, prediction_column in strategies.items():

    known_correct, known_rejected, known_wrong = (
        evaluate_single_prediction(
            known_df,
            prediction_column
        )
    )

    unknown_accepted = (
        unknown_df[prediction_column] != "Unknown"
    ).sum()

    unknown_rejected = (
        unknown_df[prediction_column] == "Unknown"
    ).sum()

    known_total = len(known_df)
    unknown_total = len(unknown_df)

    known_accuracy = (
        known_correct / known_total * 100
        if known_total else 0
    )

    frr = (
        known_rejected / known_total * 100
        if known_total else 0
    )

    far = (
        unknown_accepted / unknown_total * 100
        if unknown_total else 0
    )

    print(f"\n{strategy_name}")
    print("-" * 45)
    print(f"Known correct: {known_correct}/{known_total}")
    print(f"Known wrong identity: {known_wrong}")
    print(f"Known rejected: {known_rejected}")
    print(f"Known accuracy: {known_accuracy:.2f}%")
    print(f"FRR: {frr:.2f}%")
    print(f"Unknown correctly rejected: {unknown_rejected}/{unknown_total}")
    print(f"Unknown accepted: {unknown_accepted}")
    print(f"FAR: {far:.2f}%")


# ============================================================
# 10. ALL-FACES ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("ALL-FACES ANALYSIS")
print("=" * 65)

known_with_face = known_df[
    known_df["true_status"] == "known"
]

unknown_with_face = unknown_df[
    unknown_df["true_status"] == "unknown"
]

known_any_correct = (
    known_with_face["any_face_matches_known_person"]
).sum()

unknown_any_accepted = (
    unknown_with_face["any_face_accepted"]
).sum()

print(
    "Known images with at least one correctly identified face:",
    f"{known_any_correct}/{len(known_with_face)}"
)

print(
    "Unknown images with at least one accepted face:",
    f"{unknown_any_accepted}/{len(unknown_with_face)}"
)

print("\nIMPORTANT:")
print(
    "All-faces analysis is diagnostic only. "
    "It is not a final recognition strategy."
)

print("\n" + "=" * 65)
print("MULTI-FACE STRATEGY EVALUATION COMPLETED")
print("=" * 65)
