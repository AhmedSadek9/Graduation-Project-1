
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ==========================================
# 1. Paths and Configuration
# ==========================================

ROOT = Path(__file__).resolve().parent

DATASET_ROOT = ROOT / "data" / "faces" / "faces"

GROUND_TRUTH = ROOT / "data" / "faces" / "ground_truth.xlsx"

ENROLLMENT_FILE = ROOT / "data" / "faces" / "enrollment_embeddings.npz"

OUTPUT_FILE = ROOT / "data" / "faces" / "recognition_results.csv"

# Initial experimental threshold, NOT final
THRESHOLD = 0.50


# ==========================================
# 2. Load Enrollment Database
# ==========================================

print("=" * 55)
print("WANAS - FACE RECOGNITION TEST")
print("=" * 55)

database = np.load(ENROLLMENT_FILE)

person_ids = database["person_ids"]
person_embeddings = database["embeddings"]

print("\nRegistered persons:", len(person_ids))
print("Embedding database shape:", person_embeddings.shape)


# ==========================================
# 3. Load InsightFace
# ==========================================

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
# 5. Read Test Images
# ==========================================

df = pd.read_excel(GROUND_TRUTH)

test_df = df[df["condition"] != "enrollment"].copy()

results = []

print("\nTest images:", len(test_df))


# ==========================================
# 6. Recognition Process
# ==========================================

for _, row in test_df.iterrows():

    original_path = str(row["image_path"])

    relative_path = Path(
        original_path.replace("data/faces/", "", 1)
    )

    image_path = DATASET_ROOT / relative_path

    true_person = row["true_person"]
    true_status = row["true_status"]
    condition = row["condition"]

    print("\nProcessing:", image_path.name)

    result = {
        "image_path": original_path,
        "true_person": true_person,
        "true_status": true_status,
        "condition": condition,
        "predicted_person": None,
        "similarity": None,
        "threshold": THRESHOLD,
        "recognition_status": None,
        "number_of_faces": 0,
        "error": None
    }

    try:

        if not image_path.is_file():
            raise FileNotFoundError("Image not found")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Cannot read image")

        faces = app.get(image)

        result["number_of_faces"] = len(faces)

        if len(faces) == 0:
            raise ValueError("No face detected")

        # Temporary policy: select largest detected face
        face = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
        )

        embedding = np.asarray(
            face.embedding,
            dtype=np.float32
        )

        embedding_norm = np.linalg.norm(embedding)

        if embedding_norm == 0:
            raise ValueError("Invalid zero embedding")

        embedding /= embedding_norm

        similarities = [
            cosine_similarity(embedding, registered_embedding)
            for registered_embedding in person_embeddings
        ]

        best_index = int(np.argmax(similarities))

        best_person = str(person_ids[best_index])

        best_similarity = similarities[best_index]

        if best_similarity >= THRESHOLD:

            predicted_person = best_person
            recognition_status = "KNOWN"

        else:

            predicted_person = "Unknown"
            recognition_status = "UNKNOWN"

        result["predicted_person"] = predicted_person
        result["similarity"] = best_similarity
        result["recognition_status"] = recognition_status

        print("True identity:", true_person)
        print("Predicted identity:", predicted_person)
        print("Similarity:", round(best_similarity, 4))
        print("Result:", recognition_status)

    except Exception as error:

        result["error"] = str(error)
        result["recognition_status"] = "ERROR"

        print("ERROR:", error)

    results.append(result)


# ==========================================
# 7. Save Results
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 55)
print("RECOGNITION TEST SUMMARY")
print("=" * 55)

# Exclude detection and file errors from recognition metrics
valid = results_df[
    results_df["recognition_status"] != "ERROR"
].copy()

known = valid[
    valid["true_status"] == "known"
]

unknown = valid[
    valid["true_status"] == "unknown"
]

# Correct known identity predictions
known_correct = (
    known["predicted_person"] == known["true_person"]
).sum()

# Correct rejection of unknown identities
unknown_correct = (
    unknown["predicted_person"] == "Unknown"
).sum()

# False Acceptance: unknown person accepted as registered
false_acceptances = (
    unknown["predicted_person"] != "Unknown"
).sum()

# False Rejection: known person rejected as unknown
false_rejections = (
    known["predicted_person"] == "Unknown"
).sum()

# Wrong identity: known person matched to another registered person
wrong_known_identity = (
    (known["predicted_person"] != known["true_person"]) &
    (known["predicted_person"] != "Unknown")
).sum()


print("Total test records:", len(test_df))
print("Successful predictions:", len(valid))
print("Errors:", len(test_df) - len(valid))

print("\n--- Known Person Evaluation ---")

print("Known test images:", len(known))
print("Correct known predictions:", known_correct)

if len(known) > 0:

    known_accuracy = known_correct / len(known) * 100

    false_rejection_rate = (
        false_rejections / len(known)
    ) * 100

    print(
        "Known Identification Accuracy:",
        round(known_accuracy, 2), "%"
    )

    print(
        "False Rejection Rate (FRR):",
        round(false_rejection_rate, 2), "%"
    )


print("\n--- Unknown Person Evaluation ---")

print("Unknown test images:", len(unknown))
print("Correct unknown rejections:", unknown_correct)

if len(unknown) > 0:

    unknown_rejection_rate = (
        unknown_correct / len(unknown)
    ) * 100

    false_acceptance_rate = (
        false_acceptances / len(unknown)
    ) * 100

    print(
        "Unknown Rejection Rate:",
        round(unknown_rejection_rate, 2), "%"
    )

    print(
        "False Acceptance Rate (FAR):",
        round(false_acceptance_rate, 2), "%"
    )


print("\n--- Overall Evaluation ---")

if len(valid) > 0:

    total_correct = known_correct + unknown_correct

    overall_accuracy = (
        total_correct / len(valid)
    ) * 100

    print(
        "Total correct predictions:",
        total_correct
    )

    print(
        "Overall Accuracy:",
        round(overall_accuracy, 2), "%"
    )

print("\nWrong known identity matches:", wrong_known_identity)

print("\n--- Errors ---")

errors = results_df[
    results_df["recognition_status"] == "ERROR"
]

if len(errors) > 0:

    for _, error_row in errors.iterrows():

        print(
            error_row["image_path"],
            "->",
            error_row["error"]
        )

else:

    print("No errors")

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nRecognition test completed!")
