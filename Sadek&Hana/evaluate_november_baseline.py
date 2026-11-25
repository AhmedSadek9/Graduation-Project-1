
from pathlib import Path

import json
from time import perf_counter

import numpy as np
import cv2
import pandas as pd

from app.services.face_recognition import FaceRecognitionService


class NovemberFaceService(FaceRecognitionService):
    """Observe the October service while keeping its selection and threshold."""

    def select_best_face(self, faces):
        self.trace["all_boxes"] = [face.bbox.tolist() for face in faces]
        selected = super().select_best_face(faces)
        if selected is not None:
            self.trace["selected_box"] = selected.bbox.tolist()
        return selected

    def find_best_match(self, face_embedding):
        result = super().find_best_match(face_embedding)
        scores = self.person_embeddings @ face_embedding
        seen = set()
        for index in np.argsort(-scores, kind="stable"):
            person = str(self.person_ids[index])
            if person in seen:
                continue
            seen.add(person)
            rank = len(seen)
            self.trace[f"top{rank}_id"] = person
            self.trace[f"top{rank}_similarity"] = float(scores[index])
            if rank == 2:
                break
        return result

    def recognize(self, image):
        self.trace = {"all_boxes": [], "selected_box": None}
        return super().recognize(image)


face_service = NovemberFaceService()


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "data" / "faces" / "faces"

GROUND_TRUTH_PATH = (
    BASE_DIR / "data" / "faces" / "ground_truth.xlsx"
)

OUTPUT_PATH = (
    BASE_DIR / "data" / "faces" / "november_baseline_diagnostics.csv"
)


# =========================================================
# IMAGE PATH RESOLUTION
# =========================================================

def resolve_image_path(image_path):

    normalized = str(image_path).replace("\\", "/")

    if normalized.startswith("data/faces/"):
        normalized = normalized.replace(
            "data/faces/",
            "data/faces/faces/",
            1
        )

    return BASE_DIR / normalized


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():

    print("=" * 65)
    print("WANAS FACE RECOGNITION - TEST ONLY EVALUATION")
    print("=" * 65)

    # -----------------------------------------------------
    # 1. Load Ground Truth
    # -----------------------------------------------------

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {GROUND_TRUTH_PATH}"
        )

    ground_truth = pd.read_excel(GROUND_TRUTH_PATH)

    required_columns = {
        "image_path",
        "true_person",
        "true_status",
        "condition"
    }

    missing_columns = required_columns - set(
        ground_truth.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print(f"\nOriginal records: {len(ground_truth)}")

    # -----------------------------------------------------
    # 2. Exclude Enrollment Images
    # -----------------------------------------------------

    ground_truth = ground_truth[
        ~ground_truth["image_path"]
        .astype(str)
        .str.contains("_enroll_", case=False, na=False)
    ].reset_index(drop=True)

    print(
        f"Test-only records: {len(ground_truth)}"
    )

    # -----------------------------------------------------
    # 3. Initialize Face Recognition Service
    # -----------------------------------------------------

    print("\nInitializing Face Recognition Service...")

    face_service.initialize()

    print("Service initialized successfully.")

    # -----------------------------------------------------
    # 4. Run Recognition on Test Images
    # -----------------------------------------------------

    results = []

    for index, row in ground_truth.iterrows():

        image_path = resolve_image_path(
            row["image_path"]
        )

        true_person = str(
            row["true_person"]
        ).strip()

        true_status = str(
            row["true_status"]
        ).strip().lower()

        condition = row["condition"]

        print("\n" + "-" * 65)

        print(
            f"[{index + 1}/{len(ground_truth)}] "
            f"{image_path.name}"
        )

        record = {
            "image_path": str(row["image_path"]),
            "true_person": true_person,
            "true_status": true_status,
            "condition": condition,

            "predicted_person": None,
            "predicted_status": None,

            "similarity": None,
            "detection_score": None,
            "faces_detected": 0,

            "correct": False,
            "all_boxes": None,
            "selected_box": None,
            "top1_id": None,
            "top1_similarity": None,
            "top2_id": None,
            "top2_similarity": None,
            "recognition_ms": None,
            "error": None
        }

        # Check image existence
        if not image_path.exists():

            record["error"] = "missing_image"

            results.append(record)

            print("ERROR: Image not found.")

            continue

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:

            record["error"] = "image_read_error"

            results.append(record)

            print("ERROR: Could not read image.")

            continue

        try:

            # Perform recognition
            started = perf_counter()
            try:
                prediction = face_service.recognize(image)
            finally:
                record["recognition_ms"] = (perf_counter() - started) * 1000
                record.update(face_service.trace)
                record["all_boxes"] = json.dumps(record["all_boxes"])
                if record["selected_box"] is not None:
                    record["selected_box"] = json.dumps(record["selected_box"])

            record["predicted_person"] = (
                prediction["person_id"]
            )

            record["predicted_status"] = (
                prediction["status"]
            )

            record["similarity"] = (
                prediction["similarity"]
            )

            record["detection_score"] = (
                prediction["detection_score"]
            )

            record["faces_detected"] = (
                prediction["faces_detected"]
            )

            predicted_status = prediction["status"]

            # ---------------------------------------------
            # Evaluate Known Faces
            # ---------------------------------------------

            if true_status == "known":

                record["correct"] = (
                    predicted_status == "recognized"
                    and prediction["person_id"] == true_person
                )

            # ---------------------------------------------
            # Evaluate Unknown Faces
            # ---------------------------------------------

            elif true_status == "unknown":

                record["correct"] = (
                    predicted_status == "unknown"
                )

            else:

                record["error"] = (
                    "unsupported_ground_truth_status"
                )

            print(
                f"Expected: {true_person} ({true_status})"
            )

            print(
                f"Predicted: {prediction['person_id']} "
                f"({predicted_status})"
            )

            print(
                f"Similarity: {prediction['similarity']}"
            )

            print(
                f"Detection Score: "
                f"{prediction['detection_score']}"
            )

            print(
                f"Faces Detected: "
                f"{prediction['faces_detected']}"
            )

            print(
                f"Correct: {record['correct']}"
            )

        except Exception as error:

            record["error"] = str(error)

            print(
                f"Recognition Error: {error}"
            )

        results.append(record)

    # -----------------------------------------------------
    # 5. Save Detailed Results
    # -----------------------------------------------------

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # 6. Calculate Evaluation Metrics
    # -----------------------------------------------------

    valid = df[
        df["error"].isna()
    ].copy()

    known = valid[
        valid["true_status"] == "known"
    ]

    unknown = valid[
        valid["true_status"] == "unknown"
    ]

    known_correct = int(
        known["correct"].sum()
    )

    unknown_correct = int(
        unknown["correct"].sum()
    )

    known_wrong = (
        len(known) - known_correct
    )

    unknown_accepted = (
        len(unknown) - unknown_correct
    )

    # Known Accuracy
    known_accuracy = (
        known_correct / len(known) * 100
        if len(known) else 0
    )

    # Unknown Rejection Rate
    unknown_rejection_rate = (
        unknown_correct / len(unknown) * 100
        if len(unknown) else 0
    )

    # False Acceptance Rate
    far = (
        unknown_accepted / len(unknown) * 100
        if len(unknown) else 0
    )

    # False Rejection Rate
    frr = (
        known_wrong / len(known) * 100
        if len(known) else 0
    )

    # Overall Accuracy
    overall_accuracy = (
        valid["correct"].sum() / len(valid) * 100
        if len(valid) else 0
    )

    # -----------------------------------------------------
    # 7. Detection Statistics
    # -----------------------------------------------------

    no_face_count = int(
        (
            valid["predicted_status"]
            == "no_face_detected"
        ).sum()
    )

    no_valid_face_count = int(
        (
            valid["predicted_status"]
            == "no_valid_face"
        ).sum()
    )

    multiple_faces_count = int(
        (
            valid["faces_detected"] > 1
        ).sum()
    )

    # -----------------------------------------------------
    # 8. Final Report
    # -----------------------------------------------------

    print("\n")
    print("=" * 65)
    print("FINAL TEST-ONLY EVALUATION")
    print("=" * 65)

    
    print(f"Test-only records: {len(df)}")
    print(f"Valid evaluations: {len(valid)}")
    print(f"Errors: {df['error'].notna().sum()}")

    print("\n" + "-" * 65)
    print("KNOWN FACES")
    print("-" * 65)

    print(f"Total: {len(known)}")
    print(f"Correct: {known_correct}")
    print(f"Wrong / Rejected: {known_wrong}")

    print(
        f"Known Accuracy: {known_accuracy:.2f}%"
    )

    print(
        f"False Rejection Rate (FRR): {frr:.2f}%"
    )

    print("\n" + "-" * 65)
    print("UNKNOWN FACES")
    print("-" * 65)

    print(f"Total: {len(unknown)}")

    print(
        f"Correctly Rejected: {unknown_correct}"
    )

    print(
        f"Incorrectly Accepted: {unknown_accepted}"
    )

    print(
        f"Unknown Rejection Rate: "
        f"{unknown_rejection_rate:.2f}%"
    )

    print(
        f"False Acceptance Rate (FAR): {far:.2f}%"
    )

    print("\n" + "-" * 65)
    print("OVERALL")
    print("-" * 65)

    print(
        f"Overall Accuracy: {overall_accuracy:.2f}%"
    )

    print("\n" + "-" * 65)
    print("DETECTION STATISTICS")
    print("-" * 65)

    print(
        f"No-face cases: {no_face_count}"
    )

    print(
        f"No-valid-face cases: {no_valid_face_count}"
    )

    print(
        f"Multiple-face images: {multiple_faces_count}"
    )

    # -----------------------------------------------------
    # 9. Save Summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)

    print("EVALUATION COMPLETED")

    print("\nDetailed results saved to:")

    print(OUTPUT_PATH)

    print("=" * 65)


if __name__ == "__main__":
    main()
