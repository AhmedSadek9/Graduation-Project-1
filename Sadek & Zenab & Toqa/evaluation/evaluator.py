import csv
import sys
import time
from pathlib import Path

import cv2


# ==========================================
# 1. Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "ground_truth.csv"

RESULTS_FILE = (
    PROJECT_ROOT / "evaluation" / "evaluation_results.csv"
)

# Find Sadek & Zenab project
# Find the isolated Sadek & Zenab test project
CANDIDATE_PROJECTS = [
    PROJECT_ROOT / "Sadek & Zenab_TEST",
    PROJECT_ROOT / "Graduation-Project-1-Task_Week_1" / "Sadek & Zenab",
    PROJECT_ROOT / "Sadek & Zenab",
]

FACE_PROJECT = next(
    (
        path for path in CANDIDATE_PROJECTS
        if (path / "ai" / "face" / "face_recognition.py").is_file()
    ),
    None,
)

if FACE_PROJECT is None:
    raise FileNotFoundError(
        "Could not find the face recognition module. "
        "Check the Sadek & Zenab_TEST folder."
    )

sys.path.insert(0, str(FACE_PROJECT))

from ai.face.face_recognition import FaceRecognizer

# ==========================================
# 2. Evaluation settings
# ==========================================

OUTPUT_COLUMNS = [
    "image_path",
    "true_person_id",
    "true_status",
    "condition",
    "predicted_person_id",
    "predicted_status",
    "confidence",
    "latency_ms",
    "correct",
    "error",
]


def normalize_status(status):
    """Normalize the model's status for comparison."""
    if status in ("known", "unknown", "low_confidence"):
        return status
    return "no_face"


def evaluate_prediction(true_id, true_status, prediction):
    """
    A known image is correct only if the correct person is identified.
    An unknown image is correctly rejected if it is not classified as known.
    """
    predicted_status = normalize_status(
        prediction.get("status") if prediction else None
    )
    predicted_id = prediction.get("person_id") if prediction else None

    if true_status == "unknown":
        correct = predicted_status != "known"
    else:
        correct = (
            predicted_status == "known"
            and predicted_id == true_id
        )

    return predicted_id, predicted_status, correct


# ==========================================
# 3. Run evaluation
# ==========================================

def main():
    if not GROUND_TRUTH_FILE.is_file():
        raise FileNotFoundError(
            f"Ground-truth file not found: {GROUND_TRUTH_FILE}"
        )

    with GROUND_TRUTH_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "image_path",
            "true_person_id",
            "true_status",
            "condition",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "ground_truth.csv must contain: "
                "image_path, true_person_id, true_status, condition"
            )

        ground_truth_rows = list(reader)

    if not ground_truth_rows:
        raise ValueError("ground_truth.csv contains no test images.")

    # Initialize the recognizer once, not once per image.
    recognizer = FaceRecognizer()

    results = []

    for row in ground_truth_rows:
        image_path = PROJECT_ROOT / row["image_path"]
        true_id = (row["true_person_id"] or "").strip() or None
        true_status = row["true_status"].strip().lower()

        result_row = {
            "image_path": row["image_path"],
            "true_person_id": true_id or "",
            "true_status": true_status,
            "condition": row["condition"],
            "predicted_person_id": "",
            "predicted_status": "error",
            "confidence": "",
            "latency_ms": "",
            "correct": False,
            "error": "",
        }

        if not image_path.is_file():
            result_row["error"] = "IMAGE_NOT_FOUND"
            results.append(result_row)
            print(f"[ERROR] Image not found: {image_path}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            result_row["error"] = "IMAGE_READ_FAILED"
            results.append(result_row)
            print(f"[ERROR] Could not read: {image_path}")
            continue

        try:
            start_time = time.perf_counter()
            predictions = recognizer.recognize(image)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # The test images are expected to contain one face.
            # If multiple faces are detected, use the highest-confidence result.
            prediction = (
                max(
                    predictions,
                    key=lambda item: item.get("confidence", 0),
                )
                if predictions
                else None
            )

            predicted_id, predicted_status, correct = (
                evaluate_prediction(true_id, true_status, prediction)
            )

            result_row.update({
                "predicted_person_id": predicted_id or "",
                "predicted_status": predicted_status,
                "confidence": (
                    prediction.get("confidence", "")
                    if prediction else ""
                ),
                "latency_ms": round(latency_ms, 2),
                "correct": correct,
            })

        except Exception as exc:
            result_row["error"] = f"{type(exc).__name__}: {exc}"

        results.append(result_row)

        print(
            f"[{len(results)}/{len(ground_truth_rows)}] "
            f"{row['image_path']} -> "
            f"{result_row['predicted_status']}"
        )

    # ======================================
    # 4. Save detailed results
    # ======================================

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    # ======================================
    # 5. Calculate metrics
    # ======================================

    successful = [r for r in results if not r["error"]]
    total = len(successful)

    if total == 0:
        print("\nNo images were evaluated successfully.")
        print(f"Check the errors in: {RESULTS_FILE}")
        return

    correct_count = sum(bool(r["correct"]) for r in successful)

    # False positives: unknown person incorrectly identified as known.
    false_positives = sum(
        r["true_status"] == "unknown"
        and r["predicted_status"] == "known"
        for r in successful
    )

    # False negatives: known person not identified as that correct person.
    false_negatives = sum(
        r["true_status"] == "known"
        and not bool(r["correct"])
        for r in successful
    )

    accuracy = correct_count / total * 100

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Images in ground truth: {len(ground_truth_rows)}")
    print(f"Successfully evaluated: {total}")
    print(f"Images with errors: {len(results) - total}")
    print(f"Correct predictions: {correct_count}/{total}")
    print(f"Exact-match accuracy: {accuracy:.2f}%")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()