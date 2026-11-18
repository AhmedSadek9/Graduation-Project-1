
from pathlib import Path

import pandas as pd
import numpy as np


# ==========================================
# 1. Paths and Configuration
# ==========================================

ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "data" / "faces" / "recognition_results.csv"

OUTPUT_FILE = ROOT / "data" / "faces" / "threshold_evaluation.csv"

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


# ==========================================
# 2. Load Recognition Results
# ==========================================

print("=" * 60)
print("WANAS - THRESHOLD EVALUATION")
print("=" * 60)

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Recognition results not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

# Exclude images where detection or file processing failed
valid = df[
    (df["recognition_status"] != "ERROR") &
    (df["similarity"].notna())
].copy()

known = valid[
    valid["true_status"] == "known"
].copy()

unknown = valid[
    valid["true_status"] == "unknown"
].copy()

print("\nTotal records:", len(df))
print("Valid records:", len(valid))
print("Known records:", len(known))
print("Unknown records:", len(unknown))


# ==========================================
# 3. Evaluate Each Threshold
# ==========================================

evaluation_results = []

for threshold in THRESHOLDS:

    # Predict known/unknown based on similarity
    known_predictions = known["similarity"] >= threshold

    unknown_predictions = unknown["similarity"] >= threshold

    # Known person accepted by the system
    known_correct = (
        known_predictions &
        (known["predicted_person"] == known["true_person"])
    ).sum()

    # Known person rejected as Unknown
    false_rejections = (
        ~known_predictions
    ).sum()

    # Known person matched to the wrong registered identity
    wrong_known_identity = (
        known_predictions &
        (known["predicted_person"] != known["true_person"])
    ).sum()

    # Unknown person incorrectly accepted as a registered person
    false_acceptances = unknown_predictions.sum()

    # Unknown person correctly rejected
    unknown_correct = (
        ~unknown_predictions
    ).sum()

    known_accuracy = (
        known_correct / len(known) * 100
        if len(known) > 0 else 0
    )

    unknown_rejection_rate = (
        unknown_correct / len(unknown) * 100
        if len(unknown) > 0 else 0
    )

    far = (
        false_acceptances / len(unknown) * 100
        if len(unknown) > 0 else 0
    )

    frr = (
        false_rejections / len(known) * 100
        if len(known) > 0 else 0
    )

    total_correct = known_correct + unknown_correct

    total_accuracy = (
        total_correct / len(valid) * 100
        if len(valid) > 0 else 0
    )

    evaluation_results.append({
        "threshold": threshold,
        "known_correct": int(known_correct),
        "known_accuracy": round(known_accuracy, 2),
        "unknown_correct": int(unknown_correct),
        "unknown_rejection_rate": round(
            unknown_rejection_rate, 2
        ),
        "false_acceptances": int(false_acceptances),
        "FAR": round(far, 2),
        "false_rejections": int(false_rejections),
        "FRR": round(frr, 2),
        "wrong_known_identity": int(wrong_known_identity),
        "overall_accuracy": round(total_accuracy, 2)
    })


# ==========================================
# 4. Display Evaluation Table
# ==========================================

evaluation_df = pd.DataFrame(evaluation_results)

print("\n" + "=" * 60)
print("THRESHOLD COMPARISON")
print("=" * 60)

print(evaluation_df.to_string(index=False))


# ==========================================
# 5. Save Results
# ==========================================

evaluation_df.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nThreshold evaluation completed!")
