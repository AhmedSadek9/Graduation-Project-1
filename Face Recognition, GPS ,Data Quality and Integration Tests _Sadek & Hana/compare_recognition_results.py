
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

ORIGINAL_PATH = (
    ROOT / "data" / "faces" / "recognition_results.csv"
)

CANDIDATE_PATH = (
    ROOT / "data" / "faces" / "recognition_results_candidate.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "faces" / "enrollment_results_comparison.csv"
)


# ============================================================
# 2. LOAD FILES
# ============================================================

print("=" * 65)
print("LOADING ORIGINAL AND CANDIDATE RESULTS")
print("=" * 65)

original = pd.read_csv(ORIGINAL_PATH)
candidate = pd.read_csv(CANDIDATE_PATH)

print("Original records:", len(original))
print("Candidate records:", len(candidate))


# ============================================================
# 3. NORMALIZE IMAGE PATHS
# ============================================================

def normalize_image_path(path):

    path = str(path).replace("/", "\\").lower()

    parts = Path(path).parts

    # Locate person-XXX or unknown-XXX folder
    for i, part in enumerate(parts):

        if part.startswith("person-") or part.startswith("unknown-"):

            return "\\".join(parts[i:])

    return path


original["image_key"] = (
    original["image_path"].apply(normalize_image_path)
)

candidate["image_key"] = (
    candidate["image_path"].apply(normalize_image_path)
)


# ============================================================
# 4. PREPARE ORIGINAL RESULTS
# ============================================================

original_comparison = original[
    [
        "image_key",
        "true_person",
        "true_status",
        "predicted_person",
        "similarity",
        "recognition_status",
        "error"
    ]
].copy()

original_comparison = original_comparison.rename(
    columns={
        "predicted_person": "original_prediction",
        "similarity": "original_similarity",
        "recognition_status": "original_status",
        "error": "original_error"
    }
)


# ============================================================
# 5. PREPARE CANDIDATE RESULTS
# ============================================================

candidate_comparison = candidate[
    [
        "image_key",
        "best_match_person",
        "predicted_person",
        "similarity",
        "decision",
        "error"
    ]
].copy()

candidate_comparison = candidate_comparison.rename(
    columns={
        "best_match_person": "candidate_best_match",
        "predicted_person": "candidate_prediction",
        "similarity": "candidate_similarity",
        "decision": "candidate_status",
        "error": "candidate_error"
    }
)


# ============================================================
# 6. MERGE RESULTS
# ============================================================

comparison = pd.merge(
    original_comparison,
    candidate_comparison,
    on="image_key",
    how="outer",
    indicator=True
)

comparison["similarity_change"] = (
    comparison["candidate_similarity"]
    - comparison["original_similarity"]
)


# ============================================================
# 7. NORMALIZE RECOGNITION STATUS
# ============================================================

def normalize_status(value):

    if pd.isna(value):
        return "ERROR"

    value = str(value).strip().upper()

    if value == "KNOWN":
        return "KNOWN"

    if value == "UNKNOWN":
        return "UNKNOWN"

    return "ERROR"


comparison["original_status_normalized"] = (
    comparison["original_status"].apply(normalize_status)
)

comparison["candidate_status_normalized"] = (
    comparison["candidate_status"].apply(normalize_status)
)


# ============================================================
# 8. DETERMINE CHANGES
# ============================================================

comparison["status_changed"] = (
    comparison["original_status_normalized"]
    != comparison["candidate_status_normalized"]
)

comparison["prediction_changed"] = (
    comparison["original_prediction"].fillna("ERROR").astype(str)
    != comparison["candidate_prediction"].fillna("ERROR").astype(str)
)

comparison["similarity_changed"] = ~np.isclose(
    comparison["original_similarity"].fillna(-999),
    comparison["candidate_similarity"].fillna(-999)
)

comparison["comparison_result"] = "UNCHANGED"

comparison.loc[
    comparison["similarity_changed"],
    "comparison_result"
] = "SIMILARITY_CHANGED"

comparison.loc[
    comparison["status_changed"],
    "comparison_result"
] = "STATUS_CHANGED"

comparison.loc[
    comparison["prediction_changed"],
    "comparison_result"
] = "PREDICTION_CHANGED"

comparison.loc[
    comparison["_merge"] != "both",
    "comparison_result"
] = "MISSING_FROM_ONE_FILE"


# ============================================================
# 9. SAVE COMPARISON
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

comparison.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nComparison saved to:")
print(OUTPUT_PATH)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ORIGINAL VS CANDIDATE COMPARISON")
print("=" * 65)

print("Matched images:", (comparison["_merge"] == "both").sum())
print("Missing images:", (comparison["_merge"] != "both").sum())

print(
    "Similarity changed:",
    comparison["similarity_changed"].sum()
)

print(
    "Recognition status changed:",
    comparison["status_changed"].sum()
)

print(
    "Predictions changed:",
    comparison["prediction_changed"].sum()
)


# ============================================================
# 11. SHOW SIMILARITY CHANGES
# ============================================================

changed = comparison[
    comparison["similarity_changed"]
].copy()

print("\n" + "=" * 65)
print("IMAGES WITH CHANGED SIMILARITY")
print("=" * 65)

if len(changed) == 0:

    print("No similarity changes detected.")

else:

    print(
        changed[
            [
                "image_key",
                "original_similarity",
                "candidate_similarity",
                "similarity_change",
                "original_prediction",
                "candidate_prediction"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 12. CHECK OTHER NINE PEOPLE
# ============================================================

print("\n" + "=" * 65)
print("CHECKING OTHER NINE PEOPLE")
print("=" * 65)

other_people = comparison[
    comparison["true_person"] != "person-001"
].copy()

other_people_changed = other_people[
    other_people["similarity_changed"]
    | other_people["status_changed"]
    | other_people["prediction_changed"]
]

print(
    "Other-person records:",
    len(other_people)
)

print(
    "Other-person records with changes:",
    len(other_people_changed)
)

if len(other_people_changed) == 0:

    print("PASS: No changes detected for the other nine people.")

else:

    print("WARNING: Changes detected for other people.")

    print(
        other_people_changed[
            [
                "image_key",
                "original_similarity",
                "candidate_similarity",
                "original_prediction",
                "candidate_prediction"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 13. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("COMPARISON COMPLETED")
print("=" * 65)
