
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Paths
# ==========================================

ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "data" / "faces" / "threshold_evaluation.csv"

OUTPUT_DIR = ROOT / "data" / "faces" / "threshold_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load Evaluation Results
# ==========================================

print("=" * 55)
print("WANAS - THRESHOLD VISUALIZATION")
print("=" * 55)

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Threshold evaluation file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nLoaded threshold evaluation:")
print(df.to_string(index=False))


# ==========================================
# 3. Plot Configuration
# ==========================================

thresholds = df["threshold"]

plots = [
    {
        "column": "known_accuracy",
        "title": "Known Identification Accuracy",
        "ylabel": "Accuracy (%)",
        "filename": "known_accuracy.png"
    },
    {
        "column": "unknown_rejection_rate",
        "title": "Unknown Rejection Rate",
        "ylabel": "Rejection Rate (%)",
        "filename": "unknown_rejection_rate.png"
    },
    {
        "column": "FAR",
        "title": "False Acceptance Rate (FAR)",
        "ylabel": "FAR (%)",
        "filename": "false_acceptance_rate.png"
    },
    {
        "column": "FRR",
        "title": "False Rejection Rate (FRR)",
        "ylabel": "FRR (%)",
        "filename": "false_rejection_rate.png"
    }
]


# ==========================================
# 4. Generate Individual Plots
# ==========================================

for plot in plots:

    plt.figure(figsize=(8, 5))

    plt.plot(
        thresholds,
        df[plot["column"]],
        marker="o",
        linewidth=2
    )

    plt.title(plot["title"])
    plt.xlabel("Similarity Threshold")
    plt.ylabel(plot["ylabel"])

    plt.xticks(thresholds)
    plt.ylim(0, 105)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / plot["filename"]

    plt.savefig(output_path, dpi=300)

    plt.close()

    print("\nSaved:", output_path)


# ==========================================
# 5. Combined Accuracy Plot
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    thresholds,
    df["known_accuracy"],
    marker="o",
    label="Known Identification Accuracy"
)

plt.plot(
    thresholds,
    df["unknown_rejection_rate"],
    marker="s",
    label="Unknown Rejection Rate"
)

plt.plot(
    thresholds,
    df["overall_accuracy"],
    marker="^",
    label="Overall Accuracy"
)

plt.xlabel("Similarity Threshold")
plt.ylabel("Accuracy (%)")

plt.title("Face Recognition Performance vs Threshold")

plt.xticks(thresholds)
plt.ylim(0, 105)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

combined_path = OUTPUT_DIR / "combined_accuracy.png"

plt.savefig(combined_path, dpi=300)

plt.close()

print("\nSaved:", combined_path)


# ==========================================
# 6. Final Message
# ==========================================

print("\n" + "=" * 55)
print("VISUALIZATION COMPLETED")
print("=" * 55)

print("\nAll plots saved in:")
print(OUTPUT_DIR)
