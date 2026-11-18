import pandas as pd
from pathlib import Path

quality_path = Path(
    "data/outputs/selective_preprocessing/image_quality.csv"
)

comparison_path = Path(
    "data/outputs/preprocessing_comparison/preprocessing_comparison.csv"
)

quality = pd.read_csv(quality_path)
comparison = pd.read_csv(comparison_path)

df = quality.merge(comparison, on="Image", how="inner")

df["Contrast Group"] = pd.cut(
    df["Contrast"],
    bins=[0, 23, 30, 100],
    labels=["Low", "Medium", "High"]
)

print("\n========== CER BY CONTRAST GROUP ==========\n")

summary = df.groupby(
    "Contrast Group",
    observed=False
).agg(
    Images=("Image", "count"),
    Original_CER=("Original CER", "mean"),
    Processed_CER=("Processed CER", "mean")
)

summary["Original_CER"] *= 100
summary["Processed_CER"] *= 100

print(summary.round(2))

print("\n========== SHARPNESS AND CER ==========\n")

print(
    df[["Image", "Contrast", "Sharpness",
        "Original CER", "Processed CER"]]
    .sort_values("Sharpness")
    .to_string(index=False)
)

output = Path(
    "data/outputs/selective_preprocessing/quality_ocr_analysis.csv"
)

df.to_csv(output, index=False, encoding="utf-8-sig")

print("\nSaved:", output)
