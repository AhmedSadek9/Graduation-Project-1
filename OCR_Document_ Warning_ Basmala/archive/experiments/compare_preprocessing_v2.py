
from pathlib import Path
import cv2
import pandas as pd
import tempfile

from src.ocr.paddle_engine import extract_text
from src.preprocessing.image_processing import preprocess_image
from src.utils.evaluation import (
    calculate_cer,
    calculate_wer,
    calculate_corpus_cer,
    calculate_corpus_wer,
)


# =========================
# Configuration
# =========================

images_dir = Path(
    "data/review_subset_44/data/raw_images/real_samples"
)

ground_truth_dir = Path(
    "data/review_subset_44/data/ground_truth/real_samples"
)

# Separate output folder to preserve old results
output_dir = Path(
    "data/outputs/preprocessing_comparison_v2"
)

output_dir.mkdir(parents=True, exist_ok=True)


results = []
issues = []

image_extensions = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


# =========================
# Discover images
# =========================

if not images_dir.is_dir():
    raise FileNotFoundError(
        f"Images directory not found: {images_dir}"
    )

image_paths = sorted(
    p for p in images_dir.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
)

print(f"Found {len(image_paths)} images.")
print("Starting original vs processed comparison...\n")


# =========================
# Evaluate each image
# =========================

for index, image_path in enumerate(image_paths, start=1):

    ground_truth_path = (
        ground_truth_dir / f"{image_path.stem}.txt"
    )

    if not ground_truth_path.is_file():
        issues.append({
            "Image": image_path.name,
            "Issue": "Ground truth file not found"
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"Missing GT: {image_path.name}"
        )
        continue

    try:
        ground_truth = ground_truth_path.read_text(
            encoding="utf-8"
        ).strip()

        # OCR on original image
        original_result = extract_text(image_path)

        # Preprocessing
        processed_images = preprocess_image(image_path)

        # OCR on processed image
        with tempfile.TemporaryDirectory() as temp_dir:

            processed_path = (
                Path(temp_dir) / image_path.name
            )

            saved = cv2.imwrite(
                str(processed_path),
                processed_images["processed"]
            )

            if not saved:
                raise RuntimeError(
                    "Could not save processed image"
                )

            processed_result = extract_text(
                processed_path
            )

        original_text = (
            original_result.get("extracted_text") or ""
        )

        processed_text = (
            processed_result.get("extracted_text") or ""
        )

        original_confidence = original_result.get(
            "confidence"
        )

        processed_confidence = processed_result.get(
            "confidence"
        )

        # Per-image metrics
        original_cer = calculate_cer(
            ground_truth, original_text
        )

        processed_cer = calculate_cer(
            ground_truth, processed_text
        )

        original_wer = calculate_wer(
            ground_truth, original_text
        )

        processed_wer = calculate_wer(
            ground_truth, processed_text
        )

        results.append({
            "Image": image_path.name,
            "Ground Truth": ground_truth,

            "Original Text": original_text,
            "Original Confidence": original_confidence,
            "Original CER": original_cer,
            "Original WER": original_wer,
            "Original Status": original_result.get(
                "status", "unknown"
            ),

            "Processed Text": processed_text,
            "Processed Confidence": processed_confidence,
            "Processed CER": processed_cer,
            "Processed WER": processed_wer,
            "Processed Status": processed_result.get(
                "status", "unknown"
            ),
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"Completed: {image_path.name}"
        )

    except Exception as error:
        issues.append({
            "Image": image_path.name,
            "Issue": f"{type(error).__name__}: {error}"
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"Error: {image_path.name} - {error}"
        )


# =========================
# Save results and summary
# =========================

if results:

    df = pd.DataFrame(results)

    # Save per-image results
    csv_path = output_dir / "preprocessing_comparison_v2.csv"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Corpus metrics
    original_pairs = list(
        zip(df["Ground Truth"], df["Original Text"])
    )

    processed_pairs = list(
        zip(df["Ground Truth"], df["Processed Text"])
    )

    original_corpus_cer = calculate_corpus_cer(
        original_pairs
    )

    processed_corpus_cer = calculate_corpus_cer(
        processed_pairs
    )

    original_corpus_wer = calculate_corpus_wer(
        original_pairs
    )

    processed_corpus_wer = calculate_corpus_wer(
        processed_pairs
    )

    # Average metrics
    original_avg_cer = df["Original CER"].mean()
    processed_avg_cer = df["Processed CER"].mean()

    original_avg_wer = df["Original WER"].mean()
    processed_avg_wer = df["Processed WER"].mean()

    # Confidence: Paddle Recognition Score (0..1)
    original_conf = df["Original Confidence"].mean()
    processed_conf = df["Processed Confidence"].mean()

    # Status counts
    original_status = df["Original Status"].value_counts()
    processed_status = df["Processed Status"].value_counts()

    print("\n========== V2 COMPARISON SUMMARY ==========")

    print(f"Images found: {len(image_paths)}")
    print(f"Images evaluated: {len(df)}")
    print(f"Images skipped/failed: {len(issues)}")

    print("\n--- Average Metrics ---")

    print(
        f"Original Average CER: "
        f"{original_avg_cer * 100:.2f}%"
    )

    print(
        f"Processed Average CER: "
        f"{processed_avg_cer * 100:.2f}%"
    )

    print(
        f"Original Average WER: "
        f"{original_avg_wer * 100:.2f}%"
    )

    print(
        f"Processed Average WER: "
        f"{processed_avg_wer * 100:.2f}%"
    )

    print("\n--- Corpus Metrics ---")

    print(
        f"Original Corpus CER: "
        f"{original_corpus_cer * 100:.2f}%"
    )

    print(
        f"Processed Corpus CER: "
        f"{processed_corpus_cer * 100:.2f}%"
    )

    print(
        f"Original Corpus WER: "
        f"{original_corpus_wer * 100:.2f}%"
    )

    print(
        f"Processed Corpus WER: "
        f"{processed_corpus_wer * 100:.2f}%"
    )

    print("\n--- Paddle Recognition Score ---")

    if pd.notna(original_conf):
        print(
            f"Original Mean Score: "
            f"{original_conf:.4f} "
            f"({original_conf * 100:.2f}%)"
        )
    else:
        print("Original Mean Score: N/A")

    if pd.notna(processed_conf):
        print(
            f"Processed Mean Score: "
            f"{processed_conf:.4f} "
            f"({processed_conf * 100:.2f}%)"
        )
    else:
        print("Processed Mean Score: N/A")

    print("\n--- Original Status Counts ---")

    for status, count in original_status.items():
        print(f"{status}: {count}")

    print("\n--- Processed Status Counts ---")

    for status, count in processed_status.items():
        print(f"{status}: {count}")

    print(f"\nResults saved to: {csv_path}")

else:
    print("\nNo comparison results were generated.")


# =========================
# Save issues
# =========================

if issues:

    issues_path = output_dir / "comparison_issues_v2.csv"

    pd.DataFrame(issues).to_csv(
        issues_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Issues saved to: {issues_path}")

else:
    print("\nNo issues encountered.")


print("\nV2 preprocessing comparison completed!")
