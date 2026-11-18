
from pathlib import Path
import cv2
import pandas as pd
import tempfile

from src.ocr.paddle_engine import extract_text
from src.preprocessing.image_processing import preprocess_image
from src.utils.evaluation import calculate_cer, calculate_wer


# ============================================================
# PATHS
# ============================================================

images_dir = Path(
    "data/review_subset_44/data/raw_images/real_samples"
)

ground_truth_dir = Path(
    "data/review_subset_44/data/ground_truth/real_samples"
)

output_dir = Path(
    "data/outputs/selective_preprocessing_v3"
)

output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Minimum confidence improvement required to select preprocessing.
# Confidence is NOT a guarantee of OCR correctness.
MIN_CONFIDENCE_IMPROVEMENT = 0.01

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}


# ============================================================
# DATA
# ============================================================

results = []
issues = []

image_paths = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
)


# ============================================================
# OCR HELPERS
# ============================================================

def run_original_ocr(image_path):
    return extract_text(image_path)


def run_processed_ocr(image_path):
    processed_images = preprocess_image(image_path)

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

        result = extract_text(processed_path)

    return result


def select_result(original_result, processed_result):
    """
    Select processed OCR only if its confidence improves
    by the configured minimum amount.

    This is an experimental selection strategy.
    Confidence does not guarantee correctness.
    """

    original_confidence = float(
        original_result.get("confidence", 0)
    )

    processed_confidence = float(
        processed_result.get("confidence", 0)
    )

    confidence_improvement = (
        processed_confidence - original_confidence
    )

    if (
        confidence_improvement
        >= MIN_CONFIDENCE_IMPROVEMENT
    ):
        return processed_result, "processed"

    return original_result, "original"


# ============================================================
# EXPERIMENT
# ============================================================

print(f"Found {len(image_paths)} images.")

print(
    "\nStarting Selective Preprocessing v3...\n"
)

for index, image_path in enumerate(
    image_paths,
    start=1
):

    gt_path = (
        ground_truth_dir
        / f"{image_path.stem}.txt"
    )

    if not gt_path.exists():

        issues.append({
            "Image": image_path.name,
            "Issue": "Ground truth not found"
        })

        continue

    try:

        # Ground truth is used ONLY for evaluation.
        ground_truth = gt_path.read_text(
            encoding="utf-8"
        ).strip()

        # Run original OCR.
        original_result = run_original_ocr(
            image_path
        )

        # Run processed OCR.
        processed_result = run_processed_ocr(
            image_path
        )

        # Select without using ground truth.
        selective_result, selected_method = (
            select_result(
                original_result,
                processed_result
            )
        )

        original_text = (
            original_result["extracted_text"]
        )

        processed_text = (
            processed_result["extracted_text"]
        )

        selective_text = (
            selective_result["extracted_text"]
        )

        original_confidence = float(
            original_result.get("confidence", 0)
        )

        processed_confidence = float(
            processed_result.get("confidence", 0)
        )

        selective_confidence = float(
            selective_result.get("confidence", 0)
        )

        # Evaluation metrics.
        original_cer = calculate_cer(
            ground_truth,
            original_text
        )

        original_wer = calculate_wer(
            ground_truth,
            original_text
        )

        processed_cer = calculate_cer(
            ground_truth,
            processed_text
        )

        processed_wer = calculate_wer(
            ground_truth,
            processed_text
        )

        selective_cer = calculate_cer(
            ground_truth,
            selective_text
        )

        selective_wer = calculate_wer(
            ground_truth,
            selective_text
        )

        # Save results.
        results.append({

            "Image": image_path.name,

            "Ground Truth": ground_truth,

            "Original Text": original_text,
            "Original Confidence": original_confidence,
            "Original CER": original_cer,
            "Original WER": original_wer,

            "Processed Text": processed_text,
            "Processed Confidence": processed_confidence,
            "Processed CER": processed_cer,
            "Processed WER": processed_wer,

            "Confidence Improvement": (
                processed_confidence
                - original_confidence
            ),

            "Selected Method": selected_method,

            "Selective Text": selective_text,
            "Selective Confidence": selective_confidence,
            "Selective CER": selective_cer,
            "Selective WER": selective_wer,

            "CER Improvement": (
                original_cer - selective_cer
            ),

            "WER Improvement": (
                original_wer - selective_wer
            )
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"{selected_method} | "
            f"Original Conf={original_confidence:.4f} | "
            f"Processed Conf={processed_confidence:.4f}"
        )

    except Exception as error:

        issues.append({
            "Image": image_path.name,
            "Issue": str(error)
        })

        print(
            f"Error: {image_path.name} - {error}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

if results:

    df = pd.DataFrame(results)

    results_path = (
        output_dir
        / "selective_comparison_v3.csv"
    )

    df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Summary.
    print(
        "\n========== SELECTIVE V3 SUMMARY =========="
    )

    print(
        "Images evaluated:",
        len(df)
    )

    print(
        "Original average CER:",
        round(df["Original CER"].mean() * 100, 2),
        "%"
    )

    print(
        "Processed average CER:",
        round(df["Processed CER"].mean() * 100, 2),
        "%"
    )

    print(
        "Selective average CER:",
        round(df["Selective CER"].mean() * 100, 2),
        "%"
    )

    print(
        "Original average WER:",
        round(df["Original WER"].mean() * 100, 2),
        "%"
    )

    print(
        "Processed average WER:",
        round(df["Processed WER"].mean() * 100, 2),
        "%"
    )

    print(
        "Selective average WER:",
        round(df["Selective WER"].mean() * 100, 2),
        "%"
    )

    print(
        "Selected original:",
        (df["Selected Method"] == "original").sum()
    )

    print(
        "Selected processed:",
        (df["Selected Method"] == "processed").sum()
    )

    print(
        "Improved CER:",
        (df["CER Improvement"] > 0).sum()
    )

    print(
        "Worse CER:",
        (df["CER Improvement"] < 0).sum()
    )

    print(
        "Unchanged CER:",
        (df["CER Improvement"] == 0).sum()
    )

    print(
        "\nSaved:",
        results_path
    )


# ============================================================
# ISSUES
# ============================================================

if issues:

    issues_path = (
        output_dir / "selective_issues_v3.csv"
    )

    pd.DataFrame(issues).to_csv(
        issues_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nIssues encountered:",
        len(issues)
    )

    print(
        "Issues saved:",
        issues_path
    )


print(
    "\nSelective Preprocessing v3 completed!"
)
