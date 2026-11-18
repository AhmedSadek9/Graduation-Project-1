from pathlib import Path
import csv
from collections import defaultdict

from src.pipeline.ocr_pipeline import OCRPipeline
from src.utils.evaluation import (
    calculate_cer,
    calculate_wer,
    calculate_corpus_cer,
    calculate_corpus_wer,
)


# ============================================================
# PATHS
# ============================================================

images_dir = Path(
    "data/review_subset_44/data/raw_images/real_samples"
)

ground_truth_dir = Path(
    "data/review_subset_44/data/ground_truth/real_samples"
)

# Separate output folder - preserves previous baseline results
output_dir = Path("data/outputs/evaluation_run_02")
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "review_subset_44_results.csv"
issues_path = output_dir / "evaluation_issues.csv"
summary_path = output_dir / "evaluation_summary.csv"


# ============================================================
# OCR ENGINES
# ============================================================

engines = {
    "Tesseract": OCRPipeline(engine="tesseract"),
    "PaddleOCR": OCRPipeline(engine="paddle"),
}


# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted(
    [
        *images_dir.glob("*.jpg"),
        *images_dir.glob("*.jpeg"),
        *images_dir.glob("*.png"),
    ]
)

print("=" * 65)
print("BASMALA OCR - REVIEW SUBSET EVALUATION")
print("=" * 65)

print("Images found:", len(image_files))

if not image_files:
    raise FileNotFoundError(
        f"No images found in: {images_dir}"
    )


# ============================================================
# RESULTS STORAGE
# ============================================================

all_results = []
failed_results = []
missing_ground_truth = []

# For corpus-level evaluation
corpus_data = defaultdict(
    lambda: {"references": [], "hypotheses": []}
)


# ============================================================
# PROCESS IMAGES
# ============================================================

for image_number, image_path in enumerate(
    image_files, start=1
):

    ground_truth_path = (
        ground_truth_dir / f"{image_path.stem}.txt"
    )

    if not ground_truth_path.exists():

        print(
            f"\nMissing Ground Truth: {image_path.name}"
        )

        missing_ground_truth.append(image_path.name)

        for engine_name in engines:
            failed_results.append({
                "Image": image_path.name,
                "Engine": engine_name,
                "Status": "Missing Ground Truth",
            })

        continue

    reference_text = ground_truth_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not reference_text:

        print(
            f"\nEmpty Ground Truth: {image_path.name}"
        )

        missing_ground_truth.append(image_path.name)

        for engine_name in engines:
            failed_results.append({
                "Image": image_path.name,
                "Engine": engine_name,
                "Status": "Empty Ground Truth",
            })

        continue

    print("\n" + "-" * 65)
    print(
        f"Image {image_number}/{len(image_files)}: "
        f"{image_path.name}"
    )
    print("-" * 65)

    for engine_name, engine_function in engines.items():

        try:

            result = engine_function.run(str(image_path), ground_truth=reference_text)

            extracted_text = (
                result.get("extracted_text", "") or ""
            ).strip()

            confidence = result.get("confidence", 0)
            status = result.get("status", "unknown")

            # ----------------------------------------------------
            # OCR FAILURE
            # ----------------------------------------------------

            if status != "success":

                print(
                    f"{engine_name} failed: {status}"
                )

                failed_results.append({
                    "Image": image_path.name,
                    "Engine": engine_name,
                    "Status": status,
                })

                continue

            # ----------------------------------------------------
            # EMPTY OCR OUTPUT
            # ----------------------------------------------------

            if not extracted_text:

                print(
                    f"{engine_name}: Empty OCR output"
                )

                failed_results.append({
                    "Image": image_path.name,
                    "Engine": engine_name,
                    "Status": "Empty OCR Output",
                })

                continue

            # ----------------------------------------------------
            # CONFIDENCE NORMALIZATION FOR DISPLAY
            # ----------------------------------------------------

            scale = result.get("confidence_scale")
            if confidence is None:
                confidence_percent = 0
            elif scale == "0-1":
                confidence_percent = confidence * 100
            elif scale == "0-100":
                confidence_percent = confidence
            else:
                confidence_percent = confidence

            # ----------------------------------------------------
            # CER / WER
            # ----------------------------------------------------

            cer = calculate_cer(
                reference_text,
                extracted_text,
            )

            wer = calculate_wer(
                reference_text,
                extracted_text,
            )

            print(
                f"{engine_name} | "
                f"Confidence: {confidence_percent:.2f}% | "
                f"CER: {cer * 100:.2f}% | "
                f"WER: {wer * 100:.2f}%"
            )

            # ----------------------------------------------------
            # SAVE SUCCESSFUL RESULT
            # ----------------------------------------------------

            all_results.append({
                "Image": image_path.name,
                "Engine": engine_name,
                "Ground Truth": reference_text,
                "Extracted Text": extracted_text,
                "Confidence": confidence,
                "Confidence (%)": round(
                    confidence_percent, 2
                ),
                "CER": round(cer * 100, 2),
                "WER": round(wer * 100, 2),
                "Status": status,
            })

            # ----------------------------------------------------
            # CORPUS DATA
            # ----------------------------------------------------

            corpus_data[engine_name]["references"].append(
                reference_text
            )

            corpus_data[engine_name]["hypotheses"].append(
                extracted_text
            )

        except Exception as error:

            print(
                f"{engine_name} Error: {error}"
            )

            failed_results.append({
                "Image": image_path.name,
                "Engine": engine_name,
                "Status": f"Error: {error}",
            })


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

result_fields = [
    "Image",
    "Engine",
    "Ground Truth",
    "Extracted Text",
    "Confidence",
    "Confidence (%)",
    "CER",
    "WER",
    "Status",
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=result_fields,
    )

    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# SAVE EVALUATION ISSUES
# ============================================================

issue_fields = [
    "Image",
    "Engine",
    "Status",
]

with open(
    issues_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as issues_file:

    writer = csv.DictWriter(
        issues_file,
        fieldnames=issue_fields,
    )

    writer.writeheader()
    writer.writerows(failed_results)


# ============================================================
# CALCULATE ENGINE SUMMARY
# ============================================================

summary_results = []

print("\n" + "=" * 65)
print("FINAL EVALUATION RESULTS")
print("=" * 65)

for engine_name in engines:

    engine_results = [
        item for item in all_results
        if item["Engine"] == engine_name
    ]

    engine_failures = [
        item for item in failed_results
        if item["Engine"] == engine_name
    ]

    if not engine_results:

        print(f"\n{engine_name}: No successful results")

        continue

    # --------------------------------------------------------
    # INDIVIDUAL IMAGE AVERAGES
    # --------------------------------------------------------

    avg_confidence = sum(
        item["Confidence (%)"]
        for item in engine_results
    ) / len(engine_results)

    avg_cer = sum(
        item["CER"]
        for item in engine_results
    ) / len(engine_results)

    avg_wer = sum(
        item["WER"]
        for item in engine_results
    ) / len(engine_results)

    # --------------------------------------------------------
    # CORPUS METRICS - FIXED
    # --------------------------------------------------------

    references = corpus_data[engine_name]["references"]
    hypotheses = corpus_data[engine_name]["hypotheses"]

    # Corpus functions expect a list of (reference, hypothesis) pairs
    pairs = list(zip(references, hypotheses))

    corpus_cer = calculate_corpus_cer(pairs) * 100
    corpus_wer = calculate_corpus_wer(pairs) * 100

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print(f"\n{engine_name}")
    print("Successful images:", len(engine_results))
    print("Failed images:", len(engine_failures))
    print(f"Average Confidence: {avg_confidence:.2f}%")
    print(f"Average CER: {avg_cer:.2f}%")
    print(f"Average WER: {avg_wer:.2f}%")
    print(f"Corpus CER: {corpus_cer:.2f}%")
    print(f"Corpus WER: {corpus_wer:.2f}%")

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_results.append({
        "Engine": engine_name,
        "Successful Images": len(engine_results),
        "Failed Images": len(engine_failures),
        "Average Confidence (%)": round(
            avg_confidence, 2
        ),
        "Average CER (%)": round(avg_cer, 2),
        "Average WER (%)": round(avg_wer, 2),
        "Corpus CER (%)": round(corpus_cer, 2),
        "Corpus WER (%)": round(corpus_wer, 2),
    })


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_fields = [
    "Engine",
    "Successful Images",
    "Failed Images",
    "Average Confidence (%)",
    "Average CER (%)",
    "Average WER (%)",
    "Corpus CER (%)",
    "Corpus WER (%)",
]

with open(
    summary_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as summary_file:

    writer = csv.DictWriter(
        summary_file,
        fieldnames=summary_fields,
    )

    writer.writeheader()
    writer.writerows(summary_results)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("EVALUATION SUMMARY")
print("=" * 65)

print("Total images found:", len(image_files))
print("Missing/empty Ground Truth:", len(missing_ground_truth))
print("Failed OCR attempts:", len(failed_results))
print("Successful OCR attempts:", len(all_results))

print("\nDetailed results:", csv_path)
print("Issues:", issues_path)
print("Summary:", summary_path)

print("=" * 65)



