
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

# Separate output folder for preprocessing experiment
output_dir = Path("data/outputs/preprocessing_run_01")
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "review_subset_44_results.csv"
issues_path = output_dir / "evaluation_issues.csv"
summary_path = output_dir / "evaluation_summary.csv"

# ============================================================
# OCR PIPELINES WITH PREPROCESSING ENABLED
# ============================================================

pipelines = {
    "Tesseract": OCRPipeline(
        engine="tesseract",
        preprocess=True,
    ),
    "PaddleOCR": OCRPipeline(
        engine="paddle",
        preprocess=True,
    ),
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
print("BASMALA OCR - PREPROCESSING EVALUATION")
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

corpus_data = defaultdict(
    lambda: {"references": [], "hypotheses": []}
)

# ============================================================
# PROCESS IMAGES
# ============================================================

for image_number, image_path in enumerate(image_files, start=1):

    ground_truth_path = ground_truth_dir / f"{image_path.stem}.txt"

    if not ground_truth_path.exists():
        print(f"\nMissing Ground Truth: {image_path.name}")
        missing_ground_truth.append(image_path.name)

        for engine_name in pipelines:
            failed_results.append({
                "Image": image_path.name,
                "Engine": engine_name,
                "Status": "Missing Ground Truth",
                "Error": "",
            })
        continue

    reference_text = ground_truth_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not reference_text:
        print(f"\nEmpty Ground Truth: {image_path.name}")
        missing_ground_truth.append(image_path.name)

        for engine_name in pipelines:
            failed_results.append({
                "Image": image_path.name,
                "Engine": engine_name,
                "Status": "Empty Ground Truth",
                "Error": "",
            })
        continue

    print("\n" + "-" * 65)
    print(f"Image {image_number}/{len(image_files)}: {image_path.name}")
    print("-" * 65)

    for engine_name, pipeline in pipelines.items():

        result = pipeline.run(
            image_path,
            ground_truth=reference_text,
        )

        extracted_text = (
            result.get("extracted_text", "") or ""
        ).strip()

        status = result.get("status", "unknown")
        confidence = result.get("confidence")
        confidence_scale = result.get("confidence_scale")
        processing_time = result.get("processing_time", 0)
        error = result.get("error")

        if status != "success" or not extracted_text:

            failure_status = (
                status if status != "success" else "Empty OCR Output"
            )

            print(f"{engine_name} failed: {failure_status}")

            failed_results.append({
                "Image": image_path.name,
                "Engine": engine_name,
                "Status": failure_status,
                "Error": error or "",
            })
            continue

        cer = calculate_cer(reference_text, extracted_text)
        wer = calculate_wer(reference_text, extracted_text)

        confidence_percent = None

        if confidence is not None:
            if confidence_scale == "0-1":
                confidence_percent = confidence * 100
            elif confidence_scale == "0-100":
                confidence_percent = confidence
            else:
                confidence_percent = confidence

        print(
            f"{engine_name} | "
            f"CER: {cer * 100:.2f}% | "
            f"WER: {wer * 100:.2f}% | "
            f"Time: {processing_time:.3f}s"
        )

        all_results.append({
            "Image": image_path.name,
            "Engine": engine_name,
            "Ground Truth": reference_text,
            "Extracted Text": extracted_text,
            "Confidence": confidence,
            "Confidence Scale": confidence_scale,
            "Confidence (%)": (
                round(confidence_percent, 2)
                if confidence_percent is not None
                else ""
            ),
            "CER": round(cer * 100, 2),
            "WER": round(wer * 100, 2),
            "Processing Time (s)": processing_time,
            "Status": status,
            "Error": error or "",
        })

        corpus_data[engine_name]["references"].append(reference_text)
        corpus_data[engine_name]["hypotheses"].append(extracted_text)

# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

result_fields = [
    "Image",
    "Engine",
    "Ground Truth",
    "Extracted Text",
    "Confidence",
    "Confidence Scale",
    "Confidence (%)",
    "CER",
    "WER",
    "Processing Time (s)",
    "Status",
    "Error",
]

with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=result_fields)
    writer.writeheader()
    writer.writerows(all_results)

# ============================================================
# SAVE ISSUES
# ============================================================

issue_fields = ["Image", "Engine", "Status", "Error"]

with open(issues_path, "w", newline="", encoding="utf-8-sig") as issues_file:
    writer = csv.DictWriter(issues_file, fieldnames=issue_fields)
    writer.writeheader()
    writer.writerows(failed_results)

# ============================================================
# ENGINE SUMMARY
# ============================================================

summary_results = []

print("\n" + "=" * 65)
print("PREPROCESSING EVALUATION RESULTS")
print("=" * 65)

for engine_name in pipelines:

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

    avg_cer = sum(item["CER"] for item in engine_results) / len(engine_results)
    avg_wer = sum(item["WER"] for item in engine_results) / len(engine_results)

    avg_time = sum(
        item["Processing Time (s)"] for item in engine_results
    ) / len(engine_results)

    confidence_values = [
        item["Confidence (%)"]
        for item in engine_results
        if item["Confidence (%)"] != ""
    ]

    avg_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else None
    )

    pairs = list(zip(
        corpus_data[engine_name]["references"],
        corpus_data[engine_name]["hypotheses"],
    ))

    corpus_cer = calculate_corpus_cer(pairs) * 100
    corpus_wer = calculate_corpus_wer(pairs) * 100

    print(f"\n{engine_name}")
    print("Successful images:", len(engine_results))
    print("Failed images:", len(engine_failures))
    print(f"Average CER: {avg_cer:.2f}%")
    print(f"Average WER: {avg_wer:.2f}%")
    print(f"Corpus CER: {corpus_cer:.2f}%")
    print(f"Corpus WER: {corpus_wer:.2f}%")
    print(f"Average processing time: {avg_time:.3f}s")

    summary_results.append({
        "Engine": engine_name,
        "Successful Images": len(engine_results),
        "Failed Images": len(engine_failures),
        "Average Confidence (%)": (
            round(avg_confidence, 2)
            if avg_confidence is not None else ""
        ),
        "Average CER (%)": round(avg_cer, 2),
        "Average WER (%)": round(avg_wer, 2),
        "Corpus CER (%)": round(corpus_cer, 2),
        "Corpus WER (%)": round(corpus_wer, 2),
        "Average Processing Time (s)": round(avg_time, 3),
    })

# ============================================================
# SAVE SUMMARY
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
    "Average Processing Time (s)",
]

with open(summary_path, "w", newline="", encoding="utf-8-sig") as summary_file:
    writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_results)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("PREPROCESSING SUMMARY")
print("=" * 65)
print("Total images:", len(image_files))
print("Missing/empty Ground Truth:", len(missing_ground_truth))
print("Failed OCR attempts:", len(failed_results))
print("Successful OCR attempts:", len(all_results))
print("\nDetailed results:", csv_path)
print("Issues:", issues_path)
print("Summary:", summary_path)
print("=" * 65)
