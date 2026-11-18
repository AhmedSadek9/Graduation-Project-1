
from pathlib import Path
import csv
import json

from src.pipeline.ocr_pipeline import OCRPipeline


# Load OCR configuration
config_path = Path("config/ocr_config.json")

if not config_path.is_file():
    raise FileNotFoundError(f"OCR configuration not found: {config_path}")

with config_path.open("r", encoding="utf-8") as f:
    ocr_config = json.load(f)


# Dataset paths
images_dir = Path("data/review_subset_44/data/raw_images/real_samples")
ground_truth_dir = Path("data/review_subset_44/data/ground_truth/real_samples")

# Output directory
output_dir = Path("data/outputs/review_subset_44_results")
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "review_subset_44_results.csv"
issues_path = output_dir / "evaluation_issues.csv"


# Configuration
preprocess_enabled = ocr_config.get("preprocess", False)
paddle_config = ocr_config.get("paddle", {})
tesseract_config = ocr_config.get("tesseract", {})
selected_engine = ocr_config.get("engine", "paddle").lower()

if selected_engine not in ("paddle", "tesseract", "both"):
    raise ValueError(
        "Invalid engine in OCR config. Use paddle, tesseract, or both."
    )


# Select engines
engine_configs = {
    "paddle": ("paddle", paddle_config),
    "tesseract": ("tesseract", tesseract_config),
}

selected_engines = (
    ["paddle", "tesseract"]
    if selected_engine == "both"
    else [selected_engine]
)

pipelines = {}

for engine_key in selected_engines:
    engine_name, engine_config = engine_configs[engine_key]

    pipelines[engine_name] = OCRPipeline(
        engine=engine_key,
        engine_config=engine_config,
        preprocess=preprocess_enabled,
    )


# Find images
image_files = sorted(
    [
        *images_dir.glob("*.jpg"),
        *images_dir.glob("*.jpeg"),
        *images_dir.glob("*.png"),
    ]
)

print("=" * 60)
print("BASMALA OCR - REVIEW SUBSET EVALUATION")
print("=" * 60)
print("Selected engine:", selected_engine)
print("Preprocessing:", preprocess_enabled)
print("Number of images found:", len(image_files))

if not image_files:
    raise FileNotFoundError(f"No images found in: {images_dir}")


all_results = []
issues = []
missing_ground_truth = 0

for image_number, image_path in enumerate(image_files, start=1):
    gt_path = ground_truth_dir / f"{image_path.stem}.txt"

    if not gt_path.is_file():
        print(f"\nMissing Ground Truth: {image_path.name}")
        missing_ground_truth += 1

        issues.append({
            "Image": image_path.name,
            "Engine": "N/A",
            "Status": "Missing Ground Truth",
            "Error": "",
        })
        continue

    reference_text = gt_path.read_text(encoding="utf-8-sig").strip()

    print(f"\n[{image_number}/{len(image_files)}] {image_path.name}")

    for engine_name, pipeline in pipelines.items():
        result = pipeline.run(
            image_path=image_path,
            ground_truth=reference_text,
        )

        status = result.get("status", "unknown")
        extracted_text = result.get("extracted_text", "")
        confidence = result.get("confidence")
        confidence_scale = result.get("confidence_scale")

        if status != "success":
            print(f"{engine_name}: {status} - {result.get('error', '')}")

            issues.append({
                "Image": image_path.name,
                "Engine": engine_name,
                "Status": status,
                "Error": result.get("error", ""),
            })
            continue

        cer = result.get("cer")
        wer = result.get("wer")

        print(
            f"{engine_name} | "
            f"Confidence: {confidence} ({confidence_scale}) | "
            f"CER: {cer * 100:.2f}% | "
            f"WER: {wer * 100:.2f}%"
        )

        all_results.append({
            "Image": image_path.name,
            "Engine": engine_name,
            "Ground Truth": reference_text,
            "Extracted Text": extracted_text,
            "Confidence": confidence,
            "Confidence Scale": confidence_scale,
            "CER": round(cer * 100, 2),
            "WER": round(wer * 100, 2),
            "Status": status,
            "Processing Time": result.get("processing_time"),
        })


# Save successful results
fields = [
    "Image", "Engine", "Ground Truth", "Extracted Text",
    "Confidence", "Confidence Scale", "CER", "WER",
    "Status", "Processing Time",
]

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(all_results)


# Save issues
with issues_path.open("w", newline="", encoding="utf-8-sig") as f:
    issue_fields = ["Image", "Engine", "Status", "Error"]
    writer = csv.DictWriter(f, fieldnames=issue_fields)
    writer.writeheader()
    writer.writerows(issues)


# Summary
print("\n" + "=" * 60)
print("FINAL AVERAGE RESULTS")
print("=" * 60)

for engine_name in pipelines:
    results = [r for r in all_results if r["Engine"] == engine_name]

    if not results:
        print(f"\n{engine_name}: No successful results")
        continue

    avg_cer = sum(r["CER"] for r in results) / len(results)
    avg_wer = sum(r["WER"] for r in results) / len(results)
    avg_time = sum(r["Processing Time"] for r in results) / len(results)

    print(f"\n{engine_name}")
    print("Successful images:", len(results))
    print(f"Average CER: {avg_cer:.2f}%")
    print(f"Average WER: {avg_wer:.2f}%")
    print(f"Average processing time: {avg_time:.4f} sec/image")

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print("Total images:", len(image_files))
print("Missing Ground Truth:", missing_ground_truth)
print("Failed OCR attempts:", len(issues) - missing_ground_truth)
print("Successful OCR attempts:", len(all_results))
print("Results:", csv_path)
print("Issues:", issues_path)
print("=" * 60)

