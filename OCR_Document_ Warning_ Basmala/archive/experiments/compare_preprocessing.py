from pathlib import Path
import cv2
import pandas as pd
import tempfile

from src.ocr.paddle_engine import extract_text
from src.preprocessing.image_processing import preprocess_image
from src.utils.evaluation import calculate_cer, calculate_wer


images_dir = Path("data/review_subset_44/data/raw_images/real_samples")
ground_truth_dir = Path("data/review_subset_44/data/ground_truth/real_samples")
output_dir = Path("data/outputs/preprocessing_comparison")

output_dir.mkdir(parents=True, exist_ok=True)

results = []
issues = []

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

image_paths = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in image_extensions
)

print(f"Found {len(image_paths)} images.")
print("Starting original vs processed comparison...\n")


for index, image_path in enumerate(image_paths, start=1):

    ground_truth_path = ground_truth_dir / f"{image_path.stem}.txt"

    if not ground_truth_path.exists():
        issues.append({
            "Image": image_path.name,
            "Issue": "Ground truth file not found"
        })
        print(f"[{index}/{len(image_paths)}] Missing GT: {image_path.name}")
        continue

    try:
        ground_truth = ground_truth_path.read_text(
            encoding="utf-8"
        ).strip()

        original_result = extract_text(image_path)

        processed_images = preprocess_image(image_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            processed_path = Path(temp_dir) / image_path.name

            saved = cv2.imwrite(
                str(processed_path),
                processed_images["processed"]
            )

            if not saved:
                raise RuntimeError("Could not save processed image")

            processed_result = extract_text(processed_path)

        original_text = original_result["extracted_text"]
        processed_text = processed_result["extracted_text"]

        results.append({
            "Image": image_path.name,
            "Ground Truth": ground_truth,
            "Original Text": original_text,
            "Original Confidence": original_result["confidence"],
            "Original CER": calculate_cer(ground_truth, original_text),
            "Original WER": calculate_wer(ground_truth, original_text),
            "Original Status": original_result["status"],
            "Processed Text": processed_text,
            "Processed Confidence": processed_result["confidence"],
            "Processed CER": calculate_cer(ground_truth, processed_text),
            "Processed WER": calculate_wer(ground_truth, processed_text),
            "Processed Status": processed_result["status"]
        })

        print(f"[{index}/{len(image_paths)}] Completed: {image_path.name}")

    except Exception as error:
        issues.append({
            "Image": image_path.name,
            "Issue": str(error)
        })
        print(f"[{index}/{len(image_paths)}] Error: {image_path.name} - {error}")


if results:
    df = pd.DataFrame(results)

    csv_path = output_dir / "preprocessing_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n========== COMPARISON SUMMARY ==========")
    print(f"Images evaluated: {len(df)}")

    print("Original average CER:", round(df["Original CER"].mean(), 2), "%")
    print("Processed average CER:", round(df["Processed CER"].mean(), 2), "%")

    print("Original average WER:", round(df["Original WER"].mean(), 2), "%")
    print("Processed average WER:", round(df["Processed WER"].mean(), 2), "%")

    print("Original average confidence:", round(df["Original Confidence"].mean(), 2), "%")
    print("Processed average confidence:", round(df["Processed Confidence"].mean(), 2), "%")

    print(f"\nResults saved to: {csv_path}")

else:
    print("\nNo comparison results were generated.")


if issues:
    issues_path = output_dir / "comparison_issues.csv"
    pd.DataFrame(issues).to_csv(
        issues_path,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"Issues saved to: {issues_path}")

print("\nPreprocessing comparison completed!")
