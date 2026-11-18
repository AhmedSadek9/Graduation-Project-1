from pathlib import Path
import cv2
import pandas as pd
import tempfile

from src.ocr.paddle_engine import extract_text
from src.preprocessing.image_processing import preprocess_image
from src.utils.evaluation import calculate_cer, calculate_wer

images_dir = Path("data/review_subset_44/data/raw_images/real_samples")
ground_truth_dir = Path("data/review_subset_44/data/ground_truth/real_samples")
quality_path = Path("data/outputs/selective_preprocessing/image_quality.csv")
output_dir = Path("data/outputs/low_contrast_experiment")

output_dir.mkdir(parents=True, exist_ok=True)

quality = pd.read_csv(quality_path)
quality = quality.set_index("Image")

results = []
issues = []

image_paths = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)

for index, image_path in enumerate(image_paths, start=1):
    try:
        gt_path = ground_truth_dir / f"{image_path.stem}.txt"

        if not gt_path.exists():
            raise FileNotFoundError("Ground truth not found")

        ground_truth = gt_path.read_text(encoding="utf-8").strip()

        if image_path.name not in quality.index:
            raise ValueError("Image quality measurement not found")

        contrast = float(quality.loc[image_path.name, "Contrast"])

        original_result = extract_text(image_path)

        if contrast < 23:
            processed_images = preprocess_image(image_path)

            with tempfile.TemporaryDirectory() as temp_dir:
                processed_path = Path(temp_dir) / image_path.name

                if not cv2.imwrite(
                    str(processed_path),
                    processed_images["processed"]
                ):
                    raise RuntimeError("Could not save processed image")

                selected_result = extract_text(processed_path)

            method = "processed_low_contrast"
        else:
            selected_result = original_result
            method = "original"

        original_text = original_result["extracted_text"]
        selected_text = selected_result["extracted_text"]

        results.append({
            "Image": image_path.name,
            "Contrast": contrast,
            "Selected Method": method,
            "Ground Truth": ground_truth,
            "Original Text": original_text,
            "Original CER": calculate_cer(ground_truth, original_text),
            "Original WER": calculate_wer(ground_truth, original_text),
            "Selected Text": selected_text,
            "Selected CER": calculate_cer(ground_truth, selected_text),
            "Selected WER": calculate_wer(ground_truth, selected_text)
        })

        print(f"[{index}/{len(image_paths)}] {image_path.name}: {method}")

    except Exception as error:
        issues.append({
            "Image": image_path.name,
            "Issue": str(error)
        })
        print(f"Error: {image_path.name}: {error}")

if results:
    df = pd.DataFrame(results)

    csv_path = output_dir / "low_contrast_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n========== LOW CONTRAST EXPERIMENT ==========")
    print("Images evaluated:", len(df))
    print("Original CER:", round(df["Original CER"].mean() * 100, 2), "%")
    print("Selected CER:", round(df["Selected CER"].mean() * 100, 2), "%")
    print("Original WER:", round(df["Original WER"].mean() * 100, 2), "%")
    print("Selected WER:", round(df["Selected WER"].mean() * 100, 2), "%")
    print("Processed images:", (df["Selected Method"] == "processed_low_contrast").sum())
    print("Original images:", (df["Selected Method"] == "original").sum())
    print("Saved:", csv_path)

if issues:
    pd.DataFrame(issues).to_csv(
        output_dir / "issues.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("Issues:", len(issues))

print("\nExperiment completed!")
