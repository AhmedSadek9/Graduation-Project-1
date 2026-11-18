from pathlib import Path
import cv2
import pandas as pd
import tempfile

from src.ocr.paddle_engine import extract_text
from src.preprocessing.image_processing import preprocess_array
from src.utils.evaluation import calculate_cer, calculate_wer


# PATHS
images_dir = Path("data/review_subset_44/data/raw_images/real_samples")
ground_truth_dir = Path("data/review_subset_44/data/ground_truth/real_samples")
output_dir = Path("data/outputs/selective_preprocessing_v4")
output_dir.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# PREPROCESSING METHODS
def create_variants(image):
    stages = preprocess_array(image)

    return {
        "original": image,
        "clahe": stages["contrast"],
        "sharpen": cv2.addWeighted(stages["grayscale"], 1.6, cv2.GaussianBlur(stages["grayscale"], (0, 0), 1.0), -0.5, 0),
        "clahe_sharpen": stages["processed"],
    }


def run_ocr_on_array(image, image_name):
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / image_name

        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError("Could not save temporary image")

        return extract_text(image_path)


# EXPERIMENT
results = []
issues = []

image_paths = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
)

print(f"Found {len(image_paths)} images.")
print("Starting preprocessing variants experiment v4...\n")

for index, image_path in enumerate(image_paths, start=1):
    gt_path = ground_truth_dir / f"{image_path.stem}.txt"

    if not gt_path.exists():
        issues.append({"Image": image_path.name, "Issue": "Ground truth not found"})
        continue

    try:
        ground_truth = gt_path.read_text(encoding="utf-8").strip()
        original_image = cv2.imread(str(image_path))

        if original_image is None:
            raise ValueError("Could not read image")

        variants = create_variants(original_image)
        row = {"Image": image_path.name, "Ground Truth": ground_truth}

        for method, variant in variants.items():
            if method == "original":
                ocr_result = extract_text(image_path)
            else:
                ocr_result = run_ocr_on_array(variant, image_path.name)

            text = ocr_result.get("extracted_text", "")
            confidence = float(ocr_result.get("confidence") or 0)

            row[f"{method}_Text"] = text
            row[f"{method}_Confidence"] = confidence
            row[f"{method}_CER"] = calculate_cer(ground_truth, text)
            row[f"{method}_WER"] = calculate_wer(ground_truth, text)

        results.append(row)

        print(f"[{index}/{len(image_paths)}] {image_path.name} completed")

    except Exception as error:
        issues.append({"Image": image_path.name, "Issue": str(error)})
        print(f"Error: {image_path.name} - {error}")


# SAVE RESULTS
if results:
    df = pd.DataFrame(results)
    results_path = output_dir / "preprocessing_variants_v4.csv"
    df.to_csv(results_path, index=False, encoding="utf-8-sig")

    print("\n========== V4 SUMMARY ==========")
    for method in ["original", "clahe", "sharpen", "clahe_sharpen"]:
        print(
            f"{method}: CER={df[f'{method}_CER'].mean()*100:.2f}% | "
            f"WER={df[f'{method}_WER'].mean()*100:.2f}%"
        )

    print(f"\nSaved: {results_path}")

if issues:
    issues_path = output_dir / "v4_issues.csv"
    pd.DataFrame(issues).to_csv(issues_path, index=False, encoding="utf-8-sig")
    print(f"Issues: {len(issues)} | Saved: {issues_path}")

print("\nPreprocessing variants v4 completed!")
