from pathlib import Path
import cv2
import pandas as pd
import tempfile

from src.ocr.paddle_engine import extract_text
from src.preprocessing.image_processing import preprocess_image
from src.utils.evaluation import calculate_cer, calculate_wer


images_dir = Path("data/review_subset_44/data/raw_images/real_samples")
ground_truth_dir = Path("data/review_subset_44/data/ground_truth/real_samples")
output_dir = Path("data/outputs/selective_preprocessing")

output_dir.mkdir(parents=True, exist_ok=True)

results = []
issues = []

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

image_paths = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in extensions
)


def needs_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()
    sharpness = cv2.Laplacian(
        gray, cv2.CV_64F
    ).var()

    return contrast < 45 or sharpness < 100


print(f"Found {len(image_paths)} images.")
print("Starting selective preprocessing experiment...\n")


for index, image_path in enumerate(image_paths, start=1):

    gt_path = ground_truth_dir / f"{image_path.stem}.txt"

    if not gt_path.exists():
        issues.append({
            "Image": image_path.name,
            "Issue": "Ground truth not found"
        })
        continue

    try:
        ground_truth = gt_path.read_text(
            encoding="utf-8"
        ).strip()

        original_image = cv2.imread(str(image_path))

        if original_image is None:
            raise ValueError("Could not read image")

        original_result = extract_text(image_path)

        use_preprocessing = needs_preprocessing(original_image)

        if use_preprocessing:
            processed_images = preprocess_image(image_path)

            with tempfile.TemporaryDirectory() as temp_dir:
                processed_path = Path(temp_dir) / image_path.name

                saved = cv2.imwrite(
                    str(processed_path),
                    processed_images["processed"]
                )

                if not saved:
                    raise RuntimeError("Could not save processed image")

                selective_result = extract_text(processed_path)

            selected_method = "processed"

        else:
            selective_result = original_result
            selected_method = "original"

        original_text = original_result["extracted_text"]
        selective_text = selective_result["extracted_text"]

        results.append({
            "Image": image_path.name,
            "Selected Method": selected_method,
            "Ground Truth": ground_truth,

            "Original Text": original_text,
            "Original Confidence": original_result["confidence"],
            "Original CER": calculate_cer(
                ground_truth, original_text
            ),
            "Original WER": calculate_wer(
                ground_truth, original_text
            ),

            "Selective Text": selective_text,
            "Selective Confidence": selective_result["confidence"],
            "Selective CER": calculate_cer(
                ground_truth, selective_text
            ),
            "Selective WER": calculate_wer(
                ground_truth, selective_text
            )
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}: {selected_method}"
        )

    except Exception as error:
        issues.append({
            "Image": image_path.name,
            "Issue": str(error)
        })

        print(f"Error: {image_path.name} - {error}")


if results:
    df = pd.DataFrame(results)

    results_path = output_dir / "selective_comparison.csv"

    df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== SELECTIVE COMPARISON ==========")
    print("Images evaluated:", len(df))

    print(
        "Original average CER:",
        round(df["Original CER"].mean() * 100, 2), "%"
    )

    print(
        "Selective average CER:",
        round(df["Selective CER"].mean() * 100, 2), "%"
    )

    print(
        "Original average WER:",
        round(df["Original WER"].mean() * 100, 2), "%"
    )

    print(
        "Selective average WER:",
        round(df["Selective WER"].mean() * 100, 2), "%"
    )

    print(
        "Images using preprocessing:",
        (df["Selected Method"] == "processed").sum()
    )

    print(
        "Images using original:",
        (df["Selected Method"] == "original").sum()
    )

    print("\nSaved:", results_path)


if issues:
    pd.DataFrame(issues).to_csv(
        output_dir / "selective_issues.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Issues:", len(issues))


print("\nSelective preprocessing experiment completed!")
