import cv2
from pathlib import Path
import pandas as pd

rows = []

images_dir = Path("data/review_subset_44/data/raw_images/real_samples")

for image_path in sorted(images_dir.iterdir()):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows.append({
        "Image": image_path.name,
        "Contrast": gray.std(),
        "Sharpness": cv2.Laplacian(
            gray, cv2.CV_64F
        ).var()
    })

df = pd.DataFrame(rows)

print("\n========== IMAGE QUALITY MEASUREMENTS ==========\n")
print(df.to_string(index=False))

print("\n========== STATISTICS ==========\n")
print(df[["Contrast", "Sharpness"]].describe().round(2))

output_path = Path("data/outputs/selective_preprocessing/image_quality.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nSaved to: {output_path}")
