import cv2
from pathlib import Path

src = Path("data/review_subset_44/data/raw_images/real_samples")
dst = Path("data/outputs/crop_experiment_01")

for name in ["P0081.jpg", "P0083.jpg"]:
    img = cv2.imread(str(src / name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the non-white handwritten region
    mask = gray < 220
    ys, xs = mask.nonzero()

    if len(xs) == 0:
        print(f"{name}: no text detected")
        continue

    margin = 20
    x1 = max(0, xs.min() - margin)
    y1 = max(0, ys.min() - margin)
    x2 = min(img.shape[1], xs.max() + margin + 1)
    y2 = min(img.shape[0], ys.max() + margin + 1)

    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(dst / name), crop)
    print(f"{name}: crop saved, size={crop.shape[1]}x{crop.shape[0]}")
