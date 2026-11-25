
from pathlib import Path
import cv2
import pandas as pd

import onnxruntime as ort
from insightface.app import FaceAnalysis

ort.preload_dlls()

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))
# Project paths
ROOT = Path(__file__).resolve().parent

CSV_PATH = ROOT / "data" / "faces" / "face_detection_results.csv"
IMAGE_ROOT = ROOT / "data" / "faces" / "faces"

OUTPUT_DIR = ROOT / "data" / "faces" / "review_multiple_faces"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read detection results
df = pd.read_csv(CSV_PATH)

# Select images containing multiple detected faces
multiple = df[
    (df["file_exists"] == True) &
    (df["number_of_faces"] > 1)
]

print("Images with multiple faces:", len(multiple))

for _, row in multiple.iterrows():

    relative_path = str(row["image_path"]).replace(
        "data/faces/", "", 1
    )

    image_path = IMAGE_ROOT / relative_path

    img = cv2.imread(str(image_path))

    if img is None:
        print("Cannot read:", image_path)
        continue

    # Load InsightFace
    # The model is loaded once outside the loop below.
    faces = app.get(img)

    # Draw bounding boxes
    for i, face in enumerate(faces):

        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"Face {i+1}: {face.det_score:.2f}"

        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), img)

    print("Saved:", output_path)

print("\nReview images generated successfully!")
