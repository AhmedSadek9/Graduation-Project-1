
from pathlib import Path

import cv2
import pandas as pd
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ==========================================
# 1. Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "faces" / "ground_truth.xlsx"

DATASET_ROOT = PROJECT_ROOT / "data" / "faces" / "faces"

OUTPUT_PATH = PROJECT_ROOT / "data" / "faces" / "face_detection_results.csv"


# ==========================================
# 2. Load InsightFace
# ==========================================

print("=" * 55)
print("WANAS - FACE RECOGNITION DATASET TEST")
print("=" * 55)

ort.preload_dlls()

print("\nONNX Runtime:", ort.__version__)
print("Available Providers:", ort.get_available_providers())

print("\nLoading buffalo_l model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("Model loaded successfully!")


# ==========================================
# 3. Read Ground Truth
# ==========================================

df = pd.read_excel(GROUND_TRUTH_PATH)

print("\nTotal Ground Truth records:", len(df))


# ==========================================
# 4. Process Images
# ==========================================

results = []

for _, row in df.iterrows():

    original_path = str(row["image_path"])

    # Correct the dataset path:
    # data/faces/person-001/image.jpg
    # becomes data/faces/faces/person-001/image.jpg

    relative_path = Path(
        original_path.replace("data/faces/", "", 1)
    )

    image_path = DATASET_ROOT / relative_path

    true_person = row["true_person"]
    true_status = row["true_status"]
    condition = row["condition"]

    print("\nProcessing:", image_path.name)

    result = {
        "image_path": original_path,
        "true_person": true_person,
        "true_status": true_status,
        "condition": condition,
        "file_exists": image_path.is_file(),
        "face_detected": False,
        "number_of_faces": 0,
        "detection_score": None,
        "embedding_dimension": None,
        "error": None
    }

    try:

        if not image_path.is_file():
            raise FileNotFoundError("Image file not found")

        img = cv2.imread(str(image_path))

        if img is None:
            raise ValueError("OpenCV could not read the image")

        faces = app.get(img)

        result["number_of_faces"] = len(faces)
        result["face_detected"] = len(faces) > 0

        if len(faces) > 0:

            face = max(faces, key=lambda f: f.det_score)

            result["detection_score"] = float(face.det_score)

            result["embedding_dimension"] = len(face.embedding)

            print("Face detected!")
            print("Detection score:", face.det_score)
            print("Embedding shape:", face.embedding.shape)

        else:
            result["error"] = "No face detected"
            print("No face detected!")

    except Exception as e:

        result["error"] = str(e)
        print("ERROR:", e)

    results.append(result)


# ==========================================
# 5. Save Results
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 55)
print("DATASET TEST SUMMARY")
print("=" * 55)

print("Total records:", len(results_df))
print("Existing images:", results_df["file_exists"].sum())
print("Detected faces:", results_df["face_detected"].sum())
print("Missing images:", (~results_df["file_exists"]).sum())
print(
    "Images without detected faces:",
    (
        results_df["file_exists"]
        & ~results_df["face_detected"]
    ).sum()
)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nTest completed!")
