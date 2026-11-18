
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from insightface.app import FaceAnalysis

# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent

CSV_PATH = ROOT / "data" / "faces" / "face_detection_results.csv"
ENROLLMENT_PATH = ROOT / "data" / "faces" / "enrollment_embeddings.npz"

OUTPUT_PATH = ROOT / "data" / "faces" / "multiple_faces_diagnosis.csv"

THRESHOLD = 0.50

# ==========================================
# LOAD ENROLLMENT DATABASE
# ==========================================

database = np.load(ENROLLMENT_PATH)

person_ids = database["person_ids"]
person_embeddings = database["embeddings"]

# ==========================================
# LOAD INSIGHTFACE
# ==========================================

ort.preload_dlls()

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ==========================================
# READ MULTIPLE-FACE IMAGES
# ==========================================

df = pd.read_csv(CSV_PATH)

multiple_faces = df[
    pd.to_numeric(df["number_of_faces"], errors="coerce") > 1
].copy()

results = []

print("=" * 70)
print("WANAS - MULTIPLE FACE DIAGNOSIS")
print("=" * 70)

print(f"Images to analyze: {len(multiple_faces)}")

# ==========================================
# PROCESS EVERY FACE
# ==========================================

for _, row in multiple_faces.iterrows():

    relative_path = str(row["image_path"]).replace("/", "\\")
    image_path = ROOT / "data" / "faces" / "faces" / Path(relative_path).relative_to("data/faces")

    true_person = str(row["true_person"])
    condition = str(row["condition"])

    print("\n" + "=" * 70)
    print(f"IMAGE: {image_path.name}")
    print(f"Expected identity: {true_person}")
    print(f"Condition: {condition}")

    if not image_path.is_file():
        print("ERROR: Image not found")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERROR: Could not read image")
        continue

    faces = app.get(image)

    print(f"Detected faces: {len(faces)}")

    for face_index, face in enumerate(faces, start=1):

        embedding = np.asarray(face.embedding, dtype=np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-10

        similarities = []

        for person_id, prototype in zip(
            person_ids, person_embeddings
        ):

            similarity = cosine_similarity(
                embedding, prototype
            )

            similarities.append(
                (str(person_id), similarity)
            )

        similarities.sort(
            key=lambda item: item[1],
            reverse=True
        )

        best_person, best_similarity = similarities[0]

        x1, y1, x2, y2 = face.bbox.astype(int)

        width = x2 - x1
        height = y2 - y1
        area = width * height

        is_known = best_similarity >= THRESHOLD

        correct_identity = (
            best_person == true_person
            if not true_person.startswith("unknown-")
            else False
        )

        print(f"\nFace #{face_index}")
        print(f"Bounding Box: {face.bbox}")
        print(f"Area: {area}")
        print(f"Detection Score: {face.det_score:.4f}")
        print(f"Best Match: {best_person}")
        print(f"Similarity: {best_similarity:.4f}")
        print(f"Decision: {'KNOWN' if is_known else 'UNKNOWN'}")
        print(f"Matches expected identity: {correct_identity}")

        print("Top 3 matches:")

        for rank, (person, score) in enumerate(
            similarities[:3], start=1
        ):
            print(f"{rank}. {person}: {score:.4f}")

        results.append({
            "image_path": relative_path,
            "true_person": true_person,
            "condition": condition,
            "face_number": face_index,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "face_area": area,
            "detection_score": float(face.det_score),
            "best_match": best_person,
            "best_similarity": best_similarity,
            "decision": "KNOWN" if is_known else "UNKNOWN",
            "correct_identity": correct_identity,
            "top1_person": similarities[0][0],
            "top1_similarity": similarities[0][1],
            "top2_person": similarities[1][0],
            "top2_similarity": similarities[1][1],
            "top3_person": similarities[2][0],
            "top3_similarity": similarities[2][1]
        })


# ==========================================
# SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETED")
print("=" * 70)

print(f"Total faces analyzed: {len(results_df)}")
print(f"Results saved to: {OUTPUT_PATH}")
