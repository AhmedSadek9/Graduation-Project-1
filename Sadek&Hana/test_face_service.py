
from pathlib import Path

import cv2

from app.services.face_recognition import face_service


# ==========================================
# Test Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "data" / "faces" / "faces"

TEST_IMAGES = [
    DATASET_DIR / "person-001" / "person-001_test_angle.jpg",
    DATASET_DIR / "person-002" / "person-002_test_lighting.jpg",
    DATASET_DIR / "person-004" / "person-004_test_angle.jpg",
    DATASET_DIR / "person-005" / "person-005_test_lighting.jpg",
    DATASET_DIR / "person-010" / "person-010_test_lighting.jpg",
    DATASET_DIR / "unknown-001" / "unknown-001_test_normal.jpg",
]


# ==========================================
# Run Tests
# ==========================================

def main():

    print("=" * 60)
    print("WANAS FACE RECOGNITION SERVICE TEST")
    print("=" * 60)

    print("\nInitializing service...")

    face_service.initialize()

    print("\nService initialized successfully!")

    results = []

    for image_path in TEST_IMAGES:

        print("\n" + "-" * 60)
        print(f"Image: {image_path.name}")

        if not image_path.exists():
            print("ERROR: Image not found.")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("ERROR: Could not read image.")
            continue

        try:
            result = face_service.recognize(image)

            print(f"Status: {result['status']}")
            print(f"Person ID: {result['person_id']}")
            print(f"Similarity: {result['similarity']}")
            print(f"Detection Score: {result['detection_score']}")
            print(f"Faces Detected: {result['faces_detected']}")

            results.append({
                "image": image_path.name,
                **result
            })

        except Exception as error:
            print(f"Recognition Error: {error}")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print(f"Total test images: {len(TEST_IMAGES)}")
    print(f"Successful results: {len(results)}")

    for result in results:
        print(
            f"{result['image']} -> "
            f"{result['person_id']} | "
            f"{result['status']} | "
            f"{result['similarity']}"
        )


if __name__ == "__main__":
    main()
