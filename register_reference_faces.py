
import cv2

from ai.face.enrollment import FaceEnrollment
from ai.face.config import EMBEDDINGS_FILE, FACES_DIR


PEOPLE = [
    {
        "person_id": "person-001",
        "name": "Person 001",
        "relationship": "Test Person 001",
    },
    {
        "person_id": "person-002",
        "name": "Person 002",
        "relationship": "Test Person 002",
    },
    {
        "person_id": "person-003",
        "name": "Person 003",
        "relationship": "Test Person 003",
    },
]


def main():
    if EMBEDDINGS_FILE.exists():
        raise FileExistsError(
            f"{EMBEDDINGS_FILE} already exists. "
            "Inspect it before registering."
        )

    enrollment = FaceEnrollment()

    for person in PEOPLE:
        image_path = (
            FACES_DIR / person["person_id"] / "referance.png"
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Reference image not found: {image_path}"
            )

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        result = enrollment.enroll(
            image=image,
            person_id=person["person_id"],
            name=person["name"],
            relationship=person["relationship"],
        )

        print(f"\n{person['person_id']}: {result}")

        if not result.get("success"):
            raise RuntimeError(
                f"Enrollment failed for {person['person_id']}"
            )

    print(f"\nEmbeddings saved to: {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
