
import json
from pathlib import Path

import cv2

from ai.face.face_recognition import FaceRecognizer


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    image_path = input(
        "Enter image path (e.g. data/ahmed_test2.jpg): "
    ).strip().strip('"')

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image = cv2.imread(str(image_path))

    if image is None:
        print(json.dumps({
            "success": False,
            "error": "IMAGE_NOT_FOUND",
            "message": "Could not load the specified image."
        }, indent=2))
        return

    recognizer = FaceRecognizer()
    results = recognizer.recognize(image)

    output = {
        "success": True,
        "faces_detected": len(results),
        "results": results
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
