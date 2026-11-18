
import cv2
from pathlib import Path


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image


def resize_image(image, scale=2):
    height, width = image.shape[:2]

    new_size = (width * scale, height * scale)

    resized = cv2.resize(
        image,
        new_size,
        interpolation=cv2.INTER_CUBIC
    )

    return resized


def convert_to_grayscale(image):
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def improve_contrast(image):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    return enhanced


def sharpen_image(image):
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        image,
        1.6,
        blurred,
        -0.5,
        0
    )

    return sharpened


def preprocess_array(original):
    if original is None or getattr(original, "size", 0) == 0:
        raise ValueError("Input image is empty or invalid")

    resized = resize_image(original)

    grayscale = convert_to_grayscale(resized)

    contrast = improve_contrast(grayscale)

    processed = sharpen_image(contrast)

    return {
        "original": original,
        "resized": resized,
        "grayscale": grayscale,
        "contrast": contrast,
        "processed": processed
    }


def preprocess_image(image_path):
    """
    Load an image from a path and apply preprocessing.

    Returns:
        Dictionary containing original and processed image stages.
    """
    original = load_image(image_path)

    return preprocess_array(original)


if __name__ == "__main__":

    input_path = "data/raw_images/medicine_sample.png"

    results = preprocess_image(input_path)

    output_dir = Path("data/processed_images")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed stages only (skip original).
    for name, image in results.items():

        if name == "original":
            continue

        output_path = output_dir / f"{name}.png"

        success = cv2.imwrite(
            str(output_path),
            image
        )

        if success:
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to save: {output_path}")

    print("\nImage preprocessing completed successfully!")

