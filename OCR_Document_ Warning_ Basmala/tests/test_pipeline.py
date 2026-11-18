
import cv2
import numpy as np

from src.pipeline.ocr_pipeline import OCRPipeline


class FakeEngine:
    def extract_text(self, image_path):
        return {
            "extracted_text": "Paracetamol 500 mg",
            "confidence": 0.95,
            "confidence_scale": "0-1",
            "status": "success",
            "error": None,
        }


class ErrorEngine:
    def extract_text(self, image_path):
        return {
            "extracted_text": "",
            "confidence": None,
            "confidence_scale": None,
            "status": "error",
            "error": "Simulated OCR failure",
        }


class NoTextEngine:
    def extract_text(self, image_path):
        return {
            "extracted_text": "",
            "confidence": 0,
            "confidence_scale": "0-1",
            "status": "no_text",
            "error": None,
        }


def make_pipeline(engine=None, preprocess=False):
    pipeline = OCRPipeline.__new__(OCRPipeline)
    pipeline.engine_name = "fake"
    pipeline.preprocess = preprocess
    pipeline.engine = engine or FakeEngine()
    return pipeline


def make_image(tmp_path):
    image_path = tmp_path / "valid.png"
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert cv2.imwrite(str(image_path), image)
    return image_path


def test_missing_image(tmp_path):
    pipeline = make_pipeline()

    result = pipeline.run(tmp_path / "missing.png")

    assert result["status"] == "error"
    assert "Image not found" in result["error"]
    assert result["extracted_text"] == ""


def test_corrupt_image(tmp_path):
    pipeline = make_pipeline()
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_text("not an image")

    result = pipeline.run(corrupt_file)

    assert result["status"] == "error"
    assert "corrupt" in result["error"].lower()


def test_success_with_ground_truth(tmp_path):
    pipeline = make_pipeline()
    image_path = make_image(tmp_path)

    result = pipeline.run(
        image_path,
        ground_truth="Paracetamol 500 mg",
    )

    assert result["status"] == "success"
    assert result["extracted_text"] == "Paracetamol 500 mg"
    assert result["cer"] == 0.0
    assert result["wer"] == 0.0
    assert result["engine"] == "fake"
    assert "processing_time" in result


def test_engine_error(tmp_path):
    pipeline = make_pipeline(engine=ErrorEngine())
    image_path = make_image(tmp_path)

    result = pipeline.run(image_path)

    assert result["status"] == "error"
    assert result["error"] == "Simulated OCR failure"
    assert result["extracted_text"] == ""


def test_no_text_result(tmp_path):
    pipeline = make_pipeline(engine=NoTextEngine())
    image_path = make_image(tmp_path)

    result = pipeline.run(image_path)

    assert result["status"] == "no_text"
    assert result["extracted_text"] == ""


def test_preprocessing_enabled(tmp_path):
    pipeline = make_pipeline(preprocess=True)
    image_path = make_image(tmp_path)

    result = pipeline.run(image_path)

    assert result["status"] == "success"
    assert result["extracted_text"] == "Paracetamol 500 mg"


def test_preprocessing_temp_file_cleanup(tmp_path):
    pipeline = make_pipeline(preprocess=True)
    image_path = make_image(tmp_path)

    captured_paths = []

    original_extract = pipeline.engine.extract_text

    def capture_path(path):
        captured_paths.append(path)
        return original_extract(path)

    pipeline.engine.extract_text = capture_path

    result = pipeline.run(image_path)

    assert result["status"] == "success"
    assert len(captured_paths) == 1
    assert captured_paths[0] != str(image_path)
    assert not __import__("pathlib").Path(captured_paths[0]).exists()
