
from src.ocr.standard_output import OCRResult


def test_default_output():
    result = OCRResult()
    output = result.to_dict()

    assert output["extracted_text"] == ""
    assert output["confidence"] is None
    assert output["confidence_scale"] is None
    assert output["status"] == "error"
    assert output["engine"] == "unknown"
    assert output["processing_time"] == 0.0
    assert output["error"] is None


def test_success_output():
    result = OCRResult(
        extracted_text="Paracetamol 500 mg",
        confidence=0.95,
        confidence_scale="0-1",
        status="success",
        engine="paddle",
        processing_time=1.25,
    )

    output = result.to_dict()

    assert output["extracted_text"] == "Paracetamol 500 mg"
    assert output["confidence"] == 0.95
    assert output["confidence_scale"] == "0-1"
    assert output["status"] == "success"
    assert output["engine"] == "paddle"
    assert output["processing_time"] == 1.25
    assert output["error"] is None


def test_error_output():
    result = OCRResult(
        status="error",
        engine="tesseract",
        error="Image not found",
    )

    output = result.to_dict()

    assert output["status"] == "error"
    assert output["engine"] == "tesseract"
    assert output["error"] == "Image not found"
