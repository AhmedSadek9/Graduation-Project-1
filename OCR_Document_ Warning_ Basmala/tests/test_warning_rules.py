from src.warning.warning_rules import analyze_document


def test_successful_document():
    result = analyze_document({
        "extracted_text": "This is a complete document",
        "confidence": 0.9,
        "confidence_scale": "0-1",
        "status": "success",
    })
    assert result["warning_status"] == "clear"
    assert result["review_required"] is False


def test_low_confidence():
    result = analyze_document({
        "extracted_text": "This is a complete document",
        "confidence": 0.3,
        "confidence_scale": "0-1",
        "status": "success",
    })
    assert "low_ocr_confidence" in result["warnings"]


def test_ocr_failed():
    result = analyze_document({
        "extracted_text": "",
        "status": "error",
    })
    assert "ocr_failed" in result["warnings"]
    assert "empty_text" in result["warnings"]
    assert result["review_required"] is True


def test_insufficient_text():
    result = analyze_document({
        "extracted_text": "short",
        "confidence": 0.9,
        "confidence_scale": "0-1",
        "status": "success",
    })
    assert "insufficient_text" in result["warnings"]
