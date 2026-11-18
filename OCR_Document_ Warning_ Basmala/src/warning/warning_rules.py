
from typing import Any


def analyze_document(
    ocr_result: Any,
    confidence_threshold: float = 0.6,
    document_type: str = "unknown",
) -> dict:
    """Analyze OCR output and identify document quality warnings."""

    if hasattr(ocr_result, "to_dict"):
        ocr_result = ocr_result.to_dict()

    if isinstance(ocr_result, str):
        ocr_result = {"extracted_text": ocr_result, "status": "success"}

    if not isinstance(ocr_result, dict):
        raise TypeError("ocr_result must be a dict, string, or OCRResult")

    text = (ocr_result.get("extracted_text") or "").strip()
    status = ocr_result.get("status", "error")
    confidence = ocr_result.get("confidence")
    scale = ocr_result.get("confidence_scale")

    warnings = []

    if status != "success":
        warnings.append("ocr_failed")

    if not text:
        warnings.append("empty_text")

    if confidence is not None:
        if scale == "0-100":
            confidence = confidence / 100

        if confidence < confidence_threshold:
            warnings.append("low_ocr_confidence")

    if text and len(text) < 10:
        warnings.append("insufficient_text")

    if len(warnings) == 0:
        warning_category = None
    elif len(warnings) == 1:
        warning_category = warnings[0]
    else:
        warning_category = "multiple_warnings"

    return {
        "document_type": document_type,
        "warning_status": "warning" if warnings else "clear",
        "warning_category": warning_category,
        "warnings": warnings,
        "review_required": bool(warnings),
    }
