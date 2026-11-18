# Basmala – OCR Output Contract

## 1. Purpose
This document defines the standard OCR output format used by the Basmala OCR module and its Document Warning component.

The contract provides a consistent structure for OCR results regardless of the selected OCR engine.

## 2. OCRResult Fields

| Field | Type | Description |
|---|---|---|
| extracted_text | string | Text extracted from the document |
| confidence | float or null | OCR confidence score |
| confidence_scale | string or null | Confidence scale: `0-1` or `0-100` |
| status | string | OCR execution status |
| engine | string | OCR engine name: `paddle` or `tesseract` |
| processing_time | float | Processing time in seconds |
| error | string or null | Error message if OCR failed |

## 3. Document Warning Fields

| Field | Type | Description |
|---|---|---|
| document_type | string | Detected document type or `unknown` |
| warning_status | string | `clear` or `warning` |
| warning_category | string or null | Warning rule ID or `multiple_warnings` |
| warnings | array of strings | All triggered warning rules |
| review_required | boolean | Indicates whether manual review is required |

## 4. Successful OCR Example

```json
{
  "extracted_text": "Nexcitual",
  "confidence": 0.9130498170852661,
  "confidence_scale": "0-1",
  "status": "success",
  "engine": "paddle",
  "processing_time": 18.2819,
  "error": null,
  "document_type": "unknown",
  "warning_status": "warning",
  "warning_category": "insufficient_text",
  "warnings": ["insufficient_text"],
  "review_required": true
}
```

## 5. Failed OCR Example

```json
{
  "extracted_text": "",
  "confidence": null,
  "confidence_scale": null,
  "status": "error",
  "engine": "paddle",
  "processing_time": 0.0016,
  "error": "Unreadable or corrupt image",
  "document_type": "unknown",
  "warning_status": "warning",
  "warning_category": "multiple_warnings",
  "warnings": ["ocr_failed", "empty_text"],
  "review_required": true
}
```

## 6. Integration Notes

- OCR engines return the OCRResult structure.
- OCRResult can be converted into a dictionary using `to_dict()`.
- Document Warning analyzes OCR output and adds document warning fields.
- Confidence values must be interpreted according to `confidence_scale`.
- `review_required` indicates human review, not an autonomous medical or legal decision.
- `document_type` currently defaults to `unknown` unless supplied by the caller.
- This contract is a prototype interface and should be reviewed with Ahmed before shared integration.
