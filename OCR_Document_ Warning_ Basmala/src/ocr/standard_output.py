
"""Shared result schema for OCR engines."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class OCRResult:
    extracted_text: str = ""
    confidence: Optional[float] = None

    # Standard formats: "0-1" or "0-100"
    confidence_scale: Optional[str] = None

    status: str = "error"

    # Standard engine names: "paddle" or "tesseract"
    engine: str = "unknown"

    processing_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)
