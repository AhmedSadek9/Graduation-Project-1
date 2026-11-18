
"""Reusable image -> validation -> optional preprocessing -> OCR -> assessment pipeline."""

from pathlib import Path
import time
import tempfile

import cv2

from src.ocr.paddle_engine import PaddleEngine
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.standard_output import OCRResult
from src.utils.evaluation import calculate_cer, calculate_wer
from src.warning.warning_rules import analyze_document


class OCRPipeline:
    def __init__(self, engine="paddle", engine_config=None, preprocess=False):
        self.engine_name = engine.lower()
        self.preprocess = preprocess

        cfg = engine_config or {}

        if self.engine_name == "paddle":
            self.engine = PaddleEngine(**cfg)
        elif self.engine_name == "tesseract":
            self.engine = TesseractEngine(**cfg)
        else:
            raise ValueError("engine must be 'paddle' or 'tesseract'")

    def run(self, image_path, ground_truth=None):
        start = time.perf_counter()
        path = Path(image_path)
        temp_path = None

        def make_result(
            extracted_text="",
            confidence=None,
            confidence_scale=None,
            status="error",
            error=None,
        ):
            result = OCRResult(
                extracted_text=extracted_text or "",
                confidence=confidence,
                confidence_scale=confidence_scale,
                status=status,
                engine=self.engine_name,
                processing_time=round(time.perf_counter() - start, 4),
                error=error,
            ).to_dict()

            # Apply warnings to every result, including failures.
            result.update(analyze_document(result))
            return result

        try:
            # Validate file
            if not path.is_file():
                return make_result(error=f"Image not found: {path}")

            image = cv2.imread(str(path))
            if image is None:
                return make_result(error="Unreadable or corrupt image")

            input_for_ocr = str(path)

            # Optional preprocessing
            if self.preprocess:
                from src.preprocessing.image_processing import preprocess_array

                processed = preprocess_array(image)["processed"]

                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp:
                    temp_path = tmp.name

                saved = cv2.imwrite(temp_path, processed)
                if not saved:
                    return make_result(error="Failed to save preprocessed image")

                input_for_ocr = temp_path

            # OCR
            raw_result = self.engine.extract_text(input_for_ocr)

            if not isinstance(raw_result, dict):
                return make_result(error="OCR engine returned invalid result")

            extracted_text = raw_result.get("extracted_text") or ""
            confidence = raw_result.get("confidence")
            confidence_scale = raw_result.get("confidence_scale")
            status = raw_result.get("status", "success")
            error = raw_result.get("error")

            result = make_result(
                extracted_text=extracted_text,
                confidence=confidence,
                confidence_scale=confidence_scale,
                status=status,
                error=error,
            )

            # Evaluation
            if ground_truth is not None and status == "success":
                result["cer"] = calculate_cer(
                    ground_truth, extracted_text
                )
                result["wer"] = calculate_wer(
                    ground_truth, extracted_text
                )

            return result

        except Exception as exc:
            return make_result(
                error=f"{type(exc).__name__}: {exc}"
            )

        finally:
            # Always remove temporary preprocessed image
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)
