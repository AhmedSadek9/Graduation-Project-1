"""Configurable PaddleOCR adapter. Confidence is Paddle's recognition score (0..1)."""
from pathlib import Path
import time
from .standard_output import OCRResult

class PaddleEngine:
    def __init__(self, lang="en", **options):
        self.lang = lang
        self.options = options
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            defaults = dict(use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False)
            defaults.update(self.options)
            self._ocr = PaddleOCR(lang=self.lang, **defaults)
        return self._ocr

    def extract_text(self, image_path):
        start=time.perf_counter()
        path=Path(image_path)
        if not path.is_file():
            return OCRResult(status="error",engine="paddle",
                processing_time=time.perf_counter()-start,error=f"Image not found: {path}").to_dict()
        try:
            pages=self._get_ocr().predict(str(path))
            lines=[]; scores=[]
            for page in pages:
                lines.extend(t for t in page.get("rec_texts",[]) if str(t).strip())
                scores.extend(float(s) for s in page.get("rec_scores",[]) if s is not None)
            text="\n".join(lines)
            conf=sum(scores)/len(scores) if scores else None
            return OCRResult(text,conf,"0-1" if conf is not None else None,
                "success" if text else "no_text", "paddle",
                time.perf_counter()-start).to_dict()
        except Exception as e:
            return OCRResult(status="error",engine="paddle",
                processing_time=time.perf_counter()-start,error=f"{type(e).__name__}: {e}").to_dict()

def extract_text(image_path, lang="en", **options):
    return PaddleEngine(lang=lang,**options).extract_text(image_path)


